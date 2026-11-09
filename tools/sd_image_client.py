"""Image the board's whole SD card over SSH, one verified chunk per session.

Each chunk is read on the board with `dd` and streamed to the PC; the board
also prints the MD5 of exactly the bytes it sent (via `tee`), and the PC
compares it with its own MD5 of what arrived. A chunk cut by an Ethernet link
drop is fetched again. Output: a gzip-compressed raw image plus a JSON record
(device size, SHA-256 of the raw image, per-chunk MD5s, retries).

The card is read while Linux runs (the root filesystem cannot be remounted
read-only), so the image is crash-consistent at best: on restore, run
`fsck` on partition 2 before first boot. Keep the board idle while it runs.

    python -m tools.sd_image_client --out D:/.../zturn_sd_20261109.img.gz
    python -m tools.sd_image_client --out <same> --start-chunk 3   # resume after a link loss
    python -m tools.sd_image_client --out <same> --verify          # SHA-256 of the whole image

A resumed run appends a new gzip member (gzip readers concatenate members), so
an interrupted run must have ended cleanly (it closes its member when it gives
up). Each run writes <out>.chunks<first>.json with its chunk MD5s.
"""
import argparse
import gzip
import hashlib
import json
import subprocess
import time
from pathlib import Path

BS = 4 << 20                 # dd block size
CHUNK_BLOCKS = 256           # 1 GiB per SSH session
DEVICE = "/dev/mmcblk0"


def ssh_base(host, key):
    return ["ssh", "-i", key, "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
            "-o", "ServerAliveInterval=10", "-o", "ServerAliveCountMax=3", "root@" + host]


def fetch_chunk(base, index, expected):
    cmd = ("dd if=%s bs=%d skip=%d count=%d 2>/dev/null | tee >(md5sum >&2)"
           % (DEVICE, BS, index * CHUNK_BLOCKS, CHUNK_BLOCKS))
    p = subprocess.run(base + ["bash", "-c", "'%s'" % cmd], capture_output=True, timeout=1800)
    data = p.stdout
    board_md5 = p.stderr.decode(errors="replace").split()[0] if p.stderr.strip() else ""
    if p.returncode != 0 or len(data) != expected:
        raise OSError("rc=%s got %d of %d bytes" % (p.returncode, len(data), expected))
    pc_md5 = hashlib.md5(data).hexdigest()
    if pc_md5 != board_md5:
        raise OSError("md5 mismatch board=%s pc=%s" % (board_md5, pc_md5))
    return data, pc_md5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="192.168.2.99")
    ap.add_argument("--key", default=str(Path.home() / ".ssh" / "id_zturn"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--start-chunk", type=int, default=0)
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    if args.verify:
        verify(args.out)
        return
    base = ssh_base(args.host, args.key)

    size = int(subprocess.run(base + ["blockdev", "--getsize64", DEVICE], capture_output=True,
                              check=True, text=True).stdout.strip())
    chunk_bytes = BS * CHUNK_BLOCKS
    n_chunks = -(-size // chunk_bytes)
    md5s, retries, t0 = [], 0, time.time()
    mode = "ab" if args.start_chunk else "wb"
    with gzip.open(args.out, mode, compresslevel=1) as out:
        for i in range(args.start_chunk, n_chunks):
            expected = min(chunk_bytes, size - i * chunk_bytes)
            for attempt in range(20):
                try:
                    data, md5 = fetch_chunk(base, i, expected)
                    break
                except Exception as e:     # link loss, timeout, memory pressure: retry the chunk
                    retries += 1
                    print("chunk %d retry %d: %s" % (i, attempt + 1, e), flush=True)
                    time.sleep(10)
            else:
                raise SystemExit("giving up at chunk %d" % i)
            out.write(data)
            md5s.append(md5)
            done = i * chunk_bytes + expected
            print("chunk %d/%d  %.2f GB  %.1f MB/s" % (i + 1, n_chunks, done / 1e9,
                  done / max(time.time() - t0, 1e-6) / 1e6), flush=True)
    record = {"host": args.host, "device": DEVICE, "device_bytes": size,
              "first_chunk": args.start_chunk, "chunk_bytes": chunk_bytes,
              "chunk_md5": md5s, "retries": retries, "seconds": round(time.time() - t0, 1),
              "taken_live": True, "output": args.out,
              "restore": "gunzip, write to a >=32 GB card, then fsck.ext4 -f partition 2"}
    with open("%s.chunks%d.json" % (args.out, args.start_chunk), "w", encoding="utf-8") as f:
        json.dump(record, f, indent=1)
    print(json.dumps({k: v for k, v in record.items() if k != "chunk_md5"}))


def verify(out_path):
    """Decompress the whole image once: size, SHA-256 and every chunk MD5 vs the run records."""
    recorded = {}
    for rec_path in sorted(Path(out_path).parent.glob(Path(out_path).name + ".chunks*.json")):
        rec = json.loads(rec_path.read_text(encoding="utf-8"))
        for k, md5 in enumerate(rec["chunk_md5"]):
            recorded[rec["first_chunk"] + k] = md5
        size, chunk_bytes = rec["device_bytes"], rec["chunk_bytes"]
    sha, total, i = hashlib.sha256(), 0, 0
    bad, unrecorded = [], []
    with gzip.open(out_path, "rb") as f:
        while True:
            data = f.read(chunk_bytes)
            if not data:
                break
            sha.update(data)
            total += len(data)
            if i not in recorded:
                unrecorded.append(i)
            elif hashlib.md5(data).hexdigest() != recorded[i]:
                bad.append(i)
            i += 1
    result = {"image": out_path, "raw_bytes": total, "device_bytes": size,
              "complete": total == size, "chunks": i, "chunk_md5_mismatches": bad,
              "chunks_without_record": unrecorded,
              "note": "chunks without a record were MD5-checked board vs PC during the "
                      "transfer, but their run ended before writing its record",
              "image_sha256_raw": sha.hexdigest()}
    with open(out_path + ".verify.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1)
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
