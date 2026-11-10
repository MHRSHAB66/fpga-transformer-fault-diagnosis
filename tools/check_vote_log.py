"""Compare "VOTE <n> <class> <count>" lines of a testbench or board log with the
Python reference (host/temporal_vote.py) run over the expected window classes.

    python -m tools.check_vote_log <log> <golden_expected.txt> <k>
"""
import sys

from host.temporal_vote import vote_sequence


def main():
    log, expected, k = sys.argv[1], sys.argv[2], int(sys.argv[3])
    classes = [int(line.split()[2]) for line in open(expected).read().splitlines()[1:] if line.strip()]
    got = [tuple(int(x) for x in line.split()[2:4]) for line in open(log, errors="replace")
           if line.startswith("VOTE ")]
    ref = vote_sequence(classes, k)
    same = sum(g == r for g, r in zip(got, ref))
    print("vote outputs equal to the Python reference: %d / %d (log lines %d)" % (same, len(ref), len(got)))
    sys.exit(0 if same == len(ref) == len(got) else 1)


if __name__ == "__main__":
    main()
