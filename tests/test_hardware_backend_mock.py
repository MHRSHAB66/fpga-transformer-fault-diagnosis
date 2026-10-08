"""Mocked HardwareBackend + standalone deployment-bundle tests (no board).

A fake vendor `pynq` module (Overlay/allocate) is injected so the real
HardwareBackend transport logic is exercised without hardware. The backend under
test is loaded from the DEPLOYMENT BUNDLE driver as a standalone top-level module,
which simultaneously proves the bundle is self-contained (issue 1 + 2).

NOTE: these tests validate transport *logic and contract* only. They do NOT
validate physical DMA/UART operation on a real board.
"""
import hashlib
import importlib.util
import subprocess
import sys
import types
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
BUNDLE = REPO_ROOT / "artifacts" / "deploy" / "seq256_z7020_pynq"
DRIVER = BUNDLE / "driver" / "fault_classifier.py"
BIT = BUNDLE / "overlay" / "design_1_wrapper.bit"
PKG_HWH = REPO_ROOT / "artifacts" / "results" / "seq256_dma_system_dsp_margin_z7020" / "bitstream" / "design_1.hwh"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _make_fake_pynq(ip_dict, out_logits=None):
    """Build a fake `pynq` module recording transport events."""
    events = []
    ip_writes = []

    def allocate(shape, dtype):
        return np.zeros(shape, dtype=dtype)

    class FakeChannel:
        def __init__(self, name):
            self.name = name
            self.buf = None

        def transfer(self, buf):
            events.append((self.name, "transfer"))
            self.buf = buf

        def wait(self):
            events.append((self.name, "wait"))
            if self.name == "recv" and out_logits is not None and self.buf is not None:
                self.buf[:] = np.asarray(out_logits, dtype=self.buf.dtype)

    class FakeDMA:
        def __init__(self):
            self.sendchannel = FakeChannel("send")
            self.recvchannel = FakeChannel("recv")

    class FakeIP:
        def write(self, off, val):
            ip_writes.append((off, val))

    class FakeOverlay:
        def __init__(self, bit):
            self.bit = bit
            self.ip_dict = ip_dict
            self._dma = FakeDMA()
            self._ip = FakeIP()

        def __getattr__(self, name):
            if name == "axi_dma_0" and "axi_dma_0" in self.ip_dict:
                return self._dma
            if name == "transformer_top_0" and "transformer_top_0" in self.ip_dict:
                return self._ip
            raise AttributeError(name)

    mod = types.ModuleType("pynq")
    mod.Overlay = FakeOverlay
    mod.allocate = allocate
    mod._events = events
    mod._ip_writes = ip_writes
    return mod


def _load_bundle_driver(fake_pynq):
    """Load the bundle driver as a standalone module with the fake pynq active."""
    sys.modules["pynq"] = fake_pynq
    spec = importlib.util.spec_from_file_location("bundle_fault_classifier", DRIVER)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["bundle_fault_classifier"] = mod
    spec.loader.exec_module(mod)
    return mod


class DeploymentBundleStaticTests(unittest.TestCase):
    def test_bundle_layout_exists(self):
        for rel in ["overlay/design_1_wrapper.bit", "overlay/design_1_wrapper.hwh",
                    "overlay/seq256_dma_system.xsa", "driver/fault_classifier.py",
                    "driver/run_demo.py", "driver/frame_protocol.py",
                    "config/model_manifest.json", "config/integer_config_scales.json",
                    "golden/golden_in.txt", "golden/golden_expected.txt",
                    "README.md", "MANIFEST.sha256"]:
            self.assertTrue((BUNDLE / rel).exists(), f"missing bundle file {rel}")

    def test_bit_and_hwh_basenames_match(self):
        bit = BUNDLE / "overlay" / "design_1_wrapper.bit"
        hwh = bit.with_suffix(".hwh")
        self.assertTrue(hwh.exists(), "same-basename HWH must exist next to BIT")
        self.assertEqual(bit.stem, hwh.stem)

    def test_renamed_hwh_matches_original_same_build_hwh(self):
        self.assertEqual(_sha256(BUNDLE / "overlay" / "design_1_wrapper.hwh"),
                         _sha256(PKG_HWH),
                         "bundle HWH must be byte-identical to the same-build HWH")

    def test_manifest_verifies_and_has_no_absolute_paths(self):
        text = (BUNDLE / "MANIFEST.sha256").read_text()
        self.assertNotRegex(text, r"[A-Za-z]:\\|/home/|/Users/|mine_pc")
        checked = 0
        for line in text.splitlines():
            if not line.strip():
                continue
            digest, rel = line.split(" *./")
            path = BUNDLE / rel
            # The large overlay binaries (.bit/.xsa) are manifest-only (gitignored,
            # recorded in artifacts/manifests/artifact_manifest.json), so they are
            # absent on a fresh clone -- verify them only when present.
            if not path.exists():
                continue
            self.assertEqual(_sha256(path), digest, f"hash mismatch for {rel}")
            checked += 1
        self.assertGreater(checked, 0, "no bundle files present to verify")


class MockedHardwareBackendTests(unittest.TestCase):
    def setUp(self):
        self._saved = {k: sys.modules.get(k) for k in ("pynq", "bundle_fault_classifier")}
        self.ip_dict = {"axi_dma_0": {"phys_addr": 0x41E00000},
                        "transformer_top_0": {"phys_addr": 0x40000000}}

    def tearDown(self):
        for k, v in self._saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v

    def _backend(self, out_logits=None, ip_dict=None):
        fake = _make_fake_pynq(ip_dict if ip_dict is not None else self.ip_dict, out_logits)
        fc = _load_bundle_driver(fake)
        return fc, fake

    def test_overlay_receives_bit_path_and_looks_up_instances(self):
        fc, fake = self._backend()
        hb = fc.HardwareBackend(str(BIT))
        self.assertEqual(Path(hb.overlay.bit), BIT)
        self.assertIs(hb.dma, hb.overlay._dma)
        self.assertIs(hb.ip, hb.overlay._ip)

    def test_missing_dma_raises(self):
        fc, _ = self._backend(ip_dict={"transformer_top_0": {}})
        with self.assertRaises(RuntimeError):
            fc.HardwareBackend(str(BIT))

    def test_missing_transformer_ip_raises(self):
        fc, _ = self._backend(ip_dict={"axi_dma_0": {}})
        with self.assertRaises(RuntimeError):
            fc.HardwareBackend(str(BIT))

    def test_missing_hwh_basename_raises(self):
        fc, _ = self._backend()
        # a .bit whose sibling .hwh does not exist
        with self.assertRaises(FileNotFoundError):
            fc.HardwareBackend(str(BUNDLE / "overlay" / "seq256_dma_system.xsa"))

    def test_buffers_allocated_with_expected_shapes(self):
        fc, _ = self._backend()
        hb = fc.HardwareBackend(str(BIT))
        self.assertEqual((hb._in.shape, hb._in.dtype), ((8192,), np.dtype(np.int32)))
        self.assertEqual((hb._out.shape, hb._out.dtype), ((4,), np.dtype(np.int32)))

    def test_control_register_autostart_value(self):
        fc, fake = self._backend()
        fc.HardwareBackend(str(BIT), auto_restart=True)
        self.assertIn((0x00, 0x81), fake._ip_writes)

    def test_control_register_oneshot_value(self):
        fc, fake = self._backend()
        fc.HardwareBackend(str(BIT), auto_restart=False)
        self.assertIn((0x00, 0x01), fake._ip_writes)

    def test_recv_armed_before_send_and_row_major_serialization(self):
        fc, fake = self._backend(out_logits=[5, -3, 10, -7])
        hb = fc.HardwareBackend(str(BIT))
        codes = np.arange(1024 * 8, dtype=np.int64).reshape(1024, 8)
        logits = hb.infer_codes(codes)
        # recv transfer must precede send transfer (arm S2MM before MM2S)
        order = [e for e in fake._events if e[1] == "transfer"]
        self.assertEqual(order[0], ("recv", "transfer"))
        self.assertEqual(order[1], ("send", "transfer"))
        # row-major [sample, channel]
        self.assertEqual(hb._in[8], codes[1, 0])
        self.assertEqual(hb._in[8 * 8 + 3], codes[8, 3])
        # exact signed logits + argmax
        self.assertEqual(logits.tolist(), [5, -3, 10, -7])
        self.assertEqual(int(np.argmax(logits)), 2)

    def test_negative_logits_returned_exactly(self):
        fc, _ = self._backend(out_logits=[-214, 46, -163, 54])
        hb = fc.HardwareBackend(str(BIT))
        _, logits = fc.FaultClassifier(backend=hb).infer_from_codes(
            np.zeros((1024, 8), dtype=np.int64))
        self.assertEqual(logits.tolist(), [-214, 46, -163, 54])

    def test_bundle_resolves_config_and_golden_relative_to_itself(self):
        fc, _ = self._backend()
        self.assertEqual(fc.manifest_path(), BUNDLE / "config" / "model_manifest.json")
        self.assertEqual(fc.golden_dir(), BUNDLE / "golden")


class StandaloneImportSubprocessTests(unittest.TestCase):
    """Fresh-process proofs that the hardware path needs no torch/repo."""

    def _run(self, script: str):
        return subprocess.run([sys.executable, "-c", script],
                              capture_output=True, text=True, cwd=str(REPO_ROOT))

    def test_hardware_path_initialises_without_torch(self):
        script = f"""
import sys, types, importlib.util
from pathlib import Path
import numpy as np
fake = types.ModuleType('pynq')
class Ch:
    def transfer(self,b): self.b=b
    def wait(self): pass
class DMA:
    def __init__(self): self.sendchannel=Ch(); self.recvchannel=Ch()
class IP:
    def write(self,o,v): pass
class Ov:
    def __init__(self,bit): self.ip_dict={{'axi_dma_0':{{}}, 'transformer_top_0':{{}}}}; self._d=DMA(); self._i=IP()
    def __getattr__(self,n):
        if n=='axi_dma_0': return self._d
        if n=='transformer_top_0': return self._i
        raise AttributeError(n)
fake.Overlay=Ov; fake.allocate=lambda shape,dtype: np.zeros(shape,dtype=dtype)
sys.modules['pynq']=fake
spec=importlib.util.spec_from_file_location('fc', r'{DRIVER.as_posix()}')
m=importlib.util.module_from_spec(spec); sys.modules['fc']=m; spec.loader.exec_module(m)
hb=m.HardwareBackend(r'{BIT.as_posix()}')
assert 'torch' not in sys.modules, 'torch must NOT be imported by the hardware path'
assert 'src' not in sys.modules
assert hb._in.shape==(8192,) and hb._out.shape==(4,)
print('OK-NO-TORCH')
"""
        r = self._run(script)
        self.assertIn("OK-NO-TORCH", r.stdout, msg=r.stderr)
        self.assertEqual(r.returncode, 0, msg=r.stderr)

    def test_deployment_run_demo_imports_outside_repo_package(self):
        # Load driver/run_demo.py as a standalone script module (no `pynq`/`host`
        # packages on the path) -> its adjacent-import branch must succeed.
        script = f"""
import sys, importlib.util
from pathlib import Path
drv = Path(r'{(BUNDLE / 'driver').as_posix()}')
sys.path.insert(0, str(drv))   # board runs from inside driver/
spec=importlib.util.spec_from_file_location('run_demo', str(drv/'run_demo.py'))
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
assert hasattr(m,'main') and hasattr(m,'load_golden_windows')
w=m.load_golden_windows(); assert len(w)==16 and w[0].shape==(1024,8)
print('OK-RUNDEMO-STANDALONE')
"""
        r = self._run(script)
        self.assertIn("OK-RUNDEMO-STANDALONE", r.stdout, msg=r.stderr)
        self.assertEqual(r.returncode, 0, msg=r.stderr)


if __name__ == "__main__":
    unittest.main()
