"""Negative checks for the resource verifier; no CUDA toolkit or device needed."""

from pathlib import Path
import tempfile
import unittest

import build_cuda


class ResourceVerificationTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="jneuro-cuda-verifier-")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.ptx = self.directory / "train.ptx"
        # Structural verifier fixture only; this is deliberately not executable PTX.
        self.ptx.write_text(".target sm_75\n" + "".join(
            f".visible .entry {name}()\n" for name in build_cuda.KERNELS), encoding="utf-8")
        self.metadata = build_cuda.source_metadata() | {
            "ptx.sha256": build_cuda.sha256(self.ptx), "cuda.nvcc": build_cuda.NVCC_VERSION}
        self.write_metadata()

    def write_metadata(self):
        (self.directory / "train.properties").write_text("".join(
            f"{key}={value}\n" for key, value in self.metadata.items()), encoding="utf-8")

    def replace_ptx(self, before, after):
        self.ptx.write_text(self.ptx.read_text(encoding="utf-8").replace(before, after), encoding="utf-8")
        self.metadata["ptx.sha256"] = build_cuda.sha256(self.ptx)
        self.write_metadata()

    def test_valid_matching_bundle(self):
        build_cuda.verify(self.directory)

    def test_source_builder_and_compilation_options_cannot_be_stale(self):
        for key in build_cuda.source_metadata():
            with self.subTest(key=key):
                original = self.metadata[key]
                self.metadata[key] = "stale"
                self.write_metadata()
                with self.assertRaisesRegex(ValueError, key.replace(".", r"\.")):
                    build_cuda.verify(self.directory)
                self.metadata[key] = original

    def test_ptx_corruption_is_rejected(self):
        with self.ptx.open("a", encoding="utf-8") as output:
            output.write("changed\n")
        with self.assertRaisesRegex(ValueError, r"ptx\.sha256"):
            build_cuda.verify(self.directory)

    def test_missing_kernels_are_rejected_even_with_matching_hash(self):
        for name in build_cuda.KERNELS:
            with self.subTest(kernel=name):
                self.replace_ptx(f".entry {name}(", f".entry absent_{name}(")
                with self.assertRaisesRegex(ValueError, f"Missing CUDA kernel: {name}"):
                    build_cuda.verify(self.directory)
                self.replace_ptx(f".entry absent_{name}(", f".entry {name}(")

    def test_changed_architecture_is_rejected_even_with_matching_hash(self):
        self.replace_ptx(".target sm_75", ".target sm_86")
        with self.assertRaisesRegex(ValueError, "target sm_75"):
            build_cuda.verify(self.directory)

    def test_unpinned_compiler_is_rejected(self):
        self.metadata["cuda.nvcc"] = "Cuda compilation tools, release 13.0, V13.0.48"
        self.write_metadata()
        with self.assertRaisesRegex(ValueError, "compiler provenance"):
            build_cuda.verify(self.directory)

    def test_duplicate_metadata_is_rejected(self):
        with (self.directory / "train.properties").open("a", encoding="utf-8") as output:
            output.write("kernel.abi=1\n")
        with self.assertRaisesRegex(ValueError, "duplicate property"):
            build_cuda.verify(self.directory)

    def test_missing_resource_is_rejected(self):
        self.ptx.unlink()
        with self.assertRaises(FileNotFoundError):
            build_cuda.verify(self.directory)


if __name__ == "__main__":
    unittest.main()
