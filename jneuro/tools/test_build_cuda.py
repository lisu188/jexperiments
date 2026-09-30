"""Negative checks for the resource verifier; no CUDA toolkit or device needed."""

from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import build_cuda


class CompilerValidationTest(unittest.TestCase):
    def test_pinned_compiler_needs_no_installation_metadata(self):
        executable = "/missing-installation-metadata/bin/nvcc"
        output = subprocess.CompletedProcess([executable, "--version"], 0,
            "nvcc: NVIDIA (R) Cuda compiler driver\n" + build_cuda.NVCC_VERSION + "\n")
        with patch.object(build_cuda.shutil, "which", return_value=executable), \
                patch.object(build_cuda.subprocess, "run", return_value=output) as run:
            self.assertEqual((executable, build_cuda.NVCC_VERSION), build_cuda.compiler("nvcc"))
            run.assert_called_once_with([executable, "--version"], check=True, text=True, capture_output=True)

    def test_missing_compiler_is_rejected_without_running_a_command(self):
        with patch.object(build_cuda.shutil, "which", return_value=None), \
                patch.object(build_cuda.subprocess, "run") as run:
            with self.assertRaisesRegex(ValueError, "Cannot find nvcc"):
                build_cuda.compiler("nvcc")
            run.assert_not_called()

    def test_unexpected_versions_are_rejected(self):
        for version in ("Cuda compilation tools, release 13.0, V13.0.48",
                        "Cuda compilation tools, release 13.1, V13.1.88",
                        "unrecognized compiler output",
                        build_cuda.NVCC_VERSION + " unexpected suffix"):
            with self.subTest(version=version), \
                    patch.object(build_cuda.shutil, "which", return_value="/usr/local/cuda/bin/nvcc"), \
                    patch.object(build_cuda.subprocess, "run",
                                 return_value=subprocess.CompletedProcess([], 0, version + "\n")):
                with self.assertRaisesRegex(ValueError, "expected CUDA 13.0.88"):
                    build_cuda.compiler("nvcc")

    def test_failed_compiler_probe_is_not_accepted(self):
        with patch.object(build_cuda.shutil, "which", return_value="/usr/local/cuda/bin/nvcc"), \
                patch.object(build_cuda.subprocess, "run", side_effect=subprocess.CalledProcessError(1, "nvcc")):
            with self.assertRaises(subprocess.CalledProcessError):
                build_cuda.compiler("nvcc")


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

    def test_build_normalizes_compiler_whitespace_before_hashing(self):
        expected = self.ptx.read_text(encoding="utf-8")
        compiler_output = expected.replace("\n", " \t\r\n") + "\r\n \t\r\n"
        generated = self.directory / "generated"

        def compile_fixture(command, **_options):
            output = Path(command[command.index("--output-file") + 1])
            output.write_bytes(compiler_output.encode("utf-8"))
            return subprocess.CompletedProcess(command, 0)

        with patch.object(build_cuda, "compiler", return_value=("nvcc", build_cuda.NVCC_VERSION)), \
                patch.object(build_cuda.subprocess, "run", side_effect=compile_fixture):
            build_cuda.build(generated, "nvcc")
        ptx = generated / "train.ptx"
        self.assertEqual(expected.encode("utf-8"), ptx.read_bytes())
        self.assertEqual(build_cuda.sha256(ptx), build_cuda.read_properties(generated / "train.properties")["ptx.sha256"])
        build_cuda.normalize_ptx(ptx)
        self.assertEqual(expected.encode("utf-8"), ptx.read_bytes(), "Normalization must be idempotent")
        build_cuda.verify(generated)

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
