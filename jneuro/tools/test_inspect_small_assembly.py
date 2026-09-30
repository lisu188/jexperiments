import shutil
import unittest

from inspect_small_assembly import blocks, inspect


class AssemblyInspectionTest(unittest.TestCase):
    def test_ignores_c1_and_reports_missing_main_code(self):
        text = "Compiled method (c1) irrelevant\nCompiled method (c2) target\n"
        report = inspect(text)
        self.assertEqual([], report["methods"])
        self.assertEqual(1, len(report["incomplete_methods"]))

    def test_rejects_missing_bytes_instead_of_decoding_an_incomplete_method(self):
        text = "Compiled method (c2) target\nmain code [0x1000, 0x1004]\n  0x1000: c3\n"
        self.assertEqual("Incomplete raw main-code bytes", inspect(text)["incomplete_methods"][0]["reason"])

    def test_groups_hex_bytes_and_excludes_annotations(self):
        text = "Compiled method (c2) target\nmain code [0x1000, 0x1005]\n  0x1000: c4e2 f5b8 | c2 ; fma\n  0x1005: metadata\n"
        parsed = list(blocks(text))[0]
        self.assertEqual(bytes.fromhex("c4e2f5b8c2"), bytes(parsed["bytes"].values()))

    @unittest.skipUnless(shutil.which("objdump"), "objdump is required for disassembly")
    def test_decodes_real_avx2_packed_fma_and_stack_store(self):
        # vfmadd231pd ymm0,ymm1,ymm2; vfmadd231ps ymm0,ymm1,ymm2;
        # vmovapd YMMWORD PTR [rsp],ymm0; ret.
        raw = "c4e2f5b8c2c4e275b8c2c5fd290424c3"
        text = "Compiled method (c2) target\nmain code [0x1000, 0x1010]\n"
        text += "  0x1000: " + " ".join(raw[i:i + 8] for i in range(0, len(raw), 8)) + "\n"
        report = inspect(text)
        self.assertEqual([], report["incomplete_methods"])
        method = report["methods"][0]
        self.assertEqual({"vfmadd231pd": 1, "vfmadd231ps": 1}, method["packed_fma"])
        self.assertEqual(3, method["ymm_instructions"])
        self.assertEqual(1, method["vector_stack_store_count"])


if __name__ == "__main__":
    unittest.main()
