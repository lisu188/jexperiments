import tempfile
import unittest
from pathlib import Path
from verify_gui_paths import measure


class GuiPathCoverageTest(unittest.TestCase):
    def test_failed_skipped_missing_and_duplicate_runs_do_not_count(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog = root / "GUI_PATHS.md"
            catalog.write_text("\n".join(f"| GUI-{i:02d} | Path | `case{i}` |" for i in range(1, 6)))
            (root / "suite.xml").write_text('''<testsuite>
              <testcase classname="com.lis.neuro.NeuroGuiTest" name="case1()"/>
              <testcase classname="com.lis.neuro.NeuroGuiTest" name="case2()"><failure/></testcase>
              <testcase classname="com.lis.neuro.NeuroGuiTest" name="case3()"><skipped/></testcase>
              <testcase classname="com.lis.neuro.NeuroGuiTest" name="case4()"/>
              <testcase classname="com.lis.neuro.NeuroGuiTest" name="case4()"><error/></testcase>
              <testcase classname="OtherUnitTest" name="case5()"/>
            </testsuite>''')
            result = measure(catalog, root)
            self.assertEqual(["GUI-01"], result["covered_paths"])
            self.assertEqual(0.2, result["ratio"])
            (root / "suite.xml").unlink()
            self.assertEqual(0.0, measure(catalog, root)["ratio"])

    def test_empty_and_duplicate_catalogs_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog = root / "GUI_PATHS.md"
            catalog.write_text("")
            with self.assertRaises(ValueError):
                measure(catalog, root)
            catalog.write_text("| GUI-01 | A | `case1` |\n| GUI-01 | B | `case2` |")
            with self.assertRaises(ValueError):
                measure(catalog, root)


if __name__ == "__main__":
    unittest.main()
