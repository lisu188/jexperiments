import sys
import xml.etree.ElementTree as ET

report_path, package_name, source_name, minimum_text = sys.argv[1:]
minimum = float(minimum_text)
root = ET.parse(report_path).getroot()
package = next((item for item in root.findall("package") if item.get("name") == package_name), None)
if package is None:
    raise SystemExit(f"Package not found in coverage report: {package_name}")
source = next((item for item in package.findall("sourcefile") if item.get("name") == source_name), None)
if source is None:
    raise SystemExit(f"Source file not found in coverage report: {source_name}")
counter = next((item for item in source.findall("counter") if item.get("type") == "LINE"), None)
if counter is None:
    raise SystemExit(f"LINE counter not found for {source_name}")
missed = int(counter.get("missed", "0"))
covered = int(counter.get("covered", "0"))
total = missed + covered
ratio = 1.0 if total == 0 else covered / total
print(f"{package_name}/{source_name}: {covered}/{total} lines = {ratio:.2%}")
if ratio < minimum:
    raise SystemExit(f"Line coverage {ratio:.2%} is below required {minimum:.2%}")
