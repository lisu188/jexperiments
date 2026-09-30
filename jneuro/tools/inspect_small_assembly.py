#!/usr/bin/env python3
"""Decode selected HotSpot PrintAssembly hex blocks without installing hsdis.

Capture with -XX:+UnlockDiagnosticVMOptions and, for example,
-XX:CompileCommand=print,com.lis.neuro.SmallDoubleCompute::forward.
Do not enable global PrintAssembly: selected methods keep the evidence bounded.
Only C2 main-code ranges are decoded. Stack-store counts are diagnostics, not a
proof of register spills (HotSpot can use stack locations for other purposes).
"""
import argparse
import collections
import json
from pathlib import Path
import re
import subprocess
import tempfile

HEADER = re.compile(r"^Compiled method \(([^)]+)\)(.*)$")
MAIN = re.compile(r"main code\s+\[(0x[0-9a-fA-F]+),\s*(0x[0-9a-fA-F]+)\]")
BYTES = re.compile(r"^\s*(0x[0-9a-fA-F]+):\s*(.*)$")
INSTRUCTION = re.compile(r"^\s*[0-9a-f]+:\s+(?:[0-9a-f]{2}\s+)+\s*([a-z][a-z0-9]+)\s*(.*)$")


def blocks(text):
    """Return complete address-contiguous main-code buffers, excluding metadata/stubs."""
    current = None
    for line in text.splitlines():
        header = HEADER.match(line)
        if header:
            if current:
                yield current
            current = {"compiler": header[1], "header": line, "bytes": {}}
        elif current:
            main = MAIN.search(line)
            if main:
                current["start"], current["end"] = (int(main[i], 16) for i in (1, 2))
            raw = BYTES.match(line)
            if raw:
                # HotSpot prints groups of up to eight hex digits separated by pipes.
                data = raw[2].split(";")[0]
                tokens = data.replace("|", " ").split()
                if tokens and all(re.fullmatch(r"(?:[0-9a-fA-F]{2}){1,4}", token) for token in tokens):
                    address = int(raw[1], 16)
                    for byte in bytes.fromhex("".join(tokens)):
                        current["bytes"][address] = byte
                        address += 1
    if current:
        yield current


def inspect(text, objdump="objdump"):
    methods = []
    failures = []
    for block in blocks(text):
        if block["compiler"] != "c2":
            continue
        start, end = block.get("start"), block.get("end")
        if start is None or end is None or end <= start or end - start > 4 * 1024 * 1024:
            failures.append({"method": block["header"], "reason": "Missing or invalid main-code range"})
            continue
        if any(address not in block["bytes"] for address in range(start, end)):
            failures.append({"method": block["header"], "reason": "Incomplete raw main-code bytes"})
            continue
        code = bytes(block["bytes"][address] for address in range(start, end))
        with tempfile.TemporaryDirectory(prefix="jneuro-assembly-") as directory:
            binary = Path(directory) / "code.bin"
            binary.write_bytes(code)
            result = subprocess.run([objdump, "-D", "-b", "binary", "-m", "i386:x86-64", "-M", "intel",
                                     "--insn-width=16", f"--adjust-vma={start}", str(binary)],
                                    check=True, capture_output=True, text=True)
        instructions = []
        for line in result.stdout.splitlines():
            match = INSTRUCTION.match(line)
            if match:
                instructions.append((match[1], match[2]))
        packed_fma = collections.Counter(m for m, _ in instructions if m.startswith("vfm") and m.endswith(("pd", "ps")))
        scalar_fma = collections.Counter(m for m, _ in instructions if m.startswith("vfm") and m.endswith(("sd", "ss")))
        stack_stores = [f"{mnemonic} {operands}" for mnemonic, operands in instructions
                        if mnemonic.startswith("vmov") and re.search(r"\[[^]]*(?:rsp|rbp)[^]]*\]", operands.split(",")[0])]
        methods.append({"method": block["header"], "code_bytes": len(code), "instruction_count": len(instructions),
                        "packed_fma": dict(packed_fma), "scalar_fma": dict(scalar_fma),
                        "xmm_instructions": sum(bool(re.search(r"\bxmm\d+\b", operands)) for _, operands in instructions),
                        "ymm_instructions": sum(bool(re.search(r"\bymm\d+\b", operands)) for _, operands in instructions),
                        "vector_stack_store_count": len(stack_stores), "vector_stack_store_examples": stack_stores[:8]})
    return {"methods": methods, "incomplete_methods": failures,
            "caveat": "Counts describe decoded C2 main code. Vector stack stores are not necessarily register spills."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--objdump", default="objdump")
    parser.add_argument("--require-fp64", action="store_true")
    parser.add_argument("--require-fp32", action="store_true")
    parser.add_argument("--require-width", choices=(128, 256), type=int)
    args = parser.parse_args()
    report = inspect(args.log.read_text(encoding="utf-8", errors="replace"), args.objdump)
    methods = report["methods"]
    packed = [name for method in methods for name in method["packed_fma"]]
    passed = bool(methods) and not report["incomplete_methods"]
    passed &= not args.require_fp64 or any(name.endswith("pd") for name in packed)
    passed &= not args.require_fp32 or any(name.endswith("ps") for name in packed)
    if args.require_width:
        field = "ymm_instructions" if args.require_width == 256 else "xmm_instructions"
        passed &= any(method[field] for method in methods)
    report["requirements_passed"] = bool(passed)
    output = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(output, encoding="utf-8")
    else:
        print(output, end="")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
