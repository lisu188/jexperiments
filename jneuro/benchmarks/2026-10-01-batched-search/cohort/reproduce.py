#!/usr/bin/env python3
"""Compile the recorded harness against existing production classes; never resolve dependencies."""
import argparse
import json
import os
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--java", required=True, type=Path)
    parser.add_argument("--compiler-classpath", required=True)
    parser.add_argument("--runtime-classpath", required=True)
    parser.add_argument("--main-classes", required=True, type=Path)
    parser.add_argument("--native-cache", required=True, type=Path)
    parser.add_argument("--scratch", required=True, type=Path)
    args = parser.parse_args()
    for path in (args.java, args.main_classes, args.native_cache):
        if not path.exists():
            parser.error(f"Required existing input is missing: {path}")
    ram = Path("/run/shm").resolve()
    scratch = args.scratch.resolve()
    native = args.native_cache.resolve()
    if not scratch.is_relative_to(ram) or not native.is_relative_to(ram):
        parser.error("Use RAM-backed scratch and the already populated RAM native cache under /run/shm.")
    if not any(native.rglob("libtensorflow_cc.so.2")):
        parser.error("The native cache must already contain the TensorFlow CPU runtime.")
    scratch.mkdir(parents=True, exist_ok=True)
    classes = scratch / "classes"
    classes.mkdir(exist_ok=True)
    temporary = scratch / "tmp"
    temporary.mkdir(exist_ok=True)
    source = Path(__file__).resolve().with_name("CohortBenchmark.kt")
    main_classes = args.main_classes.resolve()
    compile_command = [str(args.java), "-Xmx512m", "--enable-native-access=ALL-UNNAMED",
        f"-Djava.io.tmpdir={temporary}", "-cp", args.compiler_classpath,
        "org.jetbrains.kotlin.cli.jvm.K2JVMCompiler", "-no-stdlib", "-no-reflect",
        "-jvm-target", "26", "-language-version", "2.4", "-api-version", "2.4", "-Werror",
        "-module-name", "jneuro", "-classpath", f"{main_classes}:{args.runtime_classpath}",
        f"-Xfriend-paths={main_classes}", "-d", str(classes), str(source)]
    run_command = [str(args.java), "-Xmx512m", "--enable-native-access=ALL-UNNAMED",
        f"-Djava.io.tmpdir={temporary}", f"-Dorg.bytedeco.javacpp.cachedir={native}",
        "-cp", f"{main_classes}:{classes}:{args.runtime_classpath}", "com.lis.neuro.CohortBenchmark", "2"]
    (scratch / "commands.json").write_text(json.dumps({"compile": compile_command, "run": run_command}, indent=2) + "\n")
    with (scratch / "compile.log").open("w") as output:
        subprocess.run(compile_command, stdout=output, stderr=subprocess.STDOUT, check=True)
    with (scratch / "results.jsonl").open("w") as output, (scratch / "runtime.log").open("w") as errors:
        subprocess.run(run_command, stdout=output, stderr=errors, check=True,
                       env=dict(os.environ, TF_CPP_MIN_LOG_LEVEL="2"))
    print(scratch / "results.jsonl")


if __name__ == "__main__":
    main()
