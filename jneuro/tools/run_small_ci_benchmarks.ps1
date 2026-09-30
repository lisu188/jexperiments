# Optional Windows CI experiment. Run from the repository root after building the
# native DLL. Each measured report comes from a fresh JVM; Gradle only builds/tests.
[CmdletBinding()]
param(
    [Parameter(Mandatory)][string] $Library,
    [Parameter(Mandatory)][string] $Output,
    [Parameter(Mandatory)][string] $Revision
)
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $false
Set-StrictMode -Version Latest
$libraryPath = (Resolve-Path $Library).Path
New-Item -ItemType Directory -Force $Output | Out-Null
$outputPath = (Resolve-Path $Output).Path
$java = Join-Path $env:JAVA_HOME 'bin/java.exe'
if (!(Test-Path $java)) { throw 'JAVA_HOME must select Java 27.' }
if (Get-ChildItem $outputPath -File) { throw 'Use an empty output directory; do not mix independent benchmark runs.' }

function Invoke-Checked([string] $Program, [string[]] $Arguments, [string] $Log) {
    Add-Content (Join-Path $outputPath 'commands.jsonl') (@{ program = $Program; arguments = $Arguments } | ConvertTo-Json -Compress)
    & $Program @Arguments *> $Log
    if ($LASTEXITCODE -ne 0) {
        Get-Content $Log -Tail 60 | Write-Host
        throw "$Program failed with exit $LASTEXITCODE; see $Log"
    }
}

function Save-HostSnapshot([string] $Name) {
    # Snapshots describe the runner; they do not prove exclusive CPU scheduling.
    [ordered]@{
        capturedUtc = [DateTime]::UtcNow.ToString('o')
        revision = $Revision
        runnerName = $env:RUNNER_NAME
        imageOs = $env:ImageOS
        imageVersion = $env:ImageVersion
        nativeLibrarySha256 = (Get-FileHash $libraryPath -Algorithm SHA256).Hash.ToLowerInvariant()
        processors = @(Get-CimInstance Win32_Processor | Select-Object Name, Manufacturer, NumberOfCores, NumberOfLogicalProcessors, MaxClockSpeed, CurrentClockSpeed, LoadPercentage)
        heavyProcesses = @(Get-Process | Sort-Object CPU -Descending | Select-Object -First 12 ProcessName, CPU, WorkingSet64)
    } | ConvertTo-Json -Depth 5 | Set-Content (Join-Path $outputPath $Name) -Encoding utf8
}

Save-HostSnapshot 'host-before.json'
Invoke-Checked $java @('-XX:+PrintFlagsFinal', '-version') (Join-Path $outputPath 'jvm-flags.log')
Get-Content (Join-Path $outputPath 'jvm-flags.log') | Select-String 'UseAVX|UseFMA|MaxVectorSize|java version|openjdk version|Runtime Environment|Server VM' |
    Set-Content (Join-Path $outputPath 'jvm-vector-flags.txt')

# Keep the build file unchanged. Export the already-defined benchmark classpath,
# which also contains main classes/resources, after the actual-DLL acceptance test.
$classpathFile = Join-Path $outputPath 'runtime-classpath.txt'
$initFile = Join-Path $outputPath 'benchmark-classpath.init.gradle'
@'
gradle.projectsEvaluated {
    def module = gradle.rootProject.project(':jneuro')
    module.tasks.register('writeSmallCiBenchmarkClasspath') {
        dependsOn module.tasks.named('jmhClasses')
        doLast {
            new File(module.findProperty('smallCiClasspath').toString()).text = module.sourceSets.jmh.runtimeClasspath.asPath
        }
    }
}
'@ | Set-Content $initFile -Encoding utf8
Invoke-Checked '.\gradlew.bat' @('--no-daemon', '--console=plain', '--max-workers=2', '-I', $initFile,
    ":jneuro:nativeSmallCheck", ':jneuro:writeSmallCiBenchmarkClasspath', "-PneuroSmallNative=$libraryPath",
    "-PsmallCiClasspath=$classpathFile") (Join-Path $outputPath 'build-and-native-check.log')
Invoke-Checked 'python' @('-B', '-m', 'unittest', 'discover', '-s', 'jneuro/tools', '-p', 'test_aggregate_small_benchmarks.py') (Join-Path $outputPath 'aggregator-tests.log')
$classpath = (Get-Content $classpathFile -Raw).Trim()
$jvmArguments = @('--add-modules=jdk.incubator.vector', '--enable-native-access=ALL-UNNAMED', '-Xms128m', '-Xmx2g',
    '-Djneuro.log.level=WARNING', '-Djneuro.log.file=false', "-Djneuro.benchmark.sourceRevision=$Revision", "-Djneuro.small.native=$libraryPath")
$small = 'SMALL_SCALAR_FP64,SMALL_128_FP64,SMALL_256_FP64,SMALL_SCALAR_FP32,SMALL_128_FP32,SMALL_256_FP32,NATIVE_FP64'
$jsonPaths = @()
$csvPaths = @()

# Rotate modes/model counts between JVMs; each harness rotates engine order inside
# measured rounds. Setup, host publication and close remain in the native gate.
for ($fork = 1; $fork -le 3; $fork++) {
    $modes = if ($fork % 2) { @('MINIBATCH', 'CHUNK') } else { @('CHUNK', 'MINIBATCH') }
    foreach ($mode in $modes) {
        $prefix = Join-Path $outputPath "$($mode.ToLowerInvariant())-fork$fork"
        $epochs = if ($mode -eq 'MINIBATCH') { '5' } else { '64' }
        $warmups = if ($mode -eq 'MINIBATCH') { '100' } else { '20' }
        $repeats = if ($mode -eq 'MINIBATCH') { '30' } else { '20' }
        $backends = if ($mode -eq 'MINIBATCH') { "CPU,CPU_LEGACY,$small" } else { "CPU,$small" }
        Invoke-Checked $java ($jvmArguments + @('-cp', $classpath, 'com.lis.neuro.NeuroCudaTrainingBenchmark',
            '--topology', '2,8,8,8,1', '--samples', '128', '--batches', '16,64', '--sigmoid', 'EXACT',
            '--retained-device', 'true', '--epochs', $epochs, '--mode', $mode, '--warmups', $warmups,
            '--repeats', $repeats, '--backends', $backends, '--output', $prefix)) "$prefix.log"
        $jsonPaths += "$prefix.json"
    }
    $modelCounts = @(1, 4, 32)
    for ($offset = 0; $offset -lt 3; $offset++) {
        $models = $modelCounts[($fork - 1 + $offset) % 3]
        $csv = Join-Path $outputPath "cohort-$models-fork$fork.csv"
        Invoke-Checked $java ($jvmArguments + @('-cp', $classpath, 'com.lis.neuro.SmallTrainingExperiments',
            '--mode', 'cohort', '--topology', '2,8,8,8,1', '--samples', '128', '--models', "$models",
            '--epochs', '64', '--batch', '64', '--parallelism', '4', '--precision', 'FP64',
            '--warmups', '10', '--repeats', '20', '--engines', 'REFERENCE_MATRIX,REFERENCE_PARALLEL,SMALL_SEQUENTIAL,SMALL_CPU,NATIVE',
            '--output', $csv)) "$csv.log"
        $csvPaths += $csv
    }
}

$jmhReport = Join-Path $outputPath 'jmh-public-epochs.json'
$engines = 'REFERENCE_SCALAR,REFERENCE_MATRIX,SMALL_SCALAR,SMALL_128,SMALL_256,SMALL_FP32_SCALAR,SMALL_FP32_128,SMALL_FP32_256'
$jmhJvm = '--add-modules=jdk.incubator.vector --enable-native-access=ALL-UNNAMED -Xms128m -Xmx2g -Djneuro.log.level=WARNING -Djneuro.log.file=false'
Invoke-Checked $java ($jvmArguments + @('-cp', $classpath, 'org.openjdk.jmh.Main',
    'com.lis.neuro.SmallTrainingJmhBenchmark.publicEpochs', '-foe', 'true', '-f', '3', '-wi', '5', '-i', '5', '-w', '1s', '-r', '1s',
    '-p', "engine=$engines", '-p', 'topology=2-8-8-8-1', '-p', 'workload=MINI_BATCH', '-p', 'sigmoid=EXACT',
    '-p', 'samples=128', '-p', 'batchSize=64', '-jvmArgs', $jmhJvm, '-rf', 'json', '-rff', $jmhReport)) (Join-Path $outputPath 'jmh-public-epochs.log')
Save-HostSnapshot 'host-after.json'

# Status 1 means the native promotion gate did not pass, and is expected evidence.
# Invalid/missing benchmark data is checked separately and always fails the job.
$summary = Join-Path $outputPath 'summary'
& python -B jneuro/tools/aggregate_small_benchmarks.py --jsonpaths @jsonPaths --csvpaths @csvPaths --output $summary
$gateExit = $LASTEXITCODE
if ($gateExit -notin @(0, 1)) { throw "Aggregation failed with exit $gateExit" }
$report = Get-Content "$summary.json" -Raw | ConvertFrom-Json
if ($report.issues.Count -ne 0 -or $report.inputs.Count -ne 15 -or $report.requiredWorkloads.Count -ne 7) {
    throw "Incomplete/invalid benchmark evidence: $($report.issues -join '; ')"
}
foreach ($workload in $report.workloads.PSObject.Properties.Value) {
    foreach ($engine in $workload.engines.PSObject.Properties.Value) {
        if ($engine.verifiedForks -ne 3 -or $engine.files -ne 3 -or $engine.unverifiedRounds -ne 0 -or $engine.missingNativeHash -ne 0) {
            throw 'Every measured engine requires three independently identified JVMs and valid native provenance.'
        }
    }
}
$jmhResults = @(Get-Content $jmhReport -Raw | ConvertFrom-Json)
if ($jmhResults.Count -ne 8 -or ($jmhResults.params.engine | Sort-Object -Unique).Count -ne 8) { throw 'Missing JMH engines.' }
foreach ($result in $jmhResults) {
    if ($result.forks -ne 3 -or $result.warmupIterations -ne 5 -or $result.measurementIterations -ne 5 -or
        $result.params.sigmoid -ne 'EXACT' -or $result.params.batchSize -ne '64' -or $result.primaryMetric.rawData.Count -ne 3) {
        throw 'Unexpected/incomplete JMH configuration.'
    }
    foreach ($measurements in $result.primaryMetric.rawData) {
        if ($measurements.Count -ne 5) { throw 'Incomplete JMH fork.' }
        foreach ($measurement in $measurements) {
            if ([double]::IsNaN($measurement) -or [double]::IsInfinity($measurement) -or $measurement -le 0) { throw 'Invalid JMH measurement.' }
        }
    }
}
if ($env:GITHUB_STEP_SUMMARY) { Get-Content "$summary.md" | Add-Content $env:GITHUB_STEP_SUMMARY }
Write-Host "Completed CPU/native evidence. Native retention gate passed: $($report.nativeGatePassed)."
