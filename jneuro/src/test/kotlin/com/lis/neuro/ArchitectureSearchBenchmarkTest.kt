package com.lis.neuro

import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.io.TempDir
import java.nio.file.Files
import java.nio.file.Path

class ArchitectureSearchBenchmarkTest {
    @TempDir lateinit var temporary: Path

    @Test fun optionsPreserveQualityBudgetAndRejectAmbiguousCases() {
        val defaults = SearchBenchmarkOptions.parse(emptyArray())
        assertEquals(2000, defaults.epochs)
        assertEquals(1_000_000, SearchBenchmarkOptions.parse(arrayOf("--mode", "quality")).epochs)
        val options = SearchBenchmarkOptions.parse(arrayOf("--mode", "fixed", "--manifest", "small", "--epochs", "25",
            "--workers", "1,4,8", "--backends", "CPU,CUDA", "--executions", "OPTIMIZED", "--warmups", "0",
            "--repeats", "2", "--order-offset", "2", "--limit-seconds", "60", "--search-seeds", "42,123", "--trials", "10",
            "--output", temporary.resolve("report.jsonl").toString()))
        assertEquals(listOf(1, 4, 8), options.workers)
        assertEquals(60L, options.limitSeconds)
        assertTrue(SearchBenchmarkOptions.usage().contains("1000000"))
        for (args in listOf(arrayOf("--bad", "1"), arrayOf("--workers"), arrayOf("--workers", "1", "--workers", "2"),
            arrayOf("--epochs", "0"), arrayOf("--workers", "0"), arrayOf("--workers", "1,1"), arrayOf("--mode", "bad"),
            arrayOf("--backends", "AUTO"), arrayOf("--manifest", "general", "--backends", "CUDA"),
            arrayOf("--mode", "quality", "--manifest", "general"), arrayOf("--trials", "6"), arrayOf("--search-seeds", "1,1"))) {
            assertThrows(IllegalArgumentException::class.java) { SearchBenchmarkOptions.parse(args) }
        }
        val config = SearchBenchmarkHarness.configuration(options, 4, TrainingBackend.CPU, ArchitectureExecution.OPTIMIZED, 123)
        assertEquals(0.01, config.targetRmse)
        assertEquals(4, config.requiredSuccesses)
        assertEquals(5, config.seeds.size)
        assertEquals(40, config.maxTrials)
        assertEquals(123L, config.searchSeed)
        assertEquals(176, ArchitectureSearchData.split(NeuroLearningSets.create(NeuroLearningSets.Kind.SPIRAL, 42)).training.size)
        assertEquals(listOf(2, 8, 8, 8, 1), SearchBenchmarkHarness.manifest("small")[2].topology().toList())
    }

    @Test fun fixedManifestMeasuresAllSeedEpochsAndReportsEquivalentParameters() {
        val records = ArrayList<Map<String, Any?>>()
        SearchBenchmarkHarness.run(SearchBenchmarkOptions(epochs = 2, workers = listOf(2), warmups = 1, repeats = 1), records::add)
        val rounds = records.filter { it["type"] == "round" }
        assertEquals(4, rounds.size)
        for (round in rounds) {
            assertEquals(40, round["completeTrials"])
            assertEquals(80L, round["committedEpochs"])
            assertEquals(14_080L, round["sampleUpdates"])
            assertEquals(false, round["targetMet"])
            assertNull(round["independentTest"])
            assertEquals(0, round["partialCandidates"])
            assertTrue((round["totalNanos"] as Long) > 0)
            assertFalse(NeuroCudaBenchmarkReports.encode(round).contains("Infinity"))
        }
        val measured = rounds.filter { it["round"] == 0 }
        fun parameters(record: Map<String, Any?>): List<List<Double>> = (record["candidates"] as List<*>).flatMap { candidate ->
            ((candidate as Map<*, *>)["trials"] as List<*>).map { trial ->
                ((trial as Map<*, *>)["parametersAtBest"] as List<*>).map { it as Double }
            }
        }
        val first = parameters(measured[0]); val second = parameters(measured[1])
        for (index in first.indices) assertArrayEquals(first[index].toDoubleArray(), second[index].toDoubleArray(), 1e-10)
        assertEquals(2, records.count { it["type"] == "summary" })
        assertTrue(records.filter { it["type"] == "summary" }.all { it["completeRounds"] == 1 })
    }

    @Test fun qualityAndGeneralRegressionUseRealSchedulerAndCliWritesIncrementalJson() {
        val quality = ArrayList<Map<String, Any?>>()
        SearchBenchmarkHarness.run(SearchBenchmarkOptions(mode = "quality", epochs = 1, workers = listOf(1),
            executions = listOf(ArchitectureExecution.OPTIMIZED), warmups = 0, repeats = 1, trials = 5,
            searchSeeds = listOf(42)), quality::add)
        assertEquals(5, quality.first { it["type"] == "round" }["completeTrials"])
        val output = temporary.resolve("nested/search.jsonl")
        ArchitectureSearchBenchmark.main(arrayOf("--manifest", "general", "--epochs", "1", "--workers", "1", "--warmups", "0",
            "--repeats", "1", "--executions", "OPTIMIZED", "--output", output.toString()))
        val lines = Files.readAllLines(output)
        assertEquals(3, lines.size)
        assertTrue(lines[1].contains("REFERENCE/CPU/OPTIMIZED"))
        assertTrue(lines[1].contains("\"completeTrials\":20"))
        ArchitectureSearchBenchmark.main(arrayOf("--help"))
    }

    @Test fun independentSpiralIsDisjointAndStatisticsAreWellDefined() {
        val train = NeuroLearningSets.create(NeuroLearningSets.Kind.SPIRAL, 42)
        val independent = SearchBenchmarkHarness.independentSpiral()
        assertEquals(218, independent.size)
        assertEquals(109, independent.count { it.target == 1.0 })
        assertTrue(independent.none { test -> train.any { it.x == test.x && it.y == test.y } })
        val network = Neuro(intArrayOf(2, 4, 1))
        val score = SearchBenchmarkHarness.independentScore(NeuroXorDiagnostics.capture(network, 0, 0.5))
        assertTrue((score.getValue("rmse") as Double).isFinite())
        assertTrue(score.getValue("classificationAccuracy") as Double in 0.0..1.0)
        assertEquals(2.5, SearchBenchmarkHarness.percentile(listOf(1.0, 2.0, 3.0, 4.0), 0.5))
        assertEquals(4.0, SearchBenchmarkHarness.percentile(listOf(1.0, 2.0, 3.0, 4.0), 0.95))
        assertEquals(1.0, SearchBenchmarkHarness.percentile(listOf(1.0), 0.0))
        assertThrows(IllegalArgumentException::class.java) { SearchBenchmarkHarness.percentile(emptyList(), 0.5) }
    }
}
