package com.lis.neuro

import java.awt.Container
import java.awt.EventQueue
import java.awt.Point
import java.awt.event.MouseEvent
import java.awt.image.BufferedImage
import javax.swing.*
import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.Test

class ArchitectureSearchPanelTest {
    @Test fun invalidatingResultsReleasesCachedCandidatesAndSelection() {
        val report = report()
        EventQueue.invokeAndWait {
            val panel = ArchitectureSearchPanel({ _, _, _, _ -> }, {}, { _, _ -> }, { _, _, _ -> })
            panel.complete(report)
            assertSame(report.candidates, field(panel, "rankedCandidates"))
            assertNotNull(field(panel, "cachedSelection"))

            panel.invalidateResults()
            for (name in listOf("rankedCandidates", "rankedPolicy", "rankedConfig", "cachedSelection")) {
                assertNull(field(panel, name), "Invalidated results must release $name")
            }
            render(panel, 1000, 750)
            assertNull(field(panel, "rankedCandidates"))
            assertNull(field(panel, "cachedSelection"))
            assertEquals(0, (field(panel, "table") as JTable).rowCount)

            panel.complete(report)
            assertSame(report.candidates, field(panel, "rankedCandidates"))
            assertEquals(report.selection.recommended?.architecture, field(panel, "selected"))
        }
    }

    @Test fun startingAnotherSearchReleasesPreviousRankingBeforeProgressArrives() {
        val report = report()
        EventQueue.invokeAndWait {
            val panel = ArchitectureSearchPanel({ _, _, _, _ -> }, {}, { _, _ -> }, { _, _, _ -> })
            panel.complete(report)
            assertSame(report.candidates, field(panel, "rankedCandidates"))
            assertNotNull(field(panel, "cachedSelection"))

            panel.started(report.config, report.data.evaluation)
            for (name in listOf("rankedCandidates", "rankedPolicy", "rankedConfig", "cachedSelection")) {
                assertNull(field(panel, name), "A new search must release $name before its first progress update")
            }
            render(panel, 1000, 750)
            assertTrue((field(panel, "rankedCandidates") as List<*>?).isNullOrEmpty())
            assertNull((field(panel, "cachedSelection") as ArchitectureSelection?)?.bestError)
            assertEquals(0, (field(panel, "table") as JTable).rowCount)
        }
    }

    @Test fun searchShowsBackendInitializationFailureAndInheritsSelection() {
        val sessions = RecordingTrainingSessions().apply { unavailable = true }
        val config = ArchitectureSearchConfig(strategy = ArchitectureSearchStrategy.EXHAUSTIVE,
            maxLayers = 1, maxWidth = 1, seeds = listOf(42), requiredSuccesses = 1,
            maxEpochs = 1, checkEvery = 1, backend = TrainingBackend.GPU)
        val report = NeuroArchitectureSearch(sessions::open).search(ArchitectureSearchData.fitting(xor()), config)
        EventQueue.invokeAndWait {
            val panel = ArchitectureSearchPanel({ _, _, _, _ -> }, {}, { _, _ -> }, { _, _, _ -> })
            panel.setSource(StudioConfig(backend = TrainingBackend.GPU, precision = Neuro.TrainingPrecision.FP32, batchSize = 7), 4)
            assertEquals(TrainingBackend.GPU, panel.readConfig().backend)
            assertEquals(Neuro.TrainingPrecision.FP32, panel.readConfig().precision)
            assertEquals(7, panel.readConfig().batchSize)
            assertTrue((field(panel, "sourceLabel") as JLabel).text.contains("TensorFlow GPU"))
            assertTrue((field(panel, "sourceLabel") as JLabel).text.contains("General topology"))
            panel.complete(report)
            assertTrue((field(panel, "summary") as JLabel).text.contains("TensorFlow GPU: GPU fixture unavailable"))
            assertFalse(button(panel, "Replay selected run").isEnabled)
        }
    }

    @Test fun rejectedReplayPreservesCompletedResultAndInspection() {
        val report = report()
        EventQueue.invokeAndWait {
            val panel = ArchitectureSearchPanel({ _, _, _, _ -> }, {}, { _, _ -> }, { _, _, _ -> })
            panel.setSource(StudioConfig(), 4)
            panel.complete(report)
            val selection = field(panel, "selected")
            val trial = field(panel, "chosenTrial")
            val surface = field(panel, "surface")
            panel.replayFailed("Recorded GPU device differs")
            assertSame(report, field(panel, "result"))
            assertSame(selection, field(panel, "selected"))
            assertSame(trial, field(panel, "chosenTrial"))
            assertSame(surface, field(panel, "surface"))
            assertTrue((field(panel, "summary") as JLabel).text.contains("Replay error: Recorded GPU device differs"))
            assertTrue((field(panel, "progressBar") as JProgressBar).string.contains("results retained"))
            assertTrue(button(panel, "Replay selected run").isEnabled)
        }
    }

    @Test fun exposesValidatedSearchSettingsProgressAndCancellation() {
        var starts = 0; var cancels = 0
        EventQueue.invokeAndWait {
            val panel = ArchitectureSearchPanel({ config, evaluation, fraction, split ->
                assertEquals(ArchitectureEvaluation.TRAINING_FIT, evaluation)
                assertEquals(0.2, fraction); assertEquals(42L, split); assertEquals(ArchitectureSearchStrategy.ADAPTIVE, config.strategy); assertEquals(listOf(6), config.initialHidden); assertEquals(1_000_000, config.plannedTrials()); starts++
            }, { cancels++ }, { _, _ -> }, { _, _, _ -> })
            panel.setSource(StudioConfig(), 4)
            val defaults = panel.readConfig()
            assertEquals(1, defaults.minLayers)
            assertEquals(5, defaults.maxLayers)
            assertEquals(1, defaults.minWidth)
            assertEquals(16, defaults.maxWidth)
            assertEquals(1_000_000, defaults.maxParameters)
            assertEquals(listOf(1L, 42L, 123L, 999L, 2026L), defaults.seeds)
            assertEquals(4, defaults.requiredSuccesses)
            assertEquals(100_000, defaults.maxEpochs)
            assertEquals(25, defaults.checkEvery)
            assertEquals(0.001, defaults.targetRmse)
            assertEquals(0.005, defaults.nearBestTolerance)
            assertEquals(32, defaults.parallelism)
            assertEquals(1_000_000, defaults.maxTrials)
            assertEquals(0L, defaults.timeLimitSeconds)
            assertEquals(42L, defaults.searchSeed)
            assertEquals(12, defaults.restartAfter)
            assertEquals(4, defaults.maxRestarts)
            assertEquals(ArchitectureSearchStrategy.ADAPTIVE, defaults.strategy)
            assertEquals(ArchitectureExecution.BATCHED, defaults.execution)
            assertEquals(ArchitectureBudgetPolicy.SUCCESSIVE_HALVING, defaults.budgetPolicy)
            assertEquals(25, defaults.initialEpochs)
            assertEquals(3, defaults.reductionFactor)
            assertEquals(128, defaults.modelsPerBatch)
            assertEquals(ArchitecturePolicy.SMALLEST_MEETING_TARGET, defaults.policy)
            val epochControl = field(panel, "epochs") as JSpinner
            assertTrue(epochControl.toolTipText.contains("Weaker architectures are pruned"))
            assertTrue(epochControl.toolTipText.contains("cannot be recommended or replayed"))
            assertEquals(epochControl.toolTipText, (epochControl.editor as JSpinner.DefaultEditor).textField.toolTipText)
            button(panel, "Start search").doClick()
            assertEquals(1, starts); assertFalse(button(panel, "Start search").isEnabled)
            val config = panel.readConfig()
            epochControl.value = 100_001 // Unapplied edits must not change the running budget shown.
            panel.updateProgress(ArchitectureSearchProgress(584, 2920, 0, emptyList(),
                listOf(ArchitectureRunningTrial(NetworkArchitecture(listOf(2)), 42, 25, 0.5)), 10))
            assertTrue((field(panel, "summary") as JLabel).text.endsWith("epoch 25/100000"))
            epochControl.value = 100_000
            panel.updateProgress(ArchitectureSearchProgress(584, 2920, 0, emptyList(), emptyList(), 20))
            render(panel, 1150, 850)
            button(panel, "Cancel search").doClick(); assertEquals(1, cancels)
            panel.complete(ArchitectureSearchResult(ArchitectureSearchData.fitting(xor()), config,
                ArchitectureTermination.CANCELLED, 584, emptyList(), 20))
            assertTrue((field(panel, "summary") as JLabel).text.contains("No fully evaluated"))
            (field(panel, "seeds") as JTextField).text = "broken"
            button(panel, "Start search").doClick()
            assertTrue((field(panel, "summary") as JLabel).text.contains("Seeds"))
            (field(panel, "seeds") as JTextField).text = "1,42,123,999,2026"
            (field(panel, "splitSeed") as JTextField).text = "bad"
            button(panel, "Start search").doClick()
            assertTrue((field(panel, "summary") as JLabel).text.contains("Split seed"))
            (field(panel, "splitSeed") as JTextField).text = "42"
            (field(panel, "evaluation") as JComboBox<*>).selectedItem = ArchitectureEvaluation.VALIDATION
            button(panel, "Start search").doClick()
            assertTrue((field(panel, "summary") as JLabel).text.contains("Truth tables"))
            panel.setSource(StudioConfig(dataset = NeuroLearningSets.Kind.CIRCLE, maxEpochs = 1), 180)
            assertEquals(100_000, panel.readConfig().maxEpochs)
            assertEquals(25, panel.readConfig().checkEvery)
            assertEquals(0.001, panel.readConfig().targetRmse)
            val advanced = descendants(panel).filterIsInstance<JCheckBox>().single()
            advanced.doClick(); render(panel, 1020, 1100); advanced.doClick()
            panel.invalidateResults(); render(panel, 800, 650)
            assertEquals(0, (field(panel, "table") as JTable).rowCount)
            panel.updateProgress(ArchitectureSearchProgress(1, 5, 0, emptyList(), emptyList(), 0))
            assertEquals(0, (field(panel, "table") as JTable).rowCount)
        }
    }

    @Test fun completedResultsSupportSortingPerSeedInspectionPoliciesAndExplicitApplyReplay() {
        val report = report()
        var applications = 0; var replays = 0
        EventQueue.invokeAndWait {
            val panel = ArchitectureSearchPanel({ _, _, _, _ -> }, {}, { result, candidate ->
                assertSame(report, result); assertTrue(candidate.valid); applications++
            }, { result, candidate, trial ->
                assertSame(report, result); assertTrue(trial in candidate.trials); replays++
            })
            panel.setSource(StudioConfig(), 4)
            panel.complete(report)
            render(panel, 1150, 850)
            assertTrue(button(panel, "Apply architecture").isEnabled)
            assertTrue(button(panel, "Replay selected run").isEnabled)
            val table = field(panel, "table") as JTable
            assertEquals(3, table.rowCount); assertEquals(9, table.columnCount)
            for (row in 0 until table.rowCount) for (column in 0 until table.columnCount) {
                assertNotNull(table.getValueAt(row, column)); assertNotNull(table.getColumnClass(column)); assertNotNull(table.getColumnName(column))
            }
            table.rowSorter.toggleSortOrder(2)
            table.rowSorter.toggleSortOrder(2)
            panel.selectArchitecture(report.candidates.last().architecture)
            assertEquals(report.candidates.last().architecture.toString(), table.getValueAt(table.selectedRow, 0))
            table.setRowSelectionInterval(0, 0)
            (field(panel, "selectedSeed") as JComboBox<*>).selectedIndex = 1
            button(panel, "Inspect result").doClick()
            button(panel, "Apply architecture").doClick()
            button(panel, "Replay selected run").doClick()
            assertEquals(1, applications); assertEquals(1, replays)
            (field(panel, "policy") as JComboBox<*>).selectedItem = ArchitecturePolicy.LOWEST_RMSE
            assertTrue((field(panel, "summary") as JLabel).text.contains("Recommended"))
            (field(panel, "policy") as JComboBox<*>).selectedItem = ArchitecturePolicy.SMALLEST_NEAR_BEST
            render(panel, 800, 650)
            val plot = field(panel, "plot") as JPanel
            @Suppress("UNCHECKED_CAST")
            val points = field(plot, "points") as List<Pair<Point, NetworkArchitecture>>
            val point = points.first().first
            plot.dispatchEvent(MouseEvent(plot, MouseEvent.MOUSE_PRESSED, 0, 0, point.x, point.y, 1, false))
            assertEquals(report.candidates.first().architecture.toString(), table.getValueAt(table.selectedRow, 0))
            plot.dispatchEvent(MouseEvent(plot, MouseEvent.MOUSE_PRESSED, 0, 0, -500, -500, 1, false))
            assertTrue((field(panel, "details") as JLabel).text.contains("trained 100"))
            assertTrue((field(panel, "summary") as JLabel).toolTipText.contains(report.data.fingerprint))
            panel.selectArchitecture(null); render(panel, 1000, 750)
            assertFalse(button(panel, "Apply architecture").isEnabled)
            panel.selectArchitecture(report.candidates.first().architecture)
            panel.invalidateResults()
            assertFalse(button(panel, "Replay selected run").isEnabled)
        }
    }

    @Test fun incompleteFailedAndAboveTargetResultsRemainHonest() {
        val original = report()
        val first = original.candidates.first()
        val partial = ArchitectureCandidate(first.architecture, first.trials.take(1), 5, 0.05)
        val good = first.trials.first()
        val failed = ArchitectureTrial(good.seed, ArchitectureTrialState.FAILED, good.epochs, good.bestEpoch, good.bestRmse,
            good.trainingRmseAtBest, good.finalRmse, good.sampleUpdates, good.elapsedNanos, good.history, good.snapshot, "Non-finite RMSE")
        val badCandidate = ArchitectureCandidate(NetworkArchitecture(listOf(4)), listOf(failed), 1, 0.05)
        val completeAbove = ArchitectureCandidate(original.candidates[1].architecture, original.candidates[1].trials, 5, 1e-12)
        val config = ArchitectureSearchConfig(maxLayers = 1, maxWidth = 4, maxEpochs = 100, targetRmse = 1e-12)
        EventQueue.invokeAndWait {
            val panel = ArchitectureSearchPanel({ _, _, _, _ -> }, {}, { _, _ -> }, { _, _, _ -> })
            panel.setSource(StudioConfig(), 4)
            panel.started(config, ArchitectureEvaluation.TRAINING_FIT)
            panel.updateProgress(ArchitectureSearchProgress(4, 20, 7, listOf(partial, completeAbove, badCandidate), emptyList(), 100))
            panel.selectArchitecture(partial.architecture); render(panel, 1050, 800)
            assertFalse(button(panel, "Apply architecture").isEnabled)
            val table = field(panel, "table") as JTable
            assertEquals("Provisional", table.model.getValueAt(0, 6))
            assertEquals("Above target", table.model.getValueAt(1, 6))
            assertEquals("Failed seed", table.model.getValueAt(2, 6))
            panel.complete(ArchitectureSearchResult(original.data, config, ArchitectureTermination.CANCELLED, 4,
                listOf(partial, completeAbove, badCandidate), 100))
            assertTrue((field(panel, "summary") as JLabel).text.contains("No evaluated architecture"))
            panel.selectArchitecture(badCandidate.architecture)
            assertFalse(button(panel, "Replay selected run").isEnabled)
            assertTrue((field(panel, "details") as JLabel).text.contains("Non-finite"))
            render(panel, 1050, 800)
            panel.failed("Observer failure")
            assertTrue((field(panel, "summary") as JLabel).text.contains("Observer failure"))
            panel.setSource(StudioConfig(dataset = NeuroLearningSets.Kind.CUSTOM), 0)
            assertFalse(button(panel, "Start search").isEnabled)
        }
    }

    @Test fun adaptiveControlsCaptureActiveTopologyAndExposeRealParentage() {
        val config = ArchitectureSearchConfig(initialHidden = listOf(2), maxLayers = 1, maxWidth = 3,
            maxEpochs = 100, maxTrials = 15, maxRestarts = 0)
        val report = NeuroArchitectureSearch().search(ArchitectureSearchData.fitting(xor()), config)
        EventQueue.invokeAndWait {
            val panel = ArchitectureSearchPanel({ _, _, _, _ -> }, {}, { _, _ -> }, { _, _, _ -> })
            panel.setSource(StudioConfig(hidden = "4,2"), 4)
            assertEquals(listOf(4, 2), panel.readConfig().initialHidden)
            assertEquals(ArchitectureSearchStrategy.ADAPTIVE, panel.readConfig().strategy)
            panel.started(config, ArchitectureEvaluation.TRAINING_FIT)
            assertFalse((field(panel, "policy") as JComboBox<*>).isEnabled)
            assertFalse((field(panel, "strategy") as JComboBox<*>).isEnabled)
            panel.updateProgress(ArchitectureSearchProgress(report.generated, 15, 5,
                report.candidates, emptyList(), 100, report.lineage))
            assertTrue((field(panel, "lineageStatus") as JLabel).text.contains("Generation"))
            panel.complete(report)
            val table = field(panel, "table") as JTable
            for (row in 0 until table.model.rowCount) {
                val proposal = report.lineage.first { it.architecture == report.candidates[row].architecture }
                assertEquals(proposal.parent?.toString() ?: "—", table.model.getValueAt(row, 7))
                assertEquals(proposal.mutation, table.model.getValueAt(row, 8))
            }
            assertTrue((field(panel, "lineageStatus") as JLabel).text.contains("not the entire space"))
            assertTrue((field(panel, "policy") as JComboBox<*>).isEnabled)
            assertTrue((field(panel, "summary") as JLabel).toolTipText.contains("search seed 42"))
            (field(panel, "strategy") as JComboBox<*>).selectedItem = ArchitectureSearchStrategy.EXHAUSTIVE
            assertEquals(ArchitectureSearchStrategy.EXHAUSTIVE, panel.readConfig().strategy)
            (field(panel, "searchSeed") as JTextField).text = "bad"
            assertThrows(IllegalArgumentException::class.java) { panel.readConfig() }
            panel.invalidateResults()
            render(panel, 1280, 980)
        }
    }

    @Test fun batchedBudgetControlsValidateInputsAndKeepFullBudgetAnExplicitAlternative() {
        EventQueue.invokeAndWait {
            val panel = ArchitectureSearchPanel({ _, _, _, _ -> }, {}, { _, _ -> }, { _, _, _ -> })
            panel.setSource(StudioConfig(batchSize = 7), 4)
            val execution = field(panel, "execution") as JComboBox<*>
            val budget = field(panel, "budgetPolicy") as JComboBox<*>
            val initial = field(panel, "initialEpochs") as JSpinner
            val reduction = field(panel, "reductionFactor") as JSpinner
            val models = field(panel, "modelsPerBatch") as JSpinner
            val threads = field(panel, "threads") as JSpinner
            val plateau = field(panel, "restartAfter") as JSpinner
            val restarts = field(panel, "restarts") as JSpinner
            assertFalse(plateau.isEnabled || restarts.isEnabled)
            assertTrue((field(panel, "lineageStatus") as JLabel).text.contains("broad population"))
            assertFalse((field(panel, "lineageStatus") as JLabel).text.contains("restarts"))
            assertFalse(threads.isEnabled, "Batched training does not use legacy trial workers")
            assertTrue(initial.isEnabled && reduction.isEnabled && models.isEnabled)
            models.value = 6; initial.value = 2; reduction.value = 2
            assertEquals(6, panel.readConfig().modelsPerBatch)
            assertEquals(7, panel.readConfig().batchSize, "Model batching must not change sample batching")
            assertEquals(2, panel.readConfig().initialEpochs)
            assertEquals(2, panel.readConfig().reductionFactor)
            for ((control, invalid) in listOf(models to 0, models to 1025, initial to 0, reduction to 1, reduction to 9)) {
                val previous = control.value
                control.value = invalid
                assertThrows(IllegalArgumentException::class.java) { panel.readConfig() }
                control.value = previous
            }
            budget.selectedItem = ArchitectureBudgetPolicy.FULL
            assertFalse(initial.isEnabled || reduction.isEnabled)
            assertTrue(models.isEnabled)
            assertTrue((field(panel, "epochs") as JSpinner).toolTipText.contains("without pruning"))
            assertEquals(ArchitectureBudgetPolicy.FULL, panel.readConfig().budgetPolicy)
            execution.selectedItem = ArchitectureExecution.REFERENCE
            assertTrue(threads.isEnabled && plateau.isEnabled && restarts.isEnabled)
            assertTrue((field(panel, "lineageStatus") as JLabel).text.contains("restarts"))
            assertFalse(budget.isEnabled || models.isEnabled)
            assertTrue((field(panel, "epochs") as JSpinner).toolTipText.contains("every seed of the first architecture"))
            execution.selectedItem = ArchitectureExecution.BATCHED
            budget.selectedItem = ArchitectureBudgetPolicy.SUCCESSIVE_HALVING
            execution.selectedItem = ArchitectureExecution.OPTIMIZED
            assertEquals(ArchitectureBudgetPolicy.FULL, panel.readConfig().budgetPolicy)
            execution.selectedItem = ArchitectureExecution.BATCHED
            budget.selectedItem = ArchitectureBudgetPolicy.SUCCESSIVE_HALVING
            val config = panel.readConfig()
            assertFalse(threads.isEnabled || plateau.isEnabled || restarts.isEnabled)
            (field(panel, "strategy") as JComboBox<*>).selectedItem = ArchitectureSearchStrategy.EXHAUSTIVE
            assertTrue((field(panel, "lineageStatus") as JLabel).text.contains("pre-enumerated population"))
            (field(panel, "strategy") as JComboBox<*>).selectedItem = ArchitectureSearchStrategy.ADAPTIVE
            panel.started(config, ArchitectureEvaluation.TRAINING_FIT)
            assertFalse(budget.isEnabled || models.isEnabled || initial.isEnabled || reduction.isEnabled)
            panel.failed("batch initialization failed")
            assertTrue(budget.isEnabled && models.isEnabled && initial.isEnabled && reduction.isEnabled)
            panel.invalidateResults()
            assertTrue((field(panel, "throughputStatus") as JLabel).text.contains("first batch"))
        }
    }

    @Test fun disabledSettingsCannotBlockApplicableModesAndTheirEditsRemainAvailable() {
        EventQueue.invokeAndWait {
            val panel = ArchitectureSearchPanel({ _, _, _, _ -> }, {}, { _, _ -> }, { _, _, _ -> })
            panel.setSource(StudioConfig(), 4)
            val execution = field(panel, "execution") as JComboBox<*>
            val budget = field(panel, "budgetPolicy") as JComboBox<*>
            val initial = field(panel, "initialEpochs") as JSpinner
            val reduction = field(panel, "reductionFactor") as JSpinner
            val models = field(panel, "modelsPerBatch") as JSpinner
            val threads = field(panel, "threads") as JSpinner
            val plateau = field(panel, "restartAfter") as JSpinner
            val restarts = field(panel, "restarts") as JSpinner
            initial.value = 0; reduction.value = 1
            (initial.editor as JSpinner.DefaultEditor).textField.text = "unfinished"
            budget.selectedItem = ArchitectureBudgetPolicy.FULL
            assertEquals(25, panel.readConfig().initialEpochs)
            assertEquals(3, panel.readConfig().reductionFactor)
            panel.started(panel.readConfig(), ArchitectureEvaluation.TRAINING_FIT)
            panel.failed("Stopped full-budget fixture")
            assertEquals(0, initial.value); assertEquals(1, reduction.value)
            assertEquals("unfinished", (initial.editor as JSpinner.DefaultEditor).textField.text)
            budget.selectedItem = ArchitectureBudgetPolicy.SUCCESSIVE_HALVING
            assertThrows(Exception::class.java) { panel.readConfig() }
            budget.selectedItem = ArchitectureBudgetPolicy.FULL

            models.value = 1025
            (models.editor as JSpinner.DefaultEditor).textField.text = "unfinished"
            execution.selectedItem = ArchitectureExecution.REFERENCE
            assertEquals(128, panel.readConfig().modelsPerBatch)
            panel.started(panel.readConfig(), ArchitectureEvaluation.TRAINING_FIT)
            panel.failed("Stopped reference fixture")
            assertEquals(1025, models.value)
            assertEquals("unfinished", (models.editor as JSpinner.DefaultEditor).textField.text)
            execution.selectedItem = ArchitectureExecution.BATCHED
            assertThrows(Exception::class.java) { panel.readConfig() }
            execution.selectedItem = ArchitectureExecution.REFERENCE

            threads.value = 0; plateau.value = 0; restarts.value = -1
            (threads.editor as JSpinner.DefaultEditor).textField.text = "unfinished"
            models.value = 6
            execution.selectedItem = ArchitectureExecution.BATCHED
            val batched = panel.readConfig()
            assertEquals(32, batched.parallelism); assertEquals(12, batched.restartAfter); assertEquals(4, batched.maxRestarts)
            assertEquals(6, batched.modelsPerBatch)
            panel.started(batched, ArchitectureEvaluation.TRAINING_FIT)
            panel.failed("Stopped batched fixture")
            assertEquals(0, threads.value); assertEquals(0, plateau.value); assertEquals(-1, restarts.value)
            assertEquals("unfinished", (threads.editor as JSpinner.DefaultEditor).textField.text)
            execution.selectedItem = ArchitectureExecution.REFERENCE
            assertThrows(Exception::class.java) { panel.readConfig() }
        }
    }

    @Test fun prunedCheckpointsRemainInspectableButNeverBecomeRecommendationsOrReplayTargets() {
        val original = report()
        val candidate = original.candidates.first()
        val seed = candidate.trials.first()
        val prunedTrial = ArchitectureTrial(seed.seed, ArchitectureTrialState.PRUNED, seed.epochs, seed.bestEpoch,
            seed.bestRmse, seed.trainingRmseAtBest, seed.finalRmse, seed.sampleUpdates, seed.elapsedNanos,
            seed.history, seed.snapshot, execution = ArchitectureExecution.BATCHED, route = ArchitectureTrialRoute.TENSOR_BATCH)
        val pruned = ArchitectureCandidate(candidate.architecture, listOf(prunedTrial), 1, 0.9)
        val pending = ArchitectureCandidate(NetworkArchitecture(listOf(5)), emptyList(), 1, 0.9)
        val active = ArchitectureCandidate(NetworkArchitecture(listOf(6)), emptyList(), 1, 0.9)
        val config = ArchitectureSearchConfig(maxLayers = 1, maxWidth = 6, seeds = listOf(seed.seed), requiredSuccesses = 1,
            execution = ArchitectureExecution.BATCHED, budgetPolicy = ArchitectureBudgetPolicy.SUCCESSIVE_HALVING)
        EventQueue.invokeAndWait {
            val panel = ArchitectureSearchPanel({ _, _, _, _ -> }, {}, { _, _ -> fail("Pruned application") },
                { _, _, _ -> fail("Pruned replay") })
            panel.setSource(StudioConfig(), 4)
            panel.started(config, ArchitectureEvaluation.TRAINING_FIT)
            panel.updateProgress(ArchitectureSearchProgress(3, 3, 1, listOf(pruned, pending, active),
                listOf(ArchitectureRunningTrial(active.architecture, seed.seed, 25, 0.5)), 1_000_000,
                nativeTrainingCalls = 7, modelsPerBatch = 3, aggregateEpochsPerSecond = 125.5))
            val table = field(panel, "table") as JTable
            assertEquals("PRUNED", table.model.getValueAt(0, 6))
            assertEquals("Pending", table.model.getValueAt(1, 6))
            assertEquals("Running", table.model.getValueAt(2, 6))
            assertEquals("Training calls: 7 · peak models per batch: 3 · aggregate model-epochs/s: 125.5",
                (field(panel, "throughputStatus") as JLabel).text)
            panel.complete(ArchitectureSearchResult(original.data, config, ArchitectureTermination.COMPLETED, 1,
                listOf(pruned), 1_000_000, nativeTrainingCalls = 8, modelsPerBatch = 3, aggregateEpochsPerSecond = 150.0))
            assertTrue((field(panel, "summary") as JLabel).text.contains("No fully evaluated"))
            assertTrue((field(panel, "progressBar") as JProgressBar).string.contains("1 pruned"))
            assertTrue((field(panel, "throughputStatus") as JLabel).text.contains("Training calls: 8"))
            panel.selectArchitecture(pruned.architecture)
            assertTrue(button(panel, "Inspect result").isEnabled)
            assertNotNull(field(panel, "surface"))
            assertFalse(button(panel, "Apply architecture").isEnabled)
            assertFalse(button(panel, "Replay selected run").isEnabled)
            assertTrue((field(panel, "details") as JLabel).text.contains("PRUNED"))
            assertTrue((field(panel, "details") as JLabel).text.contains("not eligible"))
            button(panel, "Apply architecture").doClick(); button(panel, "Replay selected run").doClick()
            render(panel, 1100, 1120)
        }
    }

    private fun report(): ArchitectureSearchResult = NeuroArchitectureSearch().search(ArchitectureSearchData.fitting(xor()),
        ArchitectureSearchConfig(strategy = ArchitectureSearchStrategy.EXHAUSTIVE, maxLayers = 1, maxWidth = 3, maxEpochs = 100, targetRmse = 0.9))
    private fun xor() = NeuroLearningSets.create(NeuroLearningSets.Kind.XOR, 42)
    private fun field(instance: Any, name: String): Any? = instance.javaClass.getDeclaredField(name).run { isAccessible = true; get(instance) }
    private fun descendants(container: Container): List<java.awt.Component> = container.components.flatMap {
        listOf(it) + if (it is Container) descendants(it) else emptyList()
    }
    private fun button(panel: Container, title: String): JButton = descendants(panel).filterIsInstance<JButton>().first { it.text == title }
    private fun render(panel: JPanel, width: Int, height: Int) {
        panel.setSize(width, height)
        fun layout(container: Container) { container.doLayout(); container.components.filterIsInstance<Container>().forEach { layout(it) } }
        layout(panel)
        val image = BufferedImage(width, height, BufferedImage.TYPE_INT_RGB)
        val graphics = image.createGraphics()
        try { panel.printAll(graphics) } finally { graphics.dispose() }
        assertNotEquals(0, image.getRGB(width / 2, height / 2))
    }
}
