# JNeuro GUI path inventory

The denominator is the finite inventory of meaningful user-visible actions and state transitions below, not every possible execution trace through loops. Each row maps to an automated native-window test with behavioral assertions. A row counts only when its entire mapped test passes in the current `guiTest` JUnit XML report. Failed, skipped, aborted, missing and non-GUI tests do not count. Duplicate executions containing a failure do not count as success.

`guiPathCoverage` enforces **at least 90%**. `guiCheck` also requires every executed GUI test to pass. JaCoCo line coverage is reported independently and is not substituted for UI scenario coverage. Update this inventory and tests when controls or state transitions change; never remove a row to hide a regression. Multiple rows map to one test only when that test explicitly drives and asserts every listed transition.

GUI tests use `AWT Robot` on a real Swing window (Xvfb + a window manager on CI). Native events run outside the EDT; Swing observations and fixture setup run on the EDT. Direct `doClick`, action-map invocation and injected frames are not used to pass these paths. Separate headless integration tests are still useful but do not enter this denominator.

| ID | User-visible path | Passing GUI test method |
|---|---|---|
| GUI-001 | Open a real window and observe the Ready state | `opensAndNavigatesEveryView` |
| GUI-002 | Navigate to Overview with a native mouse click | `opensAndNavigatesEveryView` |
| GUI-003 | Navigate to Neurons with a native mouse click | `opensAndNavigatesEveryView` |
| GUI-004 | Navigate to Learning set with a native mouse click | `opensAndNavigatesEveryView` |
| GUI-005 | Navigate to Step effect with a native mouse click | `opensAndNavigatesEveryView` |
| GUI-006 | Navigate to Parameters with a native mouse click | `opensAndNavigatesEveryView` |
| GUI-007 | Navigate to Seeds with a native mouse click | `opensAndNavigatesEveryView` |
| GUI-008 | Navigate to Timeline with a native mouse click | `opensAndNavigatesEveryView` |
| GUI-009 | Navigate to Architecture search with a native mouse click | `opensAndNavigatesEveryView` |
| GUI-010 | Open Architecture search from the sidebar Auto search action | `opensAndNavigatesEveryView` |
| GUI-011 | Reject malformed topology without applying it | `validatesConfigurationAndAcceptsUncappedInputs` |
| GUI-012 | Reject invalid seed input | `validatesConfigurationAndAcceptsUncappedInputs` |
| GUI-013 | Apply topology beyond former width and depth caps | `validatesConfigurationAndAcceptsUncappedInputs` |
| GUI-014 | Accept maximum epochs above the former cap | `validatesConfigurationAndAcceptsUncappedInputs` |
| GUI-015 | Accept a full-range Long seed | `validatesConfigurationAndAcceptsUncappedInputs` |
| GUI-016 | Preserve precise learning-rate and momentum values | `validatesConfigurationAndAcceptsUncappedInputs` |
| GUI-017 | Accept scientific notation below the old rate minimum | `validatesConfigurationAndAcceptsUncappedInputs` |
| GUI-018 | Reject zero learning rate semantically | `validatesConfigurationAndAcceptsUncappedInputs` |
| GUI-019 | Reject momentum outside its mathematical domain | `validatesConfigurationAndAcceptsUncappedInputs` |
| GUI-020 | Numeric editors have no minimum or maximum bounds | `validatesConfigurationAndAcceptsUncappedInputs` |
| GUI-021 | Commit depth, width and parameter budgets above former caps | `searchInputsAcceptValuesBeyondOldCaps` |
| GUI-022 | Commit epochs, check interval, threads and trial budgets above former caps | `searchInputsAcceptValuesBeyondOldCaps` |
| GUI-023 | Commit a time limit beyond Int range without truncation | `searchInputsAcceptValuesBeyondOldCaps` |
| GUI-024 | Commit validation fraction and tolerance beyond former preset bounds | `searchInputsAcceptValuesBeyondOldCaps` |
| GUI-025 | Commit more than twenty seeds and the matching success count | `searchInputsAcceptValuesBeyondOldCaps` |
| GUI-026 | Commit restart controls above former caps without starting unintended work | `searchInputsAcceptValuesBeyondOldCaps` |
| GUI-027 | Button: one epoch | `trainingButtonsAndKeyboardRespectState` |
| GUI-028 | Button: ten additional epochs | `trainingButtonsAndKeyboardRespectState` |
| GUI-029 | Keyboard: reset | `trainingButtonsAndKeyboardRespectState` |
| GUI-030 | Button: train and pause | `trainingButtonsAndKeyboardRespectState` |
| GUI-031 | Paused epochs stay stable | `trainingButtonsAndKeyboardRespectState` |
| GUI-032 | Keyboard: single epoch | `trainingButtonsAndKeyboardRespectState` |
| GUI-033 | Keyboard: resume and pause | `trainingButtonsAndKeyboardRespectState` |
| GUI-034 | Button: reset | `trainingButtonsAndKeyboardRespectState` |
| GUI-035 | Set custom epochs per refresh | `trainingButtonsAndKeyboardRespectState` |
| GUI-036 | Reject invalid refresh count without changing the active speed | `trainingButtonsAndKeyboardRespectState` |
| GUI-037 | Epoch-limit state and disabled training actions | `terminalTrainingStatesDisableControls` |
| GUI-038 | Target-reached state and disabled training actions | `terminalTrainingStatesDisableControls` |
| GUI-039 | All neurons in every hidden layer have reachable cards | `everyNeuronIsReachableByScrollingAndBaselineHasNoMaps` |
| GUI-040 | Native scroll to the final neuron and render its actual activation | `everyNeuronIsReachableByScrollingAndBaselineHasNoMaps` |
| GUI-041 | Native scroll back to the first hidden layer | `everyNeuronIsReachableByScrollingAndBaselineHasNoMaps` |
| GUI-042 | No manual neuron pagination remains | `everyNeuronIsReachableByScrollingAndBaselineHasNoMaps` |
| GUI-043 | Switch to the direct baseline and discard stale maps | `everyNeuronIsReachableByScrollingAndBaselineHasNoMaps` |
| GUI-044 | Add a layer to the empty editor | `everyNeuronIsReachableByScrollingAndBaselineHasNoMaps` |
| GUI-045 | Remove the final hidden layer | `everyNeuronIsReachableByScrollingAndBaselineHasNoMaps` |
| GUI-046 | Reject removal from an empty editor | `everyNeuronIsReachableByScrollingAndBaselineHasNoMaps` |
| GUI-047 | Scroll to a neuron beyond the former width cap | `everyNeuronIsReachableByScrollingAndBaselineHasNoMaps` |
| GUI-048 | Choose a parameter layer | `parameterNavigationAndSmallWindowRemainUsable` |
| GUI-049 | Navigate to a later parameter range | `parameterNavigationAndSmallWindowRemainUsable` |
| GUI-050 | Reject an invalid parameter offset without applying it | `parameterNavigationAndSmallWindowRemainUsable` |
| GUI-051 | Scroll Parameters in a small real window | `parameterNavigationAndSmallWindowRemainUsable` |
| GUI-052 | Scroll expanded search settings and results | `parameterNavigationAndSmallWindowRemainUsable` |
| GUI-053 | Empty custom dataset state disables training | `customPointsSupportMouseKeyboardUndoAndClear` |
| GUI-054 | Left-click adds class one | `customPointsSupportMouseKeyboardUndoAndClear` |
| GUI-055 | Right-click adds class zero | `customPointsSupportMouseKeyboardUndoAndClear` |
| GUI-056 | Class selector affects a native click | `customPointsSupportMouseKeyboardUndoAndClear` |
| GUI-057 | Arrow keys and Enter add a keyboard-selected point | `customPointsSupportMouseKeyboardUndoAndClear` |
| GUI-058 | Undo a point | `customPointsSupportMouseKeyboardUndoAndClear` |
| GUI-059 | Clear all custom points | `customPointsSupportMouseKeyboardUndoAndClear` |
| GUI-060 | Complete a four-seed study without changing the active model | `seedComparisonCanCompleteAndBeCancelled` |
| GUI-061 | Cancel a long seed study | `seedComparisonCanCompleteAndBeCancelled` |
| GUI-062 | Controls remain usable after seed-study cancellation | `seedComparisonCanCompleteAndBeCancelled` |
| GUI-063 | Reject unordered search bounds | `searchValidationCancellationAndStaleResultsAreHandled` |
| GUI-064 | Reject validation mode for a truth table | `searchValidationCancellationAndStaleResultsAreHandled` |
| GUI-065 | Cancel a running search and retain its partial report | `searchValidationCancellationAndStaleResultsAreHandled` |
| GUI-066 | Reset invalidates the running search | `searchValidationCancellationAndStaleResultsAreHandled` |
| GUI-067 | Late search callbacks cannot repopulate invalidated results | `searchValidationCancellationAndStaleResultsAreHandled` |
| GUI-068 | Complete an exhaustive sweep without changing the active model | `searchResultsSupportNativeSelectionInspectionReplayAndApply` |
| GUI-069 | Select a result row | `searchResultsSupportNativeSelectionInspectionReplayAndApply` |
| GUI-070 | Inspect a scored result | `searchResultsSupportNativeSelectionInspectionReplayAndApply` |
| GUI-071 | Select a different result seed | `searchResultsSupportNativeSelectionInspectionReplayAndApply` |
| GUI-072 | Sort the result table by clicking a native header | `searchResultsSupportNativeSelectionInspectionReplayAndApply` |
| GUI-073 | Change recommendation policy | `searchResultsSupportNativeSelectionInspectionReplayAndApply` |
| GUI-074 | Select an architecture through the Pareto plot | `searchResultsSupportNativeSelectionInspectionReplayAndApply` |
| GUI-075 | Replay the selected checkpoint | `searchResultsSupportNativeSelectionInspectionReplayAndApply` |
| GUI-076 | Reset a replayed run | `searchResultsSupportNativeSelectionInspectionReplayAndApply` |
| GUI-077 | Apply an architecture as a fresh model | `searchResultsSupportNativeSelectionInspectionReplayAndApply` |
| GUI-078 | Save a readable PNG using the real file dialog | `pngExportAndCancelUseTheRealFileDialog` |
| GUI-079 | Cancel the file dialog without creating another file | `pngExportAndCancelUseTheRealFileDialog` |
| GUI-080 | Select and apply every dataset through the actual selector | `allDatasetsRemainSelectableAndTrainingErrorsRecover` |
| GUI-081 | Reject an unrepresentable topology without crashing the window | `allDatasetsRemainSelectableAndTrainingErrorsRecover` |
| GUI-082 | Recover by applying a valid configuration after an allocation rejection | `allDatasetsRemainSelectableAndTrainingErrorsRecover` |
| GUI-083 | Close the real window through the window manager | `windowCloseStopsTrainingAndSearchWorkers` |
| GUI-084 | Closing terminates the Studio, search and neuron-render workers | `windowCloseStopsTrainingAndSearchWorkers` |
| GUI-085 | Empty dataset disables starting architecture search | `customPointsSupportMouseKeyboardUndoAndClear` |
| GUI-086 | Change the recommendation to Smallest near best | `searchResultsSupportNativeSelectionInspectionReplayAndApply` |
| GUI-087 | Run adaptive search through the actual strategy selector and retain ancestry | `searchResultsSupportNativeSelectionInspectionReplayAndApply` |
| GUI-088 | Apply the configuration with the actual Apply & restart button | `allDatasetsRemainSelectableAndTrainingErrorsRecover` |
| GUI-089 | Stop search at a wall-clock deadline | `searchDeadlineDoesNotRecommendIncompleteCandidates` |
| GUI-090 | Incomplete candidates do not become recommendations | `searchDeadlineDoesNotRecommendIncompleteCandidates` |
| GUI-091 | Incomplete results cannot be applied or replayed | `searchDeadlineDoesNotRecommendIncompleteCandidates` |
| GUI-092 | Non-boolean datasets default search to validation | `validationSearchReplaysHeldOutPartition` |
| GUI-093 | Run validation search on a fixed held-out partition | `validationSearchReplaysHeldOutPartition` |
| GUI-094 | Replay keeps validation samples held out | `validationSearchReplaysHeldOutPartition` |
| GUI-095 | Reset validation replay restores the full Studio dataset | `validationSearchReplaysHeldOutPartition` |
| GUI-096 | Adaptive result ancestry names an eligible, fully evaluated elite for each descendant | `searchResultsSupportNativeSelectionInspectionReplayAndApply` |
