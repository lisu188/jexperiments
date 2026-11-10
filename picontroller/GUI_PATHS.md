# Pi dashboard browser paths

The shipped dashboard uses responsive navigation, dropdown menus, ordinary
anchor navigation, and three monitor bars. Other Bootstrap theme examples are
static content. There are no forms, training, cancellation, replay, or export
controls in this module. The browser owns window shutdown. Hardware startup and
the live WebSocket server are separate Java integration concerns.

The `pathTest()` calls with the matching IDs in
[`web-tests/dashboard.test.cjs`](web-tests/dashboard.test.cjs) perform browser
clicks, keyboard input, resizing or scrolling and assert the resulting DOM or
focus state. WebSocket events use a fixture transport; they do not prove GPIO or
server behavior. Security injection probes run separately from this denominator.

| ID | User path / asserted outcome | Test assertion |
| --- | --- | --- |
| NAV-01 | Mobile page starts with all three navigation menus collapsed | `pathTest('NAV-01')` |
| NAV-02 | Click the first navigation toggle to expand it | `pathTest('NAV-02')` |
| NAV-03 | Click the first toggle again to collapse it | `pathTest('NAV-03')` |
| NAV-04 | Expand the second and third menus independently | `pathTest('NAV-04')` |
| NAV-05 | Resize to desktop and see navigation links without the mobile toggle | `pathTest('NAV-05')` |
| NAV-06 | Follow the About anchor through the navigation | `pathTest('NAV-06')` |
| DROP-01 | Click a dropdown toggle to open its menu | `pathTest('DROP-01')` |
| DROP-02 | Click the same toggle to close its menu | `pathTest('DROP-02')` |
| DROP-03 | Open another dropdown and close the previous one | `pathTest('DROP-03')` |
| DROP-04 | Click outside an open dropdown to close it | `pathTest('DROP-04')` |
| DROP-05 | Choose a menu item and close the dropdown | `pathTest('DROP-05')` |
| DROP-06 | Escape closes the menu and restores toggle focus | `pathTest('DROP-06')` |
| DROP-07 | ArrowDown opens the menu and focuses the first item | `pathTest('DROP-07')` |
| DROP-08 | Arrow keys move through items and skip a disabled item | `pathTest('DROP-08')` |
| DROP-09 | ArrowUp from the toggle focuses the final item | `pathTest('DROP-09')` |
| DROP-10 | Tab past the final item closes the menu | `pathTest('DROP-10')` |
| DROP-11 | A disabled dropdown toggle does not open the menu | `pathTest('DROP-11')` |
| DROP-12 | Click a nested caret to open the owning dropdown | `pathTest('DROP-12')` |
| DATA-01 | Observe a memory update in the dashboard bar | `pathTest('DATA-01')` |
| DATA-02 | Observe a light update in the dashboard bar | `pathTest('DATA-02')` |
| DATA-03 | Observe a temperature update in the dashboard bar | `pathTest('DATA-03')` |
| DATA-04 | A closed transport reconnects and displays a new update | `pathTest('DATA-04')` |

CI uses Chromium in an Xvfb window with Playwright input. Every run writes a
fresh `test-results/gui-path-coverage.json`, browser diagnostics and screenshots.
The verifier requires at least 90% successful paths and has negative tests for
failed, skipped, aborted, missing, duplicate, stale and non-GUI results. Any test
assertion failure fails the job even if the ratio would otherwise pass. This
browser path ratio is independent of Gradle's module line coverage.
