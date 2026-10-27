const assert = require('node:assert/strict');

function verify(inventory, report, runId) {
    const ids = [...inventory.matchAll(/^\| ([A-Z]+-\d+) \|/gm)].map(match => match[1]);
    assert(ids.length > 0, 'GUI path inventory is empty');
    assert.equal(new Set(ids).size, ids.length, 'GUI path IDs must be unique');
    assert.equal(report.runId, runId, 'Stale GUI results');
    assert.equal(report.kind, 'browser-user-input', 'Non-GUI results cannot count');
    const results = report.paths;
    assert.equal(new Set(results.map(row => row.id)).size, results.length, 'Duplicate GUI results');
    assert(results.every(row => ids.includes(row.id)), 'Unknown GUI path result');
    const passed = results.filter(row => row.status === 'passed').length;
    const ratio = passed / ids.length;
    assert(ratio >= 0.9, `GUI path coverage ${passed}/${ids.length} is below 90%`);
    return { passed, total: ids.length, ratio };
}

module.exports = { verify };
