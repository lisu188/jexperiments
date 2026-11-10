const { test } = require('node:test');
const assert = require('node:assert/strict');
const { verify } = require('./path-coverage.cjs');
const inventory = Array.from({ length: 10 }, (_, i) => `| GUI-${i + 1} | path | assertion |`).join('\n');
const report = () => ({ runId: 'fresh', kind: 'browser-user-input', paths:
    Array.from({ length: 10 }, (_, i) => ({ id: `GUI-${i + 1}`, status: 'passed' })) });

test('accepts current successful GUI results', () => {
    assert.equal(verify(inventory, report(), 'fresh').ratio, 1);
});
for (const status of ['failed', 'skipped', 'aborted', 'missing']) {
    test(`${status} paths cannot count`, () => {
        const result = report();
        result.paths[0].status = status;
        result.paths[1].status = status;
        assert.throws(() => verify(inventory, result, 'fresh'), /below 90%/);
    });
}
test('missing paths cannot count', () => {
    const result = report();
    result.paths.splice(0, 2);
    assert.throws(() => verify(inventory, result, 'fresh'), /below 90%/);
});
test('stale, duplicate and non-GUI results fail', () => {
    assert.throws(() => verify(inventory, report(), 'another-run'), /Stale/);
    const duplicate = report();
    duplicate.paths.push(duplicate.paths[0]);
    assert.throws(() => verify(inventory, duplicate, 'fresh'), /Duplicate/);
    const model = report();
    model.kind = 'model';
    assert.throws(() => verify(inventory, model, 'fresh'), /Non-GUI/);
});
