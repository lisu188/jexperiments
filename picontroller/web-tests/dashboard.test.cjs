const assert = require('node:assert/strict');
const http = require('node:http');
const fs = require('node:fs/promises');
const path = require('node:path');
const { randomUUID } = require('node:crypto');
const { chromium } = require('playwright-core');
const { verify } = require('./path-coverage.cjs');

const webRoot = path.resolve(__dirname, '../src/main/java/com/lis/pi/web');
const output = path.resolve(__dirname, 'test-results');
const runId = randomUUID();
const report = { runId, kind: 'browser-user-input', startedAt: new Date().toISOString(), paths: [], security: [] };
const types = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.png': 'image/png', '.woff': 'font/woff' };

async function main() {
    await fs.mkdir(output, { recursive: true });
    const server = http.createServer(async (request, response) => {
        try {
            const pathname = decodeURIComponent(new URL(request.url, 'http://localhost').pathname);
            const file = path.resolve(webRoot, '.' + (pathname === '/' ? '/index.html' : pathname));
            if (!file.startsWith(webRoot + path.sep)) {
                response.writeHead(403).end();
                return;
            }
            const content = await fs.readFile(file);
            response.writeHead(200, { 'Content-Type': types[path.extname(file)] || 'application/octet-stream' }).end(content);
        } catch {
            response.writeHead(404).end();
        }
    });
    await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
    let browser;
    try {
        browser = await chromium.launch({ headless: process.env.HEADLESS !== 'false',
            ...(process.env.BROWSER_EXECUTABLE ? { executablePath: process.env.BROWSER_EXECUTABLE } : {}) });
        const context = await browser.newContext({ viewport: { width: 390, height: 844 } });
        const origin = `http://127.0.0.1:${server.address().port}`;
        await context.route('**/*', route => route.request().url().startsWith(origin + '/') ? route.continue() : route.abort());
        await context.addInitScript(() => {
            window.fixtureSockets = [];
            window.WebSocket = class {
                constructor(url) { this.url = url; window.fixtureSockets.push(this); }
            };
        });
        const page = await context.newPage();
        const pageErrors = [];
        page.on('pageerror', error => pageErrors.push(error.message));
        await page.goto(origin + '/');
        const navbars = page.locator('.navbar');
        const buttons = page.locator('.navbar-toggle');
        const panels = page.locator('.navbar-collapse');
        const toggle = navbars.nth(0).locator('.dropdown-toggle');
        const dropdown = navbars.nth(0).locator('.dropdown');
        const items = dropdown.locator('.dropdown-menu a');
        const open = async (locator, expected) => assert.equal(await locator.evaluate(node => node.classList.contains('open')), expected);
        const expanded = async (locator, expected) => assert.equal(await locator.getAttribute('aria-expanded'), String(expected));
        const focused = async locator => assert.equal(await locator.evaluate(node => node === document.activeElement), true);
        const screenshot = async filename => {
            // Keep headed Chromium captures within the fixed Xvfb viewport;
            // full-page mobile captures can exceed its compositor surface.
            await page.bringToFront();
            await page.waitForFunction(() => document.visibilityState === 'visible', null, { timeout: 5000 });
            await page.evaluate(() => {
                window.fixturePaintFrames = 0;
                const painted = () => {
                    window.fixturePaintFrames++;
                    if (window.fixturePaintFrames < 2) requestAnimationFrame(painted);
                };
                requestAnimationFrame(painted);
            });
            await page.waitForFunction(() => window.fixturePaintFrames === 2, null, { timeout: 5000 });
            const viewport = page.viewportSize();
            const png = await page.screenshot({ path: path.join(output, filename), fullPage: false, timeout: 5000 });
            assert.equal(png.subarray(0, 8).toString('hex'), '89504e470d0a1a0a');
            assert.equal(png.readUInt32BE(16), viewport.width);
            assert.equal(png.readUInt32BE(20), viewport.height);
        };
        const run = async (collection, id, action) => {
            try {
                await action();
                collection.push({ id, status: 'passed' });
                console.log(`PASS ${id}`);
            } catch (error) {
                collection.push({ id, status: 'failed', error: error.message });
                console.error(`FAIL ${id}: ${error.message}`);
                try {
                    await screenshot(`${id}-failure.png`);
                } catch (captureError) {
                    collection.at(-1).screenshotError = captureError.message;
                    console.error(`Screenshot diagnostic ${id}: ${captureError.message}`);
                }
            }
        };
        const pathTest = (id, action) => run(report.paths, id, action);

        await pathTest('NAV-01', async () => {
            assert.equal(await buttons.count(), 3);
            for (let i = 0; i < 3; i++) {
                assert.equal(await panels.nth(i).isVisible(), false);
                await expanded(buttons.nth(i), false);
            }
            await screenshot('mobile.png');
        });
        await pathTest('NAV-02', async () => {
            await buttons.nth(0).click();
            assert.equal(await panels.nth(0).isVisible(), true);
            await expanded(buttons.nth(0), true);
        });
        await pathTest('NAV-03', async () => {
            await buttons.nth(0).click();
            assert.equal(await panels.nth(0).isVisible(), false);
            await expanded(buttons.nth(0), false);
        });
        await pathTest('NAV-04', async () => {
            for (let i = 1; i < 3; i++) {
                await buttons.nth(i).click();
                await expanded(buttons.nth(i), true);
                assert.equal(await panels.nth(i).isVisible(), true);
            }
            assert.equal(await panels.nth(0).isVisible(), false);
        });
        await pathTest('NAV-05', async () => {
            await page.setViewportSize({ width: 1280, height: 900 });
            for (let i = 0; i < 3; i++) {
                assert.equal(await panels.nth(i).isVisible(), true);
                assert.equal(await buttons.nth(i).isVisible(), false);
            }
            await page.evaluate(() => window.scrollTo(0, 0));
            await screenshot('desktop.png');
        });
        await pathTest('NAV-06', async () => {
            await navbars.nth(0).locator('a[href="#about"]').click();
            assert.equal(new URL(page.url()).hash, '#about');
        });
        await pathTest('DROP-01', async () => { await toggle.click(); await open(dropdown, true); await expanded(toggle, true); });
        await pathTest('DROP-02', async () => { await toggle.click(); await open(dropdown, false); await expanded(toggle, false); });
        await pathTest('DROP-03', async () => {
            await toggle.click();
            await navbars.nth(1).locator('.dropdown-toggle').click();
            await open(dropdown, false);
            await open(navbars.nth(1).locator('.dropdown'), true);
        });
        await pathTest('DROP-04', async () => {
            await toggle.click();
            await page.locator('h1').filter({ hasText: 'Buttons' }).click();
            await open(dropdown, false);
        });
        await pathTest('DROP-05', async () => { await toggle.click(); await items.nth(0).click(); await open(dropdown, false); });
        await pathTest('DROP-06', async () => {
            await toggle.click(); await page.keyboard.press('ArrowDown'); await page.keyboard.press('Escape');
            await open(dropdown, false); await focused(toggle);
        });
        await pathTest('DROP-07', async () => { await toggle.focus(); await page.keyboard.press('ArrowDown'); await open(dropdown, true); await focused(items.nth(0)); });
        await pathTest('DROP-08', async () => {
            await items.nth(1).evaluate(node => node.parentElement.classList.add('disabled'));
            try {
                await page.keyboard.press('ArrowDown'); await focused(items.nth(2));
                await page.keyboard.press('ArrowUp'); await focused(items.nth(0));
            } finally {
                await items.nth(1).evaluate(node => node.parentElement.classList.remove('disabled'));
            }
        });
        await pathTest('DROP-09', async () => {
            await page.keyboard.press('Escape'); await page.keyboard.press('ArrowUp'); await focused(items.last());
        });
        await pathTest('DROP-10', async () => { await page.keyboard.press('Tab'); await open(dropdown, false); });
        await pathTest('DROP-11', async () => {
            await toggle.evaluate(node => node.setAttribute('aria-disabled', 'true'));
            try { await toggle.focus(); await page.keyboard.press('Enter'); await open(dropdown, false); }
            finally { await toggle.evaluate(node => node.removeAttribute('aria-disabled')); }
        });
        await pathTest('DROP-12', async () => { await toggle.locator('.caret').click(); await open(dropdown, true); await page.keyboard.press('Escape'); });
        const update = async (type, value, id, width, socket = 0) => {
            const bar = page.locator('#' + id);
            await bar.scrollIntoViewIfNeeded();
            await page.evaluate(({ type, value, socket }) => window.fixtureSockets[socket].onmessage({ data: JSON.stringify({ type, value }) }), { type, value, socket });
            await page.waitForFunction(({ id, width }) => document.getElementById(id).style.width === width, { id, width });
            assert.equal(await bar.isVisible(), true);
        };
        await pathTest('DATA-01', () => update('MEMORY', 0.42, 'memoryBar', '42%'));
        await pathTest('DATA-02', () => update('LIGHT', 1, 'lightBar', '50%'));
        await pathTest('DATA-03', () => update('TEMPERATURE', 37, 'heatBar', '37%'));
        await pathTest('DATA-04', async () => {
            await page.evaluate(() => window.fixtureSockets[0].onclose());
            await page.waitForFunction(() => window.fixtureSockets.length === 2, null, { timeout: 7000 });
            await update('MEMORY', 0.25, 'memoryBar', '25%', 1);
        });

        await run(report.security, 'SEC-01', async () => {
            await page.setViewportSize({ width: 390, height: 844 });
            await buttons.nth(0).evaluate(node => node.setAttribute('data-target', '<img src=x onerror="window.injectionExecuted=true">'));
            await buttons.nth(0).click();
            assert.equal(await panels.nth(0).isVisible(), true);
            assert.equal(await page.locator('img[src="x"]').count(), 0);
            assert.equal(await page.evaluate(() => window.injectionExecuted), undefined);
        });
        await run(report.security, 'SEC-02', async () => {
            await page.setViewportSize({ width: 1280, height: 900 });
            await toggle.evaluate(node => node.setAttribute('href', 'javascript:window.injectionExecuted=true'));
            await toggle.click();
            await open(dropdown, true);
            assert.equal(await page.evaluate(() => window.injectionExecuted), undefined);
        });
        await run(report.security, 'SEC-03', async () => {
            await page.goto(origin + '/?security=hash#' + encodeURIComponent('<img src=x onerror="window.injectionExecuted=true">'));
            await toggle.click();
            await open(dropdown, true);
            assert.equal(await page.locator('img[src="x"]').count(), 0);
            assert.equal(await page.evaluate(() => window.injectionExecuted), undefined);
            assert.equal(await page.locator('script[src*="bootstrap"], script[src*="jquery"]').count(), 0);
        });
        assert.deepEqual(pageErrors, [], 'Browser script exceptions');
        report.coverage = verify(await fs.readFile(path.resolve(__dirname, '../GUI_PATHS.md'), 'utf8'), report, runId);
        assert(report.paths.every(result => result.status === 'passed'), 'A GUI path failed');
        assert(report.security.every(result => result.status === 'passed'), 'A security regression failed');
    } finally {
        report.finishedAt = new Date().toISOString();
        await fs.writeFile(path.join(output, 'gui-path-coverage.json'), JSON.stringify(report, null, 2) + '\n');
        if (browser) await browser.close();
        await new Promise(resolve => server.close(resolve));
    }
}

main().catch(error => { console.error(error); process.exitCode = 1; });
