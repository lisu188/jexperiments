#!/usr/bin/env python3
"""Compact fixed-search JSONL without losing any round/trial data or numerical evidence."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

DEFAULT_TOOL = Path(__file__).resolve().parents[2] / 'tools' / 'summarize_search_benchmark.py'
DEVICE_FIELDS = ('device', 'deviceIdentity', 'engine', 'backend', 'precision', 'sigmoid', 'kernel', 'route', 'simdBits')
PHASES = ('open', 'training', 'scoring', 'snapshot', 'close')
TIMING_COLUMNS = ['candidateManifestIndex', 'seed', 'elapsedNanos', *[key + 'Nanos' for key in PHASES], 'deviceId', 'outcomeId']
SAVED_ROUND_FIELDS = {'run', 'canonicalSha256', 'trialTimings', 'candidateSummaries'}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def load_tool(path):
    spec = importlib.util.spec_from_file_location('search_summary', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reconstruct_round(evidence, saved, lookups=None):
    if lookups is None:
        lookups = ({row['id']: row for row in evidence['runs']},
                   {row['id']: row for row in evidence['uniqueOutcomes']})
    runs, outcomes = lookups
    protocol = evidence['protocols'][runs[saved['run']]['protocol']]
    record = {key: value for key, value in saved.items() if key not in SAVED_ROUND_FIELDS}
    record['type'] = 'round'
    record['candidates'] = []
    for candidate in saved['candidateSummaries']:
        candidate = dict(candidate)
        index = protocol['manifest'].index(candidate['hidden'])
        trials = []
        for timing in saved['trialTimings']:
            if timing[0] != index:
                continue
            outcome = outcomes[timing[-1]]
            trial = dict(outcome['trial'])
            trial.update(evidence['devices'][timing[-2]])
            trial['elapsedNanos'] = timing[2]
            trial['phaseNanos'] = None if all(value is None for value in timing[3:8]) else dict(zip(PHASES, timing[3:8]))
            trials.append(trial)
        candidate['trials'] = trials
        record['candidates'].append(candidate)
    return record


def verify(evidence):
    runs = {row['id']: row for row in evidence['runs']}
    outcomes = {row['id']: row for row in evidence['uniqueOutcomes']}
    assert len(runs) == len(evidence['runs']) and len(outcomes) == len(evidence['uniqueOutcomes'])
    for saved in evidence['rounds']:
        rebuilt = reconstruct_round(evidence, saved, (runs, outcomes))
        assert digest(rebuilt) == saved['canonicalSha256'], 'Round reconstruction differs'
        assert len(saved['trialTimings']) == sum(len(candidate['trials']) for candidate in rebuilt['candidates'])
    return dict(rounds=len(evidence['rounds']), trials=sum(len(row['trialTimings']) for row in evidence['rounds']),
                distinctOutcomes=len(outcomes), allRoundCanonicalHashesMatch=True)


def build(paths, raw_root, tool_path, status, require_exit):
    tool = load_tool(tool_path)
    summary = tool.summarize(paths)
    protocols, runs, rounds, outcomes, devices, event_rows = {}, [], [], {}, {}, []
    parsed_originals = []
    seen_run_ids = set()
    for path in paths:
        path = path.resolve()
        raw = path.read_bytes()
        rows = [json.loads(line) for line in raw.decode('utf-8-sig').splitlines() if line.strip()]
        environment = rows[0]
        if environment.get('mode') != 'fixed':
            raise ValueError('Only fixed reports are supported: ' + str(path))
        protocol = tool.environment_protocol(environment)
        protocol_id = digest(protocol)[:16]
        protocols[protocol_id] = protocol
        run_id = path.stem
        if run_id in seen_run_ids:
            raise ValueError('Duplicate run name: ' + run_id)
        seen_run_ids.add(run_id)
        relative = str(path.relative_to(raw_root))
        process_path = path.with_suffix('.log.process.json')
        process = json.loads(process_path.read_text(encoding='utf-8-sig')) if process_path.exists() else None
        if require_exit and (process is None or process.get('exitCode') != 0):
            raise ValueError('Missing successful process exit: ' + str(process_path))
        if process is not None and process.get('revision') != environment['sourceRevision']:
            raise ValueError('Process/report source revision mismatch: ' + str(path))
        expected_cases = len(protocol['workers']) * len(protocol['backends']) * len(protocol['executions'])
        summaries = [row for row in rows if row['type'] == 'summary']
        complete_output = len(summaries) == expected_cases and all(row.get('completeRounds') == protocol['repeats'] for row in summaries)
        if require_exit and not complete_output:
            raise ValueError('Output is incomplete: ' + str(path))
        process_info = dict(sourceFile=str(process_path.relative_to(raw_root)), sha256=hashlib.sha256(process_path.read_bytes()).hexdigest(), record=process) if process is not None else None
        runs.append(dict(id=run_id, protocol=protocol_id, sourceFile=relative, sha256=hashlib.sha256(raw).hexdigest(),
                         bytes=len(raw), started=environment['started'], pid=environment['pid'], orderOffset=environment.get('orderOffset'),
                         process=process_info, finalSummariesComplete=complete_output, summaries=summaries))
        indices = {tuple(shape): index for index, shape in enumerate(protocol['manifest'])}
        for row in rows:
            if row['type'] != 'round':
                if row['type'] not in ('environment', 'summary'):
                    event_rows.append(dict(run=run_id, record=row))
                continue
            saved = {key: value for key, value in row.items() if key not in ('type', 'candidates')}
            saved.update(run=run_id, canonicalSha256=digest(row), trialTimings=[], candidateSummaries=[])
            for candidate in row['candidates']:
                candidate_index = indices[tuple(candidate['hidden'])]
                saved['candidateSummaries'].append({key: value for key, value in candidate.items() if key != 'trials'})
                for trial in candidate['trials']:
                    device = {key: trial[key] for key in DEVICE_FIELDS}
                    device_id = digest(device)[:16]
                    devices[device_id] = device
                    core = {key: value for key, value in trial.items() if key not in (*DEVICE_FIELDS, 'elapsedNanos', 'phaseNanos')}
                    outcome_key = dict(protocol=protocol_id, candidate=candidate_index, trial=core)
                    outcome_id = digest(outcome_key)[:20]
                    if outcome_id not in outcomes:
                        outcomes[outcome_id] = dict(id=outcome_id, **outcome_key,
                            bestParametersSha256=digest(trial['parametersAtBest']), measuredObservations=0, warmupObservations=0)
                    outcomes[outcome_id]['warmupObservations' if row['round'] < 0 else 'measuredObservations'] += 1
                    phases = trial['phaseNanos'] or {}
                    saved['trialTimings'].append([candidate_index, trial['seed'], trial['elapsedNanos'],
                        *[phases.get(key) for key in PHASES], device_id, outcome_id])
            rounds.append(saved)
            parsed_originals.append(row)
    comparisons = []
    for row in summary['comparisons']:
        row = dict(row)
        row['protocol'] = digest(row['protocol'])[:16]
        comparisons.append(row)
    host_files = [dict(sourceFile=str(path.relative_to(raw_root)), bytes=path.stat().st_size,
                       sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for path in sorted({report.resolve().parent / 'host.txt' for report in paths if (report.resolve().parent / 'host.txt').exists()})]
    revisions = sorted({protocol['sourceRevision'] for protocol in protocols.values()})
    evidence = dict(schemaVersion=2, status=status, computeRevisions=revisions,
        analysis=dict(script=str(tool_path), scriptSha256=hashlib.sha256(tool_path.read_bytes()).hexdigest(),
                      compactorSha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),
        protocols=protocols, runs=runs, hostReports=host_files, devices=devices,
        trialTimingColumns=TIMING_COLUMNS, rounds=rounds, uniqueOutcomes=list(outcomes.values()),
        comparisons=comparisons, exclusions=summary['incompleteOrFailed'], events=event_rows,
        pairingScope=summary['pairingScope'], parityScope=summary['parityScope'],
        limitations=['Qualification applies only to each compatible protocol/engine/backend/worker group; different hosts are never pooled.',
                     'Warmup-only outcomes are retained separately and excluded from measured timing/parity distributions.',
                     'Within-engine numerical parity does not establish cross-engine or cross-host arithmetic equivalence.',
                     'Per-trial phase timings overlap across models and place boundary scoring differently on CPU versus CUDA.',
                     'Raw JSONL byte hashes retain provenance; complete parsed round records can be reconstructed from this artifact without the raw files.'])
    if any(run['process'] is None for run in runs):
        evidence['limitations'].append('Some process exit sidecars were not included; verify workflow-level execution status separately.')
    evidence['reconstructionVerification'] = verify(evidence)
    lookup = ({row['id']: row for row in runs}, outcomes)
    assert all(reconstruct_round(evidence, saved, lookup) == original for saved, original in zip(rounds, parsed_originals))
    return evidence, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reports', nargs='+', type=Path)
    parser.add_argument('--raw-root', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--summary-output', required=True, type=Path)
    parser.add_argument('--tool', default=DEFAULT_TOOL, type=Path)
    parser.add_argument('--status', choices=('preliminary', 'validated-fixed-work', 'inconclusive-mixed-hardware'), required=True)
    parser.add_argument('--require-process-exit', action='store_true')
    args = parser.parse_args()
    evidence, summary = build(args.reports, args.raw_root.resolve(), args.tool.resolve(), args.status, args.require_process_exit)
    args.output.write_text(json.dumps(evidence, separators=(',', ':'), allow_nan=False) + '\n')
    args.summary_output.write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    check = verify(json.loads(args.output.read_text()))
    print(json.dumps(dict(status=args.status, bytes=args.output.stat().st_size,
        comparisons=len(summary['comparisons']), qualified=sum(row['qualifies'] for row in summary['comparisons']),
        exclusions=len(summary['incompleteOrFailed']), reconstruction=check)))


if __name__ == '__main__':
    main()
