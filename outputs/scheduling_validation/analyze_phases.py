"""Describe the completed finite burst; no additional inference or causal claims."""
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'outputs/scheduling_results'
PHASES = [('initial_short', 0, 96), ('long_context', 96, 144), ('recovery_short', 144, 240)]
rows = []
for rep in range(3):
    for policy in ['default', 'fixed32', 'context_budget']:
        folder = DATA / f'repeat-{rep}' / policy
        requests = json.loads((folder / 'requests.json').read_text())
        telemetry = json.loads((folder / 'telemetry.json').read_text())
        summary = json.loads((folder / 'summary.json').read_text())
        phase_rows = []
        for name, lo, hi in PHASES:
            subset = [r for r in requests if lo <= r['id'] < hi]
            phase_rows.append({
                'phase': name, 'request_id_start': lo, 'request_id_stop_exclusive': hi,
                'requests': len(subset), 'slo_met': sum(r['slo_met'] for r in subset),
                'slo_attainment_percent': 100 * sum(r['slo_met'] for r in subset) / len(subset),
                'median_admission_wait_s': statistics.median(r['admission_wait_s'] for r in subset),
                'median_ttft_s': statistics.median(r['ttft_s'] for r in subset),
            })
        # Reconstruct active requests from exact admission/completion timestamps.
        # A request belongs to the engine once admitted; this is not GPU kernel concurrency.
        events = sorted([(r['admitted_s'], 1) for r in requests] +
                        [(r['completed_s'], -1) for r in requests])
        active = peak = 0
        previous = area = 0.0
        for stamp, delta in events:
            area += active * (stamp - previous)
            active += delta
            peak = max(peak, active)
            previous = stamp
        rows.append({
            'repeat': rep, 'policy': policy,
            'duration_s': summary['duration_s'],
            'last_arrival_s': max(r['arrival_s'] for r in requests),
            'drain_after_last_arrival_s': summary['duration_s'] - max(r['arrival_s'] for r in requests),
            'slo_met': sum(r['slo_met'] for r in requests),
            'slo_attainment_percent': summary['slo_attainment_percent'],
            'slo_goodput_rps': summary['slo_goodput_rps'],
            'admitted_requests_time_weighted_mean_from_events': area / summary['duration_s'],
            'admitted_requests_peak_from_events': peak,
            'telemetry_admitted_requests_arithmetic_mean': statistics.mean(s['admitted_requests'] for s in telemetry),
            'telemetry_admitted_requests_max': max(s['admitted_requests'] for s in telemetry),
            'telemetry_reserved_token_estimate_max': max(s['reserved_token_estimate'] for s in telemetry),
            'phases': phase_rows,
        })
result = {
    'source': 'outputs/scheduling_results/repeat-{0,1,2}/{default,fixed32,context_budget}/{requests,telemetry,summary}.json',
    'interpretation': 'Descriptive post-hoc phase decomposition of finite replay. Admission waiting is outside vLLM; active admitted requests are not simultaneous GPU kernels. No causal or capacity inference.',
    'observations': rows,
}
destination = DATA / 'analysis/phase_diagnostics.json'
destination.write_text(json.dumps(result, indent=2) + '\n')
for r in rows:
    print(f"r{r['repeat']} {r['policy']}: pass={r['slo_met']}/240 duration={r['duration_s']:.2f}s drain={r['drain_after_last_arrival_s']:.2f}s admitted mean={r['admitted_requests_time_weighted_mean_from_events']:.1f} max={r['admitted_requests_peak_from_events']}")
    for p in r['phases']:
        print(f"  {p['phase']}: pass={p['slo_met']}/{p['requests']} median gate wait={p['median_admission_wait_s']:.3f}s TTFT={p['median_ttft_s']:.3f}s")
