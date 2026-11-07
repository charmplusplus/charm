#!/usr/bin/env python3
"""kineto_merge.py --out trace.json [--steps A:B] [--all] <dir with rank_<r>.json>

Join the per-rank torch.profiler traces of one run (fastmoe_moe.py --profile) into one Chrome/Perfetto trace
(open at https://ui.perfetto.dev): one process row per rank, its GPU streams and host thread as rows, time from
the earliest rank's start of step A. The ranks' timestamps are wall-clock based (the step starts agree across
the two nodes within a millisecond in job 23581139), so they are joined as they are. Kept: kernels, copies,
memsets, the "step N" ranges (host and GPU) and the host ops; --all keeps every event (the CUDA runtime calls
and the flow arrows, several times the size). --steps keeps the window from step A's start to step B+1's."""
import argparse, glob, json, os, re

KEEP = ('kernel', 'gpu_memcpy', 'gpu_memset', 'gpu_user_annotation', 'user_annotation', 'cpu_op')

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', required=True)
    ap.add_argument('--steps', default=None, help='first:last step kept, inclusive')
    ap.add_argument('--all', action='store_true')
    ap.add_argument('dir')
    a = ap.parse_args()
    paths = sorted(glob.glob(os.path.join(a.dir, 'rank_*.json')), key=lambda s: int(re.search(r'rank_(\d+)', s).group(1)))
    traces = [(int(re.search(r'rank_(\d+)', p).group(1)), json.load(open(p))) for p in paths]
    A = B = None
    if a.steps: A, B = (int(v) for v in a.steps.split(':'))
    def step_start(ev, n):
        c = [e['ts'] for e in ev if e.get('ph') == 'X' and e.get('name') == 'step %d' % n and e.get('cat') == 'user_annotation']
        return min(c) if c else None
    t0 = min(step_start(t['traceEvents'], A) if A else min(e['ts'] for e in t['traceEvents'] if 'ts' in e) for _, t in traces)
    out = []
    for rank, t in traces:
        ev = t['traceEvents']
        w0 = step_start(ev, A) if A else None
        w1 = step_start(ev, B + 1) if B else None
        host = t.get('host_name', '').split('.')[0]
        out.append({'ph': 'M', 'name': 'process_name', 'pid': rank, 'args': {'name': 'rank %d %s' % (rank, host)}})
        out.append({'ph': 'M', 'name': 'process_sort_index', 'pid': rank, 'args': {'sort_index': rank}})
        for e in ev:
            if e.get('ph') == 'M':
                if e.get('name') in ('thread_name', 'thread_sort_index'):
                    e = dict(e); e['pid'] = rank; out.append(e)
                continue
            if not a.all and e.get('cat') not in KEEP: continue
            ts = e.get('ts')
            if ts is None: continue
            if w0 is not None and ts + e.get('dur', 0) < w0: continue
            if w1 is not None and ts >= w1: continue
            e = dict(e); e['pid'] = rank; e['ts'] = ts - t0
            out.append(e)
    with open(a.out, 'w') as f:
        json.dump({'displayTimeUnit': 'ms', 'traceEvents': out}, f, separators=(',', ':'))
    print('%s: %d events from %d ranks, %.1f MB' % (a.out, len(out), len(traces), os.path.getsize(a.out) / 1e6))

if __name__ == '__main__':
    main()
