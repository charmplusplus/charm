#!/usr/bin/env python3
"""kernelclass.py [--steps A:B] [--md] <trace dir>...

Per-step GPU time by kernel class, per rank, from either trace the MoE comparison records:
  the runtime's +gputrace files (gputrace.<host>.<pid>.proc / .pe<N>): kernels (K), copies (M) and memsets (S) on
      the CUPTI clock; step n of a rank starts at that process's n-th Dispatcher::runStep entry (E record of its
      first PE), so the window of steps A..B is [start of runStep A, start of runStep B+1);
  torch.profiler's Chrome traces (rank_<r>.json, fastmoe_moe.py --profile): "kernel", "gpu_memcpy" and
      "gpu_memset" events; step n's GPU window is the span of its "step n" gpu_user_annotation events (the CPU
      annotation's span if the trace has none), and steps A..B run from step A's start to step B+1's start.
Classes by kernel name: gemm (cuBLAS, cutlass, xmma), nccl (ncclDevKernel), copy (memcpy/memset records), other
(the systems' own kernels: gather, combine, scatter, relu, SGD, reductions). Printed per rank: the window's
ms/step, the summed kernel time per class per step (concurrent kernels add, so a class can exceed the step),
and two busy shares of the window: kernels other than NCCL's, and all kernels. A trace of the example made with
+gputracenocopy has no copy records; its exchange is RDMA and copy-engine traffic, which occupies no SM."""
import argparse, glob, json, os, re, sys
from collections import defaultdict

GEMM = re.compile(r'gemm|cutlass|xmma|s16816|sgemm|hgemm|gemv|splitk|cublas', re.I)
NCCL = re.compile(r'nccl', re.I)

def classify(name):
    if NCCL.search(name): return 'nccl'
    if GEMM.search(name): return 'gemm'
    return 'other'

def union_ms(iv):
    """total length of the union of [s, e) intervals, in the intervals' unit"""
    tot = 0.0; cs = ce = None
    for s, e in sorted(iv):
        if cs is None or s > ce:
            if cs is not None: tot += ce - cs
            cs, ce = s, e
        elif e > ce: ce = e
    if cs is not None: tot += ce - cs
    return tot

def header(path):
    with open(path) as f:
        line = f.readline()
    return dict(kv.split('=', 1) for kv in line.split() if '=' in kv)

def load_gputrace(d, A, B):
    """-> {rank: (window_ns, {class: [(s, e)]})} on the CUPTI clock (ns)"""
    procs = {}
    for path in glob.glob(os.path.join(d, 'gputrace.*.proc')):
        h = header(path); procs[(h['host'], int(h['pid']))] = dict(path=path, pe0=int(h['pe0']), pes=[])
    for path in glob.glob(os.path.join(d, 'gputrace.*.pe*')):
        h = header(path); key = (h['host'], int(h['pid']))
        if key in procs: procs[key]['pes'].append((int(h['pe']), path))
    out = {}
    for rank, key in enumerate(sorted(procs, key=lambda k: procs[k]['pe0'])):
        p = procs[key]
        pe0path = min(p['pes'])[1]
        epnames, starts = {}, []
        with open(pe0path) as f:
            for line in f:
                if line[0] == 'P':
                    parts = line.split(None, 2); epnames[int(parts[1])] = parts[2].strip()
                elif line[0] == 'E':
                    _, s, e, ep, idx = line.split()
                    if epnames.get(int(ep)) == 'Dispatcher::runStep': starts.append(int(s))
        starts.sort()
        if len(starts) < B + 1:
            sys.exit('%s: rank %d has %d Dispatcher::runStep entries, steps %d..%d need %d' % (d, rank, len(starts), A, B, B + 1))
        w0, w1 = starts[A - 1], starts[B]
        iv = defaultdict(list); names = {}
        with open(p['path']) as f:
            for line in f:
                c = line[0]
                if c == 'N':
                    parts = line.split(None, 2); names[int(parts[1])] = parts[2].strip()
                elif c == 'K':
                    _, dev, st, s, e, nid, tok = line.split(); s = int(s); e = int(e)
                    if e <= w0 or s >= w1: continue
                    iv[classify(names.get(int(nid), ''))].append((max(s, w0), min(e, w1)))
                elif c in 'MS':
                    _, dev, st, s, e, nb, kind = line.split(); s = int(s); e = int(e)
                    if e <= w0 or s >= w1: continue
                    iv['copy'].append((max(s, w0), min(e, w1)))
        out[rank] = (w1 - w0, iv, 1e6)
    return out

def load_kineto(d, A, B):
    """-> {rank: (window_us, {class: [(s, e)]})} in microseconds"""
    out = {}
    for path in sorted(glob.glob(os.path.join(d, 'rank_*.json')), key=lambda s: int(re.search(r'rank_(\d+)', s).group(1))):
        rank = int(re.search(r'rank_(\d+)', path).group(1))
        with open(path) as f:
            ev = json.load(f)['traceEvents']
        gstep, cstep = defaultdict(list), {}
        for e in ev:
            if e.get('ph') != 'X': continue
            m = re.fullmatch(r'step (\d+)', e.get('name', ''))
            if not m: continue
            n = int(m.group(1))
            if e.get('cat') == 'gpu_user_annotation': gstep[n].append((e['ts'], e['ts'] + e['dur']))
            elif e.get('cat') == 'user_annotation': cstep[n] = (e['ts'], e['ts'] + e['dur'])
        def start(n):
            if n in gstep: return min(s for s, _ in gstep[n])
            if n in cstep: return cstep[n][0]
            sys.exit('%s: no "step %d" range in the trace' % (path, n))
        src = 'GPU' if A in gstep else 'CPU'
        w0, w1 = start(A), start(B + 1)
        iv = defaultdict(list)
        for e in ev:
            if e.get('ph') != 'X' or e.get('cat') not in ('kernel', 'gpu_memcpy', 'gpu_memset'): continue
            s = e['ts']; t = s + e['dur']
            if t <= w0 or s >= w1: continue
            c = 'copy' if e['cat'] != 'kernel' else classify(e['name'])
            iv[c].append((max(s, w0), min(t, w1)))
        out[rank] = (w1 - w0, iv, 1e3)
        if src == 'CPU': print('  %s: step windows from the CPU annotations (no gpu_user_annotation events)' % os.path.basename(path))
    return out

def report(d, data, A, B, md):
    n = B - A + 1
    rows = []
    for rank in sorted(data):
        w, iv, div = data[rank]
        ms = {c: sum(e - s for s, e in iv.get(c, [])) / div / n for c in ('gemm', 'other', 'nccl', 'copy')}
        allk = [x for c in ('gemm', 'other', 'nccl') for x in iv.get(c, [])]
        nonn = [x for c in ('gemm', 'other') for x in iv.get(c, [])]
        rows.append((rank, w / div / n, ms['gemm'], ms['other'], ms['nccl'], ms['copy'],
                     100.0 * union_ms(nonn) / w, 100.0 * union_ms(allk) / w))
    hdr = ('rank', 'ms/step', 'gemm', 'other', 'nccl', 'copies', 'busy % no nccl', 'busy % all')
    print('%s, steps %d-%d, per step:' % (d, A, B))
    if md:
        print('| ' + ' | '.join(hdr) + ' |'); print('|' + '---:|' * len(hdr))
        for r in rows: print('| %d | %.1f | %.1f | %.1f | %.1f | %.1f | %.0f | %.0f |' % r)
    else:
        print('  %5s %8s %7s %7s %7s %7s %15s %11s' % hdr)
        for r in rows: print('  %5d %8.1f %7.1f %7.1f %7.1f %7.1f %15.0f %11.0f' % r)
    import statistics as st
    cols = list(zip(*rows))
    print('  %5s %8.1f %7.1f %7.1f %7.1f %7.1f %15.0f %11.0f' % (('mean',) + tuple(st.mean(c) for c in cols[1:])))
    print('  %5s %8.1f %7.1f %7.1f %7.1f %7.1f %15.0f %11.0f' % (('max',) + tuple(max(c) for c in cols[1:])))

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--steps', default='30:41', help='first:last step, inclusive (default 30:41)')
    ap.add_argument('--md', action='store_true', help='markdown tables')
    ap.add_argument('dirs', nargs='+')
    a = ap.parse_args()
    A, B = (int(v) for v in a.steps.split(':'))
    for d in a.dirs:
        if glob.glob(os.path.join(d, 'gputrace.*.proc')): data = load_gputrace(d, A, B)
        elif glob.glob(os.path.join(d, 'rank_*.json')): data = load_kineto(d, A, B)
        else: sys.exit('%s: neither gputrace.*.proc nor rank_*.json' % d)
        report(d, data, A, B, a.md)

if __name__ == '__main__':
    main()
