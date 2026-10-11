#!/usr/bin/env python3
"""gputrace_merge.py --out trace.json [--last SECONDS | --window T0:T1] [--no-host] [--idle] <dir>

Join the +gputrace files of one run into one Chrome/Perfetto trace (open at https://ui.perfetto.dev):
  gputrace.<host>.<pid>.proc   the process's CUPTI records: K kernel, M memcpy, S memset, N kernel names,
                               T object token -> LB object id; header carries a (realtime, CUPTI) clock pair
  gputrace.<host>.<pid>.pe<N>  the PE's E entry-method and I idle ranges, P entry names, O element id -> index,
                               L message flights (send, envelope, landed, kind, tier) from the application
One process row per rank (ordered by first PE) with a thread per GPU stream and per PE. Every record is on the
CUPTI clock; the clock pair lines processes on different nodes up to the nodes' clock agreement (NTP).
--last 1.0 keeps the final second of GPU activity (about 20 sph3d steps), --window is seconds since the
earliest process start. Kernels carry the array index of the object the runtime attributed them to.
Two counter tracks per GPU, from the kernel records in --bin ms bins (default 1): "GPUd busy %" (share of the bin with
at least one kernel running) and "GPUd kernels in flight" (time-weighted mean concurrency); the per-GPU means over the
window are printed. Copies are not counted. --no-util leaves them out.
A PE row shows only its entry methods: the empty space between them is the PE's idle time. --idle also
draws the runtime's idle ranges as events (gap_report.py reads them from the files either way).
Application message records (L) become per-rank counter tracks "msg latency ms <tier>" (payload usable minus
sent, one point per message at its landing time; tiers: same process / same node / other node) and instant
events on the PE row; gap_report.py prints their distribution."""
import argparse, glob, json, os, re, sys
TIER = {0: 'same process', 1: 'same node', 2: 'other node'}
KIND = {0: 'halo', 1: 'migration', 2: 'halo2'}

COPY = {0: 'memcpy', 1: 'HtoD', 2: 'DtoH', 3: 'HtoA', 4: 'AtoH', 5: 'AtoA', 6: 'AtoD', 7: 'DtoA', 8: 'DtoD', 9: 'HtoH', 10: 'PtoP'}

def header(path):
    with open(path) as f:
        line = f.readline()
    if not line.startswith('# gputrace'):
        raise SystemExit(f'{path}: not a gputrace file')
    return dict(kv.split('=', 1) for kv in line.split() if '=' in kv)

def short_kernel(name):
    s = name
    if s.startswith('void '): s = s[5:]
    for cut in ('(', '<'):
        i = s.find(cut)
        if i > 0: s = s[:i]
    return s.strip() or name

def idx3(v):
    v = int(v)
    return f'{v // 1000000},{v // 1000 % 1000},{v % 1000}'

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--last', type=float, default=None, help='keep the last SECONDS of GPU activity')
    ap.add_argument('--window', default=None, help='T0:T1 seconds since the earliest process start')
    ap.add_argument('--no-host', action='store_true', help='GPU rows only')
    ap.add_argument('--idle', action='store_true', help='also draw the PE idle ranges as events (default: empty space is idle)')
    ap.add_argument('--streams', action='store_true', help='one row per CUDA stream instead of packed concurrency lanes per GPU')
    ap.add_argument('--bin', type=float, default=1.0, help='bin width in ms for the per-GPU busy %% and kernels-in-flight counters')
    ap.add_argument('--no-util', action='store_true', help='no per-GPU utilization counter tracks')
    ap.add_argument('dir')
    a = ap.parse_args()
    procs = {}
    for path in sorted(glob.glob(os.path.join(a.dir, 'gputrace.*.proc'))):
        h = header(path)
        key = (h['host'], int(h['pid']))
        procs[key] = dict(path=path, pe0=int(h['pe0']), off=int(h['realtime_ns']) - int(h['cupti_ns']), real=int(h['realtime_ns']), pes=[])
    if not procs:
        raise SystemExit(f'no gputrace.*.proc in {a.dir}')
    for path in sorted(glob.glob(os.path.join(a.dir, 'gputrace.*.pe*'))):
        h = header(path)
        key = (h['host'], int(h['pid']))
        if key in procs: procs[key]['pes'].append((int(h['pe']), path))
    order = sorted(procs, key=lambda k: procs[k]['pe0'])
    base = min(p['real'] for p in procs.values())
    # window in absolute ns
    w0 = w1 = None
    if a.last is not None:
        tmax = 0
        for p in procs.values():
            with open(p['path']) as f:
                for line in f:
                    if line[0] == 'K':
                        e = int(line.split()[4]) + p['off']
                        if e > tmax: tmax = e
        w1 = tmax; w0 = tmax - int(a.last * 1e9)
    elif a.window:
        s0, s1 = a.window.split(':'); w0 = base + int(float(s0) * 1e9); w1 = base + int(float(s1) * 1e9)
    def keep(s, e): return w0 is None or (e >= w0 and s <= w1)
    out = open(a.out, 'w')
    out.write('{"displayTimeUnit":"ms","traceEvents":[\n')
    first = [True]
    def emit(ev):
        if not first[0]: out.write(',\n')
        first[0] = False
        out.write(json.dumps(ev, separators=(',', ':')))
    counts = {}
    for rank, key in enumerate(order):
        p = procs[key]; off = p['off']
        host, pid = key
        names, tokens, objidx, epnames = {}, {}, {}, {}
        n = 0
        # object id -> index from the PE files (first pass over small P/O lines only)
        for pe, path in p['pes']:
            with open(path) as f:
                for line in f:
                    if line[0] == 'O':
                        _, oid, idx = line.split(); objidx[int(oid)] = int(idx)
                    elif line[0] == 'P':
                        parts = line.split(None, 2); epnames[int(parts[1])] = parts[2].strip()
        emit({'ph': 'M', 'name': 'process_name', 'pid': rank, 'args': {'name': f'rank {rank} {host} pid {pid} PE {p["pe0"]}+'}})
        emit({'ph': 'M', 'name': 'process_sort_index', 'pid': rank, 'args': {'sort_index': rank}})
        seen = set()
        def row(tid, name, sort):
            if tid in seen: return
            seen.add(tid)
            emit({'ph': 'M', 'name': 'thread_name', 'pid': rank, 'tid': tid, 'args': {'name': name}})
            emit({'ph': 'M', 'name': 'thread_sort_index', 'pid': rank, 'tid': tid, 'args': {'sort_index': sort}})
        gpu = {}   # (device, stream) -> events; emitted below as stream rows or packed lanes
        from collections import defaultdict
        gpu = defaultdict(list)
        with open(p['path']) as f:
            for line in f:
                c = line[0]
                if c == 'N':
                    parts = line.split(None, 2); names[int(parts[1])] = short_kernel(parts[2].strip())
                elif c == 'T':
                    _, tok, om, oid = line.split(); tokens[int(tok)] = int(oid)
                elif c == 'K':
                    _, dev, st, s, e, nid, tok = line.split()
                    s = int(s) + off; e = int(e) + off
                    if not keep(s, e): continue
                    ev = {'ph': 'X', 'cat': 'kernel', 'pid': rank, 'ts': (s - base) / 1000.0, 'dur': (e - s) / 1000.0,
                          'name': names.get(int(nid), f'kernel {nid}'), 'args': {'stream': int(st)}}
                    oid = tokens.get(int(tok))
                    if oid is not None and oid in objidx: ev['args']['idx'] = idx3(objidx[oid])
                    gpu[(int(dev), int(st))].append(ev); n += 1
                elif c in 'MS':
                    _, dev, st, s, e, nb, kind = line.split()
                    s = int(s) + off; e = int(e) + off
                    if not keep(s, e): continue
                    gpu[(int(dev), int(st))].append({'ph': 'X', 'cat': 'memcpy' if c == 'M' else 'memset', 'pid': rank, 'ts': (s - base) / 1000.0,
                          'dur': (e - s) / 1000.0, 'name': COPY.get(int(kind), 'memcpy') if c == 'M' else 'memset', 'args': {'bytes': int(nb), 'stream': int(st)}})
                    n += 1
        if not a.no_util:
            # Per-GPU utilization counters from the kernel intervals: sweep the starts and ends, spread each constant-
            # concurrency segment over the bins it covers.
            bydevk = defaultdict(list)
            for (dev, st), evs in gpu.items():
                bydevk[dev].extend((ev['ts'], ev['ts'] + ev['dur']) for ev in evs if ev['cat'] == 'kernel')
            binus = a.bin * 1000.0
            for dev, iv in sorted(bydevk.items()):
                if not iv: continue
                t0 = (w0 - base) / 1000.0 if w0 is not None else min(s for s, e in iv)
                t1 = (w1 - base) / 1000.0 if w1 is not None else max(e for s, e in iv)
                nb = max(1, int((t1 - t0) / binus) + 1)
                busy = [0.0] * nb; infl = [0.0] * nb
                pts = sorted([(s, 1) for s, e in iv] + [(e, -1) for s, e in iv])
                c = 0; tprev = pts[0][0]
                for t, d in pts:
                    if t > tprev and c > 0:
                        lo = max(tprev, t0); hi = min(t, t1)
                        b = int((lo - t0) / binus)
                        while lo < hi and b < nb:
                            be = t0 + (b + 1) * binus; seg = min(hi, be) - lo
                            busy[b] += seg; infl[b] += c * seg
                            lo = be; b += 1
                    c += d; tprev = t
                for b in range(nb):
                    ts = t0 + b * binus
                    emit({'ph': 'C', 'name': f'GPU{dev} busy %', 'pid': rank, 'ts': ts, 'args': {'busy': round(100.0 * busy[b] / binus, 1)}})
                    emit({'ph': 'C', 'name': f'GPU{dev} kernels in flight', 'pid': rank, 'ts': ts, 'args': {'kernels': round(infl[b] / binus, 2)}})
                span = max(binus, t1 - t0)
                print(f'  rank {rank} GPU{dev}: busy {100.0 * sum(busy) / span:.1f}%, kernels in flight {sum(infl) / span:.2f} mean over {span / 1000.0:.0f} ms')
        if a.streams:
            for (dev, st), evs in sorted(gpu.items()):
                tid = 1000 + st; row(tid, f'GPU{dev} stream {st}', tid)
                for ev in evs: ev['tid'] = tid; emit(ev)
        else:
            # Greedy lane packing per GPU: each event goes to the lowest lane that is free at its start, so a GPU
            # shows as a handful of rows (its real concurrency) instead of one row per stream.
            bydev = defaultdict(list)
            for (dev, st), evs in gpu.items(): bydev[dev].extend(evs)
            for dev, evs in sorted(bydev.items()):
                evs.sort(key=lambda ev: ev['ts'])
                lane_end = []
                for ev in evs:
                    for li, le in enumerate(lane_end):
                        if le <= ev['ts']: break
                    else:
                        li = len(lane_end); lane_end.append(0.0)
                    lane_end[li] = ev['ts'] + ev['dur']
                    tid = 1000 + 100 * dev + li; row(tid, f'GPU{dev} lane {li}', tid)
                    ev['tid'] = tid; emit(ev)
        if not a.no_host:
            for pe, path in sorted(p['pes']):
                row(pe, f'PE {pe}', pe)
                with open(path) as f:
                    for line in f:
                        c = line[0]
                        if c == 'E':
                            _, s, e, ep, idx = line.split()
                            s = int(s) + off; e = int(e) + off
                            if not keep(s, e): continue
                            ev = {'ph': 'X', 'cat': 'entry', 'pid': rank, 'tid': pe, 'ts': (s - base) / 1000.0, 'dur': (e - s) / 1000.0,
                                  'name': epnames.get(int(ep), f'ep {ep}')}
                            if int(idx) >= 0: ev['args'] = {'idx': idx3(idx)}
                            emit(ev); n += 1
                        elif c == 'L':
                            _, ts, te, tl, kind, tier = line.split()
                            ts = int(ts) + off; tl = int(tl) + off; te = int(te) + off
                            if not keep(tl, tl): continue
                            emit({'ph': 'C', 'name': f'msg latency ms {TIER.get(int(tier), tier)}', 'pid': rank, 'ts': (tl - base) / 1000.0,
                                  'args': {'ms': round((tl - ts) / 1e6, 3)}})
                            emit({'ph': 'i', 'cat': 'msg', 'pid': rank, 'tid': pe, 'ts': (tl - base) / 1000.0, 's': 't',
                                  'name': KIND.get(int(kind), 'msg'),
                                  'args': {'lat_ms': round((tl - ts) / 1e6, 3), 'env_ms': round((te - ts) / 1e6, 3), 'tier': int(tier)}})
                            n += 1
                        elif c == 'I':
                            if not a.idle: continue
                            _, s, e = line.split()
                            s = int(s) + off; e = int(e) + off
                            if not keep(s, e): continue
                            emit({'ph': 'X', 'cat': 'idle', 'pid': rank, 'tid': pe, 'ts': (s - base) / 1000.0, 'dur': (e - s) / 1000.0, 'name': 'idle'})
                            n += 1
        counts[rank] = n
    out.write('\n]}\n'); out.close()
    span = '' if w0 is None else f' window {(w0 - base) / 1e9:.2f}..{(w1 - base) / 1e9:.2f} s'
    print(f'wrote {a.out}{span}: ' + ', '.join(f'rank {r}: {n}' for r, n in sorted(counts.items())) + ' events', file=sys.stderr)

if __name__ == '__main__':
    main()
