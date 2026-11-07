#!/usr/bin/env python3
"""stepchain.py <gputrace dir> [--step S | --steps A:B] [--total N]: one MoE step's dependency chain from the
runtime's +gputrace files (moe ... +gputrace DIR [+gputracenocopy]).

Per PE the dispatcher's step S: its period, when its gather kernel ran, when its token puts had all completed,
the first and last receiveOutput of the window (more than 64 when the next step's outputs land early), the
combine kernel (its start is when the step's last output was in), the drain and the GPU's busy time by kernel
class; the PE's idle time and its host time in dispatcher and expert entries. Per expert copy on the PE its
step S: start relative to the dispatcher's, the slab notes consumed in the window, the resumed entry (every
slab in, the balancing point passed), the kernels of its compute (the lane stream with the most kernel time
between resumed and computeDone: these traces carry no per-kernel object attribution), computeDone, its
output puts' completion, the gradient exchange and apply when replicated in training, the drain, and its
period. --steps A:B prints the means per PE and the hottest copy on each. Step S is the S-th
Dispatcher::runStep entry on the PE (--total is the run's step count, warmup included, 43 for -i 40 -u 3);
an expert copy's steps are counted back from its last, so a copy that left the PE before the end is skipped.
The idle 'last computeDone' list says whose output every dispatcher waited for.
Measured 2026-11-05 on the A40 nodes (README): in serving the hot copy's period is its compute plus the
completion of its four 16 MB output puts, and every dispatcher waits for that output."""

import argparse, glob, os, re, sys
from collections import defaultdict
ap = argparse.ArgumentParser()
ap.add_argument('dir'); ap.add_argument('--step', type=int, default=35); ap.add_argument('--steps', default=None)
ap.add_argument('--total', type=int, default=43, help='runStep count per PE (last step index)')
ap.add_argument('--copies', type=float, default=None, help='with --step: list the copy records of at least this many MB on each PE in the step, on the dispatcher clock')
ap.add_argument('--group', type=int, default=None, help='print the copies of this expert on one clock: start, computeDone, partials in, applyDone')
a = ap.parse_args()

def header(path):
    with open(path) as f: line = f.readline()
    return dict(kv.split('=', 1) for kv in line.split() if '=' in kv)

def kclass(n):
    if 'gatherKernel' in n: return 'gather'
    if 'combineKernel' in n: return 'combine'
    if 'sumSq' in n or 'sumPartials' in n: return 'chk'
    if 'applySgd' in n or 'addInto' in n or 'Adam' in n: return 'apply'
    if 'gemm' in n or 'cutlass' in n or 'splitKreduce' in n or 'relu' in n: return 'gemm'
    if 'castBf16' in n or 'initUniform' in n: return 'init'
    return 'other'

procs = {}
for path in sorted(glob.glob(os.path.join(a.dir, 'gputrace.*.proc'))):
    h = header(path); off = int(h['realtime_ns']) - int(h['cupti_ns'])
    names, tok2oid, K, M = {}, {}, [], []
    with open(path) as f:
        for line in f:
            c = line[0]
            if c == 'N':
                p = line.split(None, 2); names[int(p[1])] = p[2].strip()
            elif c == 'T':
                _, tok, om, oid = line.split(); tok2oid[int(tok)] = int(oid)
            elif c == 'K':
                _, dev, st, s, e, nid, tok = line.split()
                K.append((int(s) + off, int(e) + off, int(nid), int(tok), int(st)))
            elif c == 'M':
                _, dev, st, s, e, nb, kind = line.split()
                M.append((int(s) + off, int(e) + off, int(nb), int(kind), int(st)))
    K.sort(); M.sort()
    procs[(h['host'], int(h['pid']))] = dict(pe0=int(h['pe0']), off=off, names=names, tok2oid=tok2oid, K=K, M=M)

pes = {}
for path in sorted(glob.glob(os.path.join(a.dir, 'gputrace.*.pe*'))):
    h = header(path); key = (h['host'], int(h['pid'])); P = procs[key]; off = P['off']
    ep, oidx, E, I = {}, {}, [], []
    with open(path) as f:
        for line in f:
            c = line[0]
            if c == 'E':
                _, s, e, epi, obj = line.split(); E.append((int(s) + off, int(e) + off, int(epi), int(obj)))
            elif c == 'P':
                p = line.split(None, 2); ep[p[2].strip()] = int(p[1])
            elif c == 'O':
                _, oid, idx = line.split(); oidx[int(oid)] = int(idx)
            elif c == 'I':
                _, s, e = line.split(); I.append((int(s) + off, int(e) + off))
    E.sort()
    pes[int(h['pe'])] = dict(key=key, ep=ep, oidx=oidx, E=E, I=I, P=P)

def epid(pe, name): return pes[pe]['ep'].get(name, -999)
def ms(x): return x / 1e6

def disp_steps(pe):
    r = epid(pe, 'Dispatcher::runStep')
    return [e for e in pes[pe]['E'] if e[2] == r]

def expert_runsteps(pe, idx):
    r = epid(pe, 'Expert::runStep')
    return [e for e in pes[pe]['E'] if e[2] == r and e[3] == idx]

def within(lst, t0, t1): return [x for x in lst if t0 <= x[0] < t1]

def analyze(S, verbose):
    out = {}
    groupRows = []
    lastout = []   # (time computeDone, expert idx, pe) across PEs for the last-output question
    for pe in sorted(pes):
        d = pes[pe]; P = d['P']; E = d['E']
        ds = disp_steps(pe)
        if S >= len(ds): continue
        t0 = ds[S - 1][0]; t1 = ds[S][0]
        ev = within(E, t0, t1)
        def times(name):
            i = epid(pe, name); return [x for x in ev if x[2] == i]
        gath = times('Dispatcher::gathered'); sd = times('Dispatcher::sendDoneEven') + times('Dispatcher::sendDoneOdd')
        ro = times('Dispatcher::receiveOutput'); cb = times('Dispatcher::combined'); dr = times('Dispatcher::sendsDrained')
        sd.sort()
        # kernels on this GPU during the step
        Kw = [k for k in P['K'] if k[0] < t1 and k[1] > t0]
        busy = defaultdict(float)
        for s, e, nid, tok, st in Kw:
            busy[kclass(P['names'].get(nid, ''))] += ms(min(e, t1) - max(s, t0))
        idle = sum(ms(min(e, t1) - max(s, t0)) for s, e in d['I'] if s < t1 and e > t0)
        # the dispatcher's own entry time in the step
        hostd = sum(ms(e - s) for s, e, epi, obj in ev if obj < 0)
        hoste = sum(ms(e - s) for s, e, epi, obj in ev if obj >= 0)
        row = dict(period=ms(t1 - t0), gathered=ms(gath[-1][1] - t0) if gath else None,
                   sends_done=ms(sd[-1][1] - t0) if sd else None, n_sd=len(sd),
                   out_first=ms(ro[0][0] - t0) if ro else None, out_last=ms(ro[-1][0] - t0) if ro else None, n_out=len(ro),
                   combined=ms(cb[-1][1] - t0) if cb else None, drained=ms(dr[-1][1] - t0) if dr else None,
                   busy=dict(busy), idle=idle, host_disp=hostd, host_exp=hoste)
        # the combine kernel itself
        ck = [k for k in Kw if kclass(P['names'].get(k[2], '')) == 'combine']
        if ck: row['combine_k'] = (ms(ck[0][0] - t0), ms(ck[-1][1] - t0))
        gk = [k for k in Kw if kclass(P['names'].get(k[2], '')) == 'gather']
        if gk: row['gather_k'] = (ms(gk[0][0] - t0), ms(gk[-1][1] - t0))
        # experts on this PE
        exps = []
        oid2tok = defaultdict(list)
        for tok, oid in P['tok2oid'].items(): oid2tok[oid].append(tok)
        for oid, idx in sorted(d['oidx'].items(), key=lambda kv: kv[1]):
            rs = expert_runsteps(pe, idx)
            if not rs: continue
            alive_end = rs[-1][0] >= ds[-1][0] - int(2e9)
            if not alive_end: continue
            i = len(rs) - 1 - (a.total - S)
            if i < 0: continue
            es0 = rs[i][0]; es1 = rs[i + 1][0] if i + 1 < len(rs) else t1 + int(1e9)
            eev = [x for x in E if es0 <= x[0] < es1 and x[3] == idx]
            def et(name):
                j = epid(pe, name); return [x for x in eev if x[2] == j]
            sl = et('Expert::slabIn'); res = et('Expert::resumed'); cd = et('Expert::computeDone')
            esd = et('Expert::sendDone'); edr = et('Expert::sendsDrained'); gp = et('Expert::gradPartial') + et('Expert::peerStats')
            ad = et('Expert::applyDone')
            Ke = []
            if res and cd:
                w0, w1 = res[-1][1], cd[-1][0]
                per = defaultdict(float); kin = defaultdict(list)
                for k in P['K']:
                    if k[0] >= w0 and k[1] <= w1 + int(2e5):
                        per[k[4]] += k[1] - k[0]; kin[k[4]].append(k)
                if per:
                    lane = max(per, key=per.get); Ke = kin[lane]
            kb = defaultdict(float)
            for s, e, nid, tok, st in Ke: kb[kclass(P['names'].get(nid, ''))] += ms(e - s)
            ex = dict(idx=idx, start=ms(es0 - t0), period=ms(es1 - es0),
                      slabs=(ms(sl[-1][1] - es0) if sl else None, len(sl)),
                      resumed=ms(res[-1][1] - es0) if res else None,
                      k_first=ms(Ke[0][0] - es0) if Ke else None, k_last=ms(max(k[1] for k in Ke) - es0) if Ke else None,
                      computeDone=ms(cd[-1][1] - es0) if cd else None,
                      sends=(ms(esd[-1][1] - es0) if esd else None, len(esd)),
                      grad=(ms(gp[-1][1] - es0) if gp else None, len(gp)), applyDone=ms(ad[-1][1] - es0) if ad else None,
                      drained=ms(edr[-1][1] - es0) if edr else None, busy=dict(kb))
            exps.append(ex)
            if cd: lastout.append((cd[-1][1], idx, pe, ms(cd[-1][1] - t0)))
            if a.group is not None and idx // 1000 == a.group:
                gp_abs = sorted(x[1] for x in et('Expert::gradPartial'))
                groupRows.append(dict(idx=idx, pe=pe, start=es0, resumed=res[-1][1] if res else None, cd=cd[-1][1] if cd else None,
                                      grads=gp_abs, ad=ad[-1][1] if ad else None, dr=edr[-1][1] if edr else None, nxt=es1))
        row['experts'] = exps; row['t0'] = t0; row['t1'] = t1
        out[pe] = row
    if verbose:
        tmin = min(r['t0'] for r in out.values()); tmax = max(r['t1'] for r in out.values())
        print(f'step {S}: dispatcher step starts spread {ms(max(r["t0"] for r in out.values()) - tmin):.2f} ms; '
              f'ends spread {ms(tmax - min(r["t1"] for r in out.values())):.2f} ms')
        print('PE | period | gathered | gather k | sends done (n) | outputs first..last (n) | combine k | combined | drained | GPU busy ms by class | idle | host disp/exp')
        for pe, r in out.items():
            b = ' '.join(f'{k}={v:.1f}' for k, v in sorted(r['busy'].items()))
            f = lambda v: '-' if v is None else f'{v:.1f}'
            print(f'{pe} | {r["period"]:.1f} | {f(r["gathered"])} | {r.get("gather_k", ("-", "-"))[0] if "gather_k" in r else "-"}..{f(r["gather_k"][1]) if "gather_k" in r else "-"} | {f(r["sends_done"])} ({r["n_sd"]}) | {f(r["out_first"])}..{f(r["out_last"])} ({r["n_out"]}) | '
                  f'{f(r["combine_k"][0]) if "combine_k" in r else "-"}..{f(r["combine_k"][1]) if "combine_k" in r else "-"} | {f(r["combined"])} | {f(r["drained"])} | {b} | {r["idle"]:.1f} | {r["host_disp"]:.1f}/{r["host_exp"]:.1f}')
        print('\nexperts (times relative to the copy\'s own step start; start relative to the PE\'s dispatcher step start):')
        print('PE | expert | start | slabs in (n) | resumed | kernels first..last | computeDone | sends done (n) | grad in (n) | applyDone | drained | period | busy')
        for pe, r in out.items():
            for ex in sorted(r['experts'], key=lambda x: -sum(x['busy'].values())):
                f = lambda v: '-' if v is None else f'{v:.1f}'
                b = ' '.join(f'{k}={v:.1f}' for k, v in sorted(ex['busy'].items()))
                print(f'{pe} | {ex["idx"]//1000}.{ex["idx"]%1000} | {f(ex["start"])} | {f(ex["slabs"][0])} ({ex["slabs"][1]}) | {f(ex["resumed"])} | {f(ex["k_first"])}..{f(ex["k_last"])} | {f(ex["computeDone"])} | {f(ex["sends"][0])} ({ex["sends"][1]}) | {f(ex["grad"][0])} ({ex["grad"][1]}) | {f(ex["applyDone"])} | {f(ex["drained"])} | {ex["period"]:.1f} | {b}')
        if groupRows:
            b = min(r['start'] for r in groupRows)
            f = lambda v: '-' if v is None else f'{ms(v - b):.1f}'
            print(f'\ncopies of expert {a.group} at step {S}, ms from the earliest copy\'s step start:')
            print('copy | PE | start | resumed | computeDone | partials in | applyDone | drained | next step')
            for r in sorted(groupRows, key=lambda r: r['idx']):
                print(f'{r["idx"]//1000}.{r["idx"]%1000} | {r["pe"]} | {f(r["start"])} | {f(r["resumed"])} | {f(r["cd"])} | {" ".join(f(g) for g in r["grads"])} | {f(r["ad"])} | {f(r["dr"])} | {f(r["nxt"])}')
        if a.copies is not None:
            COPY = {0: 'memcpy', 1: 'HtoD', 2: 'DtoH', 3: 'HtoA', 4: 'AtoH', 5: 'AtoA', 6: 'AtoD', 7: 'DtoA', 8: 'DtoD', 9: 'HtoH', 10: 'PtoP'}
            print(f'\ncopy records of at least {a.copies} MB, per PE, ms from that PE\'s dispatcher step start (start+dur, MB, kind, stream, GB/s):')
            for pe, r in out.items():
                P = pes[pe]['P']; t0, t1 = r['t0'], r['t1']
                rows = [m for m in P['M'] if m[0] < t1 and m[1] > t0 and m[2] >= a.copies * 1048576]
                tot = sum(m[2] for m in rows) / 1048576
                print(f'PE {pe}: {len(rows)} copies, {tot:.0f} MB: ' + ' '.join(
                    f'{ms(m[0]-t0):.1f}+{ms(m[1]-m[0]):.2f} {m[2]/1048576:.0f}MB {COPY.get(m[3], m[3])} s{m[4]} {m[2]/max(m[1]-m[0],1):.1f}' for m in rows[:24]))
        lastout.sort()
        print('\nlast computeDone in the step (absolute order, time relative to that PE\'s dispatcher step start):')
        for t, idx, pe, rel in lastout[-5:]:
            print(f'  expert {idx//1000}.{idx%1000} on PE {pe}: {rel:.1f} ms')
    return out

if a.steps:
    A, B = map(int, a.steps.split(':'))
    acc = defaultdict(lambda: defaultdict(list))
    for S in range(A, B + 1):
        o = analyze(S, False)
        for pe, r in o.items():
            for k in ('period', 'out_last', 'sends_done', 'combined', 'drained', 'idle', 'host_disp', 'host_exp'):
                if r.get(k) is not None: acc[pe][k].append(r[k])
            acc[pe]['gemm'].append(r['busy'].get('gemm', 0.0)); acc[pe]['busy_all'].append(sum(r['busy'].values()))
            hot = max(r['experts'], key=lambda x: sum(x['busy'].values()), default=None)
            if hot:
                acc[pe]['hot_idx'].append(hot['idx']); acc[pe]['hot_busy'].append(sum(hot['busy'].values()))
                acc[pe]['hot_period'].append(hot['period'])
                if hot['computeDone'] is not None: acc[pe]['hot_cd'].append(hot['computeDone'])
                if hot['slabs'][0] is not None: acc[pe]['hot_slabs'].append(hot['slabs'][0])
                if hot['drained'] is not None: acc[pe]['hot_drained'].append(hot['drained'])
                acc[pe]['hot_start'].append(hot['start'])
    print(f'means over steps {A}..{B}')
    print('PE | period | sends done | last output | combined | drained | GPU busy (gemm) | idle | host disp/exp | hottest copy: idx start slabs computeDone drained period busy')
    m = lambda l: sum(l) / len(l) if l else float('nan')
    for pe in sorted(acc):
        c = acc[pe]
        hi = max(set(c['hot_idx']), key=c['hot_idx'].count) if c['hot_idx'] else -1
        print(f'{pe} | {m(c["period"]):.1f} | {m(c["sends_done"]):.1f} | {m(c["out_last"]):.1f} | {m(c["combined"]):.1f} | {m(c["drained"]):.1f} | {m(c["busy_all"]):.1f} ({m(c["gemm"]):.1f}) | {m(c["idle"]):.1f} | {m(c["host_disp"]):.1f}/{m(c["host_exp"]):.1f} | '
              f'{hi//1000}.{hi%1000} {m(c["hot_start"]):.1f} {m(c["hot_slabs"]):.1f} {m(c["hot_cd"]):.1f} {m(c["hot_drained"]):.1f} {m(c["hot_period"]):.1f} {m(c["hot_busy"]):.1f}')
else:
    analyze(a.step, True)
