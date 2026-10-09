#!/usr/bin/env python3
"""nsys_merge.py --out trace.json[.gz] [--window T0:T1] [--no-host] rank_*.sqlite

Merge per-rank Nsight Systems sqlite exports (nsys export -t sqlite) into one Chrome/Perfetto trace:
one process row per rank, with a thread row per GPU stream (kernels, memcpys, memsets) and one per host
thread (NVTX ranges: the runtime's Chare::entry and idle ranges, with the array index as "idx").
Timestamps are aligned on each report's session-start wall clock, so ranks on different nodes line up
to the nodes' clock agreement (NTP, tens of us). --window selects seconds since the earliest session
start, e.g. --window 20:20.5 for half a second. Open the result at https://ui.perfetto.dev."""
import argparse, gzip, json, os, re, sqlite3, sys

COPY = {0: 'memcpy', 1: 'HtoD', 2: 'DtoH', 3: 'HtoA', 4: 'AtoH', 5: 'AtoA', 6: 'AtoD', 7: 'DtoA', 8: 'DtoD', 9: 'HtoH', 10: 'PtoP'}

def cols(c, table):
    return [r[1] for r in c.execute(f'PRAGMA table_info({table})')]

def tables(c):
    return {r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")}

def session_start(c, t):
    for tab in ('TARGET_INFO_SESSION_START_TIME',):
        if tab in t:
            cs = cols(c, tab)
            col = 'utcEpochNs' if 'utcEpochNs' in cs else cs[0]
            r = c.execute(f'SELECT {col} FROM {tab}').fetchone()
            if r and r[0]: return int(r[0])
    return 0

def hostname(c, t):
    for tab in ('TARGET_INFO_SYSTEM_ENV',):
        if tab in t:
            for name, val in c.execute(f'SELECT name, value FROM {tab}'):
                if str(name).lower() == 'hostname': return str(val).split('.')[0]
    return ''

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--window', default=None, help='T0:T1 seconds since the earliest session start')
    ap.add_argument('--no-host', action='store_true', help='GPU rows only (no NVTX host ranges)')
    ap.add_argument('files', nargs='+')
    a = ap.parse_args()
    dbs = []
    for f in a.files:
        m = re.search(r'rank_(\d+)', os.path.basename(f))
        rank = int(m.group(1)) if m else len(dbs)
        c = sqlite3.connect(f)
        t = tables(c)
        dbs.append((rank, f, c, t, session_start(c, t), hostname(c, t)))
    base = min(d[4] for d in dbs)
    w0 = w1 = None
    if a.window:
        s0, s1 = a.window.split(':'); w0, w1 = float(s0) * 1e9, float(s1) * 1e9
    out = gzip.open(a.out, 'wt') if a.out.endswith('.gz') else open(a.out, 'w')
    out.write('{"displayTimeUnit":"ms","traceEvents":[\n')
    first = [True]
    def emit(ev):
        if not first[0]: out.write(',\n')
        first[0] = False
        out.write(json.dumps(ev, separators=(',', ':')))
    def keep(s, e):
        return w0 is None or (e >= w0 and s <= w1)
    counts = {}
    for rank, f, c, t, t0, host in dbs:
        off = t0 - base
        strings = dict(c.execute('SELECT id, value FROM StringIds')) if 'StringIds' in t else {}
        emit({'ph': 'M', 'name': 'process_name', 'pid': rank, 'args': {'name': f'rank {rank} {host}'.strip()}})
        emit({'ph': 'M', 'name': 'process_sort_index', 'pid': rank, 'args': {'sort_index': rank}})
        n = 0
        if 'CUPTI_ACTIVITY_KIND_KERNEL' in t:
            cs = cols(c, 'CUPTI_ACTIVITY_KIND_KERNEL')
            namecol = 'shortName' if 'shortName' in cs else ('demangledName' if 'demangledName' in cs else None)
            q = f"SELECT start, end, deviceId, streamId, {namecol or 'NULL'} FROM CUPTI_ACTIVITY_KIND_KERNEL"
            seen = set()
            for s, e, dev, st, nm in c.execute(q):
                s += off; e += off
                if not keep(s, e): continue
                tid = 1000 + st
                if tid not in seen:
                    seen.add(tid)
                    emit({'ph': 'M', 'name': 'thread_name', 'pid': rank, 'tid': tid, 'args': {'name': f'GPU{dev} stream {st}'}})
                    emit({'ph': 'M', 'name': 'thread_sort_index', 'pid': rank, 'tid': tid, 'args': {'sort_index': tid}})
                emit({'ph': 'X', 'cat': 'kernel', 'pid': rank, 'tid': tid, 'ts': s / 1000.0, 'dur': (e - s) / 1000.0,
                      'name': strings.get(nm, str(nm))[:80]})
                n += 1
        for tab, cat in (('CUPTI_ACTIVITY_KIND_MEMCPY', 'memcpy'), ('CUPTI_ACTIVITY_KIND_MEMSET', 'memset')):
            if tab not in t: continue
            cs = cols(c, tab)
            extra = ', copyKind, bytes' if 'copyKind' in cs else (', 0, bytes' if 'bytes' in cs else ', 0, 0')
            seen = set()
            for s, e, dev, st, kind, nb in c.execute(f'SELECT start, end, deviceId, streamId{extra} FROM {tab}'):
                s += off; e += off
                if not keep(s, e): continue
                tid = 1000 + st
                if tid not in seen:
                    seen.add(tid)
                    emit({'ph': 'M', 'name': 'thread_name', 'pid': rank, 'tid': tid, 'args': {'name': f'GPU{dev} stream {st}'}})
                emit({'ph': 'X', 'cat': cat, 'pid': rank, 'tid': tid, 'ts': s / 1000.0, 'dur': (e - s) / 1000.0,
                      'name': (COPY.get(kind, cat) if cat == 'memcpy' else 'memset'), 'args': {'bytes': nb}})
                n += 1
        if not a.no_host and 'NVTX_EVENTS' in t:
            cs = cols(c, 'NVTX_EVENTS')
            pay = 'int64Value' if 'int64Value' in cs else 'NULL'
            textid = 'textId' if 'textId' in cs else 'NULL'
            tnames = {}
            if 'ThreadNames' in t:
                for nid, gtid in c.execute('SELECT nameId, globalTid FROM ThreadNames'):
                    tnames[gtid] = strings.get(nid, str(nid))
            seen = set()
            q = f'SELECT start, end, eventType, text, {textid}, globalTid, {pay} FROM NVTX_EVENTS WHERE eventType IN (59, 60) AND end IS NOT NULL'
            for s, e, et, text, tid_, gtid, val in c.execute(q):
                s += off; e += off
                if not keep(s, e): continue
                tid = gtid & 0xffffff
                if tid not in seen:
                    seen.add(tid)
                    emit({'ph': 'M', 'name': 'thread_name', 'pid': rank, 'tid': tid, 'args': {'name': tnames.get(gtid, f'thread {tid}')}})
                name = text if text else strings.get(tid_, str(tid_))
                ev = {'ph': 'X', 'cat': 'nvtx', 'pid': rank, 'tid': tid, 'ts': s / 1000.0, 'dur': (e - s) / 1000.0, 'name': name}
                if val is not None: ev['args'] = {'idx': val}
                emit(ev)
                n += 1
        counts[rank] = n
    out.write('\n]}\n'); out.close()
    print(f'wrote {a.out}: ' + ', '.join(f'rank {r}: {n} events' for r, n in sorted(counts.items())), file=sys.stderr)

if __name__ == '__main__':
    main()
