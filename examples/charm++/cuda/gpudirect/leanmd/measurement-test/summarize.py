#!/usr/bin/env python3
"""Summarize complete LeanMD logs without discarding diagnostic output."""
import pathlib
import re
import sys

for name in sys.argv[1:]:
    directory = pathlib.Path(name)
    logs = sorted(directory.glob('rank_*.log'))
    if not logs:
        print(f'{directory.name}: no rank logs')
        continue
    text = (directory / 'rank_0.log').read_text(errors='replace')
    times = [(int(n), float(t)) for n, t in re.findall(
        r'Step (\d+) Benchmark Time ([\d.]+)', text)]
    late = [t for n, t in times if n >= 42]
    device = re.findall(r'\[DiffusionLB\].*load dimension:.*T_g ([\d.]+)', text)
    moves = re.findall(r'Number of cross node migrations AFTER LB: (\d+)', text)
    print(f'{directory.name}: steps={len(times)} mean42={sum(late)/len(late) if late else None}')
    print(f'  T_g={device} migrations={moves}')
    for log in logs:
        data = log.read_text(errors='replace')
        print(f'  {log.stem}: ' + ' | '.join(re.findall(r'^.*(?:Energy|energy|\[TEST\] PASS|Abort|assertion|Segmentation).*$', data, re.M)))
        raw = re.findall(r'\[RAWLOAD node \d+\].*gpu sum ([\d.]+).*zero (\d+)/\d+ \| wall', data)
        if raw:
            print(f'    gpu sums/zero objects={raw}')
