# Session handoff — 2026-09-17 evening

Branch `rate-aware-gpu-lb`, repo `/u/bhosale/charm-reconverse`.
Nothing running; no allocations held (`squeue -u bhosale` empty at 17:46).
Everything below is **uncommitted**.

---

## 1. Headline result: codex's isolated receive streams are a noLB regression

Job **22160664** (N=2 weak, `16 8 8`, leanmd) vs the matched baseline **22151249**
(same LB config — critically, *neither* has `+LBMetisMaxCutRise`):

| arm | 22151249 (shared recv stream) | 22160664 (isolated recv streams) | delta |
|---|---|---|---|
| nolb  | 1586.2 | **1783.5** | **+197.3 ms (+12.4%)** |
| sync  | 1455.6 | 1428.1 | −27.5 (−1.9%) |
| async | 1421.0 | 1395.5 | −25.5 (−1.8%) |

**noLB is the clean isolate**: it never invokes a balancer and performs zero
migrations, so neither codex's LBDatabase instrumentation fix nor any LB flag can
reach it. The isolated receive streams are the only delta on that arm.

noLB history across six jobs: 1623.6, 1586.2, 1582.9, 1577.7, 1588.6 — a 46 ms
band — then **1783.5**. 160 ms outside the band. Not noise.

The LB-arm gains (−1.9% / −1.8%) sit inside the 131 ms sync spread measured
earlier, i.e. nothing. This matches codex's own hedge in
`LEANMD_RECV_STREAM_HANDOFF.md`: "A performance win or reduced variance has not
been established by the host tests... stream creation/destruction and larger
concurrent receive counts can add overhead."

**Recommendation: do not keep the isolated-stream path on by default.** If it is
kept at all it should stay behind an opt-in (it already is, per
`CkDeviceBufferPost::isolated`), and the leanmd app-side opt-ins in `Cell.cc` /
`Cell.h` / `Compute.cc` should be reverted or gated.

### Full job table for the day

| job | change under test | noLB | sync | async |
|---|---|---|---|---|
| 22137894 | relay budget (ran on batch partition) | 1623.6 | 1447.3 | 1509.5 |
| 22151249 | + global-mean floor | 1586.2 | 1455.6 | 1421.0 |
| 22154226 | + `+LBMetisMaxCutRise 1.0` | 1582.9 | 1257.6 | 1440.0 |
| 22155894 | + `CHARM_LB_ASYNC_NO_PEER_DECISION=1` | 1577.7 | 1388.9 | 1307.5 |
| 22156615 | + codex instrumentation fix | 1588.6 | **1156.6** | 1351.2 |
| 22160664 | + codex isolated recv streams, **cut-rise flag missing** | **1783.5** | 1428.1 | 1395.5 |

22160664's LB arms are **not comparable to 22156615** — the cut-rise flag was
dropped from the submission (see §4). They are only comparable to 22151249.

Migration counts are normal, not anomalous: 22160664 sync `migr=[5629 248 0 0]`
vs 22151249 sync `[5710 305 0 0]`. (An earlier in-session claim that migration
had exploded was wrong — it compared against 22156615, which had the cut-rise
gate and therefore far fewer moves.)

Rank logs preserved at
`examples/charm++/cuda/gpudirect/leanmd/ranklogs/weak_N2_{nolb,sync,async}_22160664_recvstream`
(alongside `..._22137894_relay`, `..._22151249_globalfloor`, `..._22154226_cutrise`,
`..._22155894_nopeer`, `..._22156615_instrfix`).

---

## 2. `CHARM_ZC_VALIDATE` — the 29k failures are false positives, and the gate is wrong

Codex's validator in `src/ck-core/ckrdmadevice.C` (~2486-2515) calls
`cudaPointerGetAttributes` on source and dest before the copy.

**Proof they are false positives, no rerun needed**: the noLB arm performs 0
migrations, produces 20184 validator hits (2328 `mode=1`, 17856 `mode=2`, every
one `dev_idx=-1`), and still conserves energy to 8.181E-16.

The suppression logic is ineffective:

```cpp
const bool src_will_be_read =
    !(source.device_idx != -1 && csv_gpu_manager.use_shm);
if ((sbad && src_will_be_read) || dbad) { CmiPrintf("[%d] ZC VALIDATE FAIL mode=%d ..."); }
```

Every observed hit has `dev_idx == -1`, so `src_will_be_read` is always true and
nothing is suppressed. It should gate on **mode** instead: `zcModeSlot`
(`ckrdmadevice.C:955-957`) maps MEMCPY→0, IPC→1, else→2 (RDMA); `ckrdmadevice.C:2333`
sets `zc_mode = 2` directly. Modes 1 and 2 are cross-process paths where
`source.ptr` is the *sender's* virtual address and can never classify locally.
Suppress source-side complaints for modes 1 and 2.

---

## 3. Codex's segfault (job 22158228) — status

Codex's own harness lives in
`examples/charm++/cuda/gpudirect/leanmd/recv-stream-test/` (its own `run.sbatch`,
`rankwrap.sh`, `leanmd` binary — **not** `scale.sh`).

- nolb **PASSED** (5.824 s, energy drift 8.181E-16)
- sync **SEGFAULTED** (task 2, gpub008). No Charm fatal, no assert, no core
  (`ulimit -c 0`).
- async **never ran** — codex's `set -eu` aborted the script after the sync failure.

**Did not reproduce** in my harness at the real `16 8 8` config with a clean
rebuild: all three arms `rc=0`. Job 22160664 also completed all three arms
cleanly. So the segfault is either specific to codex's `8 4 4 30 10 10` config or
intermittent.

Two dead ends already closed, don't re-walk them:
- **ABI skew is NOT the cause.** `ckrdmadevice.h` changed 16:09 and `libck.a`
  rebuilt 16:11 while `libmoduleDiffusionLB.a` is 15:25 — but no source under
  `src/ck-ldb/` references `CkDeviceBufferPost`, and `nm` shows the symbol absent
  from the archive.
- The validator output is not the fault (see §2).

Unused-but-ready gdb capture scripts, written but never run:
`recv-stream-test/rankwrap_gdb.sh` and `recv-stream-test/run_gdb.sbatch` (sync arm
only, 3 attempts, `set -u` *not* `set -e`, `--time=00:15:00`). The pattern comes
from `benchmarks/charm++/cuda/gpudirect/lbcalib/calibwrap_gdb.sh`; the essential
line is `handle SIGUSR1 SIGUSR2 SIG34 nostop noprint pass`, which is what lets gdb
survive CUDA/LCI.

---

## 4. Job-submission discipline — the open action item

The user's closing message: *"I don't like how you have been treating runs as if
allocations come for free. How can you be more careful about not making these
mistakes when submitting jobs?"*

Two concrete failures today, one cause:

- **22137894 / 22145162** went to `gpuA40x4` instead of `gpuA40x4-interactive`;
  22145162 queued 6+ hours before being cancelled.
- **22160664** was submitted without `LMD_EXTRA='+LBMetisMaxCutRise 1.0'`, making
  its LB arms non-comparable to the 1156.6 reference.

**Cause**: the exact invocation exists only in shell history. Neither the
partition nor `LMD_EXTRA` is recorded in any file. `scale-<jobid>.out` logs binary
timestamps and the cost config but **not** the LB flags, so even a careful reader
of the previous job's output cannot recover `+LBMetisMaxCutRise 1.0`. The command
gets rebuilt from `submit_scale.sh`'s documented usage rather than from the
invocation that produced the numbers being compared against. "Be more careful"
does not fix this; there is nothing to be careful with.

### Proposed fixes (agreed in principle, NOT yet implemented)

1. **`scale.sh` prints the full per-arm command line into `scale-<jobid>.out`.**
   Then every past job self-documents and the record of what produced 1156.6
   survives.
2. **Move the knobs out of the environment into `scale.sh` defaults.** An unset
   env var currently changes the experiment silently and leaves no trace; a
   default in the file changes it visibly, in git. The relevant lines are
   `scale.sh:186-187`:
   ```
   sync)  extra="$LMD_MD +LBDebug ${LMD_LBDEBUG:-${LBDEBUG:-1}} ${LMD_EXTRA:-}";;
   async) extra="$LMD_MD -lbasync -lblag $LAG +LBAsync +LBDebug ${LMD_LBDEBUG:-${LBDEBUG:-1}} ${LMD_EXTRA:-}";;
   ```
   `LMD_EXTRA` has no default and is documented only in the header comment at
   `scale.sh:13`.
3. **Pre-submit diff**: before every `sbatch`, name the job being compared against
   and diff the planned invocation against its recorded one; submit only if the
   delta is the intended one. This is what catches both failures, because it
   forces *reading* the prior command instead of recalling it.
4. **One variable per allocation, named before submitting.** 22160664 moved two
   (streams + missing cut-rise), which is exactly why its LB arms are useless.

Items 1 and 2 are edits to `scale.sh` needing no allocation. The user was asked
whether to make them; **the answer had not arrived when the session ended.** Ask
before doing it. Never edit `scale.sh` while a job is running it.

Also standing: exhaust cheap static evidence (pre-LB cut, declared loads, library
mtimes, rank logs already on disk) before spending a node, and batch several
questions into one job since each job already runs a full matrix.

---

## 5. Uncommitted work inventory

From `git status` (branch `rate-aware-gpu-lb`):

**Mine / earlier sessions**
- `src/ck-ldb/DiffusionCore.C`, `DiffusionCostModel.h`, `DiffusionFlow.h`,
  `DiffusionLB.{C,ci,h}`, `DiffusionPseudo.C` — relay budget + global-mean floor
- `src/ck-ldb/LBDatabase.C` — codex's instrumentation fix
- `tests/charm++/load_balancing/lbdriver/{Makefile,lbsim.C}`
- `examples/charm++/cuda/gpudirect/scale.sh` — harness knobs

**Codex's receive-stream work (the thing measured in §1)**
- `src/ck-core/ckrdmadevice.h` (+6) — `CkDeviceBufferPost::isolated` opt-in flag
- `src/ck-core/ckrdmadevice.C` (+34) — `CHARM_ZC_VALIDATE` validator
- `src/ck-core/ckdevicerecvpool.h` — NEW, untracked; per-worker-thread,
  per-device stream pool, caches up to 64 completed streams
- `examples/charm++/cuda/gpudirect/leanmd/{Cell.cc,Cell.h,Compute.cc}` (+11) —
  app-side opt-ins
- `tests/charm++/load_balancing/lbdriver/recv_stream_test.C` + `check-recv-stream`
  Makefile target — host-only pool test, no CUDA, no transport
- `LEANMD_RECV_STREAM_HANDOFF.md` — codex's own writeup

`LEANMD_SHARED_RECV_STREAM=1` restores the old shared-stream behaviour; unset
means isolation is ON.

---

## 6. Open, not started

- Decide the fate of the isolated-stream path given the 12.4% noLB regression.
- Fix the `CHARM_ZC_VALIDATE` gate to key on mode (§2).
- Re-run the cut-rise comparison properly, *with* `LMD_EXTRA='+LBMetisMaxCutRise 1.0'`,
  to get a real read on codex's changes against 22156615's 1156.6 — only with
  explicit permission.
- `LEANMD_JOIN_TRACE=1` run for the unexplained LB-step wall time.
- sph2d with `SPH_LBDEBUG=0` to settle the `+LBDebug` confound.
- Commit the pile in §5. A general fix branches off `main`, not this branch —
  state the branch before committing. Commit messages end with
  `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`.

---

## 7. Run rules in force (do not relax any of these)

- **ONE GPU PER PROCESS**: `-n 4 --ntasks-per-node=4`. Never `-n 1` spanning 4 GPUs.
- **Never use ReadyMigrate** in application code.
- **ALWAYS read the run's own log in the same tool call that launches it.** Never
  poll a wrapper, task file, or `squeue`. This is the #1 recurring failure.
- Never hold 2 allocations at once. A40 only (`gpuA40x4` / `gpuA40x4-interactive`),
  account `mzu-delta-gpu`, at most `-t 01:00:00`.
- **Use `PART=gpuA40x4-interactive`** — the batch partition queues for hours.
- `ulimit -c 0` before every GPU run (cores exhaust the `/u` quota).
- `+gpushm +gpupool +gpuipceventpool 256` with a pre-sized power-of-two
  `+gpupoolsize`. Do not use CUDA mallocs.
- Pin: sph2d with `numa_wrap_sph.sh`, leanmd with `ranklogs/rankwrap.sh`.
- `export LD_LIBRARY_PATH=/u/bhosale/charm-reconverse/multicore-linux-x86_64-cuda/lib:/u/bhosale/lci_install/lib64:$LD_LIBRARY_PATH`
  (required for the sph2d link).
- Absolute `cd` in every build/run command.
- Ask for the node FIRST when a run is expected. Never re-`salloc` unprompted.
  Kill my own finished allocations promptly. Never `scancel` the user's.
- **Never `pkill -f` my own command's text.**
- No 2-node allocation without permission — fresh permission for each run.
- `scale.sh:203` does `rm -rf $dir; mkdir -p $dir`, so **copy rank logs
  immediately after a run** or they are destroyed by the next one.
  (`scratchpad/save_ranklogs.sh` does this.)
- Harness chain: `submit_scale.sh` → `scale.sbatch` → `scale.sh`; output
  `scale-<jobid>.out`. `submit_scale.sh` sizes walltime from `DRY=1 scale.sh`
  EXPECTED_SECONDS (+25% +5 min), clamps MAXTIME_MIN to 58 on the interactive
  partition, and uses `CHAIN=1 --dependency=singleton`.
- User preferences: answers in ~2 lines; state the reason with confidence rather
  than proposing another run; **"it is not noise! stop suggesting noise!!"**
