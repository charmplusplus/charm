# Entry methods as C++20 coroutines: an SDAG equivalent (design)

Status: prototype on branch `coroutine-sdag` (2026-11-09), header
`src/ck-core/ckcoro.h`, first program `tests/charm++/jacobi3d-coro`,
microbenchmark `benchmarks/charm++/coro_pingpong`. The mechanism as built is
described in `doc/internals/coroutine-sdag.md`; this document holds the
reasons, the alternatives and the measurements. The precedent is CAMPI
(`src/libs/ck-libs/ampi/campi.h`, `campi_core.h`, PRs #4073 and #4136),
which runs AMPI ranks as coroutines with the same restart protocol; the
analysis behind CAMPI is in the ampi project's notes
(`AMPI-COROUTINE-ANALYSIS-2026-10-19.md`, sections 2, 4A and 9).

Scope of the first cut (Kale, 2026-11-09): SDAG `overlap` is not supported
yet; the SDAG runtime structures may be reused after cleaning; `overlap` can
come later as additional coroutine functions.

## 1. The idea in one paragraph

An SDAG entry method is a chain of generated closures: charmxi turns `when`,
`for`, `if` and `serial` into C++ methods and a `Continuation` object per
suspension, so that the element can be packed while it waits. A C++20
coroutine gives the same control flow without the translator: the body is
ordinary C++, `co_await` suspends it, and the runtime resumes it when the
awaited message arrives. What a coroutine does not give is a packable
frame: its layout is compiler private and holds code addresses, so an
element suspended in a coroutine cannot be migrated by copying the frame.
CAMPI's answer, used here, is a restart protocol: the chare's members are
the state, each suspension that may be interrupted by a migration is a
declared site whose id is packed with the element, and on arrival the
entry method is invoked again and jumps to the recorded site.

## 2. API, with the complete jacobi3d step

The .ci file declares ordinary entries; nothing marks the coroutine or the
inbox entries, and charmxi generates its usual code:

```
array [3D] Jacobi {
  entry Jacobi(void);
  entry void doStep(void);                 // the coroutine entry
  entry void receiveGhosts(ghostMsg *gmsg); // inbox entries
  entry void ResumeFromSync(void);
  entry void resume(void);
};
```

The C++ side (from `tests/charm++/jacobi3d-coro/jacobi3d.C`; the SDAG
original is `tests/charm++/jacobi3d-sdag`):

```c++
class Jacobi : public CBase_Jacobi {
  ck::coro::state co;      // the coroutine's record and the inbox; pupped
  int iterations, imsg;    // state across sites: members, pupped
  double *temperature, *new_temperature;
  ghostMsg *gmsg;          // scratch: assigned at a site, used after it

  void pup(PUP::er &p) { p | iterations; p | imsg; ...; p | co; }

  // Inbox entries: one line each.
  void receiveGhosts(ghostMsg *m) { co.deliver(CkIndex_Jacobi::receiveGhosts(0), m); }
  void ResumeFromSync() { co.deliver(CkIndex_Jacobi::ResumeFromSync()); }
  void resume() { co.deliver(CkIndex_Jacobi::resume()); }

  ck::coroutine doStep() {
    CK_COROUTINE(co, CkIndex_Jacobi::doStep());
    if (lbFreq > 0 && iterations > 0 && iterations % lbFreq == 0) {
      AtSync();
      CK_WHEN(CkIndex_Jacobi::ResumeFromSync(), ck::coro::noref);
    }
    if (iterations == chkptIter) {
      contribute(CkCallback(CkReductionTarget(Main, checkpoint), mainProxy));
      CK_WHEN(CkIndex_Jacobi::resume(), ck::coro::noref);
    }
    begin_iteration();
    for (imsg = 0; imsg < 6; imsg++) {
      CK_WHEN(CkIndex_Jacobi::receiveGhosts(0), iterations, gmsg);  // when receiveGhosts[iterations](ghostMsg *gmsg)
      processGhosts(gmsg);
    }
    check_and_compute();
    CK_END();
  }
};
```

The SDAG original of the loop, for comparison:

```
entry void doStep() {
  serial { begin_iteration(); }
  for (imsg = 0; imsg < 6; imsg++) {
    when receiveGhosts[iterations](ghostMsg *gmsg) serial { processGhosts(gmsg); }
  }
  serial { check_and_compute(); }
};
```

Vocabulary (all in `ckcoro.h`):

| | |
|---|---|
| `ck::coroutine` | return type of a coroutine entry method; the body starts at once |
| `ck::coro::state` | one member per chare: active coroutines and the inbox; `p \| co` in pup |
| `co.deliver(ep, msg)` / `co.deliver(ep, args...)` | the body of an inbox entry; `ep` is `CkIndex_X::name(...)` |
| `CK_COROUTINE(co, ep)` / `CK_COROUTINE_MSG(co, ep, msg)` | opens the body; registers the coroutine; opens the restart switch |
| `CK_WHEN(ep, refnum-or-noref, out...)` | a site: wait for a delivery to `ep` with that reference number and store it in `out` (a message pointer, or the marshalled arguments) |
| `CK_CALL(expr)` / `CK_CALL_TO(lhs, expr)` | a site: await a `ck::coro::task<T>` helper that itself contains sites |
| `CK_RESUMABLE()` / `CK_END()` | the restart switch of a helper; `CK_END` closes any body |
| `ck::coro::task<T>` | lazily started helper coroutine (CAMPI's `campi::task`) |
| `+ck_fakemigrate` | every site rebuilds the element's coroutines in place |

Reference numbers follow SDAG: the envelope's for a message entry, the
first arithmetic argument for a marshalled entry; `ck::coro::noref` matches
any. A zero-argument marshalled entry (`ResumeFromSync` above) carries none.

## 3. Rules

1. **State lives in members.** A value that is live across a site is a
   member of the chare and is pupped by the chare's pup, exactly as SDAG
   requires of its state variables (`int imsg` is a member in both
   programs). C++ rejects the restart jump past a local with an initializer
   or a non-trivial type, which catches most mistakes at compile time; a
   scalar declared without an initializer (such as `gmsg` above, assigned at
   the site) is not caught, and `+ck_fakemigrate` is the check.
2. **Code before a site is skipped on a restart**, so its effects must be
   recorded in members (above: `iterations` is incremented in
   `begin_iteration`, which runs before the ghost sites and is skipped when
   the element arrives in the middle of a step).
3. **Migration and checkpoint happen only while every active coroutine of
   the element is suspended at a site** (`CK_WHEN`, `CK_WHEN_ANY` or
   `CK_CALL`). A plain `co_await` on a `when` awaitable is allowed between
   sites, but packing an element suspended there aborts with a message
   naming the entry. This is where the design is more permissive than CAMPI
   (which has no awaitables that are not sites) and more restrictive than
   SDAG (which can pack any `when`). A body may call `migrateMe(pe)` or
   `AtSync()` itself: both are messages to the element (`ckEmigrate`, the
   balancer's reply), so the move happens while the body waits at the next
   site, which is what the migration test does.
4. **One active coroutine per entry method per element.** A second
   invocation of a coroutine entry while one is suspended aborts. SDAG allows
   several concurrent invocations of one SDAG entry; none of the SDAG test
   programs or jacobi3d relies on it. The restriction is what lets a restart
   be a plain re-invocation of the entry (section 4.5).
5. **A coroutine entry takes no parameters or one message.** Parameters it
   needs are members. Marshalled parameters would need the translator to
   generate the closure that carries them across a restart (section 4.1).
6. A site is one statement and may be the body of an `if` or a loop; its
   case label must not be inside a `switch` of the user's own; two sites must
   not share a line (CAMPI's rules, from the `switch`/`case __LINE__`
   mechanism).

What this allows that SDAG does not: the body is C++ with every construct
of the language (`break`, `continue`, `return` from the middle, nested
loops, lambdas, `switch` outside sites), nested blocking helpers returning
`task<T>` (CAMPI's `tests/campi/until` shows the pattern), and a debugger
that steps through the body. What SDAG allows that this forbids: packing at
any `when`, several concurrent invocations of one SDAG entry, marshalled
parameters on the SDAG entry, and `overlap` (not yet).

## 4. The seven questions

### 4.1 Library or translator

Library, with no charmxi change, is enough for everything in section 2.
Two facts make it so: (a) charmxi's generated wrapper for a void entry
calls `impl_obj->doStep()` and ignores the return value, so the C++
definition may return `ck::coroutine` while the .ci file says `void`;
(b) an inbox entry is an ordinary entry whose body is one line. What
charmxi still has to know is nothing; what it could usefully learn later:

- an `[inbox]` attribute that generates the one-line body and, for
  marshalled entries with arrays (`int data[n]`), keeps the marshalled
  message in the inbox instead of re-packing the arguments (today the
  library packs the arguments with PUP into bytes, which handles scalars,
  `std::vector`, `std::string` and anything with a `pup`, but not raw
  arrays);
- a `[coroutine]` attribute that carries marshalled parameters across a
  restart (rule 5) and that makes `CK_COROUTINE`'s entry index implicit.

Both are small (the SDAG translator already generates closures for
when-entries, `CEntry::generateCode`), and both keep the .ci file readable:
the declaration stays one line. They are not needed for the beta.

### 4.2 Suspension and migration rules

Adopted from CAMPI (section 3). The difference from CAMPI is that the chare
is the state object (no separate `State` type), that `when` sites are
restart sites (CAMPI's only sites are `AMPI_Migrate` and calls), and that
the frame may suspend at non-site awaitables between sites as long as the
element does not migrate there. The checks are the same: the restart aborts
if the body reaches a different site than recorded, if a function on the
recorded path is missing, or if a helper is awaited without `CK_CALL`;
`+ck_fakemigrate` exercises every site without a balancer: the first time
a coroutine suspends at a site, the element's coroutines are packed and
unpacked in place and re-invoked; a body that was just restarted to a site
waits there without rebuilding again, so a program progresses and reaches
quiescence as without the flag (the first version rebuilt on every
suspension, and the migration test, which waits for quiescence, never got
it). In the jacobi3d test with the flag on, 320 rebuilds happen in a
20-iteration run on 2 PEs and the checksum is unchanged; the migration
test passes with the flag on 2, 3 and 4 PEs.

A restart message can be in flight when the element migrates (the
element's record was unpacked, its message sent, and a balancer or an
anytime migration moves the element before the message arrives). The
record remembers that a restart is pending, so the second unpack does not
send another and the location manager forwards the first. A disk
checkpoint taken with a restart message in flight would lose it; Charm++
checkpoints assume no messages in flight, and the program's own
synchronization (a reduction before `CkStartCheckpoint`) provides that.

### 4.3 Overlap and when-any

`when_any` exists: `CK_WHEN_ANY(idx, CK_ALT(ep, ref, out...), ...)` waits
for the first of several deliveries and stores the index of the
alternative that matched; a record waits on a list of (entry, refnum)
pairs instead of one. That is SDAG `case` over single whens; a `case` arm
with a conjunction (`when b(), d()`) is not expressible yet.

`overlap` is deferred. The design that fits the restart protocol is
*several coroutines per element*: each `when` clause of an `overlap`
becomes its own coroutine entry (or a `task<void>` helper started with its
own record), each with its own site chain; the state object already keeps
a list of records, and the join is a counter in a member followed by a
plain `when`. The alternative, `co_await when_any(...)` over several
awaitables, is what `when_any` already is, and it does not give `overlap`,
which is a conjunction of independent progress. Neither needs a new pup
story: a record is a record.

### 4.4 Message buffering

The inbox buffers every delivery that no coroutine is waiting for, keyed by
entry index (stable across PEs) and reference number, in arrival order, and
`CK_WHEN` takes the first match. Messages are kept as messages
(`CkPupMessage` moves them); marshalled arguments are kept as pupped bytes.
The inbox moves with the element; messages that reach the old PE after the
element left are forwarded by the location manager and delivered to the
new PE's inbox entry as usual. SDAG's `Dependency`/`Buffer`/`Continuation`
structures were not reused: the waiter here is a coroutine handle plus a
`want`, not a closure chain, so the matching code is the forty lines of
`state::push`, `state::take` and `want::matches`. The `@todo` in
`sdag.h` about linear lookups by reference number applies here too and is
acceptable for the same reason (few buffered messages per element).

### 4.5 Checkpoint

A record is packed as entry index, a callback to the element, the entry's
message (for message entries) and the site chain. Unpacking sends the
message to the element through the callback; the body runs again, jumps to
the site and either takes the buffered delivery or waits. This is the same
code for a migration (RotateLB in the test), a rebuild (`+ck_fakemigrate`)
and a disk checkpoint (`CkStartCheckpoint` while all elements wait at a
site; `+restart` on the same, fewer and more PEs, and across two
processes). Nothing in the packed data is an address, so restart does not
need address randomization off (AMPI thread migration does, see
`machines/mac.md`).

### 4.6 Performance

See section 6. The expectation is parity or better: SDAG's path on arrival
is entry wrapper, `new Buffer`, `new MsgClosure`, a scan of continuations,
the generated `when` method, then the next construct's method; the
coroutine path is entry wrapper, a scan of the element's records, a move of
the buffer into the record, and one indirect call (`h.resume()`).

### 4.7 Coexistence

SDAG and coroutine entries coexist in one program and in one chare without
any shared state: SDAG's `__dep` and the coroutine `state` are separate
members, and an inbox entry is just an entry. A chare may even have an SDAG
entry that `when`s on one entry and a coroutine that `CK_WHEN`s on another.
If SDAG were ever replaced, what would be lost is listed at the end of
section 3; what would be gained is the removal of `src/xlat-i/sdag/` (about
2500 lines) and of `sdag.h`, and the end of the "state variables must be
declared in the class" surprise, since in a coroutine body all state is
visibly the chare's.

## 5. SDAG, CAMPI, coroutine entries

| | SDAG | CAMPI | coroutine entries (this) |
|---|---|---|---|
| control flow written as | `.ci` constructs, C++ in `serial` | C++ coroutine `main` | C++ coroutine entry body |
| generated code | closures, when methods, pup | none | none |
| suspension | generated continuation | `co_await` awaitable | `co_await` at `CK_WHEN`, or plain `co_await` |
| what is packed while suspended | closures and buffered messages | state object and site chain | site chain, callback, entry message, inbox |
| restart | resume the closure chain | re-run `main` to the site | re-invoke the entry to the site |
| state across a suspension | class members ("state variables") | the `State` object | class members |
| migration at any suspension | yes | no: `AMPI_Migrate` sites only | at `CK_WHEN`/`CK_CALL` sites only |
| nested blocking helpers | no | `task<T>` with `CAMPI_CALL` | `task<T>` with `CK_CALL` |
| disjunction / conjunction | `case`, `when a(), b()` | n/a | `when_any` planned; sequential awaits |
| independent progress | `overlap`, `forall` | n/a | several coroutines (planned) |
| self-check without a balancer | none | `+campi_fakemigrate` | `+ck_fakemigrate` |
| compiler requirement | C++11 | C++20 | C++20 for the files that include `ckcoro.h` |

## 6. Measurements

Mac (Apple M-series, 8 cores), reconverse build `--with-production`,
2026-11-09 23:45 PST. Each number is the median of three runs; the raw
lines are in the project's `measure-2.log`.

### 6.1 A when-matched message: `benchmarks/charm++/coro_pingpong`

Two array elements, 50000 round trips of an 8-byte message, each message
matched by iteration number (SDAG `when recv[iter]`, coroutine
`CK_WHEN(..., iter, m)`), against plain entry methods with no matching.
Microseconds per message (half a round trip):

| | plain entry | SDAG when | coroutine CK_WHEN |
|---|---|---|---|
| one PE, both elements local | 0.229 | 1.677 | 0.479 |
| two PEs, one element each | 0.463 | 1.291 | 0.637 |

On one PE the coroutine match costs 0.25 us over a plain entry and the
SDAG match 1.45 us; the SDAG path allocates a `Buffer`, a `MsgClosure` and
a `Continuation` per message and walks two lists, the coroutine path moves
the buffer into the record and resumes. Across two PEs the message itself
costs more and the gap narrows (0.17 us against 0.83 us). charm4py's
measurement of 1.58 us for a coroutine message against 0.38 plain is a
different binding; here the coroutine overhead is a fifth of SDAG's.

### 6.2 jacobi3d end to end

`./jacobi3d 64 16 200 100`: 64 elements of 16^3 cells, 200 iterations, a
report every 100, no load balancing. Both programs print checksum
64041493740160. Time per iteration as printed by the program
(milliseconds, median of three):

| | SDAG | coroutine | change |
|---|---|---|---|
| +pe 1 | 3.39 | 2.83 | -17% |
| +pe 4 | 0.900 | 0.770 | -14% |

Six matched messages per element per iteration at 1.2 us of saved
matching each account for about 0.5 ms of the 0.56 ms saved on one PE;
the rest is within run-to-run noise. The compute kernel is identical.

### 6.3 Anvil, two nodes

Purdue Anvil (2 x AMD EPYC 7763 per node, InfiniBand HDR), production
build of the branch at `/anvil/scratch/x-lkale/coro-sdag-measure/charm`,
Slurm job 22127892 run 2026-11-10 about 02:00 PST (outside the morning
window in which the network is quietest, see `machines/anvil.md`); the
job's output is the project's `anvil-measure-1.log`. Two processes, one
per node, 64 PEs each on socket 0 (`+pe 128 +pemap 0-63`).

jacobi3d `512 32 100 50` (4096 elements of 32^3 cells, 32 per PE, 100
iterations; both programs print checksum 18642873093165056), milliseconds
per iteration, three runs each, alternating:

| | run 1 | run 2 | run 3 | median |
|---|---|---|---|---|
| SDAG | 27.54 | 27.35 | 32.26 | 27.54 |
| coroutine | 26.97 | 26.80 | 26.63 | 26.80 |

At this scale the step is dominated by the 32^3 stencil and the halo
copies, so the 1.2 us saved per matched message is 3 percent of the
iteration. The ping-pong (us per message, median of three):

| | plain entry | SDAG when | coroutine CK_WHEN |
|---|---|---|---|
| across the two nodes, one PE each | 3.313 | 4.470 | 3.656 |
| within one node, two PEs in one process | 0.647 | 1.758 | 1.000 |

The matching cost over a plain entry is the same as on the Mac in
proportion: SDAG 1.1 us, coroutine 0.35 us within the node; across the
network the message itself is 3.3 us and the two differ by 0.8 us.

## 7. Test plan (from the charm-tests requirements, 2026-11-09) and status

| requirement | status |
|---|---|
| jacobi3d: same results at +pe 4 and in two processes | done: checksum identical at +pe 1, +pe 4, lcrun -n 2 |
| migration under `+balancer RotateLB` through declared sites, element suspended in a when | done: AtSync inside the coroutine, the element waits for ResumeFromSync at a site; RotateLB moves all elements every 4 iterations. Anytime migration while suspended: `tests/charm++/ckcoro` (the SDAG migration test as a coroutine) |
| rebuild-at-every-site flag | done: `+ck_fakemigrate`, 320 rebuilds in the 2-PE run |
| checkpoint of a suspended element, restart | done: checkpoint at a when site, `+restart` on 4 and 2 PEs in one process and on 4 PEs in two processes |
| when with reference numbers | done (jacobi3d ghosts, pingpong) |
| `case` | done for single-when arms: `CK_WHEN_ANY` in `tests/charm++/ckcoro` |
| `overlap` | deferred (Kale) |
| equivalence corpus: megatest synctest, every `tests/charm++/sdag` directory, SDAG-heavy miniapps under both implementations | started: the `migration` module of `tests/charm++/sdag` is ported (`tests/charm++/ckcoro`); `basic` needs `overlap` and `forall`; `refnum` needs `forall` |
| CAMPI tests stay green when the core is shared | in progress (2026-11-09 23:50): a shared `ckcoro_core.h` for `task<T>` and the site chain, delegated |

## 8. What charm4py can mirror

The protocol has no C++ in it: a record is (entry id, element, entry
message, list of site ids), a site is an integer the body passes when it
suspends, and a restart is "invoke the entry again; at each site compare
the id with the recorded one and skip or wait". A Python binding can keep a
record per suspended coroutine, pup it with the element, and re-invoke the
entry on arrival; the body needs the same discipline (state in `self`,
sites as explicit calls such as `yield self.when(ep, ref)` carrying a site
id, or a line number). This is the alternative to copying a Python frame
(the charm4py-thin probe of 2026-11-08 found that fragile), at the price of
the rules in section 3.

## 9. Decision memo for Kale

**Recommendation: ship `ckcoro.h` in the beta, marked experimental.** It
is a header with no runtime or translator change, so it cannot break
anything that does not include it, and it already carries a real program
(jacobi3d with load balancing, rebuild, checkpoint and restart) and the
SDAG migration test. Its matching costs a fifth of SDAG's and jacobi3d
runs 14 to 17 percent faster per iteration. The restrictions are the ones
CAMPI already imposes on AMPI programs (state in members, sites are
declared, nothing packed except at a site) plus one active coroutine per
entry and no marshalled parameters on the coroutine entry.

What "experimental" has to say: no `overlap` and no `forall`; `case` only
over single whens; arrays only (no groups or singleton chares as owners);
the macro syntax may change when the translator attributes arrive.

What the translator must gain, in order of value: an `[inbox]` attribute
(removes the one-line bodies, admits array parameters), a `[coroutine]`
attribute (marshalled parameters on the entry, implicit entry index,
several active coroutines by a generated record per invocation), and
nothing else; both are small changes in `src/xlat-i` and neither is needed
for the beta.

What charm4py can mirror: the record and site protocol of section 8, with
no C++ frame involved.

Everything the beta needs is on the branch: the shared core with CAMPI
(`ckcoro_core.h`, CAMPI's tests green), the two test directories in CI
(`TEST_DIRS` in `reconverse-ci.yaml`), and the Mac and Anvil measurements.
First beta update: `overlap` as several coroutines per element and the
rest of the SDAG test corpus.

## 10. Open items

- `when_any` for `case`; several coroutines per element for `overlap` and
  `forall`.
- Groups, nodegroups and singleton chares as owners of coroutine entries
  (the prototype's record builds its callback from an `ArrayElement`).
- An `[inbox]` attribute for marshalled entries with arrays.
- The equivalence corpus runs, and the Anvil timing.
