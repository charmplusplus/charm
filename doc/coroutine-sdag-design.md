# Entry methods as C++20 coroutines: an SDAG equivalent (design)

Status: pull request #4231 (`coroutine-sdag` into `recharm`, opened
2026-11-11), header `src/ck-core/ckcoro.h` with the charmxi attributes
`[inbox]` and `[coroutine]`, the manual section "Structured Control Flow:
Coroutine Entry Methods", the example `examples/charm++/heat1d-coro`, the
test suite `tests/charm++/ckcoro`, the SDAG programs of the tree ported
next to their originals, and the microbenchmark
`benchmarks/charm++/coro_pingpong`. The mechanism as built is described
in `doc/internals/coroutine-sdag.md`; this document holds the reasons,
the alternatives (section 4 per question, section 9a as a list) and the
measurements. The precedent is CAMPI (`src/libs/ck-libs/ampi/campi.h`,
`campi_core.h`, PRs #4073 and #4136), which runs AMPI ranks as
coroutines with the same restart protocol; the analysis behind CAMPI is
in the ampi project's notes (`AMPI-COROUTINE-ANALYSIS-2026-10-19.md`,
sections 2, 4A and 9).

History: prototype 2026-11-09 (library only, `overlap` left out by Kale's
scoping of the first cut), attributes and `CkCurrentMessage` 2026-11-10,
block form, `overlap`/`forall` as child coroutines, owners of every chare
kind, the equivalence corpus and the pull request 2026-11-11.

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

The .ci file declares the coroutine entry `[coroutine]` and the entries
it waits for `[inbox]`; the control flow is not in the .ci file:

```
array [3D] Jacobi {
  entry Jacobi(void);
  entry [coroutine] void doStep(void);      // ck::coroutine Jacobi::doStep() in jacobi3d.C
  entry [inbox] void receiveGhosts(ghostMsg *gmsg);
  entry [inbox] void ResumeFromSync(void);
  entry [inbox] void resume(void);
};
```

The C++ side (from `tests/charm++/jacobi3d-coro/jacobi3d.C`; the SDAG
original is `tests/charm++/jacobi3d-sdag`):

```c++
class Jacobi : public CBase_Jacobi {
  Jacobi_CORO_CODE         // generated: the coroutine state (pupped with the element),
                           // the inbox entry bodies, and inbox::name() / inbox::name(ref)
  int iterations, imsg;    // state across sites: members, pupped
  double *temperature, *new_temperature;
  ghostMsg *gmsg;          // scratch: assigned at a site, used after it

  void pup(PUP::er &p) { p | iterations; p | imsg; ...; }

  ck::coroutine doStep() {
    CK_MIGRATABLE {
      if (lbFreq > 0 && iterations > 0 && iterations % lbFreq == 0) {
        AtSync();
        CK_WHEN(inbox::ResumeFromSync());
      }
      if (iterations == chkptIter) {
        contribute(CkCallback(CkReductionTarget(Main, checkpoint), mainProxy));
        CK_WHEN(inbox::resume());
      }
      begin_iteration();
      for (imsg = 0; imsg < 6; imsg++) {
        CK_WHEN(inbox::receiveGhosts(iterations), gmsg);  // when receiveGhosts[iterations](ghostMsg *gmsg)
        processGhosts(gmsg);
      }
      check_and_compute();
    }
  }
};
```

The SDAG original, for comparison:

```
entry void doStep() {
  serial { begin_iteration(); }
  for (imsg = 0; imsg < 6; imsg++) {
    when receiveGhosts[iterations](ghostMsg *gmsg) serial { processGhosts(gmsg); }
  }
  serial { check_and_compute(); }
};
```

The two `if` blocks at the top of the coroutine have no counterpart in the
SDAG program, so the SDAG text is shorter for that reason and not because of
the notation. The SDAG program has no checkpoint, and it balances load
outside the SDAG code: `check_and_compute` ends with `AtSync()` instead of
`doStep()`, and the plain entry `ResumeFromSync` calls `doStep()`. The
coroutine version cannot call `doStep()` directly (a direct C++ call of a
coroutine entry aborts): it would write `thisProxy[thisIndex].doStep()` in
`ResumeFromSync`, or wait for `ResumeFromSync` as an inbox entry, which is
what it does. It waits for
the balancer and for the checkpoint inside the body on purpose, so that the
test migrates and checkpoints an element suspended at a site. Without those
two blocks the coroutine body is the loop shown, inside `CK_MIGRATABLE`.

Vocabulary (the attributes in charmxi, the rest in `ckcoro.h`):

| | |
|---|---|
| `[coroutine]`, `[inbox]` | .ci attributes: the entry is a coroutine (void; a message or marshalled parameters, re-supplied by a restart; on any chare kind); the entry is waited for (a message, or marshalled parameters, arrays included; not conditional or nocopy) |
| `X_CORO_CODE` | generated macro placed in the class: the `ck::coro::state` member (pupped through the generated pup), the inbox entry bodies, `__coro_is_entry`, and `struct inbox` |
| `inbox::name()` / `inbox::name(ref)` | a `ck::coro::want_t<inbox::name_params>`: a delivery to that entry with any, or with that, reference number; the want carries the entry's parameter types |
| `CK_WHEN_BLOCK(want, name...) { ... }` | the block form: the parameters as block-scoped variables; no site inside the block |
| `ck::coro::skip` | an out that discards its parameter |
| `ck::coroutine` | return type of a coroutine entry method; the body starts at once |
| `CK_MIGRATABLE { ... }` | the block that can be rebuilt after a migration: a frame guard and the restart switch, `switch (site) case 0: if (false) default: abort_no_site(); else` followed by the block; the same block in an entry body and in a `ck::coro::task<T>` helper, `co_return` inside it. The registration of the coroutine is done by the promise of `ck::coroutine` through `X_CORO_CODE`. (Until 2026-11-11 the body was bracketed by `CK_COROUTINE()`/`CK_COROUTINE_END()` and a helper by `CK_CORO_TASK()`/`CK_CORO_TASK_END()`; Kale asked whether C++ could remove the pair, and the block form with the registration in the promise is the answer; section 4.8) |
| `CK_WHEN(want, out...)` | a site: wait for the delivery and store it in `out` (the message pointer, or every marshalled parameter in order) |
| `CK_WHEN_ANY(idx, CK_ALT(want, out...), ...)` | a site: the first of several (SDAG `case`); `idx` gets the index that matched |
| `ck::coro::state`, `co.deliver(ep, msg)` / `co.deliver(ep, args...)`, `CK_COROUTINE_STATE(co, ep)`, `CK_WHEN(ep, ref-or-noref, out...)` | the same by hand, without the attributes: a state member pupped by the chare, inbox bodies written out, `CK_COROUTINE_STATE` as the first statement of the body (the promise registers only through `X_CORO_CODE`), entry indices spelled `CkIndex_X::name(...)` |
| `CK_CALL(expr)` / `CK_CALL_TO(lhs, expr)` | a site: await a `ck::coro::task<T>` helper that itself contains sites |
| `ck::coro::task<T>` | lazily started helper coroutine (CAMPI's `campi::task`) |
| `+ck_fakemigrate` | every site rebuilds the chare's coroutines in place |
| `ckDestroyLater()` (charm++.h, every chare) | destroy the chare after the entry method running on it returns: what a coroutine body does instead of `delete this` (Kale, 2026-11-11) |

Reference numbers follow SDAG: the envelope's for a message entry, the
first arithmetic argument for a marshalled entry; `inbox::name()` (by
hand: `ck::coro::noref`) matches any. The reference number is a
`CMK_REFNUM_TYPE` (`unsigned short` by default, a build option), the same
field SDAG matches on; it wraps, which is harmless as long as no delivery
from 65536 iterations ago is still unconsumed.

## 3. Rules

1. **State lives in members.** The outs of a when, and any other value that
   is live across a site, are members of the chare and are pupped by the
   chare's pup, exactly as SDAG requires of its state variables (`int imsg`
   is a member in both programs). A when has two forms: `CK_WHEN_BLOCK`
   declares the parameters as variables of the block that follows (an
   if-with-initializer holding a structured binding of the received
   values), and `CK_WHEN` fills existing members. A block cannot contain a
   further site: a restarted body cannot reconstruct a local, and the
   compiler rejects the jump past the binding, so SDAG's nested when that
   uses the outer parameters is written with the outer when in the member
   form. Kale weighed one rule (members only, 2026-11-10) against the
   common `for { when e(x) { work(x) } }` pattern and chose both forms
   (2026-11-11), since the misuse is a compile error, not a runtime
   surprise. C++ rejects the restart jump past a local with an initializer
   or a non-trivial type, which catches most mistakes at compile time; a
   scalar declared without an initializer, or declared before
   `CK_MIGRATABLE`, is not caught, and `+ck_fakemigrate` is the check.
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
5. **A coroutine entry's parameters are supplied again by the restart.**
   The record keeps the entry's message (a `[coroutine]` entry is not
   `nokeep`, so the state owns a marshalled message outright) and sends a
   copy on a restart; the generated wrapper unmarshals it again, so the
   body's parameters have their values at every site. They are taken by
   value in the C++ definition (a reference would bind to the wrapper's
   local); an array parameter points into the message the state keeps.
6. A site is one statement and may be the body of an `if` or a loop; its
   case label must not be inside a `switch` of the user's own; two sites must
   not share a line (CAMPI's rules, from the `switch`/`case __LINE__`
   mechanism).

What this allows that SDAG does not: the body is C++ with every construct
of the language (`break`, `continue`, `return` from the middle, nested
loops, lambdas, `switch` outside sites), nested blocking helpers returning
`task<T>` (CAMPI's `tests/campi/until` shows the pattern), and a debugger
that steps through the body. What SDAG allows that this forbids: packing at
any `when`, several concurrent invocations of one SDAG entry, inline
bodies for `overlap` clauses, and `delete this` in a body (section 9a
has the full list with the reasons).

## 4. The seven questions

### 4.1 Library or translator

Library first, then two small attributes. The library alone does
everything in section 2, because (a) charmxi's generated wrapper for a
void entry calls `impl_obj->doStep()` and ignores the return value, so the
C++ definition may return `ck::coroutine` while the .ci file says `void`,
and (b) an inbox entry is an ordinary entry whose body is one line. That
form (`ck::coro::state co` pupped by the chare, `co.deliver(...)` bodies,
`CK_COROUTINE_STATE(co, ep)`, `CK_WHEN(CkIndex_X::name(0), ref, out)`) still
works and is what the pingpong benchmark uses. Its three warts are the
entry index spelled with a placeholder argument at every site, the
one-line bodies, and the state member the chare has to declare and pup.

The two attributes remove them and nothing else. `[inbox]` and
`[coroutine]` make charmxi emit a macro, `X_CORO_CODE` (as `X_SDAG_CODE`),
that the class places in its body: the state member and its pup hook
(called by the generated recursive pup, as `_sdag_pup` is), the inbox
entry bodies, the test the promise of `ck::coroutine` uses to check that
the method runs as one of the chare's coroutine entries, and `struct
inbox` with one function per inbox entry returning a `ck::coro::want`. The generated wrapper of a
coroutine or inbox entry sets the message and entry index being delivered
(`CkEntryScope` in charm++.h, restored when the wrapper returns), and the
body learns them from `CkCurrentEntry()` and `CkCurrentMessage()`; until
the review of PR #4231 `CkInvokeEP` set them for every entry method, which
cost about 3 ns per entry method in every program. An inbox entry's
generated body is the one-liner the programmer wrote before. Validation: both attributes work on any chare
kind (chare, group, nodegroup, array element) and need a void return and
none of threaded, sync, local, inline, whenidle, iget, python, aggregate,
immediate, expedited or nokeep; `[inbox]` takes marshalled parameters, arrays included, or a
message; `[coroutine]` takes marshalled parameters (by value in the C++
definition) or a message and is not a reduction target; both reject
conditional and nocopy parameters. The whole translator change is the two
keywords, the two bits, the checks, and `Chare::genCoroCode` with
`Entry::genCoroCode` (about 120 lines in `src/xlat-i`), plus the
regenerated parser and scanner.

The generated body (reached through a reduction wrapper or a direct
call) and the hand-written form run the same delivery code, so there is
one path in the header to test and a program can drop the attributes
without touching the runtime. The generated `_call_` wrapper of a
marshalled `[inbox]` entry, however, does not unmarshal: it hands the
`CkMarshallMsg` to the state as it is (`deliver_message`, which reads
only the first parameter for the reference number), and the entry is
registered without `nokeep` so that the runtime leaves the message to
the state. A site unmarshals once, when the delivery matches. Array
parameters follow charmxi's layout (the scalars in order, an array as its
offset and byte count, then the arrays from a 16-byte aligned start): the
want carries the parameter types (`inbox::name_params`), so the site can
read the layout, and the generated body packs the same layout for the
other paths.

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

`when_any` and `when_all` exist: `CK_WHEN_ANY(idx, CK_ALT(...),
CK_ALL(CK_ALT(...), ...), ...)` waits for the first alternative whose
deliveries are all present and stores its index (SDAG `case`, arms with
conjunctions included); `CK_WHEN_ALL(CK_ALT(...), ...)` is one site that
takes several deliveries together (SDAG `when a(), b()`), so a migration
cannot fall between them. A record's wants carry an arm index; a delivery
joins the inbox and then each arm of a waiting record is tried in order
(`try_arm`: every want matched by a distinct buffer, all taken at once).
The reference number of a delivery through a reduction wrapper is the
reduction message's (set on the callback), as SDAG's closure does.

`overlap` and `forall` are child coroutines (2026-11-11): `CK_OVERLAP`
and `CK_FORALL` are one site of the parent that starts `task<void>`
helpers as child records, each with its own chain, and waits for all of
them. A child record is pupped inside its parent (chain and a done flag)
and is created again by the parent's restart: the parent body runs again
to the overlap site, the awaiter adopts the recorded children by position,
starts the unfinished ones toward their recorded sites and skips the
finished ones. So a child's results must be in pupped members, which is
rule 1 again. The join is the count of unfinished children; the last
child to finish resumes the parent. The alternative, `co_await
when_any(...)` over several awaitables, is what `when_any` already is,
and it does not give `overlap`, which is a conjunction of independent
progress. A `+ck_fakemigrate` rebuild requested by a child suspending
inside its parent's start is deferred until no coroutine of the element
is on the stack.

### 4.4 Message buffering

The inbox buffers every delivery that no coroutine is waiting for, keyed by
entry index (stable across PEs) and reference number, in arrival order, and
`CK_WHEN` takes the first match. Every buffered delivery is a message
(`CkPupMessage` moves them): the entry's own message, or the
`CkMarshallMsg` that carried the marshalled parameters, which the inbox
body keeps with a reference count instead of copying the parameters
(`CkCurrentMessage()` gives it the message being delivered; SDAG keeps the
marshalled message only for array parameters and copies the rest into its
closure). A delivery reached in another way, through the reduction
wrapper of a `[reductiontarget, inbox]` entry or by a direct call, packs
the arguments into a fresh `CkMarshallMsg` in the same layout, so a site
unpacks every buffered delivery the same way. A void entry buffers no
message.
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

### 4.8 The block instead of the begin/end pair (2026-11-11)

Until the evening of 2026-11-11 a body was bracketed by `CK_COROUTINE()`
and `CK_COROUTINE_END()`, a helper by `CK_CORO_TASK()` and
`CK_CORO_TASK_END()`: the first macro declared the guards and opened
`switch (site) { case 0:`, the second closed it with a `default:` that
aborted. This was the shape of CAMPI's `CAMPI_RESUMABLE()`/`CAMPI_END()` (now `CAMPI_MIGRATABLE { }`, #4242)
and of the C resumable-function libraries built on the same switch
(protothreads). Kale asked whether C++ could remove the pair, since a
plain coroutine needs nothing of the kind. What requires it is the
restart jump: the only standard way to jump forward to one of several
points chosen by an integer is a `switch` whose `case` labels sit at
those points, and the switch has to be in the user's function because the
labels are; a guard object's constructor and destructor cannot open and
close one. What can be removed is the closer and the registration:

- `CK_MIGRATABLE { ... }`: an object-like macro expanding to the frame
  guard and `switch (site) case 0: if (false) default: abort_no_site();
  else` with no brace of its own, so the user's braces are the else
  branch, the switch's substatement, and the sites inside them, nested in
  loops or not, are its cases (labels may sit anywhere in the
  substatement). Site 0 enters the block from the top, a recorded site
  jumps to its label, and any other value lands on `default:` and aborts
  before a statement of the block runs. The first version (pushed
  2026-11-11) had `default:` next to `case 0:`, so an unknown site ran
  the block from the top and aborted at the first site it reached; the
  CAMPI session pointed out on 2026-11-12 that everything before that
  first site then ran a second time on the restarted object (in CAMPI,
  MPI_Init and collectives the other ranks never join; in ckcoro, the
  sends before the first wait), and proposed the `if (false) default:`
  shape, which both libraries use. `abort_no_site` is `[[noreturn]]`,
  which keeps the end of a value-returning task unreachable for the
  compiler, so `co_return` inside the block is enough (without it clang
  warns that the task may fall off its end). The frame guard's destructor
  keeps its check that a restarted function reached its site.
- Registration in the promise: C++20 constructs the promise of a member
  coroutine with `*this` and the arguments when the promise has a
  constructor that accepts them, so `ck::coroutine::promise_type` has a
  constrained constructor for an owner with `__coro` and
  `__coro_is_entry` (both generated by `X_CORO_CODE`) and calls
  `state::begin` from `get_return_object`, before the body starts. The
  hand-written form without `X_CORO_CODE` keeps a registration
  statement, `CK_COROUTINE_STATE(co, ep)`, as the first line of the body.
  One compiler difference, found by Linux CI: GCC deduces the owner
  parameter of that constructor as a reference type (`Jacobi&`), clang as
  the class, so the constraint strips references (`owner_t`); with the
  constraint written on the deduced type as is, GCC default-constructs the
  promise and the body runs unregistered (the abort "CK_MIGRATABLE outside
  a coroutine").

The word says what the block buys: the code inside can be rebuilt after a
migration or a checkpoint. Every coroutine body is the block, because
each wait is a case of the switch it opens (the first text said a chare
that never migrates could leave it out; the review of PR #4231 pointed
out that a body without the block has no way to write a wait, and Kale
chose one rule over a second set of wait macros, 2026-11-12). The same
block serves entries and helpers, so the task pair is gone too. A scratch probe (block form,
labels inside loops, the promise constructor receiving `*this`, the
missed-site destructor) compiled clean under Apple clang 17 with `-Wall
-Wextra` before the conversion; the suite under `+ck_fakemigrate`,
checkpoint and restart is the real test. CAMPI can adopt the same block
in its own PR; the core change (the destructor check) is compatible with
its `default:` closer.

### 4.9 Futures, tracing and the pup mode (2026-11-12, after #4231)

The first review of #4231 listed three follow-ups; Kale asked for them in
one pull request.

**Awaiting a callback.** A reduction result or quiescence needed an
`[inbox, reductiontarget]` entry in the `.ci` file for every wait.
charmxi now adds one `[inbox]` entry, `_coro_callback(CkMessage *)`, to
every chare with a `[coroutine]` entry, and a `ck::coro::future` names
the deliveries to it that carry its id as the reference number. Two
kinds of callback were needed, which the first test showed: a reduction
delivers its result once, to the callback its contributions carry, so a
callback to each element reached only one of them. `CK_CORO_FUTURE_ALL(f)` is
a broadcast to the array, group or nodegroup, whose id every element
computes alike by counting its calls (the same discipline as
`contribute`); `CK_CORO_FUTURE(f)` is a callback to the element itself, for a
reply, with its own counter and the top bit of the id set so the two
never meet. The wait is the existing `CK_WHEN` with a future in place of
a want, not a new macro: `CK_WHEN(f, msg)`, `CK_WHEN(f)`,
`CK_WHEN_BLOCK(f, m)`, `CK_ALT(f, ...)`. Rejected: per-call ids chosen
by the user (as SDAG reference numbers), which reintroduce collisions
between independent uses; and awaiting `CkFuture`, which completes by
waking a thread and has no hook a coroutine could use. The macros say
`CORO` (Kale, 2026-11-13) because in Charm++ "future" already names
`CkFuture` and `ck::future<T>`, which differ on three counts: they live
in the table of the PE that created them, hold one value, and block a
thread; a coroutine's future is two numbers in a member of the chare,
migrates with it, takes any number of messages, and only a coroutine
waits on it. `CK_` stays as the prefix of every ckcoro macro.

**Projections.** The code after a wait ran inside whichever entry
delivered the message that resumed it, so a trace showed it under that
inbox entry (or under the load balancer's `ResumeFromSync`). The state
now traces each resume as an execution of the coroutine entry itself,
nested in the delivery, as SDAG's generated code traces a `serial` block.
No entry points are registered for this: SDAG names each `serial`, but a
coroutine's segments have no names, and the entry's own name is what the
review asked for.

A block is named with `CK_LABEL("name") { ... }`, the counterpart of
`serial "name"`: an entry `name` of the chare, nested in the execution
around it. SDAG's labels are registered by charmxi, which reads the
`serial` in the `.ci` file; a coroutine body is in the `.C` file, which
charmxi never reads. Two ways were weighed: the name only in the `.C`
file, recorded before `main` through a C++20 string template argument and
registered by `_initCharm` after the modules' entries; or the labels also
declared in the `.ci` file, for charmxi to register as SDAG's are, which
needed a grammar change and the name written twice. Kale chose the first
(2026-11-13). The order of static initialization is the same in every
process of one binary, which is what the entry indices need. A wait
inside the block is a compile error under `CK_MIGRATABLE` and an abort
otherwise, as a `serial` cannot contain a `when`.

**`+ck_fakemigrate_pup`.** `+ck_fakemigrate` rebuilds the coroutines in
place and so never runs the chare's own pup. The new flag makes an array
element migrate to the next PE (`migrateMe`) wherever `+ck_fakemigrate`
would rebuild, so the pup runs; it found a member that the ckcoro
marshall test used across waits without pupping. Two limits: the
location manager refuses a migration to the element's own PE, so the
mode needs two or more PEs (a one-PE version would need a change there);
and an element that uses AtSync is rebuilt in place, because an element
that migrates itself while waiting at the load balancer's barrier
leaves the barrier waiting for it (jacobi3d-coro with `lbFreq` 4 hung
this way). Groups, nodegroups and chares never migrate and are rebuilt
in place.

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
| disjunction / conjunction | `case`, `when a(), b()` | n/a | `CK_WHEN_ANY` with `CK_ALL` arms, `CK_WHEN_ALL` |
| independent progress | `overlap`, `forall` | n/a | `CK_OVERLAP`, `CK_FORALL` (child coroutines) |
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
| `case` | done, conjunction arms included: `CK_WHEN_ANY` in `tests/charm++/ckcoro` (basic test) |
| `overlap`, `forall` | done (2026-11-11): child coroutines, `tests/charm++/ckcoro` overlap, basic and refnum tests, with migration and `+ck_fakemigrate` |
| equivalence corpus: megatest synctest, every `tests/charm++/sdag` directory, SDAG-heavy miniapps under both implementations | done (2026-11-11): `tests/charm++/sdag` fully ported (`basic`, `refnum`, `migration` in `tests/charm++/ckcoro`), same output lines on 4 PEs; jacobi3d-coro matches jacobi3d-sdag's checksum; the SDAG programs with the most `when` clauses in the tree ported next to the originals with identical results on 1 and 4 PEs and under `+ck_fakemigrate`: `examples/charm++/leanmd-coro` (integer checksum of positions and forces), `examples/charm++/matmul-coro` (checksum), `examples/charm++/fib-coro`, `tests/charm++/sparse-coro` (same output). megatest's synctest has no SDAG (it tests sync entries) and was dropped. The ports found one defect (a conjunction kept only its last borrowed array, fixed) and two in the originals (shared `drand48` state across PE threads made matmul's and leanmd's results differ between 4-PE runs; both now seed per element) |
| CAMPI tests stay green when the core is shared | done (2026-11-10): `ckcoro_core.h` holds `task<T>` and the site chain for both; `tests/campi` green in every regression since |

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

**Recommendation: ship `ckcoro.h` with the `[inbox]` and `[coroutine]`
attributes in the beta, marked experimental.** The runtime changes are
the accessors `CkCurrentMessage()` and `CkCurrentEntry()`, set by the
generated wrappers of coroutine and inbox entries; `Chare::ckDestroyLater()`
(a message to the PE whose handler destroys the chare), with `ckDestroy`
made virtual; a node lock taken in the generated reduction wrapper of
`[exclusive]` nodegroup entries, which resends the reduction message when
the lock is busy; and a hook in the recursive pup. Two of them reach every
program: the node lock (a correction of the wrapper, which did not take
the lock before) and the virtual `ckDestroy`; `CkInvokeEP` is unchanged
since the review of PR #4231. The translator change is two attributes that
generate a macro. The branch carries a real program (jacobi3d with load
balancing, rebuild, checkpoint and restart), the SDAG migration test, and
a marshalled-inbox test (matching by the first parameter, a broadcast, a
reduction target). Matching costs a fifth of SDAG's and jacobi3d runs 14
to 17 percent faster per iteration. The restrictions are the ones CAMPI
already imposes on AMPI programs (state in members, sites are declared,
nothing packed except at a site) plus one active coroutine per entry and
no direct C++ call of a coroutine entry.

The attributes are in the beta (Kale, 2026-11-10) because they remove the
three things a reader of the hand-written form complains about first: the
entry index spelled `CkIndex_X::name(0)` at every site, the one-line
inbox bodies, and the state member to declare and pup. They were kept to
generating what the programmer would write, over the current-message
accessor, so that the hand-written form keeps working unchanged, there is
one delivery path, and a later decision about the `.ci` file in general
is not prejudged: with the `.ci` file kept, the attributes stay; without
it, the same macro can be produced by a C++20 member-pointer template
(`ck::entry<&X::name>`) over a registry, and the header does not change.

What "experimental" has to say: the site macros may still change.
Owners of every kind since 2026-11-11: the restart callback is built per
chare kind (`self_target`) and rebuilt from the owner on unpack, which
is what a checkpoint restart of a singleton chare needs; nodegroup
entries are made exclusive by charmxi, as SDAG's.

What charm4py can mirror: the record and site protocol of section 8, with
no C++ frame involved.

Everything the beta needs is on the branch: the shared core with CAMPI
(`ckcoro_core.h`, CAMPI's tests green), the two test directories in CI
(`TEST_DIRS` in `reconverse-ci.yaml`), and the Mac and Anvil measurements.
First beta update: `overlap` as several coroutines per element and the
rest of the SDAG test corpus (both done 2026-11-11).

## 9a. Alternatives considered, and SDAG features not carried over

The sections above give each decision in its place; this section lists
them in one table each so that a reviewer can check what was weighed.

### Alternatives considered

| alternative | decision | reason |
|---|---|---|
| Pack the coroutine frame on migration | rejected outright | The frame's layout is compiler private and holds code addresses and the compiler's own temporaries; there is no portable way to serialize it, and a frame restored in another process would hold the old process's addresses. The same reason rules out moving a frame between PEs of one process when the two PEs differ in anything the frame points to. (Section 1.) |
| Stackful threads: a `[threaded]` entry with isomalloc-migrated stacks, as AMPI does | rejected | Already available and unchanged; it needs isomalloc, a stack per suspended entry, and an identical program load address in every process (address-space randomization off, which macOS cannot do; `machines/mac.md`). The restart protocol packs integers and a message and needs none of that. (Section 4.5.) |
| Keep SDAG's translator and generate closures from a C++ body | rejected | It would need charmxi to parse C++ bodies, which it cannot, or a C++ front end; the point of the design is that the body is ordinary C++ compiled by the compiler, with the `.ci` file holding declarations only. (Section 4.1.) |
| A guard object whose constructor and destructor replace the begin/end macros | rejected | The pair is a `switch` and its closing brace, which an object cannot open or close; the guards are objects already. The block form of section 4.8 is what C++ allows. |
| A separate state object (CAMPI's `State`) instead of the chare | rejected | The chare already has a pup and its members already are SDAG's state variables; a second object would duplicate the rule. (Section 4.2, decision 2.) |
| Library only, no `.ci` attributes | kept as the hand-written form, attributes added | The library alone works (the pingpong benchmark still uses it) but every site spells the entry index with a placeholder argument, every inbox entry is a one-line body, and the chare declares and pups the state. Two attributes generate those three things and nothing else; the delivery path is one. (Section 4.1, Kale 2026-11-10.) |
| A C++20 member-pointer template (`ck::entry<&X::name>`) over a registry instead of attributes | deferred | Possible later without changing the header; it belongs to a decision about the `.ci` file in general, which this PR does not take. (Section 9.) |
| Reuse SDAG's `Dependency`/`Buffer`/`Continuation` structures | rejected | The waiter here is a coroutine handle plus a want, not a closure chain; the matching is forty lines, and sharing the structures would tie the two implementations' lifetimes. (Section 4.4.) |
| Copy marshalled parameters into a closure on arrival (SDAG) | rejected | The `CkMarshallMsg` is kept with a reference count and unmarshalled once at the site that takes it, so array parameters are read in place and a delivery is one buffer kind for every path. (Section 4.4.) |
| Members-only rule for received values | both forms kept | `CK_WHEN` fills members; `CK_WHEN_BLOCK` gives a block scope, since `for { when e(x) { work(x) } }` is the common pattern and the misuse (a wait inside the block) is a compile error. (Section 3 rule 1, Kale 2026-11-11.) |
| Conjunction `when a(), b()` as two sequential waits | used in the prototype, replaced | A migration can fall between the two waits and the first value, if it is not a pupped member, is lost; Linux CI found exactly that in the ported migration test. `CK_WHEN_ALL` is one site that takes both deliveries. (Section 4.3.) |
| `overlap` as `co_await when_any(...)` over several awaitables | rejected | That is `CK_WHEN_ANY`; `overlap` is independent progress of several clauses, which needs a coroutine per clause (child records pupped inside the parent). (Section 4.3.) |
| Inline bodies for `overlap` clauses, as lambda coroutines | rejected for now | A lambda coroutine outlives its closure object, so captures dangle; a restart would have to recreate the lambda in the same state. Each clause is a named `ck::coro::task<void>`. |
| Migration while suspended at any `co_await` | restricted to sites | A plain `co_await` between sites is allowed; packing there aborts with the entry's name, because only a site has a recorded id the restart can jump to. (Section 3 rule 3.) |
| Several active invocations of one coroutine entry per chare | rejected unless a program needs it | A restart is a plain re-invocation of the entry; with several invocations the restart would have to tell them apart, and no program in the corpus needs it. (Section 3 rule 4, Kale 2026-11-11.) |
| `delete this` inside a body (allowed in an SDAG `when` body) | replaced by `Chare::ckDestroyLater()` | The coroutine body is still on the stack when it would delete the chare, and the chare's state member is touched when the body ends; `fib-coro` first used a separate destroy entry, and Kale asked for a facility in the base class instead (2026-11-11). The chare is destroyed by the handler of a message `ckDestroyLater` sends to its own PE, after the call running on the chare returns (an entry method or a direct call such as the load balancer's `ResumeFromSync`). |
| `exclusive` written by hand on a nodegroup's coroutine or inbox entries | rejected, charmxi adds it | SDAG's rule: the entries are exclusive, the node lock is released at every wait because the entry method returns there, and a hand-written tag is an error so that the generated code is the only place the lock is taken. (Kale 2026-11-11.) |
| `CK_END` closing `CK_COROUTINE()`, and `CK_TASK` for a coroutine task | renamed, then replaced | First matched pairs with no synonyms (`CK_COROUTINE`/`CK_COROUTINE_END`, `CK_CORO_TASK`/`CK_CORO_TASK_END`; `CK_TASK` would collide with Charm++'s stealable tasks), then the pairs themselves were replaced by one block, `CK_MIGRATABLE { ... }`, with the registration moved into the promise. (Kale 2026-11-11; section 4.8.) |
| An awaitable future | added after #4231 (section 4.9) | `ck::coro::future` with `CK_CORO_FUTURE` / `CK_CORO_FUTURE_ALL`, through an inbox entry charmxi adds; the first review asked to await a reduction result or quiescence without an `[inbox, reductiontarget]` entry. |

### SDAG features not carried over

| SDAG feature | here | reason |
|---|---|---|
| Migration or checkpoint at any `when` | only at a site (`CK_WHEN`, `CK_WHEN_BLOCK`, `CK_WHEN_ALL`, `CK_WHEN_ANY`, `CK_CALL`, `CK_OVERLAP`, `CK_FORALL`) | A pure C++ solution cannot pack the frame; a site is where the body has a recorded id to jump to. In practice every SDAG `when` becomes a site, so the loss is only the plain `co_await` between sites. |
| Resume without re-running | the body runs again from the top to the recorded site | The frame is rebuilt by execution, not restored; code before a site must not repeat a side effect (rule 2). SDAG's closures carry the position. |
| Several concurrent invocations of one SDAG entry | an error | See the table above. |
| A `when` body that uses the outer when's parameters across a nested `when` | the outer wait in member form | A block-scoped variable cannot survive the restart jump; the compiler rejects the jump past its initializer. SDAG keeps the parameters in a packed closure; here the member is that closure. |
| `overlap` clauses and `forall` bodies written inline | named coroutine tasks | See the table above. |
| `delete this` in a `when` body | `ckDestroyLater()` | See the table above. |
| `atomic`/`serial` blocks | plain code | Nothing to carry: every statement of the body is serial code. |
| State variables declared in the class but used as if local | the same rule, visible | SDAG's surprise ("state variables must be declared in the class") remains the rule here; the difference is that the body is C++ and the reader sees the member. |
| A Projections entry per named `serial` block (`_TRACE_BEGIN_EXECUTE_DETAILED` around the generated method) | an event per resumed segment, under the coroutine entry; `CK_LABEL("name") { ... }` for a named block (section 4.9) | There is no generated method to name; the code between two waits is the user's. The state traces each resume as an execution of the coroutine entry, nested in the delivery that resumed it, and a labelled block as an entry of its own, registered at startup from the `.C` file. |
| C++11 | C++20 for the files that include `ckcoro.h` | Coroutines are a C++20 feature; the rest of the program and the runtime stay at their standard. |

## 10. Open items

- Several active invocations of one coroutine entry per chare (an error;
  stays so unless a program needs it, Kale 2026-11-11).
- Inline bodies for `overlap` clauses (see 9a).
- Under `+ck_fakemigrate_pup`, array elements that use AtSync are rebuilt
  in place (section 4.9): moving them would need the load balancer's
  barrier to accept a migration the element starts while at the barrier.
- Retiring `CkFuture` as a user API for chare code (Kale asked,
  2026-11-13). The mechanism stays: AMPI's blocking calls, `[sync]`
  entries and `CkWaitQD` block threads with it. The user API
  (`ck::future<T>`, `[sync]` in chare code) could be marked legacy once
  coroutines have a typed value (a marshalled send that `CK_WHEN(f,
  value)` unmarshals) and a form that awaits a remote entry's return
  value, as `[sync]` gives a thread.
- `CK_CORO_FUTURE_ALL` ids are counted per element: an element inserted after
  others have made collective futures starts its count at zero.
- The copy of the entry's message that the coroutine state makes for a
  coroutine entry that takes a message type: paid once per invocation
  (a restart sends a second copy), measured only for small messages
  (jacobi3d, the tests).
