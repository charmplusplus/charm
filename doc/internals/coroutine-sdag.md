# Coroutine entry methods (ckcoro)

A coroutine entry method is an entry method whose C++ definition returns
`ck::coroutine`: its body runs when the entry is invoked, suspends at
`CK_WHEN` sites until a delivery to an inbox entry matches, and finishes
like any entry method. The element's `ck::coro::state` member holds the
records of its active coroutines and the inbox, and packs them with the
element; a coroutine frame is never packed. The mechanism is in
`src/ck-core/ckcoro.h` (header-only, C++20); charmxi's `[inbox]` and
`[coroutine]` attributes generate the `X_CORO_CODE` macro that declares
the state member and the inbox entry bodies; the runtime contributes
`CkCurrentMessage()`/`CkCurrentEntry()`. The reasons and the measurements
are in `doc/coroutine-sdag-design.md`; the user-facing contract is the
header's opening comment and the manual section "Coroutine Entry Methods".

## Data structures

- `ck::coro::state` (one per chare): `std::list<record> recs_` and
  `std::vector<buffer> inbox_`.
- `record`: the coroutine entry's index `ep`, a `CkCallback target` to the
  element, the entry's message `msg` (a copy, for message entries; null for
  void entries), `pending` (a restart message is on its way; pupped), the
  restart `chain`, the frame handles `root` (the entry's frame) and `h`
  (the innermost suspended frame), and the wait state: `waiting`, `atSite`,
  `rebuilt` (restarted and not yet past the recorded site), `wants` (the
  entry and reference number pairs it waits for: one for a when, several
  for a when_any), and the delivered `readyBuf` with `readyIndex`.
- `chain`: `sites` (one site id per resumable-function depth, pupped),
  `linked` (per depth: entered through a `CK_CALL` site), `depth`,
  `restarting`, and the `noted` fields that connect a `note_site` call with
  the `co_await` that follows it. `chain`, `frame_guard`, `note_site`,
  `abort_no_site` and `task<T>` are in `ckcoro_core.h`, shared with CAMPI,
  whose rank record holds one `chain`.
- `buffer`: `ep`, `ref`, and `msg` (pupped with `CkPupMessage`): the
  entry's message, or the `CkMarshallMsg` carrying its marshalled
  parameters, or null for a void entry. `record.held` keeps the last
  delivery while a pointer out (an array parameter) points into it;
  `enter_await` releases it at the next wait.
- `want_t<params<Ps...>>`: a want carrying the entry's parameter types
  (`value_t<T>` per marshalled parameter, `array_param<T>` for an array,
  the message pointer for a message entry), generated as
  `inbox::name_params`; `when_awaiter<Schema, Outs...>` checks the outs
  against it (`static_assert` on the count) and unpacks arrays.
  `tuple_awaiter<Ps...>` (the block form) owns its outs and returns them
  as a tuple for the structured binding of `CK_WHEN_BLOCK`.
- `record.kids` (`std::list<record>`, pupped inside the parent as chain
  and `done` per child), `parent`, `done`, `kidsRunning`: the child
  coroutines of an overlap site. `state::visit` walks records and
  children; `push` and the pack checks use it.
- `state::pendingRebuild_`: a `+ck_fakemigrate` rebuild requested while
  another coroutine of the element is on the stack, done by the
  outermost `finish_run`.
- `thread_local record *current_`: the record of the coroutine running on
  this PE thread; set by `state::begin` and `state::push` around a run,
  read by the awaitables.
- `thread_local chain *current_chain_` (`ckcoro_core.h`): the chain of the
  coroutine running on this PE thread, read by `frame_guard`; set with
  `current_` by `state::begin` and `state::push`, and by CAMPI's
  `campi_resume` to the rank's chain.
- `thread_local std::coroutine_handle<> pending_root_`: set by
  `coroutine::promise_type::get_return_object`, consumed by `state::begin`
  so that the record learns its frame handle.
- `CkpvAccess(_ckCurrentMsg)`, `CkpvAccess(_ckCurrentEp)` (ck.C, set and
  restored around the call in `CkInvokeEP`, charm++.h): the message and
  entry index being delivered, read as `CkCurrentMessage()` and
  `CkCurrentEntry()`.
- The generated `X_CORO_CODE` macro (`Chare::genCoroCode`,
  `Entry::genCoroCode` in src/xlat-i): `ck::coro::state __coro`,
  `_coro_pup(p)` (called by `recursive_pup_impl` in charm++.h next to
  `_sdag_pup`), `static bool __coro_is_entry(int ep)` (true for the
  chare's `[coroutine]` entries), one method body per `[inbox]` entry
  (`__coro.deliver(CkIndex_X::idx_name(), params...)`, an array as
  `array_arg<T>{p, bytes}`), and `struct inbox` with `using name_params`
  and `static want_t<name_params> name()` / `name(ref)` per inbox entry.
  The `_call_` wrapper of a marshalled inbox entry (`Entry::genDefs`) is
  `__coro.deliver_message<X::inbox::name_params>(ep, msg)` instead of the
  unmarshalling call, and the entry is not `nokeep`.

## Protocols

**Start.** The generated entry wrapper calls the user's method, which is a
coroutine with `initial_suspend` = never: the body runs to `CK_COROUTINE`,
whose `root_guard` calls `state::begin(elt, ep, msg)`. With no arguments
(`CK_COROUTINE()`, the attribute form) the guard takes `ep` from
`CkCurrentEntry()` and `msg` from `CkCurrentMessage()` (null for a void
entry's system message), after checking `__coro_is_entry(ep)` so that a
direct C++ call of the method aborts instead of recording the caller's
entry. `begin` adopts a restarting record of the same entry if there is
one, else creates a record (aborting if an active record of that entry
exists), stores the frame handle and sets `current_`. The `frame_guard` that follows opens the switch
at `chain.sites[0]` when restarting, else at 0. When the body first
suspends or finishes, the wrapper's discarded `ck::coroutine` object is
destroyed and `~coroutine` calls `state::finish_run`, which clears
`current_`, destroys a finished frame and erases its record, or rebuilds
when `+ck_fakemigrate` asked for it.

**Delivery.** `state::deliver(ep, msg)` buffers the message (reference
number: the envelope's). `state::deliver_message<P>(ep, m)`, called by
the generated wrapper of a marshalled inbox entry, buffers the
`CkMarshallMsg` unopened (reference number: the first parameter if
arithmetic, read with `PUP::fromMem`, else the envelope's); the entry is
not `nokeep`, so the state owns the message. `state::deliver(ep,
args...)` (reference number: the first arithmetic argument, else the
envelope's), the generated body reached through a reduction wrapper or a
direct call, or a hand-written inbox body, keeps the message being
delivered when `CkCurrentEntry() == ep` and it is a `CkMarshallMsg`
(`CkReferenceMsg`), and otherwise packs `args` into a new `CkMarshallMsg`
in charmxi's layout: each scalar pupped in order, an `array_arg<T>` as
its offset (aligned to `sizeof(T)`) and byte count, then the arrays from
`CK_ALIGN(scalar size, 16)`. A void entry buffers no message. Then
`push`. `push` looks for a waiting record
whose `wants` match; if found it moves the buffer into the record, sets
`current_`, resumes `record.h` and calls `finish_run` when the body
suspends again or finishes; otherwise it appends the buffer to the inbox.

**Waiting.** `CK_WHEN` is `case __LINE__: note_site(...); co_await
state.when(ep, ref, out...)`. `enter_await` clears `chain.restarting` (a
restarted body has reached its site, remembered in `record.rebuilt`) and
records whether a site was noted (`record.atSite`). `when_awaiter::await_ready`
then takes the first matching buffer from the inbox (`state::take`, keeping
the order of the rest); if there is none, `await_suspend` stores the `want`
and `arm_wait` records the innermost handle and sets `fakeRestart` when the
flag is on and the body was not just restarted to this site. `await_resume`
unpacks the delivered buffer into the outs: a message pointer is
handed over (the coroutine owns the message); marshalled parameters are
read back in order with `PUP::fromMem` from the `CkMarshallMsg`'s buffer,
an array's offset and count read into slots and, after the scalars, its
pointer computed from the aligned array start: a pointer out gets it
(the buffer moves to `record.held` until the next wait), a container out
copies the elements with `assign`; a `ck::coro::skip` out reads into a
temporary. Without a typed want (the hand-written form) the outs are
pupped in order with no array support. `CK_WHEN_ANY` uses `any_awaiter` over several
`when_awaiter`s: `await_ready` tries each alternative in order, `await_suspend`
stores all their `want`s, and `await_resume` unpacks through the
alternative at `readyIndex` and yields that index.

**Overlap.** `CK_OVERLAP(tasks...)` / `CK_FORALL(i, lo, hi, expr)` are
`case __LINE__: note_site(...); co_await state.overlap(vector<task<void>>)`.
`overlap_awaiter::await_ready` creates the child records on the first
visit (or finds them, recorded, on a restart), counts the unfinished
ones, and for each unfinished child takes the task's frame
(`task::release`), sets `chain.restarting` when it has recorded sites,
and `state::run`s it: the child runs to its first site (its own
`frame_guard`, depth 0 of its own chain) or to its end. A child that
finishes has `finish_run` mark it `done`, decrement the parent's count
and, when the count reaches zero and the parent is waiting, resume the
parent. `await_ready` is true when no child is unfinished;
`await_suspend` arms the parent with no wants; `await_resume` clears the
children. Deliveries find waiting children through `visit`.

**Pack.** `state::pup` writes the inbox and, per record, `ep`, `target`,
`msg` and `chain.sites`. Packing aborts if a record is running or is
suspended at a `co_await` that is not a site (`waiting && atSite` is the
condition).

**Unpack and restart.** `state::pup` reads the same and calls
`send_restart` for each record: unless a restart message is already
`pending` (sent before the element moved again; the location manager
forwards it), a copy of the record's message (or a fresh system message for
a void entry) is sent through `target` and `pending` is set, so the entry
is invoked again by the array manager after the unpack completes. The body
runs to `CK_COROUTINE`, adopts the record (clearing `pending`, setting
`chain.restarting`), and each `frame_guard` on the recorded path switches
to its site; `note_site` checks that the site reached is the recorded one.
Deliveries that arrive before the restart message are buffered as usual.

**Rebuild (`+ck_fakemigrate`).** After a body suspends at a site it did
not just restart to, `finish_run` calls `state::rebuild`: the state is
packed into memory, every frame destroyed, the records and inbox cleared,
and the state unpacked from memory, which sends the restarts.

**Helpers.** A `ck::coro::task<T>` helper awaited with `CK_CALL` has its
own `CK_RESUMABLE`/`CK_END` and therefore its own depth in the chain;
`note_site(..., call = true)` marks the call site and `frame_guard` records
that the helper was entered through it (`chain.linked`), as in CAMPI.

## Limits and diagnostics

- One active coroutine per entry per element (`begin` aborts otherwise).
- Coroutine entries take a message or marshalled parameters (not
  conditional or nocopy); a `[coroutine]` entry is not `nokeep`, and
  `state::begin` takes its marshalled message as the record's own
  (`own`), replacing it on a restart so that array parameters point into
  the live copy.
- An overlap restarted with a different number of tasks aborts.
- An element packed while suspended at a non-site `co_await` aborts with
  the entry index.
- A restarted body that reaches a site other than the recorded one, or a
  helper not on the recorded path, aborts (`note_site`, `frame_guard`).
- Marshalled inbox parameters must be types with a `pup` (scalars, STL
  containers, pupable classes) or arrays; charmxi rejects conditional and
  nocopy parameters on `[inbox]` entries.
- A when's out count must equal the entry's parameter count
  (`static_assert`); a `CK_WHEN_BLOCK` containing a site fails to compile
  (jump past the structured binding); a `CK_WHEN_BLOCK` cannot be followed
  by `else`.
- `CkCurrentMessage()` sees the caller's delivery inside an `[inline]` or
  `[local]` call; `deliver` then packs the arguments instead.
- `CKCORO_DEBUG=1` in the environment prints each rebuild.

## Where in the source

- `src/ck-core/ckcoro.h`: everything above except the shared core.
- `src/ck-core/ckcoro_core.h`: `task<T>`, `chain`, `frame_guard`,
  `note_site`, `abort_no_site`, `current_chain_`; included by `ckcoro.h`
  and by CAMPI's `campi_core.h`.
- `src/ck-core/CMakeLists.txt`: installs the headers.
- `src/ck-core/charm++.h` (`CkInvokeEP`, `CkCurrentMessage`,
  `CkCurrentEntry`, `recursive_pup_impl`), `ck.C`, `init.C`: the accessor
  and the `_coro_pup` hook.
- `src/xlat-i/xi-scan.l`, `xi-grammar.y` (and the regenerated `xi-scan.C`,
  `xi-grammar.tab.C/h`), `xi-Entry.{h,C}`, `xi-Chare.{h,C}`, `xi-Member.h`:
  the two attributes, their checks, and `X_CORO_CODE`.
- `tests/charm++/jacobi3d-coro/`: the first program (migration at a site,
  rebuild, checkpoint and restart).
- `tests/charm++/ckcoro/`: the SDAG migration test as a coroutine
  (synchronized and anytime migration while suspended, `CK_WHEN_ANY`) and
  the marshalled inbox test (matching by the first parameter, a broadcast,
  a reduction target, under `+ck_fakemigrate`).
- `benchmarks/charm++/coro_pingpong/`: plain, SDAG and coroutine ping-pong.
- `src/libs/ck-libs/ampi/campi_core.h`, `campi.C`: CAMPI, the other user
  of the shared core.
