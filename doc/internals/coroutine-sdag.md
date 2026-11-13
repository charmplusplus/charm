# Coroutine entry methods (ckcoro)

A coroutine entry method is an entry method whose C++ definition returns
`ck::coroutine`: its body runs when the entry is invoked, suspends at
`CK_WHEN` sites until a delivery to an inbox entry matches, and finishes
like any entry method. The chare's `ck::coro::state` member (any chare
kind: array element, group or nodegroup branch, singleton chare,
mainchare) holds the records of its active coroutines and the inbox, and
packs them with the chare; a coroutine frame is never packed. The mechanism is in
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
  owner (`self_target`, one overload per chare kind: an array element by
  index, a group branch by PE, a nodegroup branch by node, a chare or
  mainchare by handle), the entry's message `msg` for restarts (a message
  entry's is a packed copy; a marshalled entry's is the `CkMarshallMsg`
  its parameters point into; null for void entries), `pending` (a restart
  message is on its way; pupped), the restart `chain`, the frame handles
  `root` (the entry's frame) and `h` (the innermost suspended frame),
  `frameMsg` and `frameMsgDel` (a message entry's: the message the frame
  received and its typed delete, not pupped), and the wait state: `waiting`, `atSite`,
  `rebuilt` (restarted and not yet past the recorded site), `wants` (the
  entry and reference number pairs it waits for, each tagged with the
  `arm` of a when_any it belongs to), and the delivered `readyBufs` (one
  per want of the arm) with `readyIndex`.
- `chain`: `sites` (one site id per resumable-function depth, pupped),
  `linked` (per depth: entered through a `CK_CALL` site), `depth`,
  `restarting`, and the `noted` fields that connect a `note_site` call with
  the `co_await` that follows it. `chain`, `frame_guard`, `note_site`,
  `abort_no_site` and `task<T>` are in `ckcoro_core.h`, shared with CAMPI,
  whose rank record holds one `chain`.
- `buffer`: `ep`, `ref`, and `msg` (pupped with `CkPupMessage`): the
  entry's message, or the `CkMarshallMsg` carrying its marshalled
  parameters, or null for a void entry. `record.held` (a vector) keeps
  the deliveries of the last wait while a pointer out (an array
  parameter) points into one of them, one per alternative of a
  conjunction; `enter_await` releases them at the next wait.
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
- `CkpvAccess(_ckCurrentMsg)`, `CkpvAccess(_ckCurrentEp)` (ck.C; set and
  restored by `CkEntryScope`, charm++.h, which the generated wrappers of
  coroutine and inbox entries declare around the call, in `_call_` and
  the reduction wrapper, whose scope names the target entry's own index):
  the message and entry index being delivered, read as
  `CkCurrentMessage()` and `CkCurrentEntry()`; null and -1 in other entry
  methods, and after `deliver` has taken its delivery.
- The generated `X_CORO_CODE` macro (`Chare::genCoroCode`,
  `Entry::genCoroCode` in src/xlat-i): `ck::coro::state __coro`,
  `_coro_pup(p)` (`__coro.pup(p, this)`, called by `recursive_pup_impl`
  in charm++.h next to `_sdag_pup`), `static bool __coro_is_entry(int ep)` (true for the
  chare's `[coroutine]` entries), one method body per `[inbox]` entry
  (`__coro.deliver(CkIndex_X::idx_name(), params...)`, an array as
  `array_arg<T>{p, bytes}`), and `struct inbox` with `using name_params`
  and `static want_t<name_params> name()` / `name(ref)` per inbox entry.
  The `_call_` wrapper of a marshalled inbox entry (`Entry::genDefs`) is
  `__coro.deliver_message<X::inbox::name_params>(ep, msg)` instead of the
  unmarshalling call, and the entry is not `nokeep`.

## Protocols

**Start.** The generated entry wrapper calls the user's method, which is a
coroutine with `initial_suspend` = never. Its promise is constructed with
the chare (`*this`) and the frame's copies of the parameters when the
chare has `X_CORO_CODE`; it notes the message parameter's address in the
frame and a typed delete (`entry_msg`). `get_return_object` calls
`state::begin(elt, isEntry, em)` before the body starts: `ep` comes from
`CkCurrentEntry()` and `msg` from `CkCurrentMessage()` (null for a void
entry's system message), after
checking `__coro_is_entry(ep)` so that a direct C++ call of the method
aborts instead of recording the caller's
entry. `begin` adopts a restarting record of the same entry if there is
one, else creates a record (aborting if an active record of that entry
exists), stores the frame handle and sets `current_`. For a new record of
a message entry it packs the frame's message in place, copies the packed
bytes into `msg` and unpacks it again (`packed_copy`); a custom pack
replaces the message, so the frame's parameter (`em.slot`) and
`CkCurrentMessage()` are pointed at the new one. The frame's message is
the record's `frameMsg`, deleted with the frame (`destroy_frames`), so a
message lost with a migrated frame is not leaked and the body does not
delete it. The `frame_guard` that follows opens the switch
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
args...)`, the generated body of a void entry or of a marshalled entry
reached through a reduction wrapper or a direct call, or a hand-written
inbox body, takes the reference number from the first arithmetic
argument, else from the entry's own message when `CkCurrentEntry() == ep`
(the scope its wrapper set: a void entry's system message, or a reduction
target's `CkReductionMsg`, whose reference number wins over the first
argument). It then clears the scope, so that a coroutine resumed by this
delivery and calling an inbox entry directly is not taken for it, and
packs `args` into a new `CkMarshallMsg` in charmxi's layout: each scalar pupped in order, an `array_arg<T>` as
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
(the buffer is pushed on `record.held` until the next wait), a container out
copies the elements with `assign`; a `ck::coro::skip` out reads into a
temporary. Without a typed want (the hand-written form) the outs are
pupped in order with no array support. `CK_WHEN_ALL` uses `all_awaiter`
over several `when_awaiter`s: ready only when `state::try_arm` finds a
distinct buffered match for every want (then all are taken, in want
order), else all its wants are stored under arm 0. `CK_WHEN_ANY` uses
`any_awaiter` over alternatives that are `when_awaiter`s or
`all_awaiter`s (`try_ready`, `add_wants(ws, arm)`, `unpack_ready`):
`await_ready` tries each alternative in order, `await_suspend` stores
their wants tagged with the alternative's index, and `await_resume`
unpacks through the alternative at `readyIndex` and yields that index.
`state::push` appends the delivery to the inbox and then, for each
waiting record one of whose wants matches it, tries its arms in order
with `try_arm`; the first satisfied arm resumes the record.

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
forwards it), an unpacked copy of the record's message (`restart_copy`:
a packed copy is copied as bytes and only the copy unpacked; a
`CkMarshallMsg` goes through `CkCopyMsg`), or a fresh system message for
a void entry, is sent through `target` and `pending` is set, so the entry
is invoked again by the runtime after the unpack completes. `pup(p,
owner)`, the form the generated `_coro_pup` calls, rebuilds `target`
from the owner before sending (`self_target`), so a chare restored from
a checkpoint, whose handle is new, and a group branch are reached; the
library form `p | state` keeps the packed target. The promise adopts
the record (clearing `pending`, setting `chain.restarting`), and each
`frame_guard` on the recorded path, declared by `CK_MIGRATABLE`, switches
to its site: the macro expands to the guard and `switch (site) case 0:
if (false) default: abort_no_site(); else` followed by the user's block,
whose sites are the cases. `note_site` checks that the site reached is
the recorded one; a recorded site that is not a case of the function
lands on `default:` and aborts before any statement of the block runs
(the guard's destructor also checks that a restarted function reached
its site).
Deliveries that arrive before the restart message are buffered as usual.

**Rebuild (`+ck_fakemigrate`).** After a body suspends at a site it did
not just restart to, `finish_run` calls `state::rebuild`: the state is
packed into memory, every frame destroyed, the records and inbox cleared,
and the state unpacked from memory, which sends the restarts.

**Helpers.** A `ck::coro::task<T>` helper awaited with `CK_CALL` has its
own `CK_MIGRATABLE` block and therefore its own depth in the chain;
`note_site(..., call = true)` marks the call site and `frame_guard` records
that the helper was entered through it (`chain.linked`), as in CAMPI.

## Limits and diagnostics

- One active coroutine per entry per chare (`begin` aborts otherwise).
- On a nodegroup, charmxi makes `[inbox]` and `[coroutine]` entries
  `[exclusive]` (as SDAG's when-entries of a nodegroup) and rejects the
  attribute written by hand. The generated `_call_` of a marshalled inbox
  entry and the reduction wrapper of an exclusive entry take the node
  lock with the same try-lock-and-resend as other exclusive entries; the
  resend copies the message only when the entry is nokeep. The lock is
  released when the coroutine suspends, since the entry returns there.
- Coroutine entries take a message or marshalled parameters (not
  conditional or nocopy); a `[coroutine]` entry is not `nokeep`, and
  `state::begin` takes its marshalled message as the record's own
  (`own`), replacing it on a restart so that array parameters point into
  the live copy.
- An overlap restarted with a different number of tasks aborts.
- A chare packed while suspended at a non-site `co_await` aborts with
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
- `CKCORO_DEBUG=1` in the environment prints each rebuild and each
  restart.
- A coroutine body must not delete its chare (its frame would be destroyed
  while running). `Chare::ckDestroyLater()` (charm++.h, ck.C) sends a
  Converse message (`CkDestroyLaterMsg`, handler
  `_ckDestroyLaterHandler`, registered in init.C) to its own PE, whose
  handler calls the virtual `ckDestroy()`; so the destruction follows the
  return of whatever is running on the chare, an entry method or a direct
  call (the load balancer's `ResumeFromSync`), and several chares of one
  PE may ask in one entry method. The per-PE set `_ckDestroyLaterPending`
  holds the chares whose message is on its way: `ckDestroyLater` sends
  only when it inserts, the handler destroys only what it erases, and
  `~Chare` erases its chare, so a second call sends nothing and a chare
  destroyed another way first is not destroyed again. `Chare::ckDestroy`
  deletes, the array element's detaches first, a group's aborts.

## Where in the source

- `src/ck-core/ckcoro.h`: everything above except the shared core.
- `src/ck-core/ckcoro_core.h`: `task<T>`, `chain`, `frame_guard`,
  `note_site`, `abort_no_site`, `current_chain_`; included by `ckcoro.h`
  and by CAMPI's `campi_core.h`.
- `src/ck-core/CMakeLists.txt`: installs the headers.
- `src/ck-core/charm++.h` (`CkEntryScope`, `CkCurrentMessage`,
  `CkCurrentEntry`, `ckDestroyLater`, `recursive_pup_impl`), `ck.C`,
  `init.C`: the accessors, the destroy message and the `_coro_pup` hook.
- `src/xlat-i/xi-scan.l`, `xi-grammar.y` (and the regenerated `xi-scan.C`,
  `xi-grammar.tab.C/h`), `xi-Entry.{h,C}`, `xi-Chare.{h,C}`, `xi-Member.h`:
  the two attributes, their checks, and `X_CORO_CODE`. The scanner returns
  `INBOX` and `COROUTINE` only inside the `[...]` that follows `entry`
  (`search` and `punct` in xi-scan.l), and `IDENT` everywhere else, so the
  words stay names (`tests/charm++/ckcoro/names.ci`).
- `tests/charm++/jacobi3d-coro/`: the first program (migration at a site,
  rebuild, checkpoint and restart).
- `tests/charm++/ckcoro/`: the SDAG migration test as a coroutine
  (synchronized and anytime migration while suspended, `CK_WHEN_ANY`) and
  the marshalled inbox test (matching by the first parameter, a broadcast,
  a reduction target, under `+ck_fakemigrate`), the SDAG basic and
  refnum tests, overlap under migration, and the owners module (a group,
  a nodegroup, a singleton chare and the mainchare, checkpointed and
  restarted while suspended).
- `benchmarks/charm++/coro_pingpong/`: plain, SDAG and coroutine ping-pong.
- `src/libs/ck-libs/ampi/campi_core.h`, `campi.C`: CAMPI, the other user
  of the shared core.
