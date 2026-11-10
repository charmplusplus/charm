# Coroutine entry methods (ckcoro)

A coroutine entry method is an entry method whose C++ definition returns
`ck::coroutine`: its body runs when the entry is invoked, suspends at
`CK_WHEN` sites until a delivery to an inbox entry matches, and finishes
like any entry method. The element's `ck::coro::state` member holds the
records of its active coroutines and the inbox, and packs them with the
element; a coroutine frame is never packed. Everything is in
`src/ck-core/ckcoro.h` (header-only, C++20); the reasons and the
measurements are in `doc/coroutine-sdag-design.md`. The user-facing contract
is the header's opening comment.

## Data structures

- `ck::coro::state` (one per chare): `std::list<record> recs_` and
  `std::vector<buffer> inbox_`.
- `record`: the coroutine entry's index `ep`, a `CkCallback target` to the
  element, the entry's message `msg` (a copy, for message entries; null for
  void entries), the restart `chain`, the frame handles `root` (the entry's
  frame) and `h` (the innermost suspended frame), and the wait state:
  `waiting`, `atSite`, `wants` (entry and reference number), and the
  delivered `readyBuf`.
- `chain`: `sites` (one site id per resumable-function depth, pupped),
  `linked` (per depth: entered through a `CK_CALL` site), `depth`,
  `restarting`, and the `noted` fields that connect a `note_site` call with
  the `co_await` that follows it. This is CAMPI's site chain
  (`campi_core.h`, `campi::frame_guard`) with the rank record replaced by a
  record pointer.
- `buffer`: `ep`, `hasRef`, `ref`, and either a message (`isMsg`, `msg`,
  pupped with `CkPupMessage`) or marshalled arguments as `bytes`
  (`PUP::toMem` of a `std::tuple` of the decayed arguments).
- `thread_local record *current_`: the record of the coroutine running on
  this PE thread; set by `state::begin` and `state::push` around a run,
  read by `frame_guard`, `note_site` and the awaitables.
- `thread_local std::coroutine_handle<> pending_root_`: set by
  `coroutine::promise_type::get_return_object`, consumed by `state::begin`
  so that the record learns its frame handle.

## Protocols

**Start.** The generated entry wrapper calls the user's method, which is a
coroutine with `initial_suspend` = never: the body runs to `CK_COROUTINE`,
whose `root_guard` calls `state::begin(elt, ep, msg)`. `begin` adopts a
restarting record of the same entry if there is one, else creates a record
(aborting if an active record of that entry exists), stores the frame
handle and sets `current_`. The `frame_guard` that follows opens the switch
at `chain.sites[0]` when restarting, else at 0. When the body first
suspends or finishes, the wrapper's discarded `ck::coroutine` object is
destroyed and `~coroutine` calls `state::finish_run`, which clears
`current_`, destroys a finished frame and erases its record, or rebuilds
when `+ck_fakemigrate` asked for it.

**Delivery.** `state::deliver(ep, ...)` builds a `buffer` (reference
number: the envelope's for a message, the first arithmetic argument for
marshalled arguments) and calls `push`. `push` looks for a waiting record
whose `wants` match; if found it moves the buffer into the record, sets
`current_`, resumes `record.h` and calls `finish_run` when the body
suspends again or finishes; otherwise it appends the buffer to the inbox.

**Waiting.** `CK_WHEN` is `case __LINE__: note_site(...); co_await
state.when(ep, ref, out...)`. `when_awaiter::await_ready` clears
`chain.restarting` (a restarted body has reached its site), remembers
whether a site was noted (`record.atSite`), and takes the first matching
buffer from the inbox (`state::take`, keeping the order of the rest); if
there is none, `await_suspend` records the innermost handle and the `want`
in the record and sets `fakeRestart` when the flag is on. `await_resume`
unpacks the delivered buffer into the `out` lvalues: a message pointer is
handed over (the coroutine owns the message), marshalled arguments are read
back with `PUP::fromMem`.

**Pack.** `state::pup` writes the inbox and, per record, `ep`, `target`,
`msg` and `chain.sites`. Packing aborts if a record is running or is
suspended at a `co_await` that is not a site (`waiting && atSite` is the
condition).

**Unpack and restart.** `state::pup` reads the same, marks each record
`restarting` and calls `send_restart`: a copy of the record's message (or a
fresh system message for a void entry) is sent through `target`, so the
entry is invoked again by the array manager after the unpack completes.
The body runs to `CK_COROUTINE`, adopts the record, and each `frame_guard`
on the recorded path switches to its site; `note_site` checks that the site
reached is the recorded one. Deliveries that arrive before the restart
message are buffered as usual.

**Rebuild (`+ck_fakemigrate`).** After a body suspends at a site,
`finish_run` calls `state::rebuild`: the state is packed into memory,
every frame destroyed, the records and inbox cleared, and the state
unpacked from memory, which sends the restarts.

**Helpers.** A `ck::coro::task<T>` helper awaited with `CK_CALL` has its
own `CK_RESUMABLE`/`CK_END` and therefore its own depth in the chain;
`note_site(..., call = true)` marks the call site and `frame_guard` records
that the helper was entered through it (`chain.linked`), as in CAMPI.

## Limits and diagnostics

- One active coroutine per entry per element (`begin` aborts otherwise).
- Coroutine entries take no parameters or one message.
- An element packed while suspended at a non-site `co_await` aborts with
  the entry index.
- A restarted body that reaches a site other than the recorded one, or a
  helper not on the recorded path, aborts (`note_site`, `frame_guard`).
- Marshalled inbox arguments must be types with a `pup` (scalars, STL
  containers, pupable classes); raw arrays are not handled.
- `CKCORO_DEBUG=1` in the environment prints each rebuild.

## Where in the source

- `src/ck-core/ckcoro.h`: everything above.
- `src/ck-core/CMakeLists.txt`: installs the header.
- `tests/charm++/jacobi3d-coro/`: the first program (migration at a site,
  rebuild, checkpoint and restart).
- `benchmarks/charm++/coro_pingpong/`: plain, SDAG and coroutine ping-pong.
- `src/libs/ck-libs/ampi/campi_core.h`: the origin of `task<T>` and the
  site chain.
