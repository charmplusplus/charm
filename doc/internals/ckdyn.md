# CkDyn: dynamic chares for language bindings

A dynamic chare is a Charm++ chare, group, nodegroup, array element, array
map or main chare whose class is described at run time instead of in a `.ci`
file. Six fixed C++ classes (`MainchareDyn`, `ChareDyn`, `GroupDyn`,
`NodeGroupDyn`, `ArrayElemDyn`, `ArrayMapDyn`) stand in for every class of
their kind; each instance holds an opaque `handle` that the binding owns.
The users are language bindings (the planned charm4py binding, a Julia or C
program) and the C++ test binding in `tests/charm++/ckdyn`, which exercises
the layer without Python. The contract is: tables in (a class descriptor with
an entry table per class, `CkDynRegisterClass`), hooks out (one table of
function pointers, `CkDynHooks`, through which the runtime reaches the
binding's objects), one message format (`CkDynMsg`, a header plus typed slots)
for every entry method of every dynamic class.

The reasons for each choice, the alternatives and the measurements are in the
design document, `docs/design/DESIGN.md` on branch `thin` of the charm4py
repository (github.com/charmplusplus/charm4py), section 3. This chapter
describes what the code does today. Zero-copy host and device slots, posts and
HAPI callbacks are implemented and described below. The Python binding is
charm4py 2, branch `thin` of the charm4py repository; it is not described
here.

## Files

`ckdyn.h` is the binding-facing API and includes no Charm++ header.
`ckdyn-impl.h` declares the C++ classes, for ckdyn.C, ckarray.C and C++
users. CkDyn is compiled into libck on every target; nothing in it runs until
`CkDynInit` or `CkDynRegisterClass` is called.

## Registration

A binding describes a class with a `CkDynClassDesc`: name, kind, `ndims`
(arrays only: 0 to 6, where 0 means any dimension count and the index given
at each creation or insert decides), `num_entries`, an array of `CkDynEntryDesc`
(`name`, `flags`, `user_id`) whose element 0 is the constructor,
`binding_data` (opaque, passed to every per-class hook) and `hooks` (NULL
selects the table given to `CkDynInit`). `CkDynRegisterClass(desc,
&first_ep)` returns the chare index and the entry index of `entries[0]`.

`registerClass` (ckdyn.C) does, per kind:

| kind | `CkRegisterChare` size, type | base | constructor ep (message, flags) | migration ctor |
|---|---|---|---|---|
| MAINCHARE | `MainchareDyn`, `TypeMainChare` | `CkIndex_Chare` | `CkArgMsg`, flags 0; `CkRegisterMainChare` | yes |
| CHARE | `ChareDyn`, `TypeChare` | `CkIndex_Chare` | `CkDynMsg`, `ckFlagsOf`; `CkRegisterDefaultCtor` | yes (checkpoint restart only) |
| GROUP | `GroupDyn`, `TypeGroup` | `CkIndex_IrrGroup` | `CkDynMsg`, `ckFlagsOf` | yes |
| NODEGROUP | `NodeGroupDyn`, `TypeNodeGroup` | `CkIndex_NodeGroup` | `CkDynMsg`, `ckFlagsOf` | yes |
| ARRAY | `ArrayElemDyn`, `TypeArray`; `CkRegisterArrayDimensions(idx, ndims)` | `CkIndex_ArrayElement` | `CkDynMsg`, `ckFlagsOf` | yes |
| ARRAYMAP | `ArrayMapDyn`, `TypeGroup` | `CkIndex_IrrGroup` | `CkDynMsg`, `ckFlagsOf` | yes |

The constructor is registered first. Each user entry `entries[i]`, `i >= 1`,
is then registered with `CkRegisterEp(name, X::callEntry, CkDynMsg::__idx,
idx, ckFlagsOf(flags))`, and `registerClass` aborts unless the returned index
is `first_ep + i`: the binding computes entry indices by addition, so they must
be contiguous. The migration constructor (`X::callMigrate`, message index 0,
`CkRegisterMigCtor`) is registered after the user entries so that it does not
break the contiguity.

A main chare class also gets one **relay entry** per user entry, registered
after its migration constructor with `MainchareDyn::callRelay`, the same name
and flags, and contiguous from `ClassRec::relayFirst`; its `EpRec::relayTo`
names the main chare entry it stands for. A `TARGET` callback to a main chare
is sent to the relay entry on `CkDynMgr` of the main chare's PE (see
Callbacks and futures), because the main chare's `CkChareID` holds an object
pointer (`thishandle.objPtr` is the object itself for main chares, which are
not in `chare_objs`) and a process restarted from a checkpoint allocates the
main chare elsewhere.

`ckFlagsOf` maps the entry attributes to Charm++ entry flags, for the
constructor of every kind except the main chare and for every user entry:
always
`CK_EP_NOKEEP`, plus `CK_EP_INLINE`, `CK_EP_IMMEDIATE`, `CK_EP_TRACEDISABLE`
(NOTRACE) and `CK_EP_APPWORK` when set. EXPEDITED, EXCLUSIVE and THREADED are
not Charm++ entry flags: CkDyn keeps them in its own table and applies
EXPEDITED at send time and EXCLUSIVE at delivery; THREADED is only passed back
to the binding in `ep_flags`. REDNTARGET is documentation.

Two tables in ckdyn.C hold what registration produced. Both are
process-wide, written on rank 0 only:

- `g_classes` maps a chare index to a `ClassRec`: a copy of the descriptor
  (the class and entry names are `strdup`ed), the entry array, the hooks
  pointer, and the chare index, constructor ep, first ep and migration ep.
- `g_eps` is a vector indexed by Charm++ entry index. An `EpRec` holds the
  `ClassRec` of the entry, its CkDyn flags (with `CKDYN_EP_NOKEEP` added for
  every entry except the main chare constructor) and its `user_id`. `epRec(ep)` aborts for an index that is not a
  dynamic entry.

`registerClass` finally sets `_chareTable[idx]->isDynamic` (the bit added to
`ChareInfo`, src/ck-core/register.h:224). `CkDynIsDynamicChare(chareIdx)`
reads it; the array broadcast path uses it to select the per-PE delivery
described below.

**When.** `CkDynRegisterClass` aborts unless it runs on rank 0. It must run
on every process, in the same order, before `_registerDone`
(src/ck-core/init.C:1672). A shared-library binding registers from
`hooks.register_classes`, which `_initCharm` calls in place of
`CkRegisterMainModule` (see Start-up). A C++ program registers from an
`initnode` function: init calls run on rank 0 of each process before
`_registerDone` (`InitCallTable::enumerateInitCalls`, init.C).

**Late registration.** After `CkDynAllowLateRegistration()`,
`CkDynRegisterClassLate` registers a class after start-up, in every build. It
aborts unless the process has one PE, because `_entryTable` and `_chareTable`
are process-wide vectors that another PE of the same process could be reading
while dispatching. Like `CkDynRegisterClass` it must run on every process, in
the same order, before any message for the class is sent: the test's late
phase registers from a group broadcast and waits for a reduction.

`CkRegisterEpInternal` and `CkRegisterIfNotThere` (register.C) refuse a
registration after `_registerDone` unless `lateRegistrationAllowed()`: true in
a `CMK_CHARM4PY` build (the Ext API registers classes late without telling
the runtime), and otherwise only between `_registerLateBegin()` and
`_registerLateEnd()`, which `CkDynRegisterClassLate` calls around
`registerClass`. A plain program that registers an entry late by mistake
(a templated entry method not instantiated in a `.ci` file) still aborts with
"late entry method registration happened after init". `CkDynRegisterClass`
does not rely on that check: after `_registerDone` (`_registerIsDone()`) it
aborts and names `CkDynRegisterClassLate`, because in a `CMK_CHARM4PY` build
the check above passes and the one-PE-per-process condition of the late path
would be skipped. Registration also aborts on an entry index above 65535,
which the 16-bit `CkDynHdr::ep` cannot carry.

What sizes itself by the entry or chare count, and how a late class fares:

| consumer | sized when | late class |
|---|---|---|
| `g_eps`, `g_classes` (ckdyn.C) | grow with each registration | works |
| `_entryTable`, `_chareTable`, `_msgTable` (register.C) | vectors, grow | works (one PE per process) |
| trace-projections | `.sts` written at exit from the current tables; `traceSetMsgID` checks the current size | works |
| trace-summary (`SumLogPool::initMem`, `PhaseEntry`), trace-utilization (`initMem`), trace-counter | per-entry arrays allocated when the module starts | an executed late entry aborts with "Invalid entry point!!" (summary) or is out of range; only with `CMK_TRACE_ENABLED` and those trace modes |
| `_debugEntryTable` (debug-charm.C, `CMK_CHARMDEBUG` only) | `CpdFinishInitialization`, before `_registerDone` | a breakpoint on a late entry indexes past the table |
| readonlies (`_readonlyTable`) | broadcast once at start-up | a readonly registered late is never broadcast; CkDyn registers `ckdyn_ro` in `CkDynInit`, before start |
| `_mainTable` | main chares are created at start-up | a `MAINCHARE` class registered late is never constructed |
| checkpoint restart | the restarting run registers classes at start-up | a checkpoint holding objects of a late class restarts only if the restarted program registers the same classes, in the same order, at start-up |

The `ADAPT_SCHED_MEM` table of memory-critical entries (init.C) is built at
start-up too, but CkDyn never sets `CK_EP_MEMCRITICAL`.

An array class registered with `ndims == 0` accepts any dimension count: each
creation or insert gives its own, and `CkDynCreateArray` checks the count only
for a class registered with a fixed one. A language whose arrays pick their
dimension count at creation (charm4py) registers every array class that way.

Tracing reads the dimension count from `_chareTable[idx]->ndims`: the `.sts`
file's `CHARE` line reports it, and Projections reads that many index fields
from each `BEGIN_PROCESSING` record of the class
(trace-projections.C `LogEntry::pup`; Projections `GenericLogReader`). For a
class registered with 0, `noteDims` sets `_chareTable[idx]->ndims` to the
dimension count of the first index this process sees for the class
(`CkDynCreateArray`, `CkDynArrayInsert`, the element constructor, and an
element unpacked after migration or restart), so the records carry the
element's index exactly as for a charmxi array of that dimension count.
Limits: a class used for arrays of different dimension counts in one run
keeps the first count, so the index fields of the others are truncated or
padded (the record layout still matches the `.sts` file); and the `.sts`
file, written by PE 0's process, reports 0 (no index fields) for a class of
which that process never created, inserted or held an element.

## The hook table

`CkDynInit(hooks)` copies the table into `g_hooks` and checks
`hooks->version == CKDYN_API_VERSION` and `hooks->size == sizeof(CkDynHooks)`
(the size as compiled by the binding, so a binding built against a header
with a different table is rejected even at the same version). A class whose descriptor has
`hooks == NULL` uses `&g_hooks`; otherwise its own table, kept by pointer, so
that table must outlive the program, and `registerClass` applies the same
two checks to it. The per-class hooks receive the class's
`binding_data`; the hooks without a class receive `g_hooks.global` and are
always taken from `g_hooks`.

| hook | called from | when |
|---|---|---|
| `register_classes(global)` | `CkDynCallRegisterHook`, from `_initCharm` | start-up, rank 0 of each process, when `CkDynInit` ran before `_initCharm` |
| `construct(bd, target, ctor view)` | constructors of `ChareDyn`, `GroupDyn`, `NodeGroupDyn`, `ArrayElemDyn`, `ArrayMapDyn` | object creation; returns the handle |
| `mainchare(bd, &handle, target, argc, argv)` | `MainchareDyn(CkArgMsg*)` | main chare creation on PE 0 |
| `destruct(bd, handle)` | destructors of all six classes, when the handle is not NULL | object deletion, migration away, exit |
| `deliver(bd, handle, ep, user_id, ep_flags, view)` | `dispatch`; `CkDynDeliverBroadcast` when `deliver_many` is NULL | every entry method invocation |
| `deliver_many(bd, handles, n, ep, user_id, ep_flags, view)` | `CkDynDeliverBroadcast` | an array broadcast, once per PE; rules below |
| `pack_size`, `pack`, `unpack` | `pupObj` | migration, checkpoint, restart |
| `pack_readonly_size`, `pack_readonly`, `unpack_readonly` | `ckdynRoPup` | readonly broadcast at start-up, checkpoint, restart |
| `reduce(global, reducer_id, views, n, builder)` | `ckdynForeignReduce` | combining contributions to the foreign reducer |
| `future(global, pe, id, view)` | `CkDynMgr::deposit` | a value for a binding future arrives |
| `section_joined(bd, handle, sid)` | `sectionJoin` | a member received its section's creation multicast; optional (NULL: not called) |
| `map_proc_num(bd, handle, aid, ndims, index)` | `ArrayMapDyn::procNum` | the array manager asks where an index lives |
| `resume_from_sync(bd, handle)` | `ArrayElemDyn::ResumeFromSync` | after a load balancing step, only for elements created with `CKDYN_MSG_USESATSYNC` |
| `error(global, what)` | `bindingError` | before CkDyn aborts on the binding's behalf |

The handle is whatever `construct`, `mainchare` or `unpack` returned. CkDyn
stores it in `CkDynObj::handle` (with the chare index) and never dereferences
it. A handle must be non-NULL: NULL marks a group branch that was not
constructed (outside a constrained group). A target (`CkDynTarget`) names one object of any kind: the `CkChareID`
fields for chares, group id and PE or node for groups, array id with an `int`
index for arrays. CkDyn converts array indices to and from `CkArrayIndex`
(`mkIndex`, `fillIndex`), which stores four to six dimensions as shorts;
`mkIndex` aborts on a component outside 16 bits in that case. A `MAINCHARE`
target is resolved through `_mainTable[...]->getObj()` (`localChare`),
because main chares are not in `chare_objs`.

All hooks run on the PE that owns the object, inside the scheduler, except
`reduce`, which runs inside the reduction manager on whichever PE combines.

## Messages

### Two message types

`CkDyn.ci` declares `message CkDynMsg { char buf[]; }` and `message
CkDynMcastMsg { char buf[]; }`; `ckdyn-impl.h` defines the classes.
`CkDynMsg` has the layout of `CkMarshallMsg`, a `char *` at offset 0, because
the zero-copy receive code in ckrdma.C parses `CkNcpyBuffer` descriptors from
the start of the marshalled buffer. `CkDynMcastMsg` derives from
`CkMcastBaseMsg` first, because `CkMulticastMgr` writes the section cookie
into the first bytes of a multicast message. No single layout satisfies both,
so there are two message types. Every send, broadcast, constructor message and
future value is a `CkDynMsg`; `CkDynMcastMsg` carries section sends only
(see Sections). Two of the 256 message indices are used.

### Payload layout

`buf` holds, in order:

1. `CkDynHdr`, 24 bytes: `magic_version` (`CKDYN_MAGIC`, 0xDA01), `ep`
   (uint16), `flags` (uint16), `nslots` (uint16), `user_id`, `tail_len`,
   `data_len`, `nbuf` (the number of `NCPY` and `DEVICE` slots; a message
   with any starts with a descriptor prefix, see Zero-copy).
2. `nslots` `CkDynSlot` records of 24 bytes: `type`, `elem`, `ndim`,
   `bflags`, `tag` (binding-defined, opaque to CkDyn), `len`, and a union `v`
   holding an inline scalar or an offset into the data region.
3. The data region, `data_len` bytes, padded to 8 bytes. Each item starts at
   an 8-byte aligned offset (`appendData`).
4. The tail, `tail_len` opaque bytes the binding writes in place
   (`CkDynMsgReserveTail`); the charm4py binding will put its serialized
   arguments there.

Slot types:

| type | value | data region |
|---|---|---|
| `NONE` | none (the binding's null) | none |
| `BOOL`, `I64`, `U64`, `F64` | inline in `v` | none |
| `BYTES` | `v.off`, `len` | the bytes |
| `BUFFER` | `v.off`, `len`, `elem`, `ndim`, `bflags` | `ndim` int64 shape values, then `len` bytes of contents |
| `NCPY`, `DEVICE` | `v.off`, `len`, `elem`, `ndim`, `bflags`; `reserved2` the descriptor index; `DEVICE`: `reserved` the device ordinal | `ndim` int64 shape values (`DEVICE`: preceded by the stream); the data is not in the message (see Zero-copy) |
| `FUTURE` | inline `v.u64 = (pe << 32) \| id` | none |
| `CALLBACK` | `v.off` | a `CkDynCallback` copied as bytes |
| `SECTION` | `v.off` | a `CkDynSectionID` copied as bytes |
| `PROXY` | `v.off` | a `CkDynTarget` copied as bytes |
| `TAIL` | `len` = `tail_len` | none (marks that a tail exists) |

`CkDynMsgAddBuffer` accepts a buffer with strides only when they describe a
contiguous layout in C or Fortran order (`contiguous`), and aborts otherwise.

Header flags that CkDyn reads: `CKDYN_MSG_CREATION` (acknowledge creation to
the FUTURE slot, see below), `CKDYN_MSG_USESATSYNC` (array element constructor:
set `usesAtSync`), `CKDYN_MSG_CONSTRAINED` (group constructor: the last slot
is a `BYTES` slot of int32 PE numbers; bit 6), `CKDYN_MSG_BCAST` (set by the
broadcast sends and by `CkDynCallbackSend` with a broadcast TARGET callback,
read by `dispatch`). `BLOCK`, `BCASTRET` and `RAY` are
defined for the binding and passed through. Whether a message is a reduction
result or a system message is carried by the view's `kind`, not by a flag.

Bits 12-15 of the header flags (`CKDYN_MSG_BINDING_MASK`) belong to the
binding: CkDyn neither sets nor reads them, so a binding can mark its own
message kinds (the Python binding marks foreign reduction results) without a
header extension.

### The builder

`CkDynMsgBegin(nslots_hint, data_hint, tail_hint, prio_bits, group_dep)`
returns a `CkDynMsgBuilder`: a slot vector, the data region as a list of
pieces, a tail vector, plus flags, `user_id`, priority bits and group
dependency. A piece is either small content copied into the builder when it
is added (BYTES payloads, buffer shapes, callbacks, sections, proxies) or a
buffer recorded by pointer and length (`CkDynMsgAddBuffer`): its bytes are
copied once, straight into the message, by `Finish`, so the caller's buffer
must stay valid until `Finish` or `Abandon`. A 1 MB array is therefore
written once, not into the builder and again into the message. Each PE caches one
builder in the Ckpv `_ckdynBuilder`, allocated on first use and reused. If the
cached builder is in use, which happens when a build is nested inside another
(an inline delivery that sends, or `ckdynForeignReduce` calling `reduce` while
the binding was building), `CkDynMsgBegin` allocates a second builder, and
`releaseBuilder` deletes it at `CkDynMsgFinish` or `CkDynMsgAbandon`.
The builder `ckdynForeignReduce` hands to the `reduce` hook has `copyBuffers`
set: `CkDynMsgAddBuffer` then copies the contents into the builder when it is
called, because the hook's result is written only after the hook returns, and
a hook may build it in memory it frees on return.

`CkDynMsgFinish(b, ep)` allocates the message once with the charmxi-generated
`operator new` (`new (total, prio_bits) CkDynMsg`, or the `GroupDepNum{1}`
variant with `setGroupDep` when `group_dep >= 0`), which sets `buf` to the
variable-size area. Group dependencies are given at `Begin` because the
envelope reserves their space at allocation. `writePayload` writes the
header and slots, copies each piece of the data region to its offset
(zeroing the alignment gaps), then the tail. Priority bits are filled after
`Finish` through `CkDynMsgPriorityPtr`; queueing through `CkDynMsgSetQueueing`.
`CkDynMsgPayload(m, &len)` returns the payload bytes of a finished message
(header to end of tail), which is how a C binding makes a foreign-reducer
contribution.

Size checks: a slot longer than 32 bits aborts when it is added; `checkLimits`,
called by `Finish`, `FinishMcast` and `ckdynForeignReduce` for the reduce
hook's result, aborts on more than 65535 slots (`nslots` is 16 bits) or a
payload that with the envelope, the message class and the priority field
(`msgOverhead`) exceeds 2 GB (the allocator takes an `int`). A reduce result
with an NCPY or DEVICE slot aborts as well.

### Sends

`CkDynSendChare`, `CkDynSendGroup` (pe -1: broadcast), `CkDynSendGroupMulti`,
`CkDynSendNodeGroup` (node -1: broadcast) and `CkDynSendArray` (index NULL:
broadcast) call the usual `CkSendMsg*`/`CkBroadcastMsg*` functions.
`CkDynSendTarget` selects one by the target's kind. Each first stamps `ep` into
the header (`stampEp`); the group, nodegroup and array broadcasts also set
`CKDYN_MSG_BCAST` in the header. Each then computes the send flags (`sendFlags`): the
entry's EXPEDITED, INLINE and IMMEDIATE attributes plus the
`CkDynSendOpts` flags. INLINE selects the `*Inline` send variant. Array sends
set `ForArrayEltMsg` and `CkArray_IfNotThere_buffer` on the message.

### Views

The receiver reads a message through a `CkDynMsgView`: kind, header pointer,
slots, data, tail, `refnum` (the envelope's reference number), `section`
(the section id of a `CkDynMcastMsg` delivery, NULL otherwise) and `msg` (the
runtime message). `viewAny(view,
tmp, msg)` builds it and returns the entry index:

- It reads the message index from the envelope (`env->getMsgIdx()`).
- `CkDynMsg` and `CkDynMcastMsg`: the view points into `buf`
  (`viewPayload`, which aborts when the magic is wrong); the entry index is
  the header's `ep`.
- `CkReductionMsg`: kind `REDRESULT`, one synthesized `BUFFER` slot over
  `getData()` (no copy) whose `elem` comes from the reducer table, plus a
  `FUTURE` slot `(CkMyPe(), ref)` when the envelope's reference number is not
  zero. A result of the foreign reducer is itself a CkDyn payload and is
  viewed as kind `DYN` over `getData()`. The entry index is
  `CkMessageToEpIdx(msg)`.
- Any other message (a system message from a callback sent with no data):
  kind `SYS`, no slots except the same optional `FUTURE`; entry index from
  `CkMessageToEpIdx(msg)`.

The synthesized slots live in a two-slot array on the caller's stack (`tmp`),
so a view of a reduction or system message, and every pointer the binding
read from its slots, is valid only until the hook that received it returns
(documented at `CkDynMsgKind` in ckdyn.h). `CkDynMsgRetain` keeps the message,
and so the reduction data, alive, not the synthesized slots.

The header carries the entry index because the envelope's `epIdx` can be
stale: in a production build `CkSendMsgInline` to a local chare does not
prepare the envelope (design 3.6). For reduction and system messages the
envelope is the only source, and `CkMessageToEpIdx` returns the array
broadcast ep for broadcast envelopes.

### Ownership

Every dynamic entry except the main chare constructor is registered
`CK_EP_NOKEEP` (`ckFlagsOf`), so the runtime frees the message after the call returns
(`CkDeliverMessageFree`, src/ck-core/ck.C:688), through the dealloc function
of the entry's message type, which is `CkDynMsg`'s plain `CkFreeMsg`. That
also frees a `CkReductionMsg` or system message correctly when one reaches a
dynamic entry through a callback. The main chare constructor is registered
with flags 0 and deletes its `CkArgMsg` itself.

A binding that keeps a pointer into the message beyond the hook (a zero-copy
view of a `BUFFER` slot) calls `CkDynMsgRetain(view)`, which is
`CmiReference(UsrToEnv(msg))`, and later `CkDynMsgRelease(msg)` once per
retain, which is `CmiFree(UsrToEnv(msg))`. It is not `CkFreeMsg`, because
`CkFreeMsg` aborts in error-checking builds on the nokeep message that is
still in flight. The message is freed when the last reference goes.

## Delivery

### Point-to-point and group broadcasts

The entry table points every user entry of a class at `X::callEntry(msg,
obj)`. For all kinds except nodegroups that is `dispatch(obj->d, msg)`:
look up the `ClassRec` by `d.chareIdx`, build the view with `viewAny`, look up
the `EpRec`, and call `hooks->deliver(binding_data, d.handle, ep, user_id,
flags, &view)`.

A NULL handle means the object was never constructed, which happens on a
branch outside a constrained group. Broadcasts and callback results reach
every branch, so `dispatch` drops a message there when the envelope's message
type is not `CkDynMsg` (a reduction result, binding-defined ones included,
whose view is also of kind `DYN`; a system message; a section message) or its
header has `CKDYN_MSG_BCAST`, which `CkDynSendGroup`, `CkDynSendNodeGroup`,
`CkDynSendArray` and `CkDynCallbackSend` with a broadcast TARGET callback set. A point-to-point `CkDynMsg` to such a branch is an error
and goes to `bindingError`.

`NodeGroupDyn::callEntry` first reads the entry index (`epOf`, the same rule
as `viewAny` without building a view). For an EXCLUSIVE entry it uses the
protocol charmxi generates for `[exclusive]`: `CmiTryLock(__nodelock)`; if
another PE holds the lock, copy the message (`CkCopyMsg`), send the copy to
this node's branch with `CkSendMsgNodeBranch`, and return (the original is
freed as nokeep); otherwise dispatch and `CmiUnlock`. Other entries dispatch
directly; a nodegroup entry may run on any PE of the process.

### Constructors

The constructors read the constructor message as a `DYN` view and call
`construct` with a target that names the new object.

- `MainchareDyn` takes its chare index from `_currentChareType`, calls
  `CkDynCreateManagers` (which creates the `CkDynMgr` group and stores its id
  in the readonly `_ckdynMgrGID`), then `hooks.mainchare` with argc and argv.
- Every other constructor takes its class from the constructor entry that
  `CkDynCreate*` stamped in the header (`epRec(hdr->ep).cls`).
  `_currentChareType` is set only for main chares, and
  `IrrGroup::ckGetChareType` reads the per-PE group table, which does not
  hold nodegroups.
- `ArrayElemDyn` sets `usesAtSync` from `CKDYN_MSG_USESATSYNC`.

`CkDynCreateChare`, `CkDynCreateGroup` (also for array maps),
`CkDynCreateNodeGroup` and `CkDynCreateArray` check the class's kind, stamp
the constructor ep into the header and call `CkCreateChare`,
`CkCreateGroup`, `CkCreateNodeGroup` or `CProxy_ArrayBase::ckCreateArray`.
`CkDynCreateArray` sets `ArrayEltInitMsg` and builds `CkArrayOptions` from
`CkDynArrayOpts` (dimensions, map, `bindTo`, anytime migration, static
insertion); it aborts when the dimension count differs from the class's.
`CkDynCreateChare` returns in `*out` the id `CkCreateChare` fills in, a
virtual chare id (negative PE) that is usable for sends and callbacks at
once; `construct` receives the real id. `CkDynArrayInsert` takes the
constructor ep from the message header, so the binding must have finished the
message with the class's `first_ep`; it aborts unless that ep is the
constructor of an array class, or when that class differs from the array's.
The array's class is known to a process that created the array or holds one
of its elements (`g_arrayClass`, filled by `CkDynCreateArray`, the element
constructor and unpack); elsewhere the insert is not checked. If the array
has no local branch yet, `ckInsertIdx` buffers the insert.

**Creation acknowledgement.** A constructor message with
`CKDYN_MSG_CREATION` and a `FUTURE` slot (the first one, `findFuture`)
acknowledges creation without the binding's help. A `ChareDyn` sends an
empty `CkDynMsg` to the future with `CkDynFutureSet`. A `GroupDyn`,
`NodeGroupDyn` or `ArrayElemDyn` contributes an empty reduction to a FUTURE
callback, so the future is set once every branch or element exists. The
flag belongs on whole-array creation only: on a `CkDynArrayInsert` it would
start a reduction over one element.

**Constrained groups.** When the constructor message carries
`CKDYN_MSG_CONSTRAINED`, `GroupDyn` calls `construct` only on PEs listed in
the last slot (a `BYTES` slot of int32 PE numbers). The other branches exist,
with `handle == NULL`, still contribute to the creation reduction, drop
broadcasts and callback results (`dispatch`, above), and skip the binding's
hooks when pupped (`pupObj`, below).

### Array broadcasts

An array broadcast arrives at `CkArray::recvBroadcast` once per PE and is
stored by epoch as for every array (chapter [Array broadcasts and migrating
elements](array-broadcast-migration.md)). When the first local element is
dynamic (`CkDynIsDynamicChare(localElemVec[0]->ckGetChareType())`,
src/ck-core/ckarray.C:1862; the branch requires at least one local
element), `recvBroadcast` calls
`CkArrayBroadcaster::deliverAndUpdateDyn` instead of the per-element loop.

`performDeliveryDyn` (ckarray.C:1603) walks the local elements and selects
those whose expected epoch (`getData(el)`) equals the broadcast's, advancing
each selected element's epoch by one. It skips the rest, which already have
the broadcast or have not reached that epoch, and skips an element whose
zero-copy gets after a migration are still pending
(`bufferedActiveRgetMsgs`, the same guard as `performDelivery`); the stored
broadcast reaches it on arrival. It then calls
`CkDynDeliverBroadcast(bcast, ready, n)` once for all of them.
`deliverAndUpdateDyn` (ckarray.C:1586) repeats this for each stored
broadcast that follows in epoch order, as long as the previous one reached
someone. Storing, the epoch rules and `releaseDelivered` are the code the
normal path uses.

`CkDynDeliverBroadcast` views the message once and calls
`hooks.deliver_many` with the handles of all selected elements. A handler
that destroys another local element makes that element's handle invalid for
the rest of the call (documented at the hook in ckdyn.h). Without a
`deliver_many` hook, `CkDynDeliverBroadcast` records the elements' indices
first, then before each call looks the element up again with
`CkArray::lookup`, skips it if it is gone, and calls `deliver` bracketed by
`CkCallstackPush`/`CkCallstackPop` of the element. The message is not copied
per element and not freed here; the broadcast store owns it.

This path does not go through `CkLocRec::invokeEntry`, so CkDyn does the
tracing itself. `CkArray::recvBroadcast` is registered not traced
(`CkDisableTracing`, ckarray.C `_ckArrayInit`), so no execute event is open
when `CkDynDeliverBroadcast` runs. Without `deliver_many`, when tracing is on
(`CpvAccess(traceOn)`) and the entry is traced, each element's call is
bracketed as `invokeEntry` brackets a charmxi element: `_TRACE_SET_OBJECT`
with the location manager and the element's id, then
`_TRACE_BEGIN_EXECUTE_DETAILED` with the broadcast's event, source PE and
length and the element's projection id (its index), then
`_TRACE_END_EXECUTE` and `_TRACE_CLEAR_OBJECT`; Projections shows one event
per element. With `deliver_many`, CkDyn makes one call for all elements and
neither traces nor pushes any element: time is charged to the array manager
and no per-element event is recorded unless the binding brackets each
element's method with `CkDynCoroResume(target, ep, 0)`/`CkDynCoroSuspend`
(below), or with `CkDynCallstackPush`/`Pop` for the load alone.
All local elements are assumed to belong to the class of the first one.

## Callbacks and futures

A binding describes a callback with the POD `CkDynCallback`;
`CkDynMakeCallback` converts it to a `CkCallback` wherever CkDyn needs one
(contribute, QD, checkpoint, creation acknowledgement, `CkDynCallbackSend`):

| kind | `CkCallback` |
|---|---|
| `TARGET`, element | `sendChare`, `sendGroup`, `sendNodeGroup` or `sendArray` to `cb->ep` |
| `TARGET`, `bcast != 0` | `bcastGroup`, `bcastNodeGroup` or `bcastArray` |
| `FUTURE` | `sendGroup` to `CkDynMgr[cb->pe].deposit`, reference number `cb->id` |
| `IGNORE`, `EXIT` | `CkCallback::ignore`, `CkCallback::ckExit` |
| `SECTION` | `sendGroup` to `CkDynMgr[CkMyPe()].relay`, reference number = a relay slot (see Sections); `CkDynContribute` uses `CkDynMgr[sid.root_pe].relay` with no reference number and the section-tagged reducer; `CkDynCallbackSend` sends to the section directly |

A `TARGET` callback to a main chare is the exception to the first row: it
becomes `sendGroup` to the class's relay entry for `cb->ep` (`relayFirst +
(cb->ep - firstEp) - 1`) on `CkDynMgr[t.u.chare.pe]`. `callRelay` runs on
that branch, finds the main chare through `_mainTable[...]->getObj()`, pushes
it on the running-object stack and calls `dispatch`, which delivers with
`relayTo` as the entry index when the message is not a `CkDynMsg` (whose
header already holds the main chare's entry, stamped by
`CkDynCallbackSend`). Such a callback holds the chare index and PE only, so
it survives a checkpoint.

**Checkpoint.** `CkDynCheckpoint(dir, done)` calls `CkStartCheckpoint` with
`CkDynMakeCallback(done)`. Charm++ calls the callback when the checkpoint is
written and, in a program started with `+restart dir`, once restart completes
(`CkRestartMain`, ckcheckpoint.C). The callback may target an entry of a
group, an array element or a main chare; the main chare is restored by
`CkPupMainChareData` through `MainchareDyn`'s migration constructor and
`pupObj`, so the binding's `unpack` hook receives it. Futures from the
checkpointed run are invalid after restart: no coroutine waits on them in the
new process, so a `FUTURE` callback is not a resume target. A binding that
needs the original run to continue after the checkpoint tells the two runs
apart itself (the test's `DMain` checks whether its constructor ran in this
process).

A `TARGET` callback with a non-zero `refnum` sets it on the `CkCallback`.
`CkDynCallbackSend` stamps the ep into the header for a `TARGET` callback
before sending.

**The reference number is the future id.** A `CkCallback` with a reference
number writes it into the envelope's `ref` field of whatever message it
sends. On the receiving side `viewAny` turns a non-zero `ref` on a reduction
or system message into a `FUTURE` slot `(CkMyPe(), ref)`, and every view
exposes it as `refnum`. The field is `CMK_REFNUM_TYPE`. Future ids are
32-bit, 1 to 2^32-2, in a libcharm built with `REFNUM_TYPE="unsigned int"`
(CMakeLists.txt:184), which the design requires for bindings; the default
build keeps `unsigned short`, where `checkedRefnum` aborts on an id above
65534 ("future id %u is not usable as this build's %d-byte reference
number"). `CkDynRefnumBytes()` reports the
width so that the binding can check it.
Future id 0 cannot travel this way, because 0 means "no reference number",
and neither can the all-ones value: a reduction result carries the reduction's
user flag as its reference number, -1 by default (ckreduction.C sets it before
the callback's own reference number, if any, replaces it), so `refOf` reads
all-ones as none and `checkedRefnum` rejects both values. `CkDynMsgFinish`
clears the reference number of a new message, which allocation leaves
uninitialized.

**`CkDynMgr`** is a group with one branch per PE, declared in `CkDyn.ci`,
created by `CkDynCreateManagers`. `CkDynFutureSet(pe, id, msg)` sets the
message's reference number to `id` and sends it to `CkDynMgr[pe].deposit`.
`deposit` views the message (any kind: a `CkDynMsg` from `CkDynFutureSet`, a
reduction result or a system message from a FUTURE callback), reads `id`
from the envelope, calls `g_hooks.future(global, CkMyPe(), id, &view)`, and
frees the message (`deposit` is not nokeep). `deposit` is declared
`[notrace]`: a binding resumes its coroutines from the `future` hook, and
each resumed run records its own execute event (next section), which would
otherwise nest inside one for `deposit`. `CkDynCreateManagers` creates the
group once; later calls (a program with several main chares) do nothing. It aborts if the binding has no
`future` hook. The `relay` entry serves section callbacks, and the
`section*` entries serve the section table (see Sections). The group is
`[migratable]` and has a `pup`, which holds no CkDyn state (the section
table is not checkpointed), so a checkpoint restores it.

### Coroutines: load and trace attribution

A binding coroutine (a greenlet) that suspends on a future is resumed later
from the `future` hook, inside `CkDynMgr.deposit`. Without more work its time
there is charged to `CkDynMgr`, a group the load balancer cannot move, and
tracing shows nothing for it. A binding therefore calls
`CkDynCoroResume(target, ep, coro_id)` right after switching into a
coroutine and `CkDynCoroSuspend(target)` right before switching out of it
(when it waits again or finishes). This is what Charm++ does for a
`[threaded]` entry's thread with two listeners: `CkChareThreadListener`
(ck.C) pushes and pops the chare on the running-object stack, and
`TraceThreadListener` (trace-common.C) sets the traced object and begins an
execute event on resume, and ends them on suspend.

`CkDynCoroResume` finds the local object (`localChare`; it aborts if the
target is not local) and calls `CkCallstackPush`, which stops the timing of
the object below and starts the target's (load balancer instrumentation).
When tracing is on and `ep` is traced, it also begins an execute event for
`ep`: for an array element it sets the traced object (location manager group
and element id) and uses the element's projection id (its index), so
Projections attributes the event to the element; for a group, nodegroup,
chare or main chare the object id is made from `coro_id` as `CthSetThreadID`
makes one for a thread (`{CkMyPe(), low 32 bits, high 32 bits, 0}`). A per-PE
stack (`_ckdynCoroTraced`) records whether an event was begun, so that
`CkDynCoroSuspend` ends only events Resume began. `CkDynCoroSuspend` ends the
event, clears the traced object and calls `CkCallstackPop`. With tracing
compiled out both are the running-object push and pop only. The pair must be
called outside any traced entry, as from the `future` hook: `deposit` is not
traced, so the events do not nest. The test phase `coroutine`
(tests/charm++/ckdyn) checks the running object inside and after the
bracket for an array element and for a group branch.

## Reductions

`CkDynReducers(&table)` exports a static table of the built-in reducers:
name, `CkReduction::reducerType` value, and the element kind of the data
(`sum_int` is I32, `max_double` is F64, and so on; `concat`, `set`,
`statistics` and `tuple` are BYTE and left to the binding). The same table
gives `elem` of the `BUFFER` slot in a reduction result view.

`CkDynContribute(me, data, cb)` finds the local object the target names
(`localChare`) and calls `contribute(len, data, type, callback)` on its
`Group`, `NodeGroup` or `ArrayElement`. A `nop` contribution is sent with no
data. A SECTION callback is the exception: the contribution goes through the
section-tagged reducer (Sections, Reductions).

**The foreign reducer.** `_ckdynInitNode` registers one reducer,
`CkDyn::foreign` (`CkReduction::addReducer`), on every process in module
order; `CkDynForeignReducerType()` returns its type. A contribution to it is
a CkDyn payload (header, slots, data, tail) whose header `user_id` is the
binding's reducer id; a C binding builds it as a message and passes
`CkDynMsgPayload` as the data. `ckdynForeignReduce` views every contribution, takes
the reducer id from the first, begins a builder, calls
`g_hooks.reduce(global, reducer_id, views, n, builder)`, writes the
builder's payload with the same `user_id` and returns it as a new
`CkReductionMsg` of the foreign type. The result is again a CkDyn payload,
so the next level of the reduction tree and the final view treat it the
same way.

## Sections

A section is a subset of one array, of one group, or of several arrays (a
combined section), with multicast and section reductions through
`CkMulticastMgr`. `CkDynCreateManagers` creates the one `CkMulticastMgr`
CkDyn uses (`_ckdynMcastGID`, a readonly next to `_ckdynMgrGID`), in the
main chare constructor, so it exists on every PE before any section call.

### The table and the id

A section lives in the `CkDynMgr` branch of the PE that created it, its root.
`CkDynMgr::sections` holds one `CkDynSection` per slot: the kind
(`CKDYN_ARRAY`, `CKDYN_GROUP`, 0 for combined), a generation, the member
lists (`aids` with one index list per array, or `gid` and `pes`), the
delegated `CProxySection_ArrayBase` or `CProxySection_Group`, the `done`
callback, and the state of a combine in progress. Everything else names the
section by `CkDynSectionID` `(root_pe, idx, coll_id, kind, gen)`, 16 bytes,
which messages carry as bytes: a `CkSectionID` holds its member list, and its
cookie points at an `mCastEntry` valid only on the root, so it cannot travel.
`coll_id` is the array or group id, -1 for a combined section (whose arrays a
combine with remote parts learns only later). Generation 0 is never valid.

### Creation

`CkDynSectionCreateArray(aid, n, ndims, indices, bfactor, done, &sid)` and
`CkDynSectionCreateGroup(gid, npes, pes, bfactor, done, &sid)` run on the
calling PE, which becomes the root. They remove duplicate members, take a
slot (`newSection`; a deleted slot is reused with its bumped generation),
fill `*sid`, and call `finishSection`:

1. Build the section proxy and delegate it to the `CkMulticastMgr`
   (`ckSectionDelegate`, which builds the spanning tree).
2. Multicast one `CkDynMcastMsg` with a `SECTION` slot (the id) and a
   `CALLBACK` slot (`done`) to the entry `CkDyn.sectionInit(array)` or
   `CkDyn.sectionInit(group)`.
3. Send the messages that `CkDynSendSection` held while the section was not
   ready, and answer member queries from combines (below).

The two init entries are registered once by `_ckdynInitNode`, on the base
chare types `ArrayElement` and `IrrGroup`, not per class: CkMulticast sends
one entry index to every sub-section of a combined section, whose arrays may
belong to different classes. `sectionJoin` runs on each member: it stores the
cookie from the message (`CkGetSectionInfo`) in the member's cookie map
(`CkDynObj::sectionCookies`, keyed by the section id), calls
`hooks.section_joined(bd, handle, &sid)` if set, and contributes an empty
section reduction (`nop`) whose callback is `done`. The result, a REDRESULT
view with an empty `BUFFER` slot, is the "section is ready" signal: every
member has its cookie. A group section must name a dynamic group; creation
aborts otherwise.

### Sends

`CkDynMsgFinishMcast(builder, ep)` finishes a payload as a `CkDynMcastMsg`
(the same payload as `CkDynMsgFinish`, `CkMcastBaseMsg` in front; no group
dependency). `CkDynSendSection(sid, ep, msg, opts)` writes `ep` into the
header and the id into the message's `sid` field. On the root it calls
`CkDynMgr::sendTo`: `CProxySection_ArrayBase::ckSend`, which goes to
`CkMulticastMgr::ArraySectionSend`, or `CkMulticastMgr::GroupSectionSend`.
On another PE it sends the message to `CkDynMgr[root_pe].sectionSend`, which
does the same on the root (one extra hop, design 3.9 F17). Send options are
ignored: CkMulticast delivers to members with its own flags. A send to a
section that is not ready (a combine waiting for remote parts) is held and
sent after the creation multicast.

On delivery `dispatch` sees a `CkDynMcastMsg`, refreshes the member's stored
cookie with `CkGetSectionInfo` (so that it follows tree rebuilds) and sets
`view.section` to the message's id. The view kind is `CKDYN_MSGKIND_MCAST`.
If the entry belongs to another class than the element (one entry multicast
to a combined section over arrays of different classes), `dispatch` reports a
binding error and aborts.

### Reductions

`CkDynContributeSection(me, sid, data, cb)` finds the member's cookie (it
aborts if the object is not local, not a member, or has not received the
creation multicast) and calls `CkMulticastMgr::contribute(len, data, type,
cookie, callback, userData)`. `userData` becomes the result's reference
number at the root; CkDyn passes the callback's `refnum`, or the future id of
a FUTURE callback, never the default -1 (design 3.7). The foreign reducer
works as in whole-collection reductions: CkMulticast calls the registered
reducer function. `CkDynSectionRedNo(me, sid)` returns the cookie's `redNo`,
the number of section reductions this member has contributed to, which every
member computes identically and which the binding uses as a collective future
id.

**Section callbacks of `CkDynContribute`.** A whole-collection reduction
(group, nodegroup, array) compares the callbacks of the partial results it
combines (`CkReductionMgr::reduceMessages` aborts with "mis-matched client
callbacks" under `CMK_ERROR_CHECKING`), so every contributing PE must pass a
byte-equal `CkCallback`. A per-PE relay slot cannot be one. `CkDynContribute`
therefore contributes with a second reducer that `_ckdynInitNode` registers,
`CkDyn::sectionTagged`, and the callback `CkDynMgr[sid.root_pe].relay` with no
reference number, the same on every PE. The data is a 32-byte `SecTag` (the
section id, the entry, the callback's `refnum`, and the reducer the binding
gave, the inner reducer) followed by the binding's data. The reducer function
`ckdynSectionTaggedReduce` checks that every message names the same section,
entry and refnum, builds temporary `CkReductionMsg`s of the inner data, calls
the inner reducer's function from the reducer table (built-in reducers, the
foreign reducer and any other registered one alike) and returns the result
with the same tag. `nop` contributions are skipped, as Charm++ skips them.
`set` and `tuple` tell a contribution from a partial result by `sourceFlag`,
and `reduceMessages` passes a lone message on without calling the reducer,
which would turn a raw contribution into a partial result unconverted; their
contributions are therefore converted when contributed (`preReduce`, the
inner reducer over the one contribution) and all inner messages are marked as
partial results. `CkDynReduceAccess` is the friend of `CkReduction` and
`CkReductionMsg` that reaches the reducer table and the message counts. On the
root PE `relay` recognizes the tagged reducer's result (`unwrapSectionTagged`),
builds the section message from the inner data (`mcastFromReduction`, as
`toMcast` does), sets the tag's refnum and sends it with `CkDynSendSection`.

**Other section callbacks** (`CKDYN_CB_SECTION` given to
`CkDynContributeSection`, `CkDynStartQD`, `CkDynCheckpoint` and the other calls
that take a callback) are converted, on the PE that converts them, to a group
callback to `CkDynMgr[CkMyPe()].relay` whose reference number is a slot in
that branch's relay table (`relaySlot`, slots deduplicated by `(sid, ep)`).
CkMulticast does not compare the callbacks of section contributions, so
members on different PEs may carry different slots that hold the same
`(sid, ep)`. The callback's reference number replaces the envelope's on
delivery, so the callback's own `refnum` cannot travel there; `relay`
recovers it per message kind:

- a section reduction result (`CkDynContributeSection`) carries it as the
  `CkReductionMsg` user flag: `sectionContribute` passes it as CkMulticast's
  `userData`; the all-ones flag means none. CkMulticast keeps the user flag
  through cross-array results.
- a system message (`CkDynStartQD`, `CkDynCheckpoint`) carries none, so these
  two convert with `CkDynMakeCallback(cb, true)` and get a slot of their own
  keyed by `(sid, ep, refnum)` that holds it (`CkDynRelaySlot::sys`,
  `sysRefnum`).
- a `CkDynMsg` given to `CkDynCallbackSend` does not go through the relay:
  `CkDynCallbackSend` converts it and sends it to the section itself, with
  `cb->refnum` as the reference number.

Any other message type at a reduction slot aborts. `relay` converts the
message into a `CkDynMcastMsg` (`toMcast`: a foreign result keeps its
payload; a built-in reduction result becomes one `BUFFER` slot whose `elem`
comes from the reducer table; a system message becomes an empty payload),
sets the recovered reference number, and sends it with `CkDynSendSection`.
Members see an MCAST view, with `refnum` set and no synthesized FUTURE slot.
Such a callback is pup-able, unlike Charm++'s `bcastSection` callback.

The relay table is never trimmed and does not need to be: a PE holds one
reduction slot per `(section id, entry)` pair used in a section callback
converted there (not counting `CkDynContribute`, which uses no slot), so a program that calls collectives with a fresh future id
each time reuses the same slot. The section id includes the generation, so
deleting and re-creating sections adds slots for the new ids; QD and
checkpoint with a section callback add one slot per distinct reference
number. With a 16-bit `CMK_REFNUM_TYPE` a PE has at most 65534 slots.

### Combine

`CkDynSectionCombine(nparts, parts, done, &sid)` makes a new section on the
calling PE whose members are the union of the parts, each member once. Parts
rooted on the calling PE are read from the table; for each other part, a
`sectionQuery` goes to its root, which answers with `sectionParts` (its
arrays and member indices; a part that is itself a combine still waiting for
its own parts answers when it is ready). When all parts are in, `addPart`
checks the rules and forms the union: every part must be an array section
or a combined section ("part %d is a group section; only array sections can
be combined"), and all parts must have one index dimension count ("a combined
section needs arrays of one index dimension count"). The union has one
sub-section per distinct array, in the order the parts first name them, so
an element in two parts is a member once. The section is then finished like a
created one: a `CProxySection_ArrayBase` over several `CkSectionID`s, so the
combined section has its own cookie on every member and its own reduction
numbers. CkMulticast reduces each sub-section on its own tree and combines the
sub-section results in `XArraySectionReducer` on the root; that class and
`CkMulticastMgr::reduceFragment` keep the reducer type, the user flag and the
callback of the result, which CkDyn needs (the reducer type for the `BUFFER`
slot's `elem`, the user flag for the reference number, the callback for a
foreign reducer whose result is a new message).

### Delete and generations

`CkDynSectionDelete(sid)` runs on the root (from another PE it sends
`sectionDelete` there). It frees the held messages, clears the slot and bumps
its generation. Any later use of the old id on the root (`lookup`) aborts:
"send to a deleted section (root PE %d, index %d, stale generation %u; the
slot is at generation %u, ...)". The CkMulticast trees of a deleted section
stay allocated (CkMulticastMgr has no call that frees a section), and members
keep their cookie map entries; neither is used again.

### Migration and checkpoint

`pupObj` pups the cookie map before the binding's bytes, so a member that
migrates keeps its cookies and reduction numbers. Its cookie still points at
the tree node on its old PE: its next contribution goes there and is marked
`rebuilt`, the root sets `needRebuild`, the next multicast goes to every member
point to point (`SimpleSend`) and the one after rebuilds the tree
(ckmulticast.C `sendToSection`). The root is a PE, not an element, so the
root-migration paths of CkMulticast are not used. A checkpoint writes no
cookies and the section table is not checkpointed: sections do not survive
`+restart` and a binding recreates them.

The phases `sections/*` of tests/charm++/ckdyn exercise each part: array and
group sections with both reducer kinds and every callback kind, a section
created on another PE and sent to from PE 0, combines with overlapping
members and over two arrays, delete with slot reuse (and, with
`-stalesend`, the stale-generation abort), and section reductions across
RotateLB migrations.

## Zero-copy

`NCPY` slots carry host buffers and `DEVICE` slots device buffers that the
runtime transfers without copying them into the message: the message carries
a descriptor per slot, and the receiver's runtime fetches the data. Both use
the layout and the transfer code of the generated marshalling code for
`nocopy` and `nocopydevice` parameters, so ck.C and ckrdma.C are unchanged.

### Layout and marking

A message with `nbuf > 0` has a descriptor prefix at the start of `buf`,
before the CkDyn header:

- host: `int` count, `int` root node (`CkMyNode()` of the sender, unused for
  point-to-point), then one pupped `CkNcpyBuffer` per `NCPY` slot. This is
  the layout xi-Parameter.C writes for `nocopy` parameters
  (`impl_num_rdma_fields`, `impl_num_root_node`, `pupRdma`).
- device: `int` count, then one pupped `CkDeviceBuffer` per `DEVICE` slot, as
  for `nocopydevice` parameters (`impl_num_device_rdma_fields`). The first
  descriptor's acknowledgement block (`CkDeviceBuffer::acks`) grows at send
  time, so the prefix reserves room for `CK_DEVICE_ACK_CAP` ids on it.

The prefix is padded with zeros to a 16-byte boundary and the CkDyn payload
(header, slots, data region, tail) follows. `CkDynMsgFinish` marks the
envelope `CMK_ZC_P2P_SEND_MSG` (host) or `CMK_ZC_DEVICE_MSG` (device).

`dynPayload` finds the header: when the first two bytes of `buf` are
`CKDYN_MAGIC` there is no prefix; otherwise the prefix is walked with
`PUP::fromMem` and its length is the walked length rounded up to 16 (for a
device prefix, computed as if the acknowledgement block were full). The
descriptor count is limited to 4096 so that the first `int` never reads as
the magic. The prefix format follows from the envelope's zero-copy type: a
host message is `P2P_SEND` before its transfer and `SEND_DONE` after; a
device message is `DEVICE` before and `REG_NO_ZC` after.

An `NCPY` or `DEVICE` slot has `len`, `elem`, `ndim`, `bflags` and `tag` as
a `BUFFER` slot; `v.off` points at its shape in the data region (for a
`DEVICE` slot the 8 bytes before the shape hold the stream, which a
`CkDeviceBuffer` does not pup), `reserved2` is the descriptor index, and a
`DEVICE` slot's `reserved` is the device ordinal: the sender's in an unsent
message, the receiver's after delivery. `CkDynViewBufferPtr` reads the data
pointer from the descriptor at view time.

### Sending

`CkDynMsgAddNcpy` and `CkDynMsgAddDevice` record the buffer in the builder;
`CkDynMsgFinish` builds the descriptors: `CkNcpyBuffer(ptr, len, cb,
CK_BUFFER_UNREG)` (no registration outlives the message; reconverse registers
an unregistered source on demand for a true get) or `CkDeviceBuffer(ptr, len,
cb, stream)`. The callback `cb` is a `callCFn` `CkCallback` to `zcDoneFn` on
the sending PE with a `ZcDone {done, arg}` record, so the runtime's source
acknowledgement runs `done(arg)` there, once:

- host, same process (`MEMCPY` mode in `CkRdmaIssueRgets`): from
  `performEmApiMemcpy` inside the receiver's `_processHandler`, or through
  `_ckcallbackgroup` when the receiver is another PE;
- host, other process: `handleEntryMethodApiCompletion` when the get
  completes, delivered to the source PE as a message;
- device: `CkRdmaDeviceRecvHandler` (stream completion) or the sender's own
  acknowledgement table (`deviceResolveAck`, RDMA mode).

A builder abandoned with zero-copy slots, and `CkDynMsgFree` of a finished
message never sent, send each `done` through `_ckcallbackgroup[CkMyPe()]`,
so it runs from the scheduler and never inside a CkDyn call.

Every point-to-point send (`CkDynSendChare`, `Group`, `NodeGroup`, `Array`,
`CkDynFutureSet`, `CkDynCallbackSend`) passes a message that still has an
unissued transfer through `zcBeforeSend`: it clears `CK_MSG_INLINE` (an
inline send would bypass `_processHandler`, where the host transfer
happens), and for a device message runs `CkRdmaDeviceOnSender(destPe, ...)`
with the destination PE (array: `lastKnown`; nodegroup: the node's first PE;
callback: its target) and re-pups the descriptors in place.
`CkRdmaDeviceOnSender` synchronizes the stream (same process), stages into
the IPC communication buffer (same host), or registers the buffer and mints
acknowledgement ids (other host). Broadcasts, sends to several branches,
section messages (`CkDynMsgFinishMcast`), constructor messages, reduction
payloads (`CkDynMsgPayload`), and broadcast or section callbacks abort,
naming the slot. A callback of kind `IGNORE` or `EXIT` frees the message
(running its `done` functions) and sends an empty one.

### Receiving: host

`_processHandler` sees `CMK_ZC_P2P_SEND_MSG` and calls `CkRdmaIssueRgets`
before the entry runs: it allocates a new message with room for every
buffer after the original bytes, copies (same process) or gets (other
process) each buffer into it, rewrites the descriptors' pointers to those
places, marks it `SEND_DONE`, and enqueues it. The view's pointer is
therefore inside the delivered message and lives as long as a retain, as
for a `BUFFER` slot. If the message is packed again (an array element that
moved), `CkPackRdmaPtrs` stores those pointers as offsets. `dispatch` aborts
if an `NCPY` message arrives still marked `P2P_SEND`.

### Receiving: device

There is no device step in `_processHandler`; the generated code issues the
transfers from the entry method's first delivery, and CkDyn does the same in
`zcArrive`, called by `dispatch` and `CkDynMgr::deposit` before the hook.
For a `DEVICE`-marked message it chooses a destination per slot (a posted
buffer, or `CkDeviceMalloc` from the registered device pool, recorded in a
process-wide set), writes the receiver's device ordinal into each slot, and
calls `CkRdmaDeviceIssueRgets`, which copies the message, points the copy's
descriptors at the destinations, issues the transfers, and enqueues the copy
when all complete (with the quiescence accounting of that path). The first
delivery returns without calling the hook; the runtime frees the original.
The copy's delivery calls the hook. Afterwards `zcAfterDelivery` frees the
pool memory at once if the binding kept no retain (the reference count is 1),
or records it under the message; `CkDynMsgRelease` frees it when the count
reaches 1, before the last `CmiFree`. HAPI's `hapiPoolMalloc` is pinned host
memory, not device memory, so the device pool is `CkDeviceMalloc`.

### Posts

`CkDynPostBuffer(t, ep, slot, d)` adds `{ep, slot, ptr, len}` to the local
object's `CkDynObj::posts`; a second post for the same pair aborts.
`CkDynObj`'s destructor deletes the list, so posts do not survive migration
or destruction; nodegroups are refused (a branch is shared by the node's
PEs). Of the two implementations the design allows, CkDyn keeps its own
table; it does not use `CkPostBuffer`/`CkMatchBuffer`, whose tag space is
global per PE and whose `P2P_RECV` flow delivers an entry method twice.

- Host: the sender cannot know whether the receiver posted, and the
  non-post path transfers the data in `_processHandler`, before the object is
  known. So `zcArrive` applies a host post at delivery by copying the slot
  from the delivered message into the posted buffer (one copy on the
  receiving PE) and rewriting the descriptor to point at it; the view's
  pointer is then the posted buffer. A post made any time before the message
  is delivered to the object is used, including after the send was issued.
  Receiving straight into the posted buffer would need `_processHandler` to
  consult CkDyn before `CkRdmaIssueRgets`.
- Device: the destination is chosen in `zcArrive` before the transfers are
  issued, so a device post receives the data directly.

A post's `slot` OR'd with `CKDYN_POST_ARG` names an argument instead of a slot
(`zcPostMatches`): in a message with `CKDYN_MSG_BLOCK` whose slot 0 is a
`FUTURE` slot, argument k is slot k + 1, otherwise slot k. A binding that
posts before it knows whether the sender will block uses it. A post is
consumed only by an `NCPY` or `DEVICE` slot; when the slot arrives as a
`BUFFER` (below the binding's zero-copy threshold, or in a copied context),
the post stays, and the binding removes it with `CkDynUnpostBuffer` and copies
the value itself.

### The device of a PE

`CkDynHapiMyDevice()` returns the device HAPI's mapping gave the calling PE
(`CpvAccess(my_device)`, the ordinal `hapiInit` passed to `cudaSetDevice` on that
PE's thread), -1 in a build without CUDA or HIP. `CkDynHapiUseMyDevice()` makes
it the calling thread's current device and returns it. A binding calls it before
its first user code on a PE, so that libraries that allocate on the current
device (CuPy, PyTorch, Numba) use the PE's device by default.

### Persistent device buffers

The implementation is Charm++'s `CkDevicePersistent` (ckrdmadevice.h/.C);
CkDyn's functions wrap it. `open()` takes a CUDA/HIP IPC handle of the
allocation holding the buffer and the buffer's offset in it (the base from
`cuMemGetAddressRange` through `dlsym`, or `hipMemGetAddressRange`): cupy,
torch and pool allocators hand out pieces of larger allocations, and an IPC
handle maps the whole allocation. In a program of more than one process
`open()` also registers the memory once (`CmiNcpyBuffer::init`,
`CMK_BUFFER_REG`, `CMK_BUFFER_NODEREG`); `close()` on the object that opened it
deregisters, and on a copy received from a peer does nothing. The pup carries
the pointer, length, PE, process, host, the IPC handle and offset, and the
pupped `CmiNcpyBuffer` with the registration.

Mode (`CkDevicePersistent::mode`), for a get into a local buffer or a put from
one:

- same process: `hapiMemcpyAsync` device to device on the local buffer's
  stream; `hapiAddCallback` on that stream sends each buffer's callback;
- another process on the same host, when the remote buffer has an IPC
  handle and it opens in this process (`+gpushm` is not needed: it concerns
  `DEVICE` slots): the handle is opened once per process (a table keyed by
  the handle's bytes, reused by every copy of the peer's buffer and kept
  until exit), and the copy goes through the mapping as above. If opening fails (the peer's
  GPU is not visible to this process, as with one GPU per task), `mode()`
  reports RDMA and the transfer uses it;
- otherwise RDMA on the registered memory. A get is `CmiNcpyBuffer::rdmaGet`
  with the Direct API acknowledgement, whose destination callback is a
  `callCFn` on the issuing PE that sends both buffers' callbacks. A put is
  carried out as a get by the destination's PE: the issuer sends both
  buffers in a Converse message (`persist_put_handler`), that PE gets, and
  its completion sends `persist_ack_handler` back to the issuer, which sends
  the callbacks. A true RDMA put completes on the source before the data is
  visible at the destination (CommRputLocalHandler in reconverse), so its
  callbacks could not mean "arrived". Before an RDMA transfer the local
  buffer's stream is synchronized.

In every mode both buffers' callbacks are sent from the issuing PE when the
bytes are in the destination, each with its `cb_msg` (`set_msg`), as the
manual describes. The first use of each mode and operation in a process prints
a line (`[pe] CkDevicePersistent> get: ...`).

CkDyn's `CkDynDevPersist` holds an opened `CkDevicePersistent`; the handle is
that object pupped with an ignore callback, in a fixed 512-byte record
(`CkDynDevPersistHandleSize()`), so a transfer never sends the remote side's
callback. Each get or put sets the local object's callback to a `callCFn` on
the issuing PE that sends `done` an empty message: for a `FUTURE` callback the
call's refnum is the future id, for any other kind it is the callback's
refnum. The send of a `DEVICE` message (`zcDevicePrepare`) prints the mode of
`DEVICE` slots once per process (`[pe] CkDyn> DEVICE slots: ...`), which is
how a run shows whether `+gpushm` took the IPC path. `CkDynDevPersistMode`
returns `CkDevicePersistent::mode`. Close aborts while a transfer is
outstanding. In a build without CUDA or HIP every CkDyn function aborts.

### HAPI callbacks

`CkDynHapiAddCallback(stream, cb)` builds a message with one `CALLBACK` slot
holding `cb` and calls `hapiAddCallback(stream, CkCallback(CkDynMgr::hapiDone,
CkMyPe()), msg)`. When the stream reaches that point HAPI sends the message
to `CkDynMgr::hapiDone` on the calling PE, which sends an empty `CkDynMsg` to
`cb` with `CkDynCallbackSend`. `hapiAddCallback` keeps quiescence from being
detected until then. Every CUDA- or HIP-dependent line is under
`CMK_CUDA || CMK_HIP`; other builds abort with a message.

### Tests

`tests/charm++/ckdyn` phases `ncpy/nonpost`, `ncpy/post_before`,
`ncpy/post_after` and `ncpy/samepe` send 16 MB from element 0 to the last
element and back (the last one to element 1, on element 0's PE), check the
data, that each `done` runs once on the sending PE, that a retained view's
pointer stays valid after the hook (the echo is sent from it and its `done`
releases the message), and that a post made before or after the send
receives the data. The post-after case is ordered by the test: on one PE the
post follows the send in the same method; within one process the receiving
PE spins until the sender has sent; across processes the post request
follows the data, which is delivered only after its transfer. The `-zcbcast`
run must abort on a broadcast with an `NCPY` slot. The `device` and `hapi`
phases are compiled only in a CUDA or HIP build, as are `devpersist/remote`
and `devpersist/samepe`: element 0 opens a 16 MB buffer and sends its handle
to the last element (or element 1, on its PE), which gets it into its own
buffer, checks the pattern, writes a second pattern and puts it back; element
0 checks, closes, reopens and sends the new handle, the target gets and checks
again, and both close. The done callbacks are `TARGET` callbacks to `pDone`
with the call's refnum.

## Pup

All six classes override `pup`: each calls its base class's `pup`, builds the
object's target, and calls `pupObj(p, d, target)`. `pupObj` pups the chare
index, the section cookie map (empty on a checkpoint; see Sections) and a
`hasHandle` bool. A branch outside a constrained group has no handle and stops
there, without calling the binding. Otherwise it pups the
binding's bytes:

- Reason: `CKDYN_PUP_CHECKPOINT` when `p.isCheckpoint()`, else
  `CKDYN_PUP_MIGRATE`; the sizing pass ORs in `CKDYN_PUP_SIZING`.
- Sizing: `n = pack_size(...)`, kept in `CkDynObj::sizedLen`; pup `n`, then
  count `n` bytes.
- Packing: `n = pack_size(...)`; with a `PUP::mem` pupper it aborts if `n`
  differs from `sizedLen`, because the buffer was sized by the first pass. Pup
  `n`, then `pack` writes `n` bytes.
- Unpacking: pup `n`, then `handle = unpack(bd, target, reason, ptr, n)`.

When the pupper is a `PUP::mem` (migration, in-memory buffers), `pack` and
`unpack` work directly on the pupper's buffer (`get_current_pointer`,
`advance`). Any other pupper (a checkpoint file) goes through a temporary
vector. If `pack_size` returns `CKDYN_PACK_REFUSED`, `pupObj` calls
`bindingError` ("the object cannot be packed in its current state") and
aborts; a binding returns it for an object it cannot serialize in its
current state, such as one with a suspended coroutine. `pupObj` aborts when
the `pack_size` or `unpack` hook is missing.

The migration constructors (`X(CkMigrateMessage*)`) leave the handle NULL;
`pup` sets it. `ChareDyn` has one for checkpoint restart only (the restore
path aborts on a chare without one); singleton chares do not migrate.

**Readonlies.** If the hook table has `pack_readonly_size` or
`unpack_readonly`, `CkDynInit` registers one readonly, `ckdyn_ro`, whose pup
function `ckdynRoPup` calls `pack_readonly_size`/`pack_readonly` when
packing and `unpack_readonly` when unpacking, with the same `PUP::mem` rule.
The start-up readonly broadcast and the checkpoint file both go through it.
CkDyn's own readonlies, `_ckdynMgrGID` and `_ckdynMcastGID`, are declared in
`CkDyn.ci`.

## Start-up

A shared-library binding:

1. `CkDynInit(&hooks)` copies the table, checks the version and the table
   size, and registers `ckdyn_ro` if the binding has readonly hooks. A
   mismatch prints to stderr and calls `abort()`, since this runs before
   `ConverseInit`.
2. `CkDynStart(argc, argv)` copies argv into storage it owns, calls
   `_registerConverseInitHooks()` and `ConverseInit(argc, argv, _initCharm,
   0, 0)`, and never returns.
3. In `_initCharm`, rank 0 of each process registers the core modules,
   including `_registerCkDyn()` (init.C:1601), which registers the two
   message types, `CkDynMgr`, `_ckdynMgrGID` and the two init calls.
4. `_initCharm` then calls `CkDynCallRegisterHook()`, which calls
   `hooks.register_classes` and returns true, or returns false when there is
   no hook table or no `register_classes`. In a `CMK_CHARM4PY` build it takes
   the place of `CkRegisterMainModule`, and when it returns false the Ext
   API's `CkRegisterMainModuleCallback` runs. In any other build it runs
   after `CkRegisterMainModule`, so a program with a charmxi main module that
   starts through `CkDynStart` gets both.
5. The init calls run: `_ckdynInitNode` (rank 0) registers the foreign
   reducer and the two section init entries; `_ckdynInitProc` (every PE)
   initializes the builder Ckpv.
6. `_registerDone`; PE 0 constructs the main chare. `MainchareDyn` creates
   `CkDynMgr` and the `CkMulticastMgr` before calling `hooks.mainchare`, so
   both exist on every PE before any other message is delivered.

A C++ program links CkDyn from libck and starts as any Charm++ program. It
calls `CkDynInit` and registers its classes from an `initnode` function, which
runs after `_ckdynInitNode` because the CkDyn module is registered first.
`register_classes` is not called for it, because `CkDynInit` runs after the
point in `_initCharm` where the hook is called. A program
whose main chare is an ordinary C++ main chare calls `CkDynCreateManagers()`
from its constructor.

`_createTraces` in init.C is a weak symbol in every build (empty, or in a
`CMK_CHARM4PY` build with tracing calling the projections and summary
modules), so a program without a charmc-generated moduleinit links, and a
charmc-linked program keeps its generated definition.

**What `CMK_CHARM4PY` means.** CkDyn does not depend on it: registration, the
hook, late registration, CCS (`CcsRegisterHandlerFn`) and `_createTraces` work
in a plain build. The macro selects a build of libcharm as one shared library
for a Python binding (the `charm4py` build target, built with the separate
option `REFNUM_TYPE="unsigned int"` for 32-bit reference numbers): the Ext API used by charm4py 1.x (ck.C, register.C, init.C:
`registerCkRegisterMainModuleCallback`, `_registerTreeLB` and the trace module
registration, since no charmc moduleinit exists), late registration accepted
without `_registerLateBegin`, and the charm4py CCS layout in reconverse. One
such libcharm serves both charm4py 1.x (through the Ext API) and a CkDyn
binding.

## Limits and diagnostics

- **Entries and message types.** The envelope's `epIdx` and the header's
  `ep` are 16 bits: at most 65536 entry indices per process, the runtime's
  own included. `msgIdx` is 8 bits: 256 message types, two used by CkDyn.
- **Dimensions.** Array classes have 1 to 6 dimensions (`CKDYN_MAX_DIMS`), or register with 0 and take the count from each creation;
  registration aborts with "array class %s has %d dimensions", `mkIndex` with
  "array index with %d dimensions", `CkDynCreateArray` with "array %s has %d
  dimensions, creation gave %d". A `BUFFER` slot has at most 32 dimensions
  (`CKDYN_MAX_BUF_DIMS`): "buffer with %d dimensions".
- **Sizes.** A slot of more than 32 bits of length, a payload above 2 GB and
  more than 65535 slots abort when the message is built.
- **Array indices.** Components of a 4- to 6-dimensional index must fit in
  16 bits ("index component %d does not fit in 16 bits").
- **Future ids** are 32-bit, 1 to 2^32-2: 0 means "none" and the all-ones
  value is the reduction user flag. This needs a libcharm built with
  `REFNUM_TYPE="unsigned int"`, as the design requires; a build with the
  default 16-bit `CMK_REFNUM_TYPE` accepts only 1 to 65534.
- **ABI.** The structs a binding copies as bytes into messages
  (`CkDynTarget`, `CkDynSectionID`, `CkDynCallback`, `CkDynHdr`, `CkDynSlot`)
  have fixed-width integer fields only; the enums supply constants.
  `static_assert`s in ckdyn.C fix their sizes (40, 16, 88, 24 and 24 bytes),
  so a change that alters one fails to compile until `CKDYN_API_VERSION` is
  reconsidered.
- **Group dependencies** (`group_dep` of `CkDynMsgBegin`) are set in the
  envelope, but the runtime does not honor them on point-to-point sends to
  array elements.
- **Builder.** One cached builder per PE; a nested build allocates a second
  one, which is deleted when it finishes. The builder must not be used before
  `_ckdynInitProc` has run.
- **Aborts** name their cause in the message. Registration: not on rank 0,
  no constructor entry, unknown kind, entry indices not contiguous (another
  registration interleaved), late registration not enabled or more than one
  PE per process. Lookup: a chare index or entry index that is not dynamic
  (a non-CkDyn message reached a CkDyn entry, or the header's ep is wrong),
  "message without a CkDyn header (magic 0x%x)", a slot index out of range or
  of another type. Start-up: API version or hook table size mismatch, `CkDynStart` before
  `CkDynInit`. A missing hook that the situation needs (`mainchare`,
  `unpack`, `pack_size`, `map_proc_num`, `future`, `reduce`,
  `unpack_readonly` with readonly data present).
- **Binding errors**, reported to `hooks.error` before the abort: a
  point-to-point entry method on a branch that was not constructed, and
  `CKDYN_PACK_REFUSED`. "pack_size returned %zu bytes on the packing pass but
  %zu on the sizing pass" is a binding whose two passes disagree.
- **Kind mismatches.** `CkDynCreate*` on a class of another kind,
  `CkDynContribute` from a chare or main chare, `CkDynGetReductionNumber` on a
  kind other than group, nodegroup or array element (an array map included), `CkDynAtSync`, `CkDynSetMigratable`,
  `CkDynMigrateMe` and `CkDynSetObjLoad` on a target that is not a local array
  element, and `CkDynArrayInsert` with a message not finished with an array
  class's constructor entry.
- **Sections** abort on: an empty member list, a group section over a group
  that is not dynamic, a PE out of range, a send, delete or combine naming a
  deleted section (stale generation) or an unknown slot, a contribution or
  `CkDynSectionRedNo` from an object that is not a member, a combine with a
  group part or with parts of different index dimension counts, a section
  message with a group dependency, and a section message for an entry of
  another class than the receiving element.
- **Section callback relay table.** Per PE, one slot per `(section id with
  generation, entry)` used in a section callback converted on that PE (other
  than by `CkDynContribute`), plus
  one per distinct reference number of a QD or checkpoint section callback;
  slots are never freed. Fresh future ids per collective call do not add
  slots. At most 65534 slots with a 16-bit `CMK_REFNUM_TYPE`; the next one
  aborts with "future id %u is not usable as this build's %d-byte reference
  number".
- **Zero-copy** (see Zero-copy) aborts on: more than 4096 zero-copy slots
  in a message, `NCPY` and `DEVICE` slots in one message, a zero-copy slot
  in a broadcast, multi-branch send, section message, constructor message,
  reduction payload or broadcast or section callback ("slot %d is an NCPY
  slot; a message with zero-copy slots is sent point to point only, not as
  ..."), a device send whose destination PE is unknown, a posted length
  that differs from the slot's, a second post for one (object, entry, slot),
  a post for an object that is not local or for a nodegroup, a host post for
  a `DEVICE` slot or the reverse, an `NCPY` message delivered without its
  transfer, and any device call in a build without CUDA or HIP.

## Where in the source

- src/ck-core/ckdyn.h: the C API: descriptors, `CkDynTarget`,
  `CkDynCallback`, `CkDynHdr`, `CkDynSlot`, `CkDynHooks`, all functions.
- src/ck-core/ckdyn-impl.h: the message classes, `CkDynObj`, the six chare
  classes, `CkDynMgr`, the helpers used by ckarray.C and init.C.
- src/ck-core/ckmulticast.C, XArraySectionReducer.h: CkMulticast, which
  carries section multicasts and reductions.
- src/ck-core/CkDyn.ci: the message types, `_ckdynMgrGID`, `_ckdynMcastGID`, `_ckdynInitNode`,
  `_ckdynInitProc`, the `CkDynMgr` group.
- src/ck-core/ckdyn.C: the class and entry tables (`ClassRec`, `EpRec`,
  `g_classes`, `g_eps`), `registerClass`, the builder (`CkDynMsgBegin`,
  `writePayload`, `CkDynMsgFinish`), the views (`viewPayload`,
  `viewReduction`, `viewSys`, `viewAny`), `dispatch`,
  `CkDynDeliverBroadcast`, the chare classes, `MainchareDyn::callRelay`,
  `noteDims`, `pupObj`, the sends and creation functions,
  `CkDynMakeCallback`, `CkDynMgr::deposit`, the section table and protocol
  (`CkDynMgr::newSection`, `finishSection`, `sendTo`, `addPart`,
  `deleteSection`, `relay`, `relaySlot`, `toMcast`, `sectionJoin`,
  `sectionContribute`), `CkDynCoroResume`,
  `CkDynCoroSuspend`, the reducer
  table and `ckdynForeignReduce`, `ckdynRoPup`, `CkDynInit`, `CkDynStart`,
  and the zero-copy part (`zcBuildPrefix`, `dynPayload`, `zcPrefixLen`,
  `zcBeforeSend`, `zcDevicePrepare`, `zcArrive`, `zcApplyHostPosts`,
  `zcDeviceIssue`, `zcAfterDelivery`, `zcViewPtr`, the posts,
  `CkDynHapiAddCallback`, `CkDynMgr::hapiDone`).
- src/ck-core/ckrdma.C, ckrdmadevice.C: the transfers CkDyn's zero-copy slots
  use (`CkRdmaIssueRgets`, `CkRdmaDeviceOnSender`, `CkRdmaDeviceIssueRgets`).
- src/ck-core/register.h: `ChareInfo::isDynamic`.
- src/ck-core/ckarray.C, ckarray.h: `CkArrayBroadcaster::deliverAndUpdateDyn`,
  `performDeliveryDyn`, and the branch in `CkArray::recvBroadcast` that
  selects them.
- src/ck-core/init.C: `_registerCkDyn` and `CkDynCallRegisterHook` in
  `_initCharm`; the weak `_createTraces`.
- tests/charm++/ckdyn: the C++ test binding (`cbinding.h`, `cbinding.C`): a
  hook table over C++ method tables, installed with `CkDynInit` from an
  `initnode`, and the tests that use it.
