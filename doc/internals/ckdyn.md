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
design document, [`doc/charm4py-thin-design.md`](../charm4py-thin-design.md)
(until it moves into this repository it is `DESIGN.md` of the charm4py-thin
project), section 3. This chapter describes what the code does today. Sections
over CkMulticast (design 3.9), zero-copy and device buffers (3.14) and the
Python binding (section 4) are planned; their entry points exist in `ckdyn.h`
and abort, and they are not described here.

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
(src/ck-core/init.C:1668). A shared-library binding registers from
`hooks.register_classes`, which `_initCharm` calls in place of
`CkRegisterMainModule` (see Start-up). A C++ program registers from an
`initnode` function: init calls run on rank 0 of each process before
`_registerDone` (`InitCallTable::enumerateInitCalls`, init.C).

**Late registration.** After `CkDynAllowLateRegistration()`,
`CkDynRegisterClassLate` registers a class after start-up. It aborts unless
the process has one PE, because `_entryTable` and `_chareTable` are
process-wide vectors that another PE of the same process could be reading
while dispatching. Only builds with `CMK_CHARM4PY` accept it:
`CkRegisterEpInternal` (src/ck-core/register.C:48-53) aborts on any
registration after `_registerDone` otherwise.

An array class registered with `ndims == 0` accepts any dimension count: each
creation or insert gives its own, and `CkDynCreateArray` checks the count only
for a class registered with a fixed one. A language whose arrays pick their
dimension count at creation (charm4py) registers every array class that way.

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
| `register_classes(global)` | `CkDynCallRegisterHook`, from `_initCharm` | start-up, rank 0 of each process, `CMK_CHARM4PY` builds |
| `construct(bd, target, ctor view)` | constructors of `ChareDyn`, `GroupDyn`, `NodeGroupDyn`, `ArrayElemDyn`, `ArrayMapDyn` | object creation; returns the handle |
| `mainchare(bd, &handle, target, argc, argv)` | `MainchareDyn(CkArgMsg*)` | main chare creation on PE 0 |
| `destruct(bd, handle)` | destructors of all six classes, when the handle is not NULL | object deletion, migration away, exit |
| `deliver(bd, handle, ep, user_id, ep_flags, view)` | `dispatch`; `CkDynDeliverBroadcast` when `deliver_many` is NULL | every entry method invocation |
| `deliver_many(bd, handles, n, ep, user_id, ep_flags, view)` | `CkDynDeliverBroadcast` | an array broadcast, once per PE; rules below |
| `pack_size`, `pack`, `unpack` | `pupObj` | migration, checkpoint, restart |
| `pack_readonly_size`, `pack_readonly`, `unpack_readonly` | `ckdynRoPup` | readonly broadcast at start-up, checkpoint, restart |
| `reduce(global, reducer_id, views, n, builder)` | `ckdynForeignReduce` | combining contributions to the foreign reducer |
| `future(global, pe, id, view)` | `CkDynMgr::deposit` | a value for a binding future arrives |
| `section_joined` | not called | sections are planned (design 3.9) |
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
future value is a `CkDynMsg`; `CkDynMcastMsg` is reserved for section sends
(planned). Two of the 256 message indices are used.

### Payload layout

`buf` holds, in order:

1. `CkDynHdr`, 24 bytes: `magic_version` (`CKDYN_MAGIC`, 0xDA01), `ep`
   (uint16), `flags` (uint16), `nslots` (uint16), `user_id`, `tail_len`,
   `data_len`, `nbuf` (reserved for zero-copy descriptors).
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
| `NCPY`, `DEVICE` | planned (design 3.14) | |
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
broadcast sends, read by `dispatch`). `BLOCK`, `BCASTRET` and `RAY` are
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

Size checks: a slot longer than 32 bits aborts when it is added; `Finish`
aborts on a payload above 2 GB (the allocator takes an `int`) or more than
65535 slots (`nslots` is 16 bits).

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
(NULL until sections exist) and `msg` (the runtime message). `viewAny(view,
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
every branch, so `dispatch` drops a message there when its view is not of
kind `DYN` (a reduction result or system message) or its header has
`CKDYN_MSG_BCAST`. A point-to-point `CkDynMsg` to such a branch is an error
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
constructor of an array class. If the array has no local branch yet,
`ckInsertIdx` buffers the insert.

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

This path does not go through `CkDeliverMessageFree`, so the entry method is
not traced per element and, with `deliver_many`, no element is on the
running-object stack during the call: time is charged to the array manager
unless the binding brackets each element with `CkDynCallstackPush`/`Pop`.
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
| `SECTION` | aborts (planned) |

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
frees the message (`deposit` is not nokeep). It aborts if the binding has no
`future` hook. The `relay` entry is reserved for section callbacks and
aborts. The group is `[migratable]` and has a `pup` (which holds no CkDyn
state), so a checkpoint restores it.

## Reductions

`CkDynReducers(&table)` exports a static table of the built-in reducers:
name, `CkReduction::reducerType` value, and the element kind of the data
(`sum_int` is I32, `max_double` is F64, and so on; `concat`, `set`,
`statistics` and `tuple` are BYTE and left to the binding). The same table
gives `elem` of the `BUFFER` slot in a reduction result view.

`CkDynContribute(me, data, cb)` finds the local object the target names
(`localChare`) and calls `contribute(len, data, type, callback)` on its
`Group`, `NodeGroup` or `ArrayElement`. A `nop` contribution is sent with no
data.

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

## Pup

All six classes override `pup`: each calls its base class's `pup`, builds the
object's target, and calls `pupObj(p, d, target)`. `pupObj` pups the chare
index and a `hasHandle` bool. A branch outside a constrained group has no
handle and stops there, without calling the binding. Otherwise it pups the
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
CkDyn's own readonly, `_ckdynMgrGID`, is declared in `CkDyn.ci`.

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
   including `_registerCkDyn()` (init.C:1600), which registers the two
   message types, `CkDynMgr`, `_ckdynMgrGID` and the two init calls.
4. In a `CMK_CHARM4PY` build `_initCharm` then calls `CkDynCallRegisterHook()`
   (init.C:1634), which calls `hooks.register_classes` and returns true. When
   it returns false (no hook table, or no `register_classes`), the Ext API's
   `CkRegisterMainModuleCallback` runs as before.
5. The init calls run: `_ckdynInitNode` (rank 0) registers the foreign
   reducer; `_ckdynInitProc` (every PE) initializes the builder Ckpv.
6. `_registerDone`; PE 0 constructs the main chare. `MainchareDyn` creates
   `CkDynMgr` before calling `hooks.mainchare`, so the manager exists on
   every PE before any other message is delivered.

A C++ program links CkDyn from libck and starts as any Charm++ program. It
calls `CkDynInit` and registers its classes from an `initnode` function, which
runs after `_ckdynInitNode` because the CkDyn module is registered first. In a
build without `CMK_CHARM4PY`, `register_classes` is never called. A program
whose main chare is an ordinary C++ main chare calls `CkDynCreateManagers()`
from its constructor.

`_createTraces` in init.C is a weak symbol (init.C:103, 109), so a
charmc-linked program, which generates its own definition, links against a
libck that also defines it.

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
  kind other than group or array, `CkDynAtSync`, `CkDynSetMigratable`,
  `CkDynMigrateMe` and `CkDynSetObjLoad` on a target that is not a local array
  element, and `CkDynArrayInsert` with a message not finished with an array
  class's constructor entry.
- **Not implemented yet.** Sections (`CkDynSendSection`,
  `CkDynSectionCreate*`, `CkDynSectionCombine`, `CkDynSectionDelete`,
  `CkDynSectionRedNo`, `CkDynContributeSection`, `SECTION` callbacks,
  `CkDynMgr::relay`) abort with "... not implemented yet (step 2)". Zero-copy
  and device buffers (`CkDynMsgAddNcpy`, `CkDynMsgAddDevice`,
  `CkDynPostBuffer`, `CkDynPostDeviceBuffer`) and `CkDynHapiAddCallback`
  abort with "... not implemented yet (step 3)".

## Where in the source

- src/ck-core/ckdyn.h: the C API: descriptors, `CkDynTarget`,
  `CkDynCallback`, `CkDynHdr`, `CkDynSlot`, `CkDynHooks`, all functions.
- src/ck-core/ckdyn-impl.h: the message classes, `CkDynObj`, the six chare
  classes, `CkDynMgr`, the helpers used by ckarray.C and init.C.
- src/ck-core/CkDyn.ci: the message types, `_ckdynMgrGID`, `_ckdynInitNode`,
  `_ckdynInitProc`, the `CkDynMgr` group.
- src/ck-core/ckdyn.C: the class and entry tables (`ClassRec`, `EpRec`,
  `g_classes`, `g_eps`), `registerClass`, the builder (`CkDynMsgBegin`,
  `writePayload`, `CkDynMsgFinish`), the views (`viewPayload`,
  `viewReduction`, `viewSys`, `viewAny`), `dispatch`,
  `CkDynDeliverBroadcast`, the chare classes, `pupObj`, the sends and
  creation functions, `CkDynMakeCallback`, `CkDynMgr::deposit`, the reducer
  table and `ckdynForeignReduce`, `ckdynRoPup`, `CkDynInit`, `CkDynStart`.
- src/ck-core/register.h: `ChareInfo::isDynamic`.
- src/ck-core/ckarray.C, ckarray.h: `CkArrayBroadcaster::deliverAndUpdateDyn`,
  `performDeliveryDyn`, and the branch in `CkArray::recvBroadcast` that
  selects them.
- src/ck-core/init.C: `_registerCkDyn` and `CkDynCallRegisterHook` in
  `_initCharm`; the weak `_createTraces`.
- tests/charm++/ckdyn: the C++ test binding (`cbinding.h`, `cbinding.C`): a
  hook table over C++ method tables, installed with `CkDynInit` from an
  `initnode`, and the tests that use it.
