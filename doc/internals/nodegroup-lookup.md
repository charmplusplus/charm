# Nodegroup lookup

Files: `src/ck-core/ck.C` (`CkLocalNodeBranch`, `_processForNodeBocMsg`,
`CkSendMsgNodeBranchInline`, `CkCreateLocalNodeGroup`, `IrrGroup::~IrrGroup`),
`src/ck-core/init.h` (`TableEntry`, `GroupIdxArray`).

## Two tables

A process keeps its nodegroups in `_nodeGroupTable`, a `GroupIdxArray` of
`TableEntry` (object pointer, chare type, messages held until the object
exists, ready flag). Every PE of the process can reach it, `find()` inserts
a missing entry, and the storage moves when it grows, so every access to it
is made under the process-wide lock `_nodeGroupTableImmLock`.

Looking up a nodegroup that already exists is far more frequent than
creating one: `CkLocalNodeBranch` (what `proxy.ckLocalBranch()` calls on a
nodegroup proxy) and the delivery of every nodegroup message do it. For
those reads the process keeps a second table, `_nodeGroupPublished`, that
is read without the lock:

- a fixed top-level array of 4,096 atomic pointers to blocks, each block an
  array of 256 atomic object pointers; nodegroup id `i` is slot `i & 255` of
  block `i >> 8`;
- a block is allocated when the first nodegroup in its range becomes ready
  and is never moved or freed, so a reader may hold a block pointer without
  any protection;
- it covers ids 1 to 2^20 - 1. Positive ids are the ones created on node 0
  (the usual case: created from a mainchare). Ids created on another node
  are negative and are not in it.

## Writers

Both writers already hold `_nodeGroupTableImmLock`, which orders them:

- `CkCreateLocalNodeGroup` stores the object in the published table as the
  last step of the critical section in which it fills the `TableEntry`,
  sets the ready flag and releases the held messages. A reader therefore
  sees the object only after its constructor has returned.
- `IrrGroup::~IrrGroup` clears the slot when a nodegroup is destroyed.

## Readers

`_publishedNodeGroup(id)` returns the object, or null when the nodegroup is
not ready or the id is outside the published range. The three readers try it
first and take the lock only on null:

- `CkLocalNodeBranch`: on null, the locked lookup of `_nodeGroupTable`, and
  if the nodegroup is not ready the PE runs the scheduler and tries again.
- `_processForNodeBocMsg` (delivery of a nodegroup message): on null, the
  locked path, which queues the message in the `TableEntry` if the object
  does not exist yet.
- `CkSendMsgNodeBranchInline`: on null, the locked lookup.

A lookup from inside the nodegroup's own constructor does not use either
table: `CkLocalNodeBranch` returns `_currentNodeGroupObj`.

A pointer returned by a lookup stays valid only while the nodegroup exists;
this was also the case when every lookup took the lock, which was released
before the pointer was used.

## Why it matters

With one lock per process, an application that looks up a nodegroup in
every entry method serializes all PEs of the process on it. SpECTRE reaches
its `GlobalCache` nodegroup this way; at 64 PEs in one process most worker
threads were found waiting on this lock (measurements in the pull request
that added the published table).

## Test

`tests/charm++/nodegroup_lookup`: every PE looks up 600 nodegroups (three
blocks) repeatedly while they are being created, and checks that each lookup
returns the object built with that nodegroup's tag.
