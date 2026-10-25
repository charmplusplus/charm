# Array broadcasts and migrating elements

A broadcast to a chare array is one message per PE, delivered by that PE's
branch of the array manager (`CkArray::recvBroadcast`, src/ck-core/ckarray.C)
to every element that is local when the message is processed. Elements move
between PEs, by `migrateMe` or at a load balancing step, while broadcasts are
in flight. The `CkArrayBroadcaster` listener (ckarray.C, ckarray.h) makes
every element receive every broadcast exactly once regardless.

## Epochs

PE 0 serializes the array's broadcasts (`CkArray::sendBroadcast`): each gets
the next epoch number before it is sent to all PEs. Each element carries the
epoch of the next broadcast it expects, in the broadcaster's listener data
(`bcastNo`, pupped with the element). A new element is stamped with the
highest epoch its PE has seen plus one (`ckElementStamp`), so it receives
only broadcasts sent after its creation.

## Delivery

On arrival a broadcast is stored in the PE's `storedBcasts` queue, indexed by
epoch (`ingestIncoming`), then offered to every local element
(`attemptDelivery`). An element whose expected epoch is lower has missed an
earlier broadcast and is skipped for now; one whose epoch is higher has
already received this broadcast elsewhere and is skipped for good. An element
that takes the broadcast then also takes every stored broadcast that follows
in sequence (`bringUpToDate`).

When an element arrives on a PE, after its pup, the broadcaster runs the same
catch-up (`ckElementArriving`). This covers both orders in which a broadcast
and a migration can meet: a broadcast that reached the destination PE before
the element is in the store and is delivered on arrival; one that reached the
source PE before the element left was delivered there, and its epoch tells
the destination to skip it.

Stored broadcasts are freed by spring cleaning (`CkArrayBroadcaster::
springCleaning`, on the array's periodic timer): a broadcast is kept until a
cleaning pass finds it older than the newest broadcast seen at the previous
pass.

## Order

The epochs also order broadcasts. A PE processes broadcasts in whatever order
the machine layer delivers them, and on reconverse two broadcasts from one PE
can arrive in either order: a group broadcast travels within a process as one
node-level message, which whichever PE is free takes from the process's node
queue and fans out to the PEs, so two broadcasts taken by two PEs are fanned
out concurrently. The element that expects epoch n is skipped for epoch n+1,
which waits in the store, and takes both in sequence when n arrives. Every
element therefore receives the array's broadcasts in one order, the order in
which PE 0 stamped them, in both modes below.

## Arrays with stable locations

`CkArray::stableLocations` is set when both hold at array creation:

- the array was created with static insertion (`CkArrayOptions::
  setStaticInsertion(true)`, the default for a `ckNew` with a nonzero
  initial size);
- anytime migration is off (`CkArrayOptions::setAnytimeMigration(false)`, or
  `+noAnytimeMigration` on the command line, which makes that the default).

These flags are the program's promise that no broadcast to the array is in
flight while an element moves. Elements of such an array may still migrate at
a load balancing step, so the promise covers that step: every broadcast,
including the result of a reduction whose callback broadcasts to the array,
must have been delivered everywhere before the elements call `AtSync`. A
broadcast that reaches the source PE after an element left and the destination
before it arrived is lost to that element; in the other order the element
receives it twice.

Under the promise no element will ever need a broadcast that every element on
its PE has received, so such an array keeps no spring cleaning timer and
instead frees the stored broadcasts after each delivery
(`CkArrayBroadcaster::releaseDelivered`): every stored broadcast older than
the lowest expected epoch among the PE's elements. A broadcast is then held
only while an earlier one is still on its way, and the ordering above holds.
Everything else, storing, the epoch check and the catch-up, is the same code
in both modes.

In the tree, TCharm, AMPI and CkIO sessions create their arrays with these
flags. AMPI keeps the promise by refusing `AMPI_Migrate` with a non-blocking
collective outstanding (`checkNoCollectiveOutstanding`, ampi.C), because the
results of `MPI_Iallreduce`, `MPI_Ibarrier` and the others are broadcasts to
the ampi array. With `+ampiMigrateWithCollectives`, or
`+ampiEnableMigrateToPe` which implies it, AMPI creates its arrays with
anytime migration on and allows collectives to be outstanding across a
migration, at the memory cost of broadcasts kept until spring cleaning.

## Where in the source

- src/ck-core/ckarray.C: `CkArray::CkArray` (sets `stableLocations`),
  `CkArray::sendBroadcast`, `CkArray::recvBroadcast`,
  `CkArrayBroadcaster::ingestIncoming`, `attemptDelivery`,
  `bringUpToDate`, `releaseDelivered`, `springCleaning`.
- src/ck-core/ckarray.h: `BcastQueue` (`releaseBelow`).
- src/ck-core/ckarray.h: `CkArrayBroadcaster` (`ckElementStamp`,
  `ckElementArriving`), `CkArray::stableLocations`.
- src/ck-core/init.C: `+noAnytimeMigration`.
- src/libs/ck-libs/ampi/ampi.C: `isMigrateWithCollsEnabled`,
  `checkNoCollectiveOutstanding`, the `setAnytimeMigration` calls.
- tests/charm++/atsync_reduction: reductions in flight across an AtSync
  balancing step; with `-s -d`, an array with stable locations keeping the
  promise; `-o` checks the delivery order.
- contrib/reconverse/README.md, "Messages to one destination can be
  reordered".
