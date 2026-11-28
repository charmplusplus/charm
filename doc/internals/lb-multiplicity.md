# Replication as a load-balancing decision

A placement balancer moves objects between processors; it cannot bring the
maximum load below the heaviest object. This chapter describes the extension
that lets a balancer change how many copies of an object exist: the
declaration an object makes, the two decision kinds a strategy can emit
beside its moves, how they reach the object, how the memory contract
accounts for them, and what ShedLB does with them. The reasons and the
measurements are in the pull request that introduced it.

The division of responsibility is the one migration already has. The
application decides *whether* an object type can be replicated and *how*
the copies are made and kept consistent, the way it decides whether an
object is migratable and provides its `pup`. The balancer decides *when*,
*which* object, *how many* copies and *where*, jointly with its moves. An
object that never declares itself replicable is never replicated, and no
other application is affected.

## The declaration

`LDObjData` (`src/ck-ldb/lbdb.h`) carries eleven fields per object:
`replicaGroup`, `replicaRank`, `replicaCount`, `replicaMax`, `replicaParts`,
`replicaExchangeBytes`, `replicaExchangeMsgs`, `replicaExtraBytesPerPeer`,
`replicaExtraBytes`, `replicaCostNode`, `replicaCostRemote`. `replicaGroup
< 0` means not replicable, which the `LBObj` constructor sets. Objects
sharing a group are the copies of one logical object; `replicaRank` is the
copy's index, `replicaCount` how many copies exist, `replicaMax` how many
the application can keep consistent, `replicaExtraBytes` what one copy
holds on its device beyond the object's own footprint (the buffers the
copies exchange), and the two costs what the exchange with one peer, on the
same node or on another node, measured on this copy per balancing interval
(zero while unmeasured, or when the copies exchange nothing, as when
serving). `replicaParts`, `replicaExchangeBytes`, `replicaExchangeMsgs` and
`replicaExtraBytesPerPeer` are described under ShedLB's group pass below.
The fields are pupped with the stats (format version 8, the costs 9, the
rest 10) and so appear in every `LDStats::objData` entry a strategy reads.

An element declares them with `CkMigratable::setLBReplica(group, rank,
count, maxCount, extraBytes, costNode, costRemote, parts, exchangeBytes,
exchangeMsgs, extraPerPeer)` (`src/ck-core/cklocation.C`), through
`CkLocRec::setObjReplica` and `LBManager::SetObjReplica` into its `LBObj`.
The values are also members of `CkMigratable`, pupped with the element and
declared again to the database of the PE it arrives on, since the `LBObj`
is per PE. The application calls it before every `AtSync`, as it does
`setObjGPUTime`.

## Decision kinds

A decision record, `MigrateInfo` (`src/ck-ldb/BaseLB.h`) and its scatter
form `MigrateDecision`, carries `kind` (`LBDecisionKind`: move, clone,
retire) and `rank`. A clone asks the object on `from_pe` to make a copy of
itself on `to_pe` with replica rank `rank`; a retire asks the copy on
`from_pe` to go away. Every producer of records sets the kind; `MigrateInfo`
constructs as a move, and the one producer that fills a varsize message's
records in place (`DistBaseLB::BroadcastSingleLocationUpdate`) sets it
explicitly.

A central strategy does not build records; it writes `stats->to_proc` and
`CentralLB::createMigrateMsg` derives the moves. Replication decisions have
their own storage beside it: `LDStats::clones` and `LDStats::retires`, each
an `LBAction {obj, fromPe, toPe, rank}` with `obj` the stats index of the
copy addressed (the source of a clone, the copy a retire removes).
`createMigrateMsg` appends them after the moves. The batch planner, which
rebuilds one message per batch from `to_proc`, puts them all in the last
batch: they stage nothing, and the application executes them once the step
has resumed.

Everything that counts arrivals skips them: the tallies of
`InitiateScatter`, the `migrates_expected` counting in
`ProcessReceiveMigration` and `ProcessMigrationDecision`,
`getPredictedLoadWithMsg`. A copy made through array insertion never calls
`LBManager::Migrated`, so counting it would hang the step.

## Delivery

On the PE named by `from_pe`, `ProcessReceiveMigration` (broadcast) or
`ProcessMigrationDecision` (scatter) calls
`CentralLB::executeReplicaDecision`, which calls `LBManager::Clone` or
`LBManager::Retire`. `LBDatabase::Clone/Retire` check that the handle's slot
still holds the object with the handle's id (slots are reused when objects
die; `Migrate` tolerates an empty slot but does not make this check) and
hand the decision to the object manager's callbacks, `LDCallbacks::clone`
and `LDCallbacks::retire` (`lbdb.h`). The callback table is filled in
`CkLocMgr::initLB`, with `CkLocRec::staticClone/staticRetire`, which call
`CkMigratable::ckClone(int toPe, int rank)` and `ckRetire()` on every
element bound to the location.
The defaults abort for an element that declared itself replicable and do
nothing for any other.

The decision is delivered at once, with no deferral, and can arrive at any
point of the element's step. The element records it and acts at a point of
its own choosing. `replica_test` acts at the end of the step the balancer
released it in (after `AtSync` has resumed it), so that by then every PE's
barrier for the round has fired and the round's migrations have been
released: a copy inserted then registers into the next round's epoch
(`cksyncbarrier.C`, `addClient`) and no decision of the current round can
still arrive. The paths that hold a deferred move behind
`CkLocRec::recvMigrate` are unchanged.

Two facts about the runtime an implementation of `ckClone`/`ckRetire`
relies on:

- A copy is a new array element. A 2-D array created as `(n, 1)` has no
  bit in its element id for the second index; it needs
  `CkArrayOptions::setBounds(n, 2)` or `(e, 1)` silently takes another
  element's id. Insertion after creation needs no `beginInserting` /
  `doneInserting`, and calling them mid-run would stall the array's
  reductions and the LB barrier. A message sent to the new element before
  it exists is buffered and delivered after insertion. A freshly inserted
  element can receive the round's `ResumeFromSync`; it should ignore a
  resume it did not ask for.
- Destruction waits for no device work and closes no channel. A retiring
  copy must finish its sends, have its senders drop their channel copies
  (destroy them: an assignment over a `CkDeviceChannel` leaves its
  registration behind), `close()` its own, and then `ckDestroy()`, from an
  entry method other than the one that decided it.

## Memory accounting

A clone is new device state at the destination with no departure anywhere.
`MemoryLedger::feasibleArrival(bytes, pe)` and `commitArrival` (`src/ck-ldb/
LBMemoryContract.h`) debit `footprint + replicaExtraBytes` at the
destination's device; the source's device is debited `replicaExtraBytes`
when the object was alone. `ContractVerifier::verifyAndRepair` takes the
clone list beside the moves, accounts them the same way, ranks them with
the other arrivals by load per byte (a copy carrying the group's load split
one way further), and removes a refused clone from the list. A retire frees
bytes, but the credit is not spent in the same step (the two-phase rule):
the copy is still there when the step's moves execute.

## ShedLB's group pass

`ShedLB::work` (`src/ck-ldb/ShedLB.C`) runs a group pass before shedding.
For every group of replicable objects, with `L_g` the sum of its copies'
loads, `k` their count and `K` the declared maximum: the target is
`k* = min(K, max(1, ceil(L_g / T)))` with `T = (1 + eps) * mean`.

- `k* > k`: every count from `k + 1` to `k*` is planned in turn, each plan
  extending the previous one by one copy, and the count whose plan leaves
  the fullest bin lowest is taken if it passes the critical-path test below;
  a count that gains nothing on the way is passed over (three copies of a
  four-partition object leave one copy with half of it: `replicaParts`,
  declared by the application, gives each copy's share as its block of the
  partitions, `LDObjData::replicaShare`; without it the shares are `1/k`).
  The ranks run on from the first unused; the clones' sources and the
  notices' targets are the copies that exist now. The source is the copy on
  the fullest bin.
  The destination is the fullest bin that still clears `T` with the copy's
  share `L_g / k*`, else the emptiest, never a bin the group is already
  on, only where `feasibleArrival` says the copy fits (and the source's
  device the extra), and only where the PE's host time stays under the
  device target (`hostCap`, the larger of `T` and the hosts' mean with the
  same slack): an object's entry-method time runs on the host whatever its
  device load, a PE's step is bound by the larger of the two, and a round
  that packed ten experts on one PE by device load alone gave its host 10 ms
  a step and the serving step (job 23612682). The same cap applies to every
  move of the shed loop. The cap is for bins on device time only; on host
  time a bin's load is its PE's host time, `T` already bounds it, and a cap
  would refuse every copy of an object above `T`. Every copy is priced as its share `L_g / k*` plus its
  exchanges: for each other copy, the declared `replicaCostNode` or
  `replicaCostRemote` by whether the two sit on the same node (`CkNodeOf`).
  The measured loads leave the exchange out, so the group pass first adds
  each existing copy's exchanges to its bin. What one peer costs comes from
  two places: a copy that has exchanged declares what its exchange measured
  (`replicaCostNode` / `replicaCostRemote`: its wait per interval and peer);
  a copy that has not declares only what it would send each peer per
  interval (`replicaExchangeBytes`, `replicaExchangeMsgs`), and the balancer
  prices that with the runtime's calibrated transfer-cost table at the
  group's size (`DiffusionCostConfig::exchangePerPeer`: the table's fan
  entries, `xchg_fan<k>_host_*` / `_inter_*`, falling back to the tier alone
  where a table has no fan entry). The fan matters on a PCIe fabric: one 134 MB
  partial alone takes 5.5 ms on PCIe-attached A40s, three copies on one node
  waited 50 ms a step for two of them and four copies 300 ms for three (jobs
  23581818, 23582210), the concurrent peer reads serialising on the fabric;
  on NVLink-attached A100s three and four copies exchange at 75 to 90 GB/s a
  peer, as two do (job 23601306). The built-in table is chosen by the
  device's name (`DiffusionCostConfig::loadBuiltin`), so each node type is
  priced from its own measurement. What the table cannot price is the copies
  waiting for each other: the copies serve different dispatcher partitions,
  start their steps as their slabs arrive, and every copy's apply waits for
  the latest copy's partial. On the A100 nodes four copies with 7 ms of
  compute each had 40 ms periods that way and the step did not fall below
  the unreplicated one; the wait a copy measures and declares contains it,
  the table's price does not.
  The application names no copy count: it declares what it can keep
  consistent (`replicaMax`) and the
  device bytes each further peer adds to a copy (`replicaExtraBytesPerPeer`,
  priced into the ledger's check as `replicaExtraFor(k)`), and the balancer
  chooses. The copy is made only if the
  step's critical path, the fullest bin anywhere with those loads, drops by
  more than `eps * mean`. A copy is paid for every step, so a copy that
  relieves a bin other than the fullest, or leaves another bin as full as
  the one it relieves, buys nothing: eight experts on eight GPUs, each 0.7
  to 1.4 times the mean, got two copies that lowered nothing (+29% in bf16
  training, +35% in bf16 serving); on Mixtral-8x7B's real routing (hottest
  expert 1.27 times the mean, the next 1.18) a copy that lowered its own
  bin but not the fullest cost +15% in training and, where the copies
  exchange nothing, gained 10% in serving: the declared cost is what tells
  the two apart. The declined copy is printed with its exchange and the two
  loads. Every copy's load becomes the share; the staged bytes of the
  objective grow by the object's bytes.
- `k > 1`: one retire record for the highest-ranked copy (so an
  application that partitions its work over ranks `0..k-1` contiguously
  keeps a contiguous partition), when the group fits fewer copies,
  `L_g / (k - 1) <= T (1 - eps)`, or when the exchanges cost the critical
  path more than the sharing saves it (the fullest bin without the copy
  and its exchanges is lower by more than `eps * mean`). Rank 0 is never
  retired: it keeps the object's identity for the application. The
  remaining copies' loads become `L_g / (k - 1)` plus their exchanges. The
  margin keeps the two decisions from alternating on a load near `T`.
  When the group declares partitions and no exchange, the bins of the host
  that originates most of the copy's block of partitions are tried first
  (`ProcStats::partition`, set by `LBManager::setPartition` on the feeding
  PEs: copy `r` of `k` serves `[P r / k, P (r + 1) / k)`), and the other
  hosts' bins only when none of those can take the copy, so a copy that
  exchanges nothing sits with its inputs. Measured 2026-11-07 on two A100
  nodes serving SmartMoE's routing: every copy of the hot expert on one host
  sent 100 MB a step each way over the link, the step 21.8 ms against
  FasterMoE's 16.0. With an exchange the fan-priced cost decides as before
  (copies on one host exchange over NVLink).

Clone sources and retiring copies are left out of the shedding that
follows, and a copy is never placed on a bin that holds another copy of its
group. The pass runs before the early return on "already within the
threshold", so a group can shrink when nothing else needs to move. One line
is printed per decision (`CharmLB> ShedLB: replicate ...` / `retire ...`).

## What an application implements

`tests/charm++/load_balancing/replica_test` is the reference
implementation, on the host: a 2-D array `(worker, copy)`; a worker's items
form two fixed partitions (even and odd items), and copy `r` of a replicated
worker computes partition `r`, sends its partial to the other copy and adds
the two in partition order, so the copies' running sums stay identical. An
unreplicated worker computes both partitions and adds them in the same
order, which keeps the sum identical whether or not the balancer
replicated. `ckClone` records the decision; at the end of the step `(w, 0)`
inserts `(w, 1)` on the destination and sends it the sum so far, and both
serve one partition from the next step. `ckRetire` is announced in the
copy's partial, after which the copy leaves its loop and destroys itself
from a separate entry method.

What a copy costs the application is the consistency protocol, and that
sets where the balancer's decision pays. A copy that must agree with its
peers every step (a training expert of a mixture-of-experts layer, which
exchanges a gradient partial and applies the update twice) declares the
bytes of that exchange, and at a shape where the step is short against the
exchange a copy can cost more than the load it halves. A copy that
exchanges nothing after its one state send (a serving expert, forward
only) declares no extra bytes and pays off wherever the balancer finds an
object hotter than a processor's share. The balancer sees the difference
only through the declaration and the measured load; the policy is
unchanged.

## Where in the source

- `src/ck-ldb/lbdb.h`: `LDObjData` replica fields, `LDCallbacks::clone/retire`.
- `src/ck-ldb/LBObj.h`, `LBDatabase.h/.C`, `LBManager.h`, `LBOM.h`: the
  setter chain and `Clone`/`Retire` with the id check.
- `src/ck-ldb/BaseLB.h`: `LBDecisionKind`, `LBAction`, `MigrateInfo::kind`,
  `LDStats::clones/retires`.
- `src/ck-ldb/CentralLB.C`: `createMigrateMsg`, the batch planner's last
  batch, `executeReplicaDecision`, the skips in the arrival tallies.
- `src/ck-ldb/LBMemoryContract.h`: `feasibleArrival`, the verifier's clones.
- `src/ck-ldb/ShedLB.C`: the group pass.
- `src/ck-ldb/LBSimulation.C`: decision printing.
- `src/ck-core/ckmigratable.h`, `cklocation.C`, `cklocrec.h`:
  `setLBReplica`, `ckClone`/`ckRetire`, the callback registration.
- `tests/charm++/load_balancing/replica_test`: the reference implementation.

## A round that decides nothing

An application that holds work back around a balancing round for the sake
of migrations (sends no data ahead across the round's window, so that a
migrating object has nothing of a later step in flight) pays for that even
when the round moves nothing. Every PE learns how many
decisions the round made, job-wide, before its `MigrationDone` callbacks
run: `CentralLB` records the migrate message's count (`LBMigrateMsg::n_moves`
on the broadcast path; `LBScatterMsg::totalDecisions`, carried down the
scatter tree, on the scatter path) in `LBManager::SetLastRoundDecisions`,
and `LBManager::LastRoundDecisions()` returns it (-1 before the first round).
Clones and retires are in the count, since they are entries of the same
list. Such an application registers an `AddMigrationDoneFn` callback and, on
a count of 0, releases the work it held back.

## Copies beyond two

The two-copy protocol generalises to `k <= replicaMax` copies. What the
runtime and ShedLB guarantee for it:

- An application that splits an object's work into `P` fixed partitions
  (`replicaParts`) assigns copy `r` of `k` the contiguous block of
  partitions `[floor(P r / k), floor(P (r + 1) / k))`, and the balancer
  prices each copy with its block's share of the load
  (`LDObjData::replicaShare`). Keeping the partitions fixed and combining
  partial results in a fixed order makes the result independent of `k`.
- A clone's rank is `k` (the next unused; several clones of one round take
  `k`, `k+1`, ..., and `CentralLB` keeps the ranks contiguous when the
  memory verifier refuses one) and a retire takes the highest rank (ShedLB
  does both), so the block mapping only ever splits the last block or
  merges the last two.
- Every copy must announce the same count for the same step, so the
  decision reaches all of them before the round releases any: besides the
  clone or retire record for the copy concerned, ShedLB emits a notice
  record (`LDStats::cloneNotices` / `retireNotices`, kinds
  `LB_DECISION_CLONE_NOTICE` / `RETIRE_NOTICE`) for each other member,
  delivered by the same path to `CkMigratable::ckCloneNotice(toPe, rank)` /
  `ckRetireNotice(rank)` (virtuals that do nothing by default). The notices
  of a clone the memory verifier refused are dropped with it. The copies act
  at the end of the step the round released them in; the new copy starts,
  or the retired one ends, at the next.
