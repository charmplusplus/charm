# Routing a send to an array element

A message to a chare array element is routed on the sending PE. `CkSendMsgArray`
(src/ck-core/ckarray.C) looks up the sending PE's copy of the array's manager
group (`CkArray`, which holds the `CkLocMgr`), asks it where the element is,
and forwards the message to that PE or to the element's home PE. The lookup
needs the manager group to exist on the sending PE.

An array manager is a group, created by a broadcast from the PE that called
`ckNew`. Messages between two PEs are not delivered in order (Converse makes
no such promise, and reconverse's shared per-process receive queue, drained by
every PE thread, does reorder them), so a PE can receive a message that names
an array, such as a `CkCallback` into it or a proxy in a marshalled parameter,
before it has processed the broadcast that creates the array's manager. Both
sides of array messaging allow for this.

## Receiving for a group that does not exist yet

A message that arrives for a group, or for an array whose manager group is
not yet created on the receiving PE, is held in that PE's group table
(`_lookupGroupAndBufferIfNotThere`, src/ck-core/ck.C). When
`CkCreateLocalGroup` creates the group, `CkReadyEntry` re-enqueues the held
messages. Array element messages arriving at a PE without the manager take
this path through `_processArrayEltMsg`.

## Sending from a PE that has no manager yet

`CkSendMsgArray`, `CkSendMsgArrayInline` and `CProxyElement_ArrayBase::ckSend`
(the path of a generated proxy call such as `a[i].foo()`) hold the send when
the sending PE has no local branch of the manager. The send (entry index, message,
element index, options) goes into a per-PE table keyed by the manager's group
id (`deferredArraySends`, ckarray.C). `CkCreateLocalGroup` calls
`_ckArrayFlushDeferredSends` after the new group is ready, and every held
send for that group id is re-issued through `CkSendMsgArray` in the order it
was held. A held inline send (`CkSendMsgArrayInline`, or `ckSend` with
`CK_MSG_INLINE`) is re-issued as a queued send, since there was no manager to
deliver inline to when it was called.

The table is per PE and keyed by group id. A group creation costs one hash
lookup in it, and nothing else.

## Quiescence

A send counts toward quiescence (`CpvAccess(_qd)->create()`) in
`_prepareOutgoingArrayMsg`, when the message is handed to the machine layer,
which is after the manager lookup. A held send is therefore not counted yet.
The manager's creation broadcast is counted by `_createGroup`
(`create(CkNumPes()-1)`) and is unprocessed on this PE for as long as the send
is held; the flush runs inside the handler that processes that broadcast,
before control returns to the scheduler. So at every point where a
quiescence sample can be taken, either the creation message or the re-issued
send is counted as outstanding.

A held send cannot wait indefinitely: the creation broadcast reaches every
PE, and the flush is unconditional when it is processed.

## Where in the source

- src/ck-core/ckarray.C: `DeferredArraySend`, `deferArraySend`,
  `_ckArrayFlushDeferredSends`, `CkSendMsgArray`, `CkSendMsgArrayInline`,
  `CProxyElement_ArrayBase::ckSend`, `CkArray::sendMsg`.
- src/ck-core/ck.C: `CkCreateLocalGroup` (calls the flush after
  `CkReadyEntry`), `_lookupGroupAndBufferIfNotThere`, `_processArrayEltMsg`,
  `_prepareOutgoingArrayMsg` (quiescence count), `_createGroup`.
- contrib/reconverse/README.md, "Messages to one destination can be
  reordered", for the delivery-order property this design allows for.
