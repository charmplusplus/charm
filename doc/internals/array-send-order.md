# Sending to an array element before the manager exists

A send to a chare array element is routed on the sending PE: `CkSendMsgArray`
(src/ck-core/ckarray.C) looks up the PE's copy of the array's manager group
(`CkArray`, which holds the `CkLocMgr`) and asks it where the element is, then
forwards the message to that PE or to the element's home. That lookup needs
the manager group to exist on the sending PE.

An array manager is a group, created by a broadcast from the PE that called
`ckNew`. Until every PE has processed that broadcast, a PE can hold a message
that names the array (a `CkCallback` into it, a proxy in a marshalled
parameter) before it holds the manager. Converse does not order messages
between a pair of PEs, and reconverse in particular delivers two messages for
one PE in either order when different threads take them off the network
(contrib/reconverse/README.md, "Messages to one destination can be
reordered"). So a group entry method on PE 2 can run, and send into the
array, before PE 2 has created the array's manager.

## The receive side already waits

A message that arrives for a group or an array manager that does not exist
yet on the receiving PE is held in the group table
(`_lookupGroupAndBufferIfNotThere`, src/ck-core/ck.C) and re-enqueued when
`CkCreateLocalGroup` creates that group (`CkReadyEntry`). Array element
messages arriving at a PE without the manager take the same path through
`_processArrayEltMsg`.

## The send side holds the message

`CkSendMsgArray` and `CkSendMsgArrayInline` do the same when the sending PE
has no local branch of the manager: the send (entry index, message, element
index, options) goes into a per-PE table keyed by the manager's group id
(`deferredArraySends` in ckarray.C). `CkCreateLocalGroup` calls
`_ckArrayFlushDeferredSends` after the new group is ready, and every held
send for that group id is re-issued through `CkSendMsgArray`, in the order it
was held. An inline send that was held is re-issued as a queued send; there
is no manager to deliver inline to at the time of the call.

The table is per PE and keyed by group id, so a PE that creates many arrays
pays one hash lookup per group creation and nothing else.

## Quiescence

A send counts toward quiescence (`CpvAccess(_qd)->create()`) in
`_prepareOutgoingArrayMsg`, at the moment the message is actually handed to
the machine layer, after the manager lookup. A held send is therefore not yet
counted. Nothing is lost: the manager's creation broadcast was counted by
`_createGroup` (`create(CkNumPes()-1)`) and is still unprocessed on this PE
while the send is held. The flush runs inside the handler that processes that
creation message, before control returns to the scheduler, so no quiescence
sample can see the creation processed and the held send still uncounted.

A held send cannot wait forever: the manager's creation broadcast is a
reliable message to every PE, and the flush is unconditional when it is
processed. No timeout is needed.

## Where in the source

- src/ck-core/ckarray.C: `DeferredArraySend`, `deferArraySend`,
  `_ckArrayFlushDeferredSends`, `CkSendMsgArray`, `CkSendMsgArrayInline`,
  `CkArray::sendMsg`.
- src/ck-core/ck.C: `CkCreateLocalGroup` (calls the flush after
  `CkReadyEntry`), `_lookupGroupAndBufferIfNotThere`, `_processArrayEltMsg`,
  `_prepareOutgoingArrayMsg` (quiescence count), `_createGroup`.
- First sighting and the ordering analysis: charm issue #4088.
