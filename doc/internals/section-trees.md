# Section multicast trees: build, rebuild, retire, free

How `CkMulticastMgr` (`src/ck-core/ckmulticast.C`, `.h`, `.ci`) keeps the
spanning tree of an array or group section alive across element migration,
and when it frees a tree it has replaced. The user-facing side (section
proxies, `resetSection`, delegation) is in the manual.

## Entries

Every PE in a section's spanning tree holds one `mCastEntry` per tree it
takes part in. The entry records its parent (`parentGrp`, null at the root),
its children (`children`, one `CkSectionInfo` per child entry on another
PE), the local members (`localElem`, or `localGrpElem` for a group
section), the root's id (`rootSid`), buffers for messages that arrive before
the tree is built (`msgBuf`, `packetBuf`), the reduction state (`red`), and
a flag: `COOKIE_NOTREADY` while the subtree is being built, `COOKIE_READY`
when the children have reported in (`recvCookie`, `childrenReady`), and
`COOKIE_OLD` once the tree is replaced.

A section proxy's cookie (`CkSectionInfo`) names the root entry by PE and
pointer. A multicast is handed to the root (`sendToSection`), which copies it
to each child (`recvMsg`, or `recvPacket` for a message above
`split_threshold`) and delivers it to the local members (`sendToLocal`).

## Replacing a tree

Two events replace a tree:

- **Rebuild** (`rebuild`, from `sendToSection` when `needRebuild` says
  enough members moved): the root PE allocates a new entry, links it behind
  the old one (`oldc`/`newc`), marks the old one obsolete and tears it down
  (`teardown` to each child, recursively), and builds the new tree
  (`initCookie`, `setup`).
- **Root migration** (`resetSection` on the new PE): the new PE allocates a
  new root entry, remembers the old root (`oldtree`), asks the old PE for the
  reduction number and retires the old root in one message
  (`retrieveCookie`, `retire`). `retire` is `teardown` plus setting the old
  root's `rootSid` to the new root, so reductions arriving at the old tree
  are forwarded to the new root (`recvRedMsg`, obsolete branch). A root
  retired while its own tree was still being built keeps that `rootSid`
  (`setup`) and tears its children down when they report (`childrenReady`).

An obsolete entry still delivers the multicasts that were already inside the
old tree: `recvMsg` on an obsolete entry forwards to the old children and to
the local members exactly as on a live one. Reconverse does not order
messages between a pair of PEs, so a multicast can arrive at a child after
the teardown that made it obsolete.

## Freeing the old tree

The old tree is freed after the first reduction completes over the new tree
(`recvRedMsg`, `currentTreeUp`): the root sends `freeup` to the old root
entry on its own PE (`oldc` chain) or on the old PE (`oldtree`). Freeing is
by count, not by that message alone, because multicasts may still be in
flight inside the old tree:

- Every entry counts the multicasts it has delivered (`received`, in
  `delivered`) and the ones it has forwarded to each child (`sentToChild`,
  same order as `children`). A root also counts the multicasts handed to it
  (`injected`, in `sendToSection`).
- `freeup(cookie, expected)` records `expected`, the number of multicasts the
  parent forwarded to this entry, or the root's own `injected` when the
  argument is -1, and sets `freeRequested`.
- `tryFree` deletes the entry when `received == expected`. Only then are its
  own `sentToChild` counts final, so only then does it send `freeup` with
  those counts to its children. `tryFree` runs from `freeup` and from
  `delivered`, whichever completes the count last.

Invariant: an entry is deleted only after the last multicast its parent
forwarded to it has been delivered through it, so no message reaches a freed
entry. `sendToLocal` asserts that an entry with no local members has no
parent: only a root, current or retired, is in that state.

Not covered by the count: a section proxy copy on another PE that keeps
sending to a root that has since retired (`sendToSection` with the cookie's
PE not the local PE, "Multicast not optimized" warning). Such a proxy must be
reset after the root migrates.
