/*
 * Parallel state space search library
 *
 * Jonathan A. Booth
 * Fri Mar 28 19:01:25 CST 2003
 */

#include "cklibs/search.h"
#include "cklibs/search.decl.h"

/* ************************************************************************
 *
 * ************************************************************************ */
// The unpack function gets called on the remote processor after the message
// has been received. I use it to fill the solution pointer with a pointer
// to a local copy of the solution. I then tell the group that it can go
// jump out a window.
searchResults * searchResults::unpack(void *inbuf) {
  searchResults *me = CMessage_searchResults::unpack(inbuf);

  // Grab a copy of the solution.
  CProxy_idaStarGroup group(me->GroupID);
  idaStarGroup *local = group.ckLocalBranch();
  me->Solution = local->Solution->clone();

  // Don't need the group anymore.
  group.Terminate();

  // Return the new message.
  return me;
}

// search.def.h is a local file and not placed in cklibs.
#include "search.def.h"
