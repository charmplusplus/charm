/*
 * Parallel state space search library
 *
 * Jonathan A. Booth
 * Fri Mar 28 19:01:25 CST 2003
 */

#ifndef __UIUC_CHARM_SEARCH_H
#define __UIUC_CHARM_SEARCH_H

// Grab all the charm stuff.
#include <charm++.h>

// Define the problem class. We'll be passing 'problem *' pointers around.
// The reason we do this is that we need it to keep the sub-class type
// of the 'problem *' correct rather than having it tend to revert to a
// plain old 'problem'.
//
// Of course that's sort of a pain with charm.
#include "cklibs/problem.h"

// The serial tree class priovides a way to do a full-tree search of a node
// for a depth up to depth whatever. I use this both internally in each
// interior chare, and at the leaves of the tree.
#include "cklibs/serialtree.h"

// Draw in the charm interface declarations.
#include "cklibs/search.decl.h"

// This class definition is used to pass around the solution and nodes and
// chares when a search is done.
class searchResults : public CMessage_searchResults {
 public:
  // These three are real data that gets sent over the network
  CkGroupID GroupID;
  unsigned int NodesExpanded;
  unsigned int CharesExpanded;

  // This is dummy data.
  problem * Solution;

  // Create me. Set solution to null
  searchResults() : Solution(NULL) {};

  // Delete me, and the solution
  ~searchResults() {
    if ( Solution ) { delete Solution; }
  }

  // The unpack function gets called on the remote processor after the message
  // has been received. I use it to fill the solution pointer with a pointer
  // to a local copy of the solution. I then tell the group that it can go
  // jump out a window.
  static searchResults * unpack(void *inbuf);
};

// The idaStar search, as well as fakeing the astar search are included
// here. In this file are all the definitions you need to run an astar
// or idastar search.
#include "cklibs/idastar.h"

// If I had another search type like branch and bound, you'd include
// it's header file here. You'd write the code in branchandbound.C
// add that to the makefile and be happy.
// #include "branchandbound.h"

#endif /* __UIUC_CHARM_SEARCH_H */
