// File: cancel.h
// CancelList stores cancellation of event notices processed by a strat
// Last Modified: 5.29.01 by Terry L. Wilmarth

#ifndef CANCEL_H
#define CANCEL_H

class CancelNode { // CancelList is comprised of these nodes
 public:
  int timestamp;   // timestamp of event to be cancelled
  eventID evID;    // eventID of event to be cancelled
  CancelNode *next;
  CancelNode() { timestamp = -1; next = NULL; }
  CancelNode(int ts, eventID e) { timestamp = ts; evID = e; next = NULL; }
  void dump() {
    CkPrintf("[timestamp=%d ", timestamp);
    evID.dump();
    CkPrintf("] ");
  }
  void pup(PUP::er &p) { p(timestamp); evID.pup(p); }
};

class CancelList { // List to store incoming cancellations
 public:
  int count, earliest;          // count of items in list; earliest timestamp 
  CancelNode *cancellations, *current; // cancellations is the list of nodes;
                                      // current points somewhere in the list
  CancelList() { count = 0; earliest = -1; cancellations = current = NULL; }
  void Insert(int ts, eventID e);    // inserts an event at beginning of list
  CancelNode *GetItem(int eGVT);     // returns a pointer to a node in list;
                                     // uses current to cycle through nodes
  void RemoveItem(CancelNode *item);      // remove a node that was cancelled
  int IsEmpty();                          // tests if CancelList is empty
  void dump(int pdb_level);               // print the list contents
  void pup(PUP::er &p);                   // pup the entire cancel list
};

#endif
