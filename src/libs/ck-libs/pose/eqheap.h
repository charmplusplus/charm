// File: eqheap.h
// Defines: EqHeap class and methods; used for the unexecuted portion of the
//          eventQueue for rapid insertion/deletion (hopefully)
// Last Modified: 5.31.01 by Terry L. Wilmarth

#ifndef EQHEAP_H
#define EQHEAP_H

class HeapNode  // EqHeap contains nodes of this type
{
 public:
  int subheapsize;         // size of this subheap, including this node
  Event *e;                // the event stored here
  HeapNode *left, *right;  // left and right subheaps
  HeapNode() { subheapsize = 0; e = NULL; left = right = NULL; }
  HeapNode(Event *ev, int sz, HeapNode *l, HeapNode *r) { 
    subheapsize = sz; e = ev; left = l; right = r; 
  }
  ~HeapNode() { if (left) delete left; if (right) delete right; delete e; }
  void insert(Event *e);                    // insert event in heap
  HeapNode *conjoin(HeapNode *h);           // join this heap with h
  int remove(eventID evID, int timestamp);  // remove node matching evID
  void dump(int pdb_level);                 // print this node and subheaps
  void pup(PUP::er &p);                     // packs/sizes recursively
};

class EqHeap {  // Heap of events for unexecuted portion of eventQueue
 public:
  int heapSize;   // size of heap
  HeapNode *top;  // top of heap

  EqHeap() { heapSize = 0; top = NULL; }
  ~EqHeap() { if (top) delete top; }    
  void InsertEvent(Event *e);                    // insert e in heap
  Event *GetAndRemoveTopEvent();           // return top event, delete top node
  int DeleteEvent(eventID evID, int timestamp);  // delete evID event from heap
  void dump(int pdb_level);                      // print entire heap
  void pup(PUP::er &p);                          // pup entire heap
};

#endif
