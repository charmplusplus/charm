// File: eqheap.C
// Defines: EqHeap class and methods; used for the unexecuted portion of the
//          eventQueue for rapid insertion/deletion (hopefully)
// Last Modified: 5.31.01 by Terry L. Wilmarth

#include "pose.h"

// Insert event e in this subheap; designed to find insertion position
// quickly, at the cost of creating unbalanced or high & narrow heaps
void HeapNode::insert(Event *e)
{
  // PRE: this != NULL && e->timestamp > this->timestamp
  HeapNode *eh;
  if (left == NULL) {  // make it the left subheap
    eh = new HeapNode(e, 1, NULL, NULL);
    left = eh;
    subheapsize += 1;
  }
  else if (right == NULL) {  // make it the right subheap
    eh = new HeapNode(e, 1, NULL, NULL);
    right = eh;
    subheapsize += 1;
  }
  else if (left->e->timestamp >= e->timestamp) {
    // make it the root of the left subtree
    eh = new HeapNode(e, left->subheapsize+1, left, NULL);
    left = eh;
    subheapsize += 1;
  }
  else if (right->e->timestamp >= e->timestamp) {
    // make it the root of the right subtree
    eh = new HeapNode(e, right->subheapsize+1, right, NULL);
    right = eh;
    subheapsize += 1;
  }
  else if (left->subheapsize < right->subheapsize) {
    // insert it in the smaller of the left or right subtrees
    subheapsize += 1;
    left->insert(e);
  }
  else {
    subheapsize += 1;
    right->insert(e);
  }
}

// Join this heap with h and return the new heap; uses quickest join method 
// possible at expense of creating unbalanced tree
HeapNode *HeapNode::conjoin(HeapNode *h)
{
  // CAREFUL:  this version allows "this" to be NULL
  if (!this)
    return h;
  else if (!h)
    return this;
  else if (e->timestamp < h->e->timestamp) {  // make this the root
    // conjoin this's kids into this's left and make this's right h
    left = left->conjoin(right);
    right = h;
    subheapsize += h->subheapsize;
    return this;
  }
  else {  // make h the root
    // conjoin h's kids into h's right and make h's left this
    h->right = h->left->conjoin(h->right);
    h->left = this;
    h->subheapsize += subheapsize;
    return h;
  }
}

// Remove node matching evID and timestamp in this subheap
int HeapNode::remove(eventID evID, int timestamp)
{
  // PRE: top of this subheap is not the node to be removed; this != NULL
  int found = 0;                                 // return status
  HeapNode *tmp;
  if (left) {                                    // search left subheap first
    if (timestamp < left->e->timestamp)          // subheap elements too high
      found = 0;                                 // set return status
    else if ((timestamp == left->e->timestamp) && (evID == left->e->evID)) {
      // found element on top
      tmp = left;                                // save a pointer to it
      left = left->left->conjoin(left->right);   // remove it from heap
      subheapsize--;
      tmp->left = tmp->right = NULL;             // foil recursive destructor
      delete tmp;                                // delete it
      found = 1;                                 // set return status
    }
    else if (timestamp >= left->e->timestamp) {  // need to look deeper
      found = left->remove(evID, timestamp);     // set return status
      if (found)                                 // must decrement heap size
	subheapsize--;                           // as we pop up from recursion
    }
  }
  if (found)                                     // found in left subheap
    return 1;                                    // so exit with status 1
  else if (right) {                              // search in right subheap
    if (timestamp < right->e->timestamp)         // subheap elements too high
      found = 0;                                 // set return status
    else if ((timestamp == right->e->timestamp) && (evID == right->e->evID)) {
      // founf element on top
      tmp = right;                               // save a pointer to it
      right = right->left->conjoin(right->right);// remove it from heap
      subheapsize--;                 
      tmp->left = tmp->right = NULL;             // foil recursive destructor
      delete tmp;                                // delete it
      found = 1;                                 // set return status
    }
    else if (timestamp >= right->e->timestamp) { // need to look deeper
      found = right->remove(evID, timestamp);    // set return status
      if (found)                                 // must decrement heap size
	subheapsize--;                           // as we pop up from recursion
    }
  }
  return found;                                  // exit with found status
}

// Recursively prints entire subheap
void HeapNode::dump(int pdb_level)
{
  pdb_indent(pdb_level);
  CkPrintf("[HpNd: sz=%d event=(%d.%d.%d) ", subheapsize, e->evID.id, e->evID.pe);
  if (left)
    left->dump(pdb_level);
  else CkPrintf("[NULL] ");
  if (right)
    right->dump(pdb_level);
  else CkPrintf("[NULL]");
  pdb_indent(pdb_level);
  CkPrintf(" end HpNd]\n");
}

// Recursively packs/sizes entire subheap; DOES NOT UNPACK HEAP!!!
void HeapNode::pup(PUP::er &p)
{
  // PRE: assumes this node is not NULL
  if (!p.isUnpacking()) {
    e->pup(p);
    if (left) left->pup(p);
    if (right) right->pup(p);
  }
  else CkPrintf("ERROR: HeapNode::pup: Use only for packing/sizing.\n");
}

// Insert event e in heap; low timestamps at top of heap
void EqHeap::InsertEvent(Event *e)
{
  HeapNode *eh;

  if (top == NULL) {  // make the top of the heap
    top = new HeapNode(e, 1, NULL, NULL);
  }
  else if (top->subheapsize < 1)
    CkPrintf("ERROR: EqHeap::InsertEvent: top of heap corrupted\n");
  else if (e->timestamp <= top->e->timestamp) {  // insert at top of heap
    if (top->subheapsize == 1)                   // only one node in heap
      top = new HeapNode(e, 2, top, NULL);    // make old top into left subheap
    else if (top->left && top->right) {          // full(ish) heap
      // try to improve the balance by one
      if (top->left->subheapsize < top->right->subheapsize) {
	eh = new HeapNode(e, top->subheapsize+1, top, top->right);
	top->subheapsize -= top->right->subheapsize;
	top->right = NULL;
	top = eh;
      }
      else {
	eh = new HeapNode(e, top->subheapsize+1, top->left, top);
	top->subheapsize -= top->left->subheapsize;
	top->left = NULL;
	top = eh;
      }
    }
    else if (top->left) {           // at least keep the balance about the same
      eh = new HeapNode(e, top->subheapsize+1, top->left, top);
      top->subheapsize = 1;
      top->left = NULL;
      top = eh;
    }
    else if (top->right) {          // at least keep the balance about the same
      eh = new HeapNode(e, top->subheapsize+1, top, top->right);
      top->subheapsize = 1;
      top->right = NULL;
      top = eh;
    }
  }
  else                              // insert somewhere below the top node
    top->insert(e);
  heapSize++;
}

// Returns event at top of heap if one exists, null otherwise; deletes top
// node in heap, conjoining left and right subheaps
Event *EqHeap::GetAndRemoveTopEvent()
{
  // PRE: top is not NULL
  HeapNode *tmp = top;
  Event *result;

  if (top == NULL) CkPrintf("ERROR: GetAndRemoveTopEvent has NULL heap.\n");
  else if ((top != NULL) && (top->subheapsize < 1))
    CkPrintf("ERROR: EqHeap::GetAndRemoveTopEvent: corrupt top\n");
  top = top->left->conjoin(top->right);
  result = tmp->e;
  tmp->e = NULL;
  tmp->left = tmp->right = NULL;
  delete(tmp);
  heapSize--;
  return result;
}

// Delete the node and event in the heap corresponding to evID and timestamp; 
// returns 1 if an event was successfully deleted, 0 if the event was not
// found in the heap
int EqHeap::DeleteEvent(eventID evID, int timestamp)
{
  int result;
  if (!top || (timestamp < top->e->timestamp))
    return 0;
  else if ((top->e->timestamp == timestamp) && (top->e->evID == evID)) {
    HeapNode *tmp = top;                        // top is the match
    top = top->left->conjoin(top->right);       // remove node from heap
    tmp->left = tmp->right = NULL;              // foil recursive destructor
    delete tmp;                                 // delete it
    heapSize--;
    return 1;                                   // return status
  }
  else {                                        // search deeper in heap
    result = top->remove(evID, timestamp);      // return status
    if (result) heapSize--;
    return result;
  }
}

// Pups entire heap relying on recursive HeapNode::pup
void EqHeap::pup(PUP::er &p) { 
  int i=0, hs;
  Event *e;

  if (p.isUnpacking()) {  // UNPACK
    p(hs);
    top = NULL;
    while (i < hs) {
      e = new Event();
      e->pup(p);
      InsertEvent(e);
      i++;
    }
  }
  else {  // PACK / SIZE
    p(heapSize);
    if (top) top->pup(p); // HeapNode::pup recursively packs/sizes events
  }
}

// Dumps entire heap relying on recursive HeapNode::dump
void EqHeap::dump(int pdb_level)
{
  pdb_indent(pdb_level);
  CkPrintf("[EQHEAP: ");
  if (top)
    top->dump(pdb_level+1);
  else
    CkPrintf("NULL");
  pdb_indent(pdb_level);
  CkPrintf(" end EQHEAP]\n");
}

