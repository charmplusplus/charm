/// Heap structure used for unexecuted portion of the eventQueue
#include "pose.h"

/// Insert event in heap
void HeapNode::insert(Event *e)
{
  CmiAssert(this != NULL);
  CmiAssert(e->timestamp > this->e->timestamp);
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
  else if (left->e->timestamp >= e->timestamp) { // make root of left subtree
    eh = new HeapNode(e, left->subheapsize+1, left, NULL);
    left = eh;
    subheapsize += 1;
  }
  else if (right->e->timestamp >= e->timestamp) { // make root of right subtree
    eh = new HeapNode(e, right->subheapsize+1, right, NULL);
    right = eh;
    subheapsize += 1;
  }
  else if (left->subheapsize < right->subheapsize) { // insert in left subtree
    subheapsize += 1;
    left->insert(e);
  }
  else { // insert in right subtree
    subheapsize += 1;
    right->insert(e);
  }
}

/// Join this heap with h
HeapNode *HeapNode::conjoin(HeapNode *h)
{
  if (!this) return h;
  else if (!h) return this;
  else if (e->timestamp < h->e->timestamp) { // make this the root
    // conjoin this's kids into this's left and make this's right h
    left = left->conjoin(right);
    right = h;
    subheapsize += h->subheapsize;
    return this;
  }
  else { // make h the root
    // conjoin h's kids into h's right and make h's left this
    h->right = h->left->conjoin(h->right);
    h->left = this;
    h->subheapsize += subheapsize;
    return h;
  }
}

/// Remove heap node matching evID
int HeapNode::remove(eventID evID, POSE_TimeType timestamp)
{
  CmiAssert(this != NULL);
  CmiAssert(!(this->e->evID == evID));
  int found = 0; // return status
  HeapNode *tmp;
  if (left) { // search left subheap first
    if (timestamp < left->e->timestamp) found = 0; // subheap elements too high
    else if ((timestamp == left->e->timestamp) && (evID == left->e->evID)) {
      // found element on top
      tmp = left; // save a pointer to it
      left = left->left->conjoin(left->right); // remove it from heap
      subheapsize--;
      tmp->left = tmp->right = NULL; // foil recursive destructor
      delete tmp; // delete it
      found = 1;
    }
    else if (timestamp >= left->e->timestamp) { // need to look deeper
      found = left->remove(evID, timestamp);
      if (found) subheapsize--; // must decrement heap size
    }
  }
  if (found) return 1; // found in left subheap; exit with status 1
  else if (right) { // search in right subheap
    if (timestamp < right->e->timestamp) found = 0; //subheap elements too high
    else if ((timestamp == right->e->timestamp) && (evID == right->e->evID)) {
      // found element on top
      tmp = right; // save a pointer to it
      right = right->left->conjoin(right->right); // remove it from heap
      subheapsize--;                 
      tmp->left = tmp->right = NULL; // foil recursive destructor
      delete tmp; // delete it
      found = 1; 
    }
    else if (timestamp >= right->e->timestamp) { // need to look deeper
      found = right->remove(evID, timestamp); // set return status
      if (found) subheapsize--; // must decrement heap size
    }
  }
  return found; // exit with found status
}

/// Find maximum element
POSE_TimeType HeapNode::findMax()
{
  POSE_TimeType max = e->timestamp, leftTS, rightTS;
  leftTS = rightTS = POSE_UnsetTS;
  if (left) leftTS = left->findMax();
  if (right) rightTS = right->findMax();
  if (max < leftTS) max = leftTS;
  if (max < rightTS) max = rightTS;
  return max;
}

/// Dump all data fields in entire subheap
void HeapNode::dump()
{
  CkPrintf("[HpNd: sz=%d event=(", subheapsize);
  e->evID.dump();
  CkPrintf(") ");
  if (left) left->dump();
  else CkPrintf("[NULL] ");
  if (right) right->dump();
  else CkPrintf("[NULL]");
  CkPrintf(" end HpNd]\n");
}

/// Pack/unpack/sizing operator
/** Recursively packs/sizes entire subheap; DOES NOT UNPACK HEAP!!! */
void HeapNode::pup(PUP::er &p)
{
  CmiAssert(this != NULL);
  CmiAssert(!p.isUnpacking());
  e->pup(p);
  if (left) left->pup(p);
  if (right) right->pup(p);
}

/// Insert event e in heap with low timestamps at top of heap
void EqHeap::InsertEvent(Event *e)
{
  HeapNode *eh;

  CmiAssert((top == NULL) || (top->subheapsize > 0));
  if (top == NULL) // make the top of the heap
    top = new HeapNode(e, 1, NULL, NULL);
  else if (e->timestamp <= top->e->timestamp) { // insert at top of heap
    if (top->subheapsize == 1) // only one node in heap
      top = new HeapNode(e, 2, top, NULL); // make old top into left subheap
    else if (top->left && top->right) { // full(ish) heap
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
    else if (top->left) { // at least keep the balance about the same
      eh = new HeapNode(e, top->subheapsize+1, top->left, top);
      top->subheapsize = 1;
      top->left = NULL;
      top = eh;
    }
    else if (top->right) { // at least keep the balance about the same
      eh = new HeapNode(e, top->subheapsize+1, top, top->right);
      top->subheapsize = 1;
      top->right = NULL;
      top = eh;
    }
  }
  else // insert somewhere below the top node
    top->insert(e);
  heapSize++;
}

/// Return event on top of heap, deleting it from the heap
Event *EqHeap::GetAndRemoveTopEvent()
{
  CmiAssert(top != NULL);
  CmiAssert(top->subheapsize > 0);
  HeapNode *tmp = top;
  Event *result;

  top = top->left->conjoin(top->right);
  result = tmp->e;
  tmp->e = NULL;
  tmp->left = tmp->right = NULL;
  delete(tmp);
  heapSize--;
  return result;
}

/// Delete event from heap
int EqHeap::DeleteEvent(eventID evID, POSE_TimeType timestamp)
{
  int result;
  if (!top || (timestamp < top->e->timestamp))
    return 0;
  else if ((top->e->timestamp == timestamp) && (top->e->evID == evID)) {
    HeapNode *tmp = top; // top is the match
    top = top->left->conjoin(top->right); // remove node from heap
    tmp->left = tmp->right = NULL; // foil recursive destructor
    delete tmp;
    heapSize--;
    return 1;
  }
  else { // search deeper in heap
    result = top->remove(evID, timestamp);
    if (result) heapSize--;
    return result;
  }
}

/// Find maximum element
POSE_TimeType EqHeap::FindMax()
{
  if (top) return top->findMax();
  return POSE_UnsetTS;
}

/// Pack/unpack/sizing operator
void EqHeap::pup(PUP::er &p) { 
  int i=0, hs;
  Event *e;

  if (p.isUnpacking()) {  // UNPACK entire heap right here
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

/// Dump entire heap
void EqHeap::dump()
{
  CkPrintf("[EQHEAP: ");
  if (top) top->dump();
  else CkPrintf("NULL");
  CkPrintf(" end EQHEAP]\n");
}

/// Check validity of data fields
void EqHeap::sanitize()
{
}
