/// Heap structure used for unexecuted portion of the eventQueue
#include "pose.h"

//#define EH_SANITIZE 1

/// Insert event in heap
void HeapNode::insert(Event *e)
{
  if(pose_config.deterministic)
    {
      insertDeterministic(e);
    }
  else
    {
#ifdef EH_SANITIZE
      sanitize();
#endif
      CmiAssert(this != NULL);
      CmiAssert(e->timestamp > this->e->timestamp || (e->timestamp == this->e->timestamp && e->evID >= this->e->evID));
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
      else if (e->timestamp < left->e->timestamp || (e->timestamp == left->e->timestamp && e->evID <= left->e->evID)) { // make root of left subtree
	eh = new HeapNode(e, left->subheapsize+1, left, NULL);
	left = eh;
	subheapsize += 1;
      }
      else if (e->timestamp < right->e->timestamp || (e->timestamp == right->e->timestamp && e->evID <= right->e->evID)) { // make root of right subtree
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
#ifdef EH_SANITIZE
      sanitize();
#endif
    }
}

/// Insert event in heap deterministically
void HeapNode::insertDeterministic(Event *e)
{
#ifdef EH_SANITIZE
  sanitize();
#endif
  CmiAssert(this != NULL);
  CmiAssert(e->timestamp > this->e->timestamp || (e->timestamp == this->e->timestamp && e->evID >= this->e->evID));
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
  else if ((e->timestamp < left->e->timestamp) ||
	   ((e->timestamp == left->e->timestamp) &&
	    (e->evID <= left->e->evID))) { // make root of left subtree
    eh = new HeapNode(e, left->subheapsize+1, left, NULL);
    left = eh;
    subheapsize += 1;
  }
  else if ((e->timestamp < right->e->timestamp) ||
	   ((e->timestamp == right->e->timestamp) &&
	    (e->evID <= right->e->evID))) { // make root of right subtree
    eh = new HeapNode(e, right->subheapsize+1, right, NULL);
    right = eh;
    subheapsize += 1;
  }
  else if (left->subheapsize < right->subheapsize) { // insert in left subtree
    subheapsize += 1;
    left->insertDeterministic(e);
  }
  else { // insert in right subtree
    subheapsize += 1;
    right->insertDeterministic(e);
  }
#ifdef EH_SANITIZE
  sanitize();
#endif
}

/// Join this heap with h
HeapNode *HeapNode::conjoin(HeapNode *h)
{
#ifdef EH_SANITIZE
  sanitize();
#endif
#ifdef EH_SANITIZE
  if (h) h->sanitize();
#endif
  if (!this) return h;
  else if (!h) return this;
  else if (((pose_config.deterministic) && (e->timestamp < h->e->timestamp) ||
	    ((e->timestamp == h->e->timestamp) && (e->evID <= h->e->evID))) ||
	   (e->timestamp < h->e->timestamp || (e->timestamp == h->e->timestamp && e->evID <= h->e->evID)))
    { 
      // conjoin this's kids into this's left and make this's right h
      if (!left) left = right;
      else left = left->conjoin(right);
      right = h;
      subheapsize += h->subheapsize;
#ifdef EH_SANITIZE
      sanitize();
#endif
      return this;
    }
  else { // make h the root
    // conjoin h's kids into h's right and make h's left this
    if (h->left) h->right = h->left->conjoin(h->right);
    h->left = this;
    h->subheapsize += subheapsize;
#ifdef EH_SANITIZE
    h->sanitize();
#endif
    return h;
  }
}

/// Remove heap node matching evID
int HeapNode::remove(eventID evID, POSE_TimeType timestamp)
{
#ifdef EH_SANITIZE
    sanitize();
#endif
  CmiAssert(this != NULL);
  CmiAssert(!(this->e->evID == evID));
  int found = 0; // return status
  HeapNode *tmp;
  if (left) { // search left subheap first
    if (timestamp < left->e->timestamp) found = 0; // subheap elements too high
    else if ((timestamp == left->e->timestamp) && (evID == left->e->evID)) {
      // found element on top
      tmp = left; // save a pointer to it
      if (left->left)
	left = left->left->conjoin(left->right); // remove it from heap
      else left = left->right;
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
      if (right->left)
	right = right->left->conjoin(right->right); // remove it from heap
      else right = right->right;
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
#ifdef EH_SANITIZE
    sanitize();
#endif
  return found; // exit with found status
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

/// Dump all data fields in entire subheap
char *HeapNode::dumpString() {
  char *str=new char[PVT_DEBUG_BUFFER_LINE_LENGTH];
#if USE_LONG_TIMESTAMPS
  snprintf(str, PVT_DEBUG_BUFFER_LINE_LENGTH,"[HpNd: sz=%d event=%lld(%u.%d) ", subheapsize, e->timestamp, e->evID.id, e->evID.getPE());
#else
  snprintf(str, PVT_DEBUG_BUFFER_LINE_LENGTH, "[HpNd: sz=%d event=%d(%u.%d) ", subheapsize, e->timestamp, e->evID.id, e->evID.getPE());
#endif
  if (left) {
    char *lstring=left->dumpString();
    strncat(str, lstring, PVT_DEBUG_BUFFER_LINE_LENGTH);
    delete [] lstring;
  } else {
    strncat(str, "[NULL] ",  PVT_DEBUG_BUFFER_LINE_LENGTH);
  }
  if (right) {
    char *rstring=right->dumpString();
    strncat(str, rstring, PVT_DEBUG_BUFFER_LINE_LENGTH);
    delete [] rstring;
  } else {
    strncat(str, "[NULL]", PVT_DEBUG_BUFFER_LINE_LENGTH);
  }
  strncat(str, "] ", PVT_DEBUG_BUFFER_LINE_LENGTH);
  return str;
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

void HeapNode::sanitize()
{
  if (e == NULL) CkPrintf("WARNING: uninitialized HeapNode!\n");
  CmiAssert(((e==NULL) && (subheapsize==0) && (left==NULL) && (right==NULL)) ||
	    ((e!=NULL) && (subheapsize==1) && (left==NULL) && (right==NULL)) ||
	    ((e!=NULL) && (subheapsize>1)));
  if (e!=NULL) {
    e->sanitize();
    if (left) left->sanitize();
    if (right) right->sanitize();
  }
}

/// Insert event e in heap with low timestamps at top of heap
void EqHeap::InsertEvent(Event *e)
{
  HeapNode *eh;

  if(pose_config.deterministic){
    InsertDeterministic(e);
  }
  else
    {
#ifdef EH_SANITIZE
      sanitize();
#endif
      CmiAssert((top == NULL) || (top->subheapsize > 0));
      if (top == NULL) // make the top of the heap
	top = new HeapNode(e, 1, NULL, NULL);
      else if (e->timestamp < top->e->timestamp || (e->timestamp == top->e->timestamp && e->evID < top->e->evID)) { // insert at top of heap
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
#ifdef EH_SANITIZE
      sanitize();
#endif
    }
}

/// Insert event e in heap with low timestamps at top of heap
void EqHeap::InsertDeterministic(Event *e)
{
#ifdef EH_SANITIZE
    sanitize();
#endif
  HeapNode *eh;

  CmiAssert((top == NULL) || (top->subheapsize > 0));
  if (top == NULL) // make the top of the heap
    top = new HeapNode(e, 1, NULL, NULL);
  else if ((e->timestamp < top->e->timestamp) || 
           ((e->timestamp == top->e->timestamp) && 
	    (e->evID <= top->e->evID))) { // insert at top of heap
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
#ifdef EH_SANITIZE
    sanitize();
#endif
}

/// Return event on top of heap, deleting it from the heap
Event *EqHeap::GetAndRemoveTopEvent()
{
#ifdef EH_SANITIZE
    sanitize();
#endif
  CmiAssert(top != NULL);
  CmiAssert(top->subheapsize > 0);
  HeapNode *tmp = top;
  Event *result;

  if (top->left) top = top->left->conjoin(top->right);
  else top = top->right;
  result = tmp->e;
  tmp->e = NULL;
  tmp->left = tmp->right = NULL;
  delete(tmp);
  heapSize--;
#ifdef EH_SANITIZE
    sanitize();
#endif
  return result;
}

/// Delete event from heap
int EqHeap::DeleteEvent(eventID evID, POSE_TimeType timestamp)
{
#ifdef EH_SANITIZE
    sanitize();
#endif
  int result;
  if (!top || (timestamp < top->e->timestamp))  // NOTE: Skipping evID comparison... if control not set in parameter evID then
    return 0;                                   //   search will fail... not having the check shouldn't cause too much un-needed work
  else if ((top->e->timestamp == timestamp) && (top->e->evID == evID)) {
    HeapNode *tmp = top; // top is the match
    if (top->left)
      top = top->left->conjoin(top->right); // remove node from heap
    else top = top->right;
    tmp->left = tmp->right = NULL; // foil recursive destructor
    delete tmp;
    heapSize--;
#ifdef EH_SANITIZE
    sanitize();
#endif
    return 1;
  }
  else { // search deeper in heap
    result = top->remove(evID, timestamp);
    if (result) heapSize--;
#ifdef EH_SANITIZE
    sanitize();
#endif
    return result;
  }
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

/// Dump entire heap to a string
char *EqHeap::dumpString() {
  char *str= new char[8192];
  sprintf(str, "[EQHEAP: ");
  //if (top) {
  //  strcat(str, top->dumpString());
  //} else {
  //  strcat(str, "NULL");
  //}
  strcat(str, "<not printed right now>");
  strcat(str, " end EQHEAP] ");
  return str;
}

/// Check validity of data fields
void EqHeap::sanitize()
{
  CkAssert(((top==NULL) && (heapSize==0)) ||
	   ((top!=NULL) && (heapSize>0)));
  if (top!=NULL) top->sanitize();
}
