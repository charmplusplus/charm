/// Heap structure used for unexecuted portion of the eventQueue
/** This should provide rapid insertion/deletion of new events to the 
    event queue */
#ifndef EQHEAP_H
#define EQHEAP_H

/// Structure for storing events on a heap
class HeapNode
{
 public:
  /// Size of subheap topped by this node
  int subheapsize;         
  /// The event stored here
  Event *e;                
  /// Left and right subheaps
  HeapNode *left, *right;  
  /// Basic Constructor
  HeapNode() : subheapsize ( 0), e(NULL),left(NULL),right( NULL){ }
  /// Initializing Constructor
  HeapNode(Event *ev, int sz, HeapNode *l, HeapNode *r) :
    subheapsize(sz), e(ev), left(l), right(r) 
  {}
  /// Destructor
  ~HeapNode(){if (left) delete left; if (right) delete right; if (e) delete e;}
  /// Insert event in heap
  /** Insert event e in this subheap; designed to find insertion position
      quickly, at the cost of creating unbalanced or high & narrow heaps */
  void insert(Event *e);                    
  /// Insert event in heap deterministically
  /** Insert event e in this subheap; designed to find insertion position
      quickly, at the cost of creating unbalanced or high & narrow heaps */
  void insertDeterministic(Event *e);                    
  /// Join this heap with h
  /** Join this heap with h and return the new heap; uses quickest join method 
      possible at expense of creating unbalanced tree */
  HeapNode *conjoin(HeapNode *h);           
  /// Remove heap node matching evID
  int remove(eventID evID, POSE_TimeType timestamp);  
  /// Find maximum element
  POSE_TimeType findMax() {
    POSE_TimeType max = e->timestamp, leftTS, rightTS;
    leftTS = rightTS = POSE_UnsetTS;
    if (left) leftTS = left->findMax();
    if (right) rightTS = right->findMax();
    if (max < leftTS) max = leftTS;
    if (max < rightTS) max = rightTS;
    return max;
  }
  /// Dump all data fields
  void dump();
  /// Dump all data fields to a string
  char *dumpString();
  /// Pack/unpack/sizing operator
  /** Packs/sizes the entire heap, DOES NOT UNPACK HEAP!!! */
  void pup(PUP::er &p);                     
  /// Check validity of data fields
  void sanitize();
};

/// Heap structure to store events in unexecuted portion of event queue
class EqHeap {  
  /// Size of heap
  int heapSize; 
 public:
  /// Top node of heap  
  HeapNode *top;
  /// Basic Constructor
  EqHeap() : heapSize(0),top(NULL){ }
  /// Destructor
  ~EqHeap() { if (top) delete top; }
  /// Heap size
  inline int size() { return heapSize; }
  /// Insert event e in heap with low timestamps at top of heap
  void InsertEvent(Event *e);              
  /// Insert event e in heap deterministically with low timestamps at top
  void InsertDeterministic(Event *e);              
  /// Return event on top of heap, deleting it from the heap
  /** Returns event at top of heap if one exists, null otherwise; deletes top
      node in heap, conjoining left and right subheaps */
  Event *GetAndRemoveTopEvent();           
  /// Delete event from heap
  /** Delete the node with event corresponding to evID and timestamp; 
      returns 1 if an event was successfully deleted, 0 if the event was not
      found in the heap */
  int DeleteEvent(eventID evID, POSE_TimeType timestamp);  
  /// Find maximum element
  inline POSE_TimeType FindMax() {
    if (top) return top->findMax();
    return POSE_UnsetTS;
  }
  /// Dump entire heap
  void dump();
  /// Dump entire heap to a string
  char *dumpString();
  /// Pack/unpack/sizing operator
  /** Pups entire heap relying on recursive HeapNode::pup */
  void pup(PUP::er &p);     
  /// Check validity of data fields
  void sanitize();
};

#endif
