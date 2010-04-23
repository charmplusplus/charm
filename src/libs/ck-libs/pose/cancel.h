/// List to store event cancellations
#ifndef CANCEL_H
#define CANCEL_H

/// A single event cancellation
class CancelNode { 
 public:
  /// Timestamp of event to be cancelled
  POSE_TimeType timestamp;   
  /// Event ID of event to be cancelled
  eventID evID;    
  /// Next cancellation in list
  CancelNode *next;
  /// Basic Constructor
  CancelNode() : timestamp(POSE_UnsetTS), next(NULL) { }
  /// Initializing Constructor
  CancelNode(POSE_TimeType ts, eventID e) :timestamp (ts), evID (e), next(NULL){ }
  /// Dump all datafields
  void dump() {
#if USE_LONG_TIMESTAMPS
    CkPrintf("[timestamp=%lld ", timestamp); evID.dump();  CkPrintf("] ");
#else
    CkPrintf("[timestamp=%d ", timestamp); evID.dump();  CkPrintf("] ");
#endif
  }
  /// Pack/unpack/sizing operator
  void pup(PUP::er &p) { p(timestamp); evID.pup(p); }
};

/// A list of event cancellations
class CancelList { 
  /// Number of cancellations in list
  int count;
  /// Timestamp of earliest cancellation in list
  /** Set to -1 if no cancellations present */
  POSE_TimeType earliest;        
  /// The list of cancellations
  CancelNode *cancellations;
  /// Pointer to a particular node in cancellations
  CancelNode *current; 
 public:
  /// Initializing Constructor
  CancelList() :
    count (0), earliest(POSE_UnsetTS), cancellations(NULL), current(NULL) 
  {}
  /// Insert an event at beginning of cancellations list
  /** Inserts an event at beginning of list; increments count and sets earliest
      if applicable; sets current if list was previously empty */
  void Insert(POSE_TimeType ts, eventID e) {
    CancelNode *newnode = new CancelNode(ts, e);
    count++;
    /*if (count%1000 == 0) 
      CkPrintf("WARNING: CancelList has %d events!\n", count); */
    if ((ts < earliest) || (earliest < 0)) // new event has earliest timestamp
      earliest = ts;
    newnode->next = cancellations; // place at front of list
    cancellations = newnode;
    if (!current) current = newnode; // set current if list was empty
  }    
  /// Return a pointer to a node in list
  /** Use current to cycle through nodes and give a different node each time
      GetItem is called */
  CancelNode *GetItem() {
    CancelNode *result;
    if (!current) 
      CkPrintf("ERROR: CancelList::GetItem: CancelList is empty\n");
    result = current;
    if (current->next) current = current->next;
    else current = cancellations;
    return result;
  }
  /// Remove a specific cancellation from the list
  void RemoveItem(CancelNode *item);      
  /// Test if cancellations is empty
  inline int IsEmpty() { 
    CmiAssert(((count == 0) && (cancellations == NULL)) ||
	      ((count > 0) && (cancellations != NULL)));
    return (count == 0);
  }    
  /// Return earliest timestamp
  inline POSE_TimeType getEarliest() { return earliest; }
  /// Dump all data fields
  void dump();               
  /// Pack/unpack/sizing operator
  void pup(PUP::er &p);                   
};

#endif
