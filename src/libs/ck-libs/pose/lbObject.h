// File: lbObject.h
// Defines lbObjects, a list that holds records of objects registered with 
// a load balancer.
// Last Modified: 11.05.01 by Terry L. Wilmarth

#ifndef LBOBJ1_H
#define LBOBJ1_H

class lbObjectNode {  // lbObjects is a list of these nodes
 public:
  POSE_TimeType ovt;  // last reported OVT of object; 
  int index;  // in POSE_Objects array
  sim *localObjPtr;
  short int present, sync;  // present==1 indicates that this node 
  // contains a valid object, present==0 indicates the node can be recycled; 
  // sync is object's synchronization strategy (OPTIMISTIC or CONSERVATIVE)
  POSE_TimeType eet;
  int ne, execPrio;
  double rbOh;
  int *comm; // communication with other PEs
  int totalComm, localComm, remoteComm, maxComm, maxCommPE;
  lbObjectNode() {  // basic init
    present = 0; eet = 0; rbOh = -1.0; 
    ne=0;
    comm = (int *)malloc(CkNumPes() * sizeof(int));
    for (int i=0; i<CkNumPes(); i++) comm[i] = 0;
    totalComm = localComm = remoteComm = maxComm = 0;
    maxCommPE = -1;
  }
  inline void Set(POSE_TimeType ts, int idx, short int on, short int s) {
    ovt = ts;  index = idx;  present = on;  sync = s;
  }
  void dump() {  // print the node contents
#if USE_LONG_TIMESTAMPS
    CkPrintf("ovt=%lld index=%d present=%d sync=%s",
	     ovt, index, present, (sync==0)?"OPT":"CON");
#else
    CkPrintf("ovt=%d index=%d present=%d sync=%s",
	     ovt, index, present, (sync==0)?"OPT":"CON");
#endif
  }
};

class lbObjects {  // expandable list of nodes
 public:
  int numObjs, numSpaces, size, firstEmpty; // see pvtobj.h
  lbObjectNode *objs;                       // the list of objects
  lbObjects();                              // basic initialization
  int Insert(int sync, int index, sim *myPtr);  // insert an object in the list
  void Delete(int idx);                     // delete an object from the list
  void UpdateEntry(int idx, POSE_TimeType ovt, POSE_TimeType eet, int ne, double rbOh, int *srVec);
  void AddComm(int idx, int pe, int sr);   // add sr msgs s/r to obj idx t/f pe
  void ResetComm();                         // reset comm array entries to 0
  void RequestReport();                     // request reports from all objects
  void dump();                              // print out the list contents
};
  
#endif
