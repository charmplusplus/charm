// File: pvtobj.h
// Defines pvtObjects, a list that holds records of objects registered with 
// a PVT.
// Last Modified: 5.30.01 by Terry L. Wilmarth

#ifndef PVTOBJ_H
#define PVTOBJ_H

class pvtObjectNode {  // pvtObjects is a list of these nodes
 public:
  int ovt, index;  // last reported OVT of object; index in POSE_Objects array
  sim *localObjPtr;
  short int present, sync;  // present==1 indicates that this node 
  // contains a valid object, present==0 indicates the node can be recycled; 
  // sync refers to the synchronization strategy of the object (OPTIMISTIC or 
  // CONSERVATIVE)
  pvtObjectNode() { present = 0; }  // basic initialization
  void Set(int ts, int idx, short int on, short int s) {
    ovt = ts;  index = idx;  present = on;  sync = s;
  }
  void dump() {  // print the node contents
    CkPrintf("ovt=%d index=%d present=%d sync=%s",
	     ovt, index, present, (sync==0)?"OPT":"CON");
  }
};

class pvtObjects {  // expandable list of nodes
 public:
  int numObjs, numSpaces, size, firstEmpty;  // numObjs is the actual number of
  // objects present in the list;  numSpaces is the number of consecutive
  // spaces in the list that are or have been occupied by object data; size is
  // the space allocated in objs; firstEmpty is the lowest index of an empty 
  // slot in objs
  pvtObjectNode *objs;                        // the list of objects
  pvtObjects();                               // basic initialization
  void SetIdle();                             // sets objects to idle (ovt==-1)
  void Wake();                                // wake up all objects
  void Commit();                              // commit all objects
  int Insert(int index, int ovt, int sync, sim *myPtr); //insert object in list
  void Delete(int idx);                       // delete an object from the list
  void dump();                                // print out the list contents
};
  
#endif
