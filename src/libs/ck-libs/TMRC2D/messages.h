#include "charm++.h"
#include "refine.decl.h"
#include "tcharm.h"

// ------------------------------ Messages ----------------------------------

// chunkMsg: pass in number of elements and ghost elements
class chunkMsg : public CMessage_chunkMsg {
public:
  int nEl, nGhost, nChunks;
  CProxy_TCharm myThreads;
};

// splitOutMsg: output data from split edge
class splitOutMsg : public CMessage_splitOutMsg {
public:
  edgeRef e;
  int n;
  int result, local, first, nullNbr;
};

// refMsg: generic message for sending/receiving a reference to/from an element
class refMsg : public CMessage_refMsg {
public:
  objRef aRef;
  int idx;
};

// intMsg: used to send parameterless messages to an element anInt
class intMsg : public CMessage_intMsg {
public:
  int anInt;
};

// doubleMsg: used to send a double to an element idx
class doubleMsg : public CMessage_doubleMsg {
public:
  int idx;
  double aDouble;
};

// boolMsg: used to send a bool
class boolMsg : public CMessage_boolMsg {
public:
  int aBool;
};
