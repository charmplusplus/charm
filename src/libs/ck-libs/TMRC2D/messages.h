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


// nodeMsg: coordinates of a node
class nodeMsg : public CMessage_nodeMsg {
public:
  int idx;
  double x, y;
};

// edgeMsg: references of nodes defining and elements sharing an edge
class edgeMsg : public CMessage_edgeMsg {
public:
  nodeRef nodes[2];
  elemRef elements[2];
};

// remoteEdgeMsg: used my FEM interface to add an edge reference to a
// remote element
class remoteEdgeMsg : public CMessage_remoteEdgeMsg {
public:
  int elem, localEdge;
  edgeRef er;
};

// elementMsg: references of nodes and edges defining an element; see
// element class below for relationship between nodes and edges
class elementMsg : public CMessage_elementMsg {
public: 
  nodeRef nodes[3];
  edgeRef edges[3];
};

// femElementMsg: references of nodes and edges defining an element; see
// element class below for relationship between nodes and edges
class femElementMsg : public CMessage_femElementMsg {
public: 
  int idx;
  nodeRef nodes[3];
  edgeRef edges[3];
};

// ghostElementMsg: conn & gid for ghost element
class ghostElementMsg : public CMessage_ghostElementMsg {
public: 
  int index;
  int conn[3];
  int gid[2];
};

// refineMsg: index of element to refine and area to refine it under
class refineMsg : public CMessage_refineMsg {
public:
  int idx;
  double area;
};

// splitInMsg: index of edge to split and input data
class splitInMsg : public CMessage_splitInMsg {
public:
  int idx;
  elemRef e;
  nodeRef n;
};

// collapseMsg: specs for edge to collapse
class collapseMsg : public CMessage_collapseMsg {
public:
  int idx;
  edgeRef er;
  nodeRef nr1, nr2;
};

// splitOutMsg: output data from split edge
class splitOutMsg : public CMessage_splitOutMsg {
public:
  edgeRef e;
  nodeRef n;
  int result;
};

// coarsenMsg: index of element to coarsen and area of surrounding elements
class coarsenMsg : public CMessage_coarsenMsg {
public:
  int idx;
  double area;
};

// updateMsg: update a reference for a node or edge in element idx
class updateMsg : public CMessage_updateMsg {
public:
  int idx;
  objRef oldval, newval;
};

// specialRequestMsg: requester makes a special request (to refine) of
// requestee
class specialRequestMsg : public CMessage_specialRequestMsg {
public:
  int requestee;
  elemRef requester;
};

// specialResponseMsg: response to special request with refinement info
class specialResponseMsg : public CMessage_specialRequestMsg {
public:
  int idx;
  nodeRef newNodeRef, otherNodeRef;
  edgeRef newLongEdgeRef;
};

// refMsg: generic message for sending/receiving a reference to/from an element
class refMsg : public CMessage_refMsg {
public:
  objRef aRef;
  int idx;
};

// refMsg: generic message for sending/receiving a reference to/from an element
class drefMsg : public CMessage_drefMsg {
public:
  objRef aRef1, aRef2;
  int idx;
};

// edgeUpdateMsg: update the edges of an element idx
class edgeUpdateMsg : public CMessage_edgeUpdateMsg {
public:
  int idx;
  edgeRef e0, e1, e2;
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
