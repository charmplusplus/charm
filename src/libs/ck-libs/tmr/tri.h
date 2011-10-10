// Triangular Mesh Refinement Framework - 2D (TMR)
// Created by: Terry L. Wilmarth

#include <math.h>
#include <vector>
#include "charm++.h"
#include "tcharm.h"
#include "charm-api.h"

// Constants to tell FEM interface whether node is on a boudary between chunks
// and if it is the first of two split operations
#define LOCAL_FIRST 0x2
#define LOCAL_SECOND 0x0
#define BOUND_FIRST 0x3
#define BOUND_SECOND 0x1

class node;
class chunk;
class elemRef;

// --------------------------- Helper Classes -------------------------------
// objRef: References to mesh data require a chunk ID (cid) and an
// index on that chunk (idx). Subclasses for nodeRefs, edgeRefs and
// elemRefs define the same operators as the data they reference.
class objRef {
 public:
  int cid, idx;
  objRef() { cid = -1;  idx = -1; }
  objRef(int chunkId, int objIdx) { cid = chunkId; idx = objIdx; }
  void init() { cid = -1; idx = -1; }
  void init(int chunkId, int objIdx) { cid = chunkId; idx = objIdx; }
  int operator==(const objRef& o) const { return((cid == o.cid) && (idx == o.idx)); }
  int operator!=(const objRef& o) const { return !( (*this)==o ); }
  int isNull(void) const {return cid==-1;}
  void sanityCheck(chunk *C);
  void pup(PUP::er &p) { p(cid); p(idx); }
};

// edge and element References: see actual classes below for
// method descriptions
class edgeRef : public objRef {
 public:
  edgeRef() :objRef() {}
  edgeRef(int c,int i) :objRef(c,i) {}

  void updateElement(chunk *C, elemRef oldval, elemRef newval);
  int lock(chunk *C);
  void unlock(chunk *C);
  int locked(chunk *C) const;
};

class elemRef : public objRef {
 public:
  elemRef() : objRef() {}
  elemRef(int c, int i) : objRef(c,i) {}

  int checkIfLongEdge(chunk *C, edgeRef e);
  double getArea(chunk *C);
  void setTargetArea(chunk *C, double ta);
  void updateEdges(chunk *C, edgeRef e0, edgeRef e1, edgeRef e2);
  void unsetDependency(chunk *C);
  void setDependent(chunk *C, int anIdx, int aCid);
  int hasDependent(chunk *C);
};

#include "refine.decl.h"
// ------------------------ Global Read-only Data ---------------------------
extern CProxy_chunk mesh;
CtvExtern(chunk *, _refineChunk);

// ------------------------------ Messages ----------------------------------
// chunkMsg: information needed at startup
class chunkMsg : public CMessage_chunkMsg {
 public:
  int nChunks;
  CProxy_TCharm myThreads;
};

// refMsg: generic message for sending/receiving a reference to/from an element
class refMsg : public CMessage_refMsg {
 public:
  objRef aRef;
  int idx;
};

// doubleMsg: used to send a double to an element idx
class doubleMsg : public CMessage_doubleMsg {
 public:
  double aDouble;
};

// intMsg: used to return integers from sync methods
class intMsg : public CMessage_intMsg {
 public:
  int anInt;
};

// ---------------------------- Data Classes --------------------------------

class node {
  // node: a coordinate

  // Each node has coordinates (x, y) and a local pointer (C).
  double x, y;  
  chunk *C;

 public:
  // Basic node constructor
  node() { C = NULL; }
  node(double a, double b) { init(a,b); }

  // Initializers for a node: can set coordinate separately, basic
  // chunk and index info separately, or everything at once. 
  void init() { x = -1.0; y = -1.0; C = NULL; }
  void init(double a, double b) { x = a; y = b; }
  void init(chunk *cPtr) { C = cPtr; }
  void init(double a, double b, chunk *cPtr) { x = a; y = b; C = cPtr; }

  // Assignment
  node& operator=(const node& n) { x = n.x; y = n.y; return *this; }
  // Equality
  int operator==(const node& n) const { return ((x == n.x) && (y == n.y)); }

  // Access X and Y coordinates
  double X() const { return x; }
  double Y() const { return y; }

  // distance: compute distance between this and another node
  double distance(const node& n) const {
    double dx = n.x - x, dy = n.y - y;
    return (sqrt ((dx * dx) + (dy * dy)));
  }

  // midpoint: compute midpoint between this and another node
  void midpoint(const node& n, node *result) const {
    result->x = (x + n.x) / 2.0;  result->y = (y + n.y) / 2.0;
  }
};


class edge {
  // edge: acts as a passageway between elements; thus, we can lock
  // passageways to prevent multiple refinement paths from accessing a
  // single element simultaneously.  

  // * Only ONE edge exists between two REAL nodes. 
  // * An edge has indices of its two endpoints on the chunk on which the 
  // edge exists. 
  // * An edge has references to the two elements that on either side of it. 
  // * Also, it has a reference to itself (myRef).
  // in addition, it has a pointer to the local chunk (C) on which it resides.
  // * Finally, it has a lock which can cut off access to the elements on 
  // either side of the edge. 
  edgeRef myRef;
  chunk *C;
  int theLock;  // 0 if open; 1 if locked

 public:
  elemRef elements[2];

  // Basic edge contructor: unlocked by default
  edge() { 
    for (int i=0; i<2; i++)
      elements[i].init();
    myRef.init();  C = NULL;  theLock = 0; 
  }
  
  void sanityCheck(chunk *C,edgeRef ref);
  // Initializers for an edge: either set all fields, or just the
  // index and chunk info.  Element references can be set or
  // modified later with updateElement
  void init() { theLock = 0; } // equivalent to constructor initializations
  void init(int i, chunk *cPtr);
  void init(elemRef *e, int i, chunk *cPtr);

  // updateElement: Set or modify the element references of the edge.  
  // If the edge has not been initialized, passing
  // a -1 or the null reference as oldval will initialize any uninitialized
  // field in the edge
  void updateElement(elemRef oldval, elemRef newval);
  const edgeRef &getRef() const { return myRef; } // get reference to this edge

  // getNbrRef: called by an element on one of it's edges to get a
  // reference to the neighbor on the other side of the edge; the
  // element passes in its own reference for comparison
  const elemRef &getNbrRef(const elemRef &er) { 
    if (er == elements[0])
      return elements[1];
    else if (!(er == elements[1]))
      CkAbort("ERROR: edge::getNbrRef: input edgeRef not on edge\n");
    return elements[0];
  }
  // lock, unlock, and locked control access to the edge's lock
  void lock() { theLock = 1; }
  void unlock() { theLock = 0; }
  int locked() { return theLock; }
};

class element {
  // element: triangular elements defined by three node references,
  // and having three edge references to control refinement access to
  // this element, and provide connectivity to adjacent elements

  // targetArea is the area to attempt to refine this element
  // below. It's unset state is -1.0, and this will be detected as a
  // 'no refinement desired' state. currentArea is the actual area of
  // the triangle that was cached most recently.  This gets updated by
  // calls to either getArea or calculateArea.
  double targetArea, currentArea;

  // When refining this element, it may discover that an adjacent
  // element needs to be refined first. This element then sets its
  // depend field to indicate that it is dependent on another element's
  // refinement.  It also tells the adjacent element that it is
  // dependent, and sends its reference to the adjacent element.  The
  // adjacent element sets its dependent field to that reference. THUS:
  // depend=1 if this element depends on another to refine first, 0 otherwise
  int depend;
  // dependent is a valid reference if some adjacent element depends
  // on this one to refine first; dependent = {-1, -1} (nullRef) otherwise
  elemRef dependent;

  // elements on different chunks use the following data fields to
  // negotiate and communicate refinements; see below for more details
  int specialRequest, pendingRequest, requestResponse;
  elemRef specialRequester;
  node newNode, otherNode;
  edgeRef newLongEdgeRef;
  
  // the reference for this edge and its chunk
  elemRef myRef;
  chunk *C;

 public:
  /* node and edge numberings follow this convention
                         0
                        / \
                      2/   \0
                      /     \
                     2_______1
                         1
  */
  int nodes[3];
  edgeRef edges[3];
  
  element(); // Basic element constructor
  
  void sanityCheck(chunk *C,elemRef ref);
    
  // Initializers: specifying edges is optional; they can be added
  // with updateEdges later on
  void init(); // equivalent to constructor initializations
  void init(int *n, edgeRef *e, int index, chunk *chk);
  void init(int *n, int index, chunk *chk);
  void updateEdge(int idx, edgeRef e) { edges[idx] = e; }
  void updateEdges(edgeRef e0, edgeRef e1, edgeRef e2);
  
  // Access a particular node or edge reference
  node getNode(int i) const;
  const edgeRef &getEdge(int i) const { return edges[i]; }
  // getOpNode returns the node index for node opposite edge[e]
  int getOpNode(int e) { return (e+2)%3; }
  // getOtherNode returns the node index for one endpoint on edge[e]
  int getOtherNode(int e) { return e; }
  // getNeighbor returns the neighboring element reference along the edge[e].
  elemRef getNeighbor(int e) const;
  
  // getArea & calculateArea both set currentArea; getArea returns it
  // setTargetArea initializes or minimizes targetArea
  // getTargetArea & getCachedArea provide access to targetArea & currentArea
  double getArea();
  void calculateArea();
  void setTargetArea(double area) {
    if (((targetArea > area) || (targetArea < 0.0))  &&  (area >= 0.0))
      targetArea = area;
  }
  double getTargetArea() { return targetArea; }
  double getCachedArea() { return currentArea; }

  // These methods manipulate this element's dependence on another
  void setDependency() { depend = 1; }
  void unsetDependency() { depend = 0; }
  int hasDependency() { return (depend); }

  // These methods manipulate what element is dependent on this one
  void setDependent(elemRef e) { dependent = e; }
  void setDependent(int cId, int i) { dependent.cid = cId; dependent.idx = i; }
  void unsetDependent() { dependent.idx = dependent.cid = -1; }
  int hasDependent() { return ((dependent.idx!=-1) && (dependent.cid!=-1)); }
  void tellDepend() {
    if (hasDependent()) {
      dependent.unsetDependency(C);
      unsetDependent();
    }
  }
	
  // These methods handle the negotiation of refinement between two
  // elements on different chunks.  In addition, once the refinement
  // relationship has been established, they control how information
  // on that refinement is passed between the two elements.  Elements
  // A needs to refine along with element B, but they are on different
  // chunks. Element A makes a special request of B to refine first. A
  // labels itself as 'pendingRequest' to avoid resending the request.
  // B receives the request and performs a test based on local id and
  // chunk id.  The greater index gets to refine first.  If B fails,
  // it sends a special request to A, otherwise it dos half of the
  // refinement sends the refinement info to A in the form of a
  // request response.  When A receives the response, it completes its
  // half of the refinement. Note that both A & B could have sent
  // special requests simultaneously.  The test resolves which one
  // gets to proceed first, and when one element is preparing a
  // request response, it ignores incoming special requests.
  void setSpecialRequest(elemRef r) { specialRequest=1; specialRequester=r; }
  int isSpecialRequest() { return (specialRequest == 1); }
  int isPendingRequest() { return (pendingRequest == 1); }
  int isRequestResponse() { return (requestResponse == 1); }
  void setRequestResponse(node n, node o, edgeRef e) { 
    requestResponse = 1; 
    newNode = n;
    otherNode = o;
    newLongEdgeRef = e;
  }

  // These methods handle various types and stages of refinements.
  //
  // refine is the method by which a refinement on an element is
  // initiated. refine first checks if there are any requests arrived
  // or pending, and handles these appropriately.  If there are none,
  // the longest edge is determined, and some tests are performed to
  // determine:
  // 1. if the edge is on a border, splitBorder is called
  // 2. if the edge is also the neighbor's longest, splitNeighbors is called
  // 3. if the edge is not the neighbor's longest, refineNeighbor is called
  void refine();
  // refineNeighbor has to tell this element's neighbor to refine, and
  // tell the neighbor that this element depends on it for its own
  // refinement.  In addition, this element notes that it is dependent
  // on a neighbor, and does not attempt further refinement until it
  // hears from the neighbor
  void refineNeighbor(int longEdge);
  // splitBorder and splitNeighbors set up a locked perimeter of edges
  // around the area of refinement and call splitBorderLocal and
  // splitNeighborsLocal respectively.  splitNeighbors handles the
  // sending of special requests for refinement should the neighbor
  // element be located on a different chunk.  In this case, locking
  // the perimeter is put off until the refinement actually happens.
  void splitBorder(int longEdge);
  void splitNeighbors(int longEdge);
  // These methods handle local refinement -- they create a new node,
  // new edge(s), new element(s), and make sure everything is properly
  // connected.
  void splitBorderLocal(int longEdge, int opnode, int othernode, int modEdge);
  void splitNeighborsLocal(int longEdge, int opnode, int othernode, 
			   int modEdge, int nbrLongEdge, int nbrOpnode,
			   int nbrOthernode, int nbrModEdge, const elemRef &nbr);
  
  // These methods handle the two-phase split of neighboring elements
  // on differing chunks.  splitResponse is called by the element that
  // has precedence, while splitHelp is called by the secondary
  // element to complete the refinement.  Note that an element that
  // receives a special request may choose to simply accept it and
  // continue with the splitResponse regardless of precedence when it
  // is not scheduled for any refinements.  This is because the status
  // of the elements are only checked if the elements need to be
  // refined.
  void splitHelp(int longEdge);
  void splitResponse(int longEdge);

  // These are helper methods.
  //
  // Finds the element's longes edge and returns its index in edges.
  int findLongestEdge();
  // Checks the status of neighbor on longEdge: returns -1 if there is
  // no neighbor (border case), 1 if there is a neighbor and that
  // neighbor has the same longEdge, and 0 otherwise (neighbor does
  // have longEdge as its longest edge)
  int checkNeighbor(int longEdge);
  // Checks if e is the edge reference for the longest edge of this element
  int checkIfLongEdge(edgeRef e);
};

/**
 * The user inherits from this class to receive "split" calls,
 * and to be informed when the refinement is complete.
 */
class refineClient {
public:
  virtual ~refineClient() {}

  /**
   * This triangle of our chunk is being split along this edge.
   *
   * For our purposes, edges are numbered 0 (connecting nodes 0 and 1), 
   * 1 (connecting 1 and 2), and 2 (connecting 2 and 0).
   * 
   * Taking as A and B the (triangle-order) nodes of the splitting edge:
   *
   *                     C                      C                 
   *                    / \                    /|\                  
   *                   /   \                  / | \                 
   *                  /     \      =>        /  |  \                
   *                 /       \              /   |   \               
   *                /         \            /old | new\            
   *               B --------- A          B --- D --- A         
   *
   *   The original triangle's node A should be replaced by D;
   * while a new triangle should be inserted with nodes CAD.
   *
   *   The new node D's location should equal A*(1-frac)+B*frac.
   * For a simple splitter, frac will always be 0.5.
   *
   *   If nodes A and B are shared with some other processor,
   * that processor will also receive a "split" call for the
   * same edge.  If nodes A and B are shared by some other local
   * triangle, that triangle will immediately receive a "split" call
   * for the same edge.  
	 *
	 * flag denotes the properties of the new node added by the split	
   * 0x1 - node is on the chunk boundary
	 * 0x2 - since split will be called twice for each new node,
	 *       this bit shows whether it is the first time or not
	 *
   * Client's responsibilities:
   *   -Add the new node D.  Since both sides of a shared local edge
   *      will receive a "split" call, you must ensure the node is
   *      not added twice.
   *   -Update connectivity for source triangle
   *   -Add new triangle.
   */
  virtual void split(int triNo,int edgeOfTri,int movingNode,double frac) =0;
  virtual void split(int triNo,int edgeOfTri,int movingNode,double frac,int flags) =0;

};

class refineResults; //Used by refinement API to store intermediate results

// ---------------------------- Chare Arrays -------------------------------
class chunk : public TCharmClient1D {
  // Data fields for this chunk's array index, and counts of elements,
  // edges, and nodes located on this chunk; numGhosts is numElements
  // plus number of ghost elements surrounding this chunk

  // current sizes of arrays allocated for the mesh
  int sizeElements, sizeEdges, sizeNodes;
  
  // debug_counter is used to print successive snapshots of the chunk
  // and match them up to other chunk snapshots; refineInProgress
  // flags that the refinement loop is active; modified flags that a
  // target area for some element on this chunk has been modified
  int debug_counter, refineInProgress, modified;

  // meshLock is used to lock the mesh for expansion; if meshlock is
  // zero, the mesh can be either accessed or locked; accesses to the
  // mesh (by a chunk method) decrement the lock, and when the
  // accesses are complete, the lock is incremented; when an expansion
  // of the mesh is required, the meshExpandFlag is set, indicating
  // that no more accesses will be allowed to the mesh until the
  // adjuster gets control and completes the expansion; when the
  // adjuster gets control, it sets meshLock to 1 and when it is
  // finished, it resets both variables to zero.  See methods below.
  int meshLock, meshExpandFlag;

  // private helper methods used by FEM interface functions
  void deriveNodes();
  int edgeLocal(elemRef e1, elemRef e2);
  int findEdge(int n1, int n2);
  int addNewEdge();
  int getNbrRefOnEdge(int n1, int n2, int *conn, int nGhost, int *gid, 
		      int idx, elemRef *er);
  int hasEdge(int n1, int n2, int *conn, int idx);
  
 public:
  refineResults *refineResultsStorage;
 
  // the chunk's components, left public for sanity's sake
  int cid;
  int numElements, numEdges, numNodes, numGhosts, numChunks;
  std::vector<element> theElements;
  std::vector<edge> theEdges;
  std::vector<node> theNodes;

  // client to report refinement split information to
  refineClient *theClient;

  // Basic constructor
  chunk(chunkMsg *);
  chunk(CkMigrateMessage *m) : TCharmClient1D(m) { };
  
  void sanityCheck(void);
  
  void setupThreadPrivate(CthThread forThread) {
    CtvAccessOther(forThread, _refineChunk) = this;
  }
  // entry methods
  
  // Initiates a refinement for a single element
  void refineElement(int i, double area);
  // Loops through all elements performing refinements as needed
  void refiningElements();

  // The following methods simply provide remote access to local data
  // See above for details of each
  void updateElement(int i, objRef oldval, objRef newval);
  void specialRequest(int reqestee, elemRef requester);
  void specialRequestResponse(int i, double newNodeX, double newNodeY, 
			      double otherNodeX, double otherNodeY, 
			      edgeRef newLongEdgeRef);
  doubleMsg *getArea(int i);
  intMsg *lock(int i);
  void unlock(int i);
  intMsg *locked(int i);
  intMsg *checkElement(objRef oR, int i);
  refMsg *getNeighbor(objRef oR, int i);
  void setTargetArea(int i, double area);
  void updateEdges(int i, edgeRef e0, edgeRef e1, edgeRef e2);
  void unsetDependency(int i);
  void setDependent(objRef oR, int i);
  intMsg *hasDependent(int i);

  // meshLock methods
  void accessLock();  // waits until meshExpandFlag not set, then decs meshLock
  void releaseLock(); // incs meshLock
  void adjustFlag();  // sets meshExpandFlag
  void adjustLock();  // waits until meshLock is 0, then sets it to 1
  void adjustRelease();  // resets meshLock and meshExpandFlag to 0

  // used to print snapshots of all chunks at once (more or less)
  void print();

  // local methods

  // These methods are part of the interface with the FEM framework:

  // Sets the node coordinates and recalculates element areas
  void updateNodeCoords(int nNode, double *coord, int nEl);
  // multipleRefine sets target areas of elements as specified by
  // desiredArea, and starts refining.  Each split that occurs is
  // reported to client, and the methods returns only when all
  // refinement is completed.
  void multipleRefine(double *desiredArea, refineClient *client);
  void newMesh(int nEl, int nGhost,const int *conn_,const int *gid_, int idxOffset);
  void addRemoteEdge(int elem, int localEdge, edgeRef er);

  
  // These access and set local flags
  void setModified() { modified = 1; }
  int isModified() { return modified; }
  void setRefining() { refineInProgress = 1; }
  int isRefining() { return refineInProgress; }

  // these methods allow for run-time additions/modifications to the chunk
  void allocMesh(int nEl);
  void adjustMesh();
  int addNode(node n);
  edgeRef addEdge();
  elemRef addElement(int n1, int n2, int n3);
  elemRef addElement(int n1, int n2, int n3,
		     edgeRef er1, edgeRef er2, edgeRef er3);

  void debug_print(int c);  // prints a snapshot of the chunk to file
  void out_print();  // prints a readable meshfile that can be used as input
};
