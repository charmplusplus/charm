#ifndef ELEMENT_H
#define ELEMENT_H

#include "ref.h"
#include "refine.decl.h"

extern CProxy_chunk mesh;

class element {  // triangular elements defined by three node references,
  // having three edge references to control refinement access to
  // this element, and provide connectivity to adjacent elements

  // targetArea is the area to attempt to refine this element
  // below. It's unset state is -1.0, and this will be detected as a
  // 'no refinement desired' state. currentArea is the actual area of
  // the triangle that was cached most recently.  This gets updated by
  // calls to either getArea or calculateArea.
  double targetArea, currentArea;
  int present;  // indicates this is an element present in the mesh

  /* node and edge numberings follow this convention regardless of 
     orientation:
                       0
                      / \
                    2/   \0
                    /     \
                   2_______1
	               1                              */
 public:
  nodeRef nodes[3];
  edgeRef edges[3];
  elemRef myRef;
  chunk *C;

  element(); // Basic element constructor
  element(int cid, int idx, chunk *C) { set(); set(cid, idx, C); }
  element(nodeRef *n);
  element(nodeRef *n, edgeRef *e);
  element(nodeRef& n1, nodeRef& n2, nodeRef& n3, 
	  edgeRef& e1, edgeRef& e2, edgeRef& e3);
  element(int cid, int idx, chunk *C, nodeRef& n1, nodeRef& n2, nodeRef& n3, 
	  edgeRef& e1, edgeRef& e2, edgeRef& e3);
  element(const element& e);
  element& operator=(const element& e);
    
  void set(); // equivalent to constructor initializations
  void set(int cid, int idx, chunk *myChk) { set(); myRef.set(cid, idx); C = myChk;}
  void set(nodeRef *n);
  void set(nodeRef *n, edgeRef *e);
  void set(nodeRef& n1, nodeRef& n2, nodeRef& n3, 
	  edgeRef& e1, edgeRef& e2, edgeRef& e3);
  void set(int idx, edgeRef e) { edges[idx] = e; }
  void set(edgeRef& e1, edgeRef& e2, edgeRef& e3);

  void update(edgeRef& oldval, edgeRef& newval);
  void update(nodeRef& oldval, nodeRef& newval);

  nodeRef& getNode(int nodeIdx) { return nodes[nodeIdx]; }
  edgeRef& getEdge(int edgeIdx) { return edges[edgeIdx]; }
  int getEdgeIdx(edgeRef e);
  int getNodeIdx(nodeRef n);
  elemRef getElement(int edgeIdx);
  edgeRef& getEdge(edgeRef eR, nodeRef nR);
  nodeRef& getOpnode(edgeRef& e);
  
  void clear() { present = 0; }
  int isPresent() { return present; }

  // getArea & calculateArea both set currentArea; getArea returns it
  // minimizeTargetArea initializes or minimizes targetArea
  // getTargetArea & getCachedArea provide access to targetArea & currentArea
  double getArea();
  void calculateArea();
  void minimizeTargetArea(double area);
  void resetTargetArea(double area) { targetArea = area; }
  void setTargetArea(double area) { 
    if ((area < targetArea) || (targetArea < 0.0)) targetArea = area; }
  double getTargetArea() { return targetArea; }
  double getCachedArea() { return currentArea; }

  // These methods handle various types and stages of refinements.
  void refine();
  void split(int longEdge);

  // coarsen will delete this element (and possibly a neighbor) by squishing 
  // its shortest edge to that edge's midpoint.
  void coarsen();
  void collapse(int shortEdge, int n1, int n2, int e1, int e2);
  void collapseHelp(edgeRef shortEdgeRef, nodeRef n1ref, nodeRef n2ref);

  // These are helper methods.
  int findLongestEdge();
  int findShortestEdge();
  int isLongestEdge(edgeRef& e);

  // Mesh improvement stuff
  void tweakNodes();
  node tweak(node n[3], int i);

  void sanityCheck(chunk *c,elemRef shouldRef);
};

#endif
