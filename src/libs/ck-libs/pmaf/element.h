/// Element class for PMAF3D Framework
#ifndef ELEMENT_H
#define ELEMENT_H

#include <vector.h>
#include "ref.h"
#include "PMAF.decl.h"

extern CProxy_chunk mesh;

/// Tetrahedral mesh element
/* Nodes:    0: top    1: left    2: back    3: right
   Edges:    0: {0,1}  1: {0,2}   2: {0,3}   3: {1,2}   4: {1,3}   5: {2,3}
   Faces:    0: {0,1,2} (left)     1: {0,1,3} (front) 
             2: {0,2,3} (right)    3: {1,2,3} (bottom)
  
                            0
                            o
                           /|\
                          / ' \
                         /  |  \
                        /   '   \
                       /   2|    \
                      /  __-o-__  \
                     /__-       -__\
                  1 o_______________o 3
*/
class element {  
  double targetVolume, currentVolume;
  elemRef myRef;
  chunk *C;
 public:
  int present, lock;  // indicates this is an element present in the mesh
  nodeRef nodes[4];
  elemRef faceElements[4];
  elemRef dependents[4];
  int numDepends;

  element() { 
    targetVolume = currentVolume = -1.0; present = 1; numDepends = lock = 0; 
    C = NULL;
  }
  element(nodeRef *n) { 
    targetVolume=currentVolume=-1.0; present = 1; numDepends = lock = 0; 
    C = NULL; set(n);
  }
  void set(int cid, int idx, chunk *cptr) {
    myRef.set(cid, idx); 
    C = cptr; 
    numDepends = lock = 0;
  }
  void set(nodeRef *n) { 
    for (int i=0; i<4; i++)  nodes[i] = n[i]; 
  }
  
  void update(nodeRef& oldval, nodeRef& newval);
  int hasNode(nodeRef n);
  int hasNode(node n);
  int hasNodes(nodeRef n1, nodeRef n2, nodeRef n3);
  int hasNodes(double nodeCoords[3][3]);
  elemRef getFace(int face[3]) {
    return(faceElements[face[0]+face[1]+face[2]-3]);
  }
  elemRef getFace(int a, int b, int c) {
    return(faceElements[a+b+c-3]);
  }
  void setFace(int a, int b, int c, elemRef er) {
    faceElements[a+b+c-3] = er;
  }
  void setFace(int idx, elemRef er) {
    faceElements[idx] = er;
  }

  int lockElement() {
    //CkPrintf("L%d.%dL ", myRef.idx, myRef.cid);
    if (lock) return 0;
    else {
      lock = 1;
      CkPrintf("LOCKED %d on %d. \n", myRef.idx, myRef.cid);
    }
    return 1;
  }
  void unlockElement() {
    //CkPrintf("U%d.%dU ", myRef.idx, myRef.cid);
    lock = 0;
  }
  int lockedElement() {
    return lock;
  }
  void addDepend(elemRef d) {
    if ((d == dependents[0]) || (d == dependents[1]) || 
	(d == dependents[2]) || (d == dependents[3]))
       return;
    if (numDepends == 4) {
      CkPrintf("ERROR: can't have more than 4 dependents!\n");
      return;
    }
    dependents[numDepends] = d;
    numDepends++;
  }
  elemRef getDepend() {
    numDepends--;
    return dependents[numDepends];
  }
  void fireDependents() {
    elemRef er;
    while (numDepends > 0) {
      er = getDepend();
      CkPrintf(".......Element %d on %d firing dependent %d on %d\n",
	       myRef.idx, myRef.cid, er.idx, er.cid);
      mesh[er.cid].refineElement(er.idx);
    }
  }
  
  nodeRef& getNode(int nodeIdx) { return nodes[nodeIdx]; }
  int getNodeIdx(nodeRef n);
  int getNode(node n);
  elemRef& getMyRef() { return myRef; }
  
  void clear() { present = 0; }
  int isPresent() { return present; }
  
  // getVolume & calculateVolume both set currentVolume; getVolume returns it
  // getTargetVolume & getCachedVolume provide access to targetVolume & currentVolume
  double getVolume();
  void calculateVolume();
  double getArea(int n1, int n2, int n3);
  void resetTargetVolume(double volume) { 
    targetVolume = volume; 
  }
  void setTargetVolume(double volume);/* { 
    if (myRef.cid != 0)
      CkPrintf("Trying to set target volume of element %d on chunk %d to %f\n",
	       myRef.idx, myRef.cid, volume);
    if ((volume < targetVolume) || (targetVolume < 0.0)) targetVolume = volume;
    }*/
  double getTargetVolume() { return targetVolume; }
  double getCachedVolume() { return currentVolume; }
  double findLongestEdge(int *le1, int *le2, int *nl1, int *nl2);
  
  // Delaunay operations
  // perform 2->3 flip on this element with element neighboring on face
  void flip23(int face[3]);
  flip23response *flip23remote(flip23request *fr);
  // perform 3->2 flip on this element with elements neighboring on edge
  void flip32(int edge[2]);
  flip32response *flip32remote(flip32request *fr);
  flip32response *remove32element(flip32request *fr);
  // test if this element should perform 2->3 flip with element neighboring 
  // on face
  int test23(int face[3]);
  // test if this element should perform 3->2 flip with elements neighboring 
  // on edge
  int test32(int edge[2]);
  
  // Refinement operations
  // Largest face methods
  void refineLF();
  splitResponse *element::splitLF(node in1, node in2, node in3, node in4, 
				  elemRef requester);
  // Longest edge methods
  void refineLE();
  LEsplitResult *LEsplit(elemRef root, elemRef parent, nodeRef newNodeRef, 
			 node newNode, elemRef newRootElem, elemRef newElem, 
			 elemRef targetElem, node aIn, node bIn);
  lockResult *lockArc(elemRef prioRef, elemRef parentRef, double prio,
		      elemRef destRef, node aNode, node bNode);
  void unlockArc1(int prio, elemRef parentRef, elemRef destRef, node aNode, node bNode);
  void unlockArc2(int prio, elemRef parentRef, elemRef destRef, node aNode, node bNode);
  // Centerpoint methods
  void refineCP();

  void updateFace(int cid, int idx) {
    faceElements[0].cid = cid;
    faceElements[0].idx = idx;
    lock = 0; CkPrintf("%d on %d lock %d\n", myRef.idx, myRef.cid, lock);
  }
  void updateFace(elemRef oldElem, elemRef newElem) {
    for (int i=0; i<4; i++) {
      if (faceElements[i] == oldElem) {
	faceElements[i] = newElem;
	return;
      }
    }
  }
  
  // Coarsen operations
  void coarsen();

  // Mesh improvement operations
  void improveElement();
  void improveInternalNode(int n);
  void improveSurfaceNode(int n);
  void improveSurfaceNodeHelp(int n, int ot1, int ot2);

  // element refine tests
  int LEtest();
  int LFtest() {
    double lf=0.0, lf2=0.0, f[4];
    f[0] = getArea(0,1,2);
    f[1] = getArea(0,1,3);
    f[2] = getArea(0,2,3);
    f[3] = getArea(1,2,3);
    for (int i=0; i<4; i++) if (f[i] > lf) { lf2 = lf; lf = f[i]; }
    if (lf2 + (0.1*lf2) <= lf) return 1;
    return 0;
  }
  int CPtest() { return 1; }
  int twoNodesLocked();
  int anyNodeLocked();
};

#endif
