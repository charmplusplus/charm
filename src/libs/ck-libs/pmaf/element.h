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
  int present;  // indicates this is an element present in the mesh
  nodeRef nodes[4];
  elemRef faceElements[4];
  element() { targetVolume = currentVolume = -1.0; present = 1; C = NULL; }
  element(nodeRef *n) { 
    targetVolume=currentVolume=-1.0; present = 1; C = NULL; set(n);
  }
  void set(int cid, int idx, chunk *cptr) { myRef.set(cid, idx); C = cptr; }
  void set(nodeRef *n) { for (int i=0; i<4; i++)  nodes[i] = n[i]; }
  void update(nodeRef& oldval, nodeRef& newval);
  int hasNode(nodeRef n);
  int hasNode(node n);
  int hasNodes(nodeRef n1, nodeRef n2, nodeRef n3);
  int hasNodes(double nodeCoords[3][3]);
  elemRef getFace(int face[3]) {
    return(faceElements[face[0]+face[1]+face[2]-3]);
  }
  elemRef getFace(int a, int b, int c) { return(faceElements[a+b+c-3]); }
  void setFace(int a, int b, int c, elemRef er) { faceElements[a+b+c-3] = er; }
  void setFace(int idx, elemRef er) { faceElements[idx] = er; }
  int hasFace(elemRef face);
  int checkFace(node n1, node n2, node n3, elemRef nbr);
  void updateFace(int cid, int idx) {
    faceElements[0].cid = cid; faceElements[0].idx = idx;
  }
  void updateFace(elemRef oldElem, elemRef newElem) {
    for (int i=0; i<4; i++)
      if (faceElements[i] == oldElem) {
	faceElements[i] = newElem;
	return;
      }
  }
  nodeRef& getNode(int nodeIdx) { return nodes[nodeIdx]; }
  int getNodeIdx(nodeRef n);
  int getNode(node n);
  elemRef& getMyRef() { return myRef; }
  void clear() { present = 0; }
  int isPresent() { return present; }
  double getVolume();
  void calculateVolume();
  double getArea(int n1, int n2, int n3);
  void resetTargetVolume(double volume) { targetVolume = volume; }
  void setTargetVolume(double volume);
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
  int connectTest();

  // Refinement operations
  // Largest face methods
  void refineLF();
  int lockLF(node n1, node n2, node n3, node n4, elemRef requester, 
	     double prio);
  splitResponse *element::splitLF(node in1, node in2, node in3, node in4, 
				  elemRef requester);
  // Longest edge methods
  void refineLE();
  LEsplitResult *LEsplit(elemRef root, elemRef parent, nodeRef newNodeRef, 
			 node newNode, elemRef newRootElem, elemRef newElem, 
			 elemRef targetElem, double targetVol, 
			 node aIn, node bIn);
  lockResult *lockArc(elemRef prioRef, elemRef parentRef, double prio,
		      elemRef destRef, node aNode, node bNode);
  void unlockArc1(int prio, elemRef parentRef, elemRef destRef, node aNode, 
		  node bNode);
  void unlockArc2(int prio, elemRef parentRef, elemRef destRef, node aNode, 
		  node bNode);
  // Centerpoint methods
  void refineCP();


  // Coarsen operations
  void coarsen();

  // Mesh improvement operations
  void improveElement();
  void improveInternalNode(int n);
  void improveSurfaceNode(int n);
  void improveSurfaceNodeHelp(int n, int ot1, int ot2);

  // element refine tests
  int LEtest();
  int LFtest();
  int CPtest();
};

#endif
