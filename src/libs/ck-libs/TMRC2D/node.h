// Node class for TMRC2D Framework
// Created by: Terry L. Wilmarth
#ifndef NODE_H
#define NODE_H

#include <math.h>
#include "ref.h"
#define DIM 2

class elemRef;
class edgeRef;

class node {  // a 2D double coordinate
  double x, y;  // the double 2D coordinate
  int reports;  // for smoothing
  double sumReports[DIM];  // for smoothing
  int theLock; // locking is for coarsening
  double lockLength;
  edgeRef lockHolder;
 public:
  int present;  // indicates this is an edge present in the mesh
  int boundary; // mesh boundary info: 0 for internal pos int for boundary
  int fixed;   // mesh fixed node info: 0 for non-fixed, 1 for fixed
  
  node() {
    x=-1.0; y=-1.0; theLock = reports = boundary = fixed = 0; 
    sumReports[0]=sumReports[1]=0.0; present = 0;
  }
  node(double a, double b) { 
    x = a;  y = b;  reports = boundary = fixed = theLock = 0; 
    sumReports[0] = sumReports[1] = 0.0; present = 1;
  }
  node(const node& n) { 
    x = n.x;  y = n.y;  reports = n.reports; boundary = n.boundary; 
    theLock = n.theLock; present = n.present; fixed = n.fixed;
    sumReports[0] = n.sumReports[0];  sumReports[1] = n.sumReports[1];
  }
  void set(double a, double b) { 
    x = a;  y = b; present = 1;
  }
  void reset() { 
    theLock = reports = boundary = fixed = 0;
    sumReports[0] = sumReports[1] = 0.0; present = 0;
  }
  int operator==(const node& n) { return ((x == n.x) && (y == n.y)); }
  node& operator=(const node& n) { 
    x = n.x;  y = n.y;  reports = n.reports;  boundary = n.boundary;
    theLock = n.theLock; fixed = n.fixed;
    present = n.present;
    sumReports[0] = n.sumReports[0];  sumReports[1] = n.sumReports[1];
    return *this; 
  }
  void pup(PUP::er &p) {
    p(x); p(y); p(present); p(reports); p(theLock); p(boundary); 
    p(sumReports, DIM); p(fixed);
  }
  int isPresent() { return present; }
  double X() { return x; }
  double Y() { return y; }
  int lock(double l, edgeRef e) { 
    if (theLock == 0) {
      theLock = 1;
      lockLength = l;
      lockHolder = e;
      return 1;
    }
    else if (e == lockHolder) {
      return 1;
    }
    else if (e.cid == lockHolder.cid) {
      return 0;
    }
    else if (l >= lockLength) {
      return 0;
    }
    else if (l < lockLength) {
      /*CkPrintf("TMRC2D: .........SPIN w/l=%f edge=%d,%d node %f,%f... held w/l=%f edge=%d,%d\n", l, e.idx, e.cid, x, y, lockLength, lockHolder.idx, lockHolder.cid);
      while (theLock) CthYield();
      theLock = 1;
      lockLength = l;
      lockHolder = e;
      CkPrintf("TMRC2D: .........(c) LOCK w/l=%f edge=%d,%d node %f,%f\n", l, e.idx, e.cid, x, y);
      return 1;*/
      return 0;
    }
    CkPrintf("WARNING: node::lock: unhandled case.\n");
    return 0;
  }
  void unlock() { 
    theLock = 0; 
  }
  double distance(const node& n) { // get distance to n
    double dx = n.x - x, dy = n.y - y;
    double d = (sqrt ((dx * dx) + (dy * dy)));
    CkAssert(d > 0.0);
    return d;
  }
  void midpoint(const node& n, node& result) { // get midpoint between this & n
    result.x = (x + n.x) / 2.0;  result.y = (y + n.y) / 2.0;
    CkAssert(result.x >= 0.0);
    CkAssert(result.y >= 0.0);
  }
  node midpoint(const node& n) { // get midpoint between this & n
    double a=(x + n.x) / 2.0, b=(y + n.y) / 2.0;
    return node(a, b);
  }

  // mesh smoothing methods (NOT USED)
  void improvePos() {
    x = sumReports[0]/reports;
    y = sumReports[1]/reports;
    reports = 0;
    sumReports[0] = sumReports[1] = 0.0;
  }
  void reportPos(const node& n) {
    sumReports[0] += n.x;
    sumReports[1] += n.y;
    reports++;
  }
  int safeToMove(node m) {
    if (boundary) return 0;  // this node on boundary; don't move it
    return 1;
  }
  int safeToMove(node m, elemRef E0, edgeRef e0, edgeRef e1, 
		 node n1, node n2, node n3) {
    /*    node nbrs[100], intersectionPt;
    int nbrCount = 2, result;
    elemRef Ei, Eold = E0;
    edgeRef ei, eold = e1;
    node ni;
    */
    if (boundary) return 0;  // this node on boundary; don't move it
    return 1;
    /*
    else { // check if this node can flip edges when moved to m
      // acquire all neighbor nodes in order from e0 to e1 around to e0 again.
      nbrs[0] = n2.get();
      nbrs[1] = n3.get();
      while (!(eold == e0)) {
	Ei = eold.getNot(Eold);
	//ei = Ei.get(eold, n1);
	ni = ei.getNot(n1);
	nbrs[nbrCount] = ni.get();
	nbrCount++;
	eold = ei;
	Eold = Ei;
      }
      nbrCount--;

      // compute if adjacent neigboring nodes form line that
      // intersects e0 between this node and m
      for (int i=1; i<nbrCount-1; i++) {
	result = findIntersection(m, nbrs[i], nbrs[i+1], intersectionPt);
	if (result && between(m, intersectionPt))
	  return 0;
      }
    }
    return 1;
    */
  }
  int findIntersection(node m, node pi, node pj, node mi) {
    // find intersection point mi of lines defined by (this,m) and (pi,pj)
    double num, den, ua;
    
    num = ((pj.X() - pi.X()) * (y - pi.Y())) -
      ((pj.Y() - pi.Y()) * (x - pi.X()));
    den = ((pj.Y() - pi.Y()) * (m.X() - x)) - 
      ((pj.X() - pi.X()) * (m.Y() - y));
    if (den == 0) return 0;  // lines are parallel or coincident
    ua = num / den;
    mi.set(x + (ua * (m.X() - x)), y + (ua * (m.Y() - y)));
    return 1;
  }
  int between(node m, node mi) {
    // check if mi is between this node and m
    return((((mi.X() >= x) && (mi.X() <= m.X())) ||
	    ((mi.X() >= m.X()) && (mi.X() <= x)))
	   && (((mi.Y() >= y) && (mi.Y() <= m.Y())) ||
	       ((mi.Y() >= m.Y()) && (mi.Y() <= y))));
  }
  void sanityCheck(int cid, int idx) {
    if ((x == -1.0) && (y == -1.0)) {
      CkPrintf("TMRC2D: [%d] node::sanityCheck WARNING: node %d has default coordinate values.\n", cid, idx);
    }	
    if (theLock)
      CkAbort("TMRC2D: node::sanityCheck: WARNING: node is locked.\n");
  }
  void dump() {
    CkPrintf("[%5.9f,%5.9f]", x, y);
  }
};

#endif
