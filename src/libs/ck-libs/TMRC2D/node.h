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
 public:
  int border; // mesh boundary info
  
  node() {
    x=-1.0; y=-1.0; theLock = reports = border = 0; 
    sumReports[0]=sumReports[1]=0.0;
  }
  node(double a, double b) { 
    x = a;  y = b;  reports = border = theLock = 0; 
    sumReports[0] = sumReports[1] = 0.0; 
  }
  node(const node& n) { 
    x = n.x;  y = n.y;  reports = n.reports; border = n.border; 
    theLock = n.theLock;
    sumReports[0] = n.sumReports[0];  sumReports[1] = n.sumReports[1];
  }
  void set(double a, double b) { 
    x = a;  y = b;
  }
  void setBorder() { border = 1; }
  void reset() { 
    x = -1.0; y = -1.0;  theLock = reports = border = 0;
    sumReports[0] = sumReports[1] = 0.0;
  }
  int operator==(const node& n) { return ((x == n.x) && (y == n.y)); }
  node& operator=(const node& n) { 
    x = n.x;  y = n.y;  reports = n.reports;  border = n.border;
    theLock = n.theLock;
    sumReports[0] = n.sumReports[0];  sumReports[1] = n.sumReports[1];
    return *this; 
  }
  void pup(PUP::er &p) {
    p(x); p(y); p(reports); p(theLock); p(border); p(sumReports, DIM);
  }
  double X() { return x; }
  double Y() { return y; }
  int lock() { return (theLock ? 0 : theLock = 1); }
  void unlock() { theLock = 0; }
  double distance(const node& n) { // get distance to n
    double dx = n.x - x, dy = n.y - y;
    return (sqrt ((dx * dx) + (dy * dy)));
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
    if (border) return 0;  // this node on border; don't move it
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
    if (border) return 0;  // this node on border; don't move it
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
    if ((x == -1.0) && (y == -1.0))
      CkPrintf("TMRC2D: sanityCheck WARNING: node %d on chunk %d has default coordinate values.\n", idx, cid);
  }
};

#endif
