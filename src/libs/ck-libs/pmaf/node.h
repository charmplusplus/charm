/// Node class for PMAF3D Framework
#ifndef NODE_H
#define NODE_H

#include <math.h>
#include "ref.h"
#include "charm.h"

class chunk;

/// A node in the mesh with a 3D double coordinate
class node { 
  /// The 3D Cartesian coordinate
  double coord[3];       
  /// Flag for fixed node
  /** One (1) flags this as fixed node; zero (0) is not fixed; 
      negative (-1) is uninitialized  */
  int fixed;             
  /// Flag for surface node
  /** One (1) flags this as a surface node; zero (0) is not surface;
      negative (-1) is uninitialized  */
  int surface;           
  /// Number of point relocation reports received
  /** For mesh improvement, we receive new point votes for relocating this
      point.  This keeps track of how many we have received.  */
  int reports;           
  /// Sum of point relocation reports received
  /** For mesh improvement, we sum new point votes for each coordinate
      in this array.  When voting is complete we compute an average for
      each coordinate using reports (above).  */
  double sumReports[3];  
  /// A reference to this node
  nodeRef myRef;         
  /// A pointer to the chunk on which this node resides
  chunk *C;              
 public:
  /// Basic constructor
  /** Initializes all data members except for coordinates.  */
  node() {
    reports = 0;
    fixed = surface = -1; 
    for (int i=0; i<3; i++) sumReports[i] = 0.0;
    myRef.reset();
    C = NULL;
  }
  /// Initializing constructor 1
  /** Initializes all data members, accepting three (3) double parameters
      to initialize the coordinates.  */
  node(double x, double y, double z) { 
    coord[0] = x; coord[1] = y;  coord[2] = z;
    reports = 0;
    fixed = surface = -1; 
    for (int i=0; i<3; i++) sumReports[i] = 0.0;
    myRef.reset();
    C = NULL;
  }
  /// Initializing constructor 2
  /** Initializes all data members, accepting an array of three (3) doubles 
      as a parameter to initialize the coordinates.  */
  node(double inNode[3]) { 
    reports = 0;
    fixed = surface = -1; 
    for (int i=0; i<3; i++) {
      coord[i] = inNode[i];
      sumReports[i] = 0.0;
    }
    myRef.reset();
    C = NULL;
  }
  /// Pupper
  /** Packs/unpacks/sizes the node for use in messages via parameter
      marshalling.  myRef and C become irrelevant remotely and are not
      needed here.  */
  void pup(PUP::er &p) { 
    p|reports; p|fixed; p|surface;
    p(coord,3);
    p(sumReports,3);
  }
  /// Set operation 1
  /** Initializes myRef and C. */
  void set(int cid, int idx, chunk *cptr) { myRef.set(cid, idx);  C = cptr; }
  /// Set operation 2
  /** Accepts three (3) double parameters to initialize the coordinates.  */
  void set(double x, double y, double z) { 
    coord[0] = x; coord[1] = y;  coord[2] = z;
  }
  /// Set operation 3
  /** Accepts and array of three (3) doubles to initialize the coordinates.  */
  void set(double inNode[3]) { 
    for (int i=0; i<3; i++) coord[i] = inNode[i];
  }
  /// Reset operation
  /** Reinitializes all data except for the coordinates.  */
  void reset() { 
    reports = 0;
    fixed = surface = -1; 
    for (int i=0; i<3; i++) sumReports[i] = 0.0;
    myRef.reset();
    C = NULL;
  }
  /// Equality comparison operation
  /** Compares only coordinate values.  */
  int operator==(const node& n) { 
    return ((coord[0] == n.coord[0]) && (coord[1] == n.coord[1]) 
       && (coord[2] == n.coord[2])); }
  /// Assignment operation
  /** Assigns only coordinates and flags.  */
  node& operator=(const node& n) { 
    fixed = n.fixed;  surface = n.surface;
    for (int i=0; i<3; i++) {
      coord[i] = n.coord[i];
    }
    return *this; 
  }
  /// Coordinate value access operation
  /** Input should be 0, 1 or 2.  Returns the double that is the dth entry
      in the coord array. Prints error message if d is out of range.  */
  double getCoord(int d) { 
    CmiAssert((d<=2) && (d>=0));
    return coord[d]; 
  }
  /// Set fixed flag
  void fix() { fixed = 1; }
  /// Test fixed flag
  int isFixed() { return(fixed); }
  /// Unset fixed flag
  void notFixed() { fixed = 0; }
  /// Set surface flag
  void setSurface() { surface = 1; }
  /// Test surface flag
  int onSurface() { return(surface); }
  /// Unset surface flag
  void notSurface() { surface = 0; }
  /// Get distance to node n
  double distance(const node& n) {
    double dx = n.coord[0] - coord[0], dy = n.coord[1] - coord[1], 
      dz = n.coord[2] - coord[2];
    return (sqrt ((dx * dx) + (dy * dy) + (dz * dz)));
  }
  /// Find midpoint between this and node n; place in result
  void midpoint(const node& n, node& result) { // get midpoint between this & n
    result.coord[0] = (coord[0] + n.coord[0]) / 2.0;  
    result.coord[1] = (coord[1] + n.coord[1]) / 2.0;
    result.coord[2] = (coord[2] + n.coord[2]) / 2.0;
  }
  /// Return midpoint between this and node n
  node midpoint(const node& n) { 
    double x = (coord[0] + n.coord[0]) / 2.0;
    double y = (coord[1] + n.coord[1]) / 2.0;
    double z = (coord[2] + n.coord[2]) / 2.0;
    return node(x, y, z);
  }
  /// Project line through point n to get new point
  /** Get point at end of projection of line amd
      place new point in result.  */
  //  o------------------o------------------o
  // this      L         n         L      result
  void project(const node& n, node& result) { 
    result.coord[0] = n.coord[0] + (n.coord[0] - coord[0]);  
    result.coord[1] = n.coord[1] + (n.coord[1] - coord[1]);
    result.coord[2] = n.coord[2] + (n.coord[2] - coord[2]);
  }
  /// Shorten line to length l; place new point in result
  /** Get a point on the line this-->n that is distance l from this
      node; place the resulting node in result */
  void shortenLine(const node& n, double l, node& result) {
    double m = distance(n);
    result.coord[0] = (l/m)*(n.coord[0] - coord[0]) + coord[0];
    result.coord[1] = (l/m)*(n.coord[1] - coord[1]) + coord[1];
    result.coord[2] = (l/m)*(n.coord[2] - coord[2]) + coord[2];
  }
  /// Adjust node coordinates with collected data
  /** Calculate a new position for this node from the collected
      reports of desired node locations; assumes all possible
      reports are in and that the node is movable.  */
  void relocateNode() { // assumes all reports are in
    // calculates a new position from the report data
    if (fixed || (reports==0)) return;
    for (int i=0; i<3; i++) {
      coord[i] = sumReports[i]/reports;
      sumReports[i] = 0.0;
    }
    reports = 0;
  }
  /// Receive a position report
  /** Receive a node position report and update the locally
      collected data.  */
  void relocationVote(const node& n) {
    for (int i=0; i<3; i++) sumReports[i] += n.coord[i];
    reports++;
  }
};

#endif
