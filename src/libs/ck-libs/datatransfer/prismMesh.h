/* A prism mesh.
   (Re-)Created 11 Sept 2006 - Terry L. Wilmarth
 */
#ifndef __UIUC_CHARM_PRISMMESH_H
#define __UIUC_CHARM_PRISMMESH_H

#include <stdio.h> // For FILE *
#include "ckvector3d.h"
#include <vector>

#define OSL_TETMESH_DEBUG 0

/* A 3d prism mesh.  Contains the connectivity only--no data. */
class PrismMesh {
 public:
  enum {nodePer=6}; //!< Nodes per prism.
  
  //! Connectivity: 0-based node indices around our prism.
  class conn_t {
  public:
    int nodes[PrismMesh::nodePer];
    conn_t() {nodes[0]=-1;}
    conn_t(int a,int b,int c,int d, int e, int f)
      {nodes[0]=a; nodes[1]=b; nodes[2]=c; nodes[3]=d;nodes[4]=e;nodes[5]=f;}
  };
  
  //! Create a new empty mesh.
  PrismMesh() { }
  //! Create a new mesh with this many prisms and points.
  PrismMesh(int nt,int np) { allocate(nt,np); }
  virtual ~PrismMesh() { }
  
  //! Set size of mesh to be nt prisms and np points. Throws away previous mesh.
  virtual void allocate(int nt,int np) {
    prism.resize(nt);
    pts.resize(np);
  }
  
  //! Return the number of prisms in the mesh
  inline int getPrisms(void) const {return prism.size();}
  //! Return the t'th prism's 0-based node indices
  inline int *getPrism(int t) {return &(prism[t].nodes[0]);}
  inline const int *getPrism(int t) const {return &(prism[t].nodes[0]);}
  inline int *getPrismConn(void) {return getPrism(0);}
  inline const int *getPrismConn(void) const {return getPrism(0);}
  
  //! Return the number of points (vertices, nodes) in the mesh
  inline int getPoints(void) const {return pts.size();}
  //! Return the p'th vertex (0..getPoints()-1)
  inline CkVector3d &getPoint(int p) {return pts[p];}
  inline const CkVector3d &getPoint(int p) const {return pts[p];}
  CkVector3d *getPointArray(void);
  const CkVector3d *getPointArray(void) const;
  
  void cleanup() {
    pts.erase(pts.begin(), pts.end());
    std::vector<CkVector3d>(pts).swap(pts);
    prism.erase(prism.begin(), prism.end());
    std::vector<conn_t>(prism).swap(prism);
  }
  
  //! Simple mesh modification. The new number of the added object is returned.
  int addPrism(const conn_t &c) {prism.push_back(c); nonGhostPrism++; return prism.size()-1;}
  int addPoint(const CkVector3d &pt) {pts.push_back(pt); nonGhostPt++; return pts.size()-1;}
  
  int nonGhostPrism, nonGhostPt;
  void writeToTecplot(char *fname) {
    FILE *file = fopen(fname, "w");
    // Header
    fprintf (file, "TITLE=\"%s\"\n", fname);
    fprintf (file, "ZONE N=%d E=%d ET=BRICK F=FEPOINT\n",
	     pts.size(), prism.size());
    // Mesh vertices
    int i,n;
    n=pts.size();
    for (i=0; i<n; ++i) {
      fprintf(file,"%lf %lf %lf\n",pts[i][0],pts[i][1],pts[i][2]);
    }
    // Mesh prisms
    n=prism.size();
    for (i=0; i<n; ++i) {
      fprintf(file,"%d %d %d %d %d %d %d %d\n", prism[i].nodes[0]+1, 
	      prism[i].nodes[1]+1, prism[i].nodes[2]+1, prism[i].nodes[2]+1, 
	      prism[i].nodes[3]+1, prism[i].nodes[4]+1, prism[i].nodes[5]+1, 
	      prism[i].nodes[5]+1);  
    }
    fclose(file);
  }
  
  
 private:
  std::vector<conn_t> prism;     //!< Connectivity
  std::vector<CkVector3d> pts; //!< nPts 3d node locations.
};
#endif
