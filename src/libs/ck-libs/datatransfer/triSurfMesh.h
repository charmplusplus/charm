/* A triangle surface mesh.
   Created 4 Oct 2006 - Terry L. Wilmarth
 */
#ifndef __UIUC_CHARM_SURFACEMESH_H
#define __UIUC_CHARM_SURFACEMESH_H

#include <stdio.h> // For FILE *
#include "ckvector3d.h"
#include <vector>

/* A 3d surface mesh.  Contains the connectivity only--no data. */
class TriangleSurfaceMesh {
 public:
  enum {nodePer=3}; //!< Nodes per surface triangle
  
  //! Connectivity: 0-based node indices around our triangle
  class conn_t {
  public:
    int nodes[TriangleSurfaceMesh::nodePer];
    conn_t() {nodes[0]=-1;}
    conn_t(int a,int b,int c)
      {nodes[0]=a; nodes[1]=b; nodes[2]=c;}
  };
  
  //! Create a new empty mesh.
  TriangleSurfaceMesh() { }
  //! Create a new mesh with this many triangles and points.
  TriangleSurfaceMesh(int nt,int np) { allocate(nt,np); }
  virtual ~TriangleSurfaceMesh() { }
  
  //! Set size of mesh to be nt triangles and np points. Throws away previous mesh.
  virtual void allocate(int nt,int np) {
    tris.resize(nt);
    pts.resize(np);
  }
  
  //! Return the number of triangles in the mesh
  inline int getTriangles(void) const {return tris.size();}
  //! Return the t'th triangle's 0-based node indices
  inline int *getTriangle(int t) {return &(tris[t].nodes[0]);}
  inline const int *getTriangle(int t) const {return &(tris[t].nodes[0]);}
  inline int *getTriangleConn(void) {return getTriangle(0);}
  inline const int *getTriangleConn(void) const {return getTriangle(0);}
  
  inline const double getArea(int t) const {
    CkVector3d n1_coord = pts[tris[t].nodes[0]];
    CkVector3d n2_coord = pts[tris[t].nodes[1]];
    CkVector3d n3_coord = pts[tris[t].nodes[2]];
    double area=0.0;
    double aLen, bLen, cLen, sLen, d, ds_sum;
    
    ds_sum = 0.0;
    for (int i=0; i<3; i++) {
      d = n1_coord[i] - n2_coord[i];
      ds_sum += d*d;
    }
    aLen = sqrt(ds_sum);
    ds_sum = 0.0;
    for (int i=0; i<3; i++) {
      d = n2_coord[i] - n3_coord[i];
      ds_sum += d*d;
    }
    bLen = sqrt(ds_sum);
    ds_sum = 0.0;
    for (int i=0; i<3; i++) {
      d = n3_coord[i] - n1_coord[i];
      ds_sum += d*d;
    }
    cLen = sqrt(ds_sum);
    sLen = (aLen+bLen+cLen)/2;
    return (sqrt(sLen*(sLen-aLen)*(sLen-bLen)*(sLen-cLen)));
  }

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
    tris.erase(tris.begin(), tris.end());
    std::vector<conn_t>(tris).swap(tris);
  }
  
  //! Simple mesh modification. The new number of the added object is returned.
  int addTriangle(const conn_t &c) {tris.push_back(c); nonGhostTri++; return tris.size()-1;}
  int addPoint(const CkVector3d &pt) {pts.push_back(pt); nonGhostPt++; return pts.size()-1;}
  
  int nonGhostTri, nonGhostPt;
  void writeToTecplot(char *fname) {
    FILE *file = fopen(fname, "w");
    // Header
    fprintf (file, "TITLE=\"%s\"\n", fname);
    fprintf (file, "ZONE N=%d E=%d ET=TRIANGLE F=FEPOINT\n",
	     pts.size(), tris.size());
    // Mesh vertices
    int i,n;
    n=pts.size();
    for (i=0; i<n; ++i) {
      fprintf(file,"%lf %lf %lf\n",pts[i][0],pts[i][1],pts[i][2]);
    }
    // Mesh triangles
    n=tris.size();
    for (i=0; i<n; ++i) {
      fprintf(file,"%d %d %d\n", tris[i].nodes[0]+1, tris[i].nodes[1]+1, 
	      tris[i].nodes[2]+1);  
    }
    fclose(file);
  }
  
  
 private:
  std::vector<conn_t> tris;     //!< Connectivity
  std::vector<CkVector3d> pts; //!< nPts 3d node locations.
};
#endif
