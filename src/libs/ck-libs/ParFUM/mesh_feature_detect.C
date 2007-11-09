/***************************************************
 * fem_feature_detect.C
 *
 * Feature Detection
 *
 * Authors include: Isaac
 *
 * The experimental new code for feature detection below has not yet been widely tested, but it does work for some example
 */

#include "ParFUM.h"
#include "ParFUM_internals.h"
#include "import.h"

#include <map>
#include <set>
#include <utility>
#include <cmath>
 
#define CORNER_ANGLE_CUTOFF  (3.0/4.0*3.14159)


using namespace std;

CDECL void
FEM_Mesh_detect_features(int fem_mesh) {
  const char *caller="FEM_Mesh_detect_features"; FEMAPI(caller);
  FEM_Mesh *m=FEM_Mesh_lookup(fem_mesh,caller);
  m->detectFeatures();
}
FORTRAN_AS_C(FEM_MESH_DETECT_FEATURES,
             FEM_Mesh_detect_features,
             fem_mesh_detect_features,
             (int *fem_mesh), (*fem_mesh))

    
/**
  A feature detection routine that will eventually work on triangular and tetrahedral meshes. Currently it just works on oriented triangle meshes in serial.

  @note The oriented restriction is here because I am considering the following two edges to be distinct (n1,n2) and (n2,n1). To fix this I could use a class which representst the edges, so that when I put the edges in their container things would work out, or I could order the nodes in the edge by node id, or I could search for (n2,n1) before inserting (n1,n2).

*/
void FEM_Mesh::detectFeatures() {
  CkPrintf("detectFeatures() has been called\n");

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
     
  
  /** The vertices that look like they are on a boundary of {local mesh unioned with its ghosts} 
      This is used as an intermediate result.
  */
  std::set<std::pair<int,int> > edgesOnBoundaryWithGhosts;
  
  
  // Assume for now we have an oriented triangular mesh in FEM_ELEM+0
  int nodesPerElem = elem[0].getNodesPer();
  CkAssert(nodesPerElem==3);
  
  // Make sure the valid attributes have been set for the elements
  // This is actually not sufficient. We should also verify that the adjacencies have been computed
  elem[0].allocateValid();
  node.allocateValid();
  
 CkPrintf("%d:  %d Nodes, %d Ghost Nodes\n", rank, node.size(), node.ghost->size());
 CkPrintf("%d:  %d Elem, %d Ghost Elem\n", rank, elem[0].size(), elem[0].ghost->size());
   
  
  for(int n=0; n<node.size(); n++){
  // Get the coordinates of the 3 nodes
    double x,y;
      
    node.get_coord(n,x,y);
    CkPrintf("Node %d is at %.1lf,%.1lf\n", n, x, y);    
  }

  // Find topological edges of this mesh

  for (int ele=0;ele<elem[0].size();ele++) {
    // if this is a valid element
    if (elem[0].is_valid(ele)) {
      
      const int *conn = elem[0].connFor(ele);
        
      int n1 = conn[0];
      int n2 = conn[1];
      int n3 = conn[2];

      int numElementsOnEdge;
      
      numElementsOnEdge = countElementsOnEdge(n1,n2);
      if (numElementsOnEdge==1) {
        edgesOnBoundaryWithGhosts.insert(std::make_pair(n1,n2));
      }
      
      numElementsOnEdge = countElementsOnEdge(n2,n3);
      if (numElementsOnEdge==1) {
        edgesOnBoundaryWithGhosts.insert(std::make_pair(n2,n3));
      }
      
      numElementsOnEdge = countElementsOnEdge(n3,n1);
      if (numElementsOnEdge==1) {
        edgesOnBoundaryWithGhosts.insert(std::make_pair(n3,n1));
      }
         
    }
    }
    
    // And find anything that looks like an edge around the ghost layer
    for (int ele=0;ele<elem[0].ghost->size();ele++) {
    // if this is a valid element
      if (elem[0].ghost->is_valid(ele)) {
      
        FEM_Elem *ghosts = (FEM_Elem*)elem[0].ghost;
        const int *conn = ghosts->connFor(ele);
        
        int n1 = conn[0];
        int n2 = conn[1];
        int n3 = conn[2];

        int numElementsOnEdge;
      
        numElementsOnEdge = countElementsOnEdge(n1,n2);
        if (numElementsOnEdge==1) {
          edgesOnBoundaryWithGhosts.insert(std::make_pair(n1,n2));
        }
      
        numElementsOnEdge = countElementsOnEdge(n2,n3);
        if (numElementsOnEdge==1) {
          edgesOnBoundaryWithGhosts.insert(std::make_pair(n2,n3));
        }
      
        numElementsOnEdge = countElementsOnEdge(n3,n1);
        if (numElementsOnEdge==1) {
          edgesOnBoundaryWithGhosts.insert(std::make_pair(n3,n1));
        }
         
      }
    }
    
    
    
    // Produce the final list of edges on the boundary(pruning things on the boundary that we can't verify are on the real boundary)
  
    for(std::set<std::pair<int,int> >::iterator iter=edgesOnBoundaryWithGhosts.begin(); iter!=edgesOnBoundaryWithGhosts.end(); ++iter){
      int n1 = (*iter).first;  // The experimental new code for feature detection below has not yet been widely tested, but it does work for some example
  
      int n2 = (*iter).second;
  
      // Include any edge that adjacent to at least one local node. With a 1-deep node-based ghost layer, all such edges will be guaranteed to be in the true boundary
      if(n1>=0 || n2>=0){
        edgesOnBoundary.insert(make_pair(n1,n2));
        verticesOnBoundary.insert(n1);
        verticesOnBoundary.insert(n2);
      }
      
    }
   
    
  CkPrintf("%d: Found a total of %d edges(with at least one local node) on boundary of mesh: ", rank, edgesOnBoundary.size());
  for(std::set<std::pair<int,int> >::iterator iter=edgesOnBoundary.begin();iter!=edgesOnBoundary.end();++iter){
    int n1 = (*iter).first;
    int n2 = (*iter).second;
    double x1,y1,x2,y2;
    node.get_coord(n1,x1,y1);
    node.get_coord(n2,x2,y2);
    CkPrintf("    %.2lf,%.2lf---%.2lf,%.2lf", x1,y1, x2,y2);
  }
  CkPrintf("\n");
  
  
  
  CkPrintf("%d: Found a total of %d nodes on boundary described earlier:", rank, verticesOnBoundary.size());
  for(std::set<int>::iterator iter=verticesOnBoundary.begin();iter!=verticesOnBoundary.end();++iter){
    int n = (*iter);
    double x,y;
    node.get_coord(n,x,y);
    CkPrintf("    %.2lf,%.2lf  ", x,y);
  }
  CkPrintf("\n");
    
  
  
  // Find geometric corners of this mesh

  double COS_CORNER_ANGLE_CUTOFF = cos(CORNER_ANGLE_CUTOFF);
  
  
// build map from boundary vertex to adjacent boundary vertices
//   std::map<int,int> verticesOnBoundary;
//   std::map<std::pair<int,int>,int> edgesOnBoundary;
//   
  std::map<int,std::pair<int,int> >  vertexToAdjacentBoundaryVertices;
  
  for(std::set<std::pair<int,int> >::iterator iter=edgesOnBoundaryWithGhosts.begin();iter!=edgesOnBoundaryWithGhosts.end();++iter){
    int n1 = (*iter).first;
    int n2 = (*iter).second;
    
    // --------------- add n1's neighbor if n1 is local ---------------
    if(n1>=0){
       // see if we have an existing entry
      std::map<int,std::pair<int,int> >::iterator v = vertexToAdjacentBoundaryVertices.find(n1);
      if( v == vertexToAdjacentBoundaryVertices.end() ){
        // no entry yet, create one
        vertexToAdjacentBoundaryVertices[n1] = make_pair( n2, -1 );
      } 
      else if((*v).first == n2){
        // do nothing, we've seen this one already
      }
      else{
        // we have recorded a different adjacent vertex 
        // already, but now we add this one
        CkAssert((*v).second.second==-1);
        (*v).second.second = n2;
      }
    }
    
    // --------------- add n2's neighbor if n2 is local---------------
    // see if we have an existing entry
    if(n2>=0){
      std::map<int,std::pair<int,int> >::iterator v = vertexToAdjacentBoundaryVertices.find(n1);
      v = vertexToAdjacentBoundaryVertices.find(n2);
      if( v == vertexToAdjacentBoundaryVertices.end() ){
        // no entry yet, create one
        vertexToAdjacentBoundaryVertices[n2] = make_pair( n1, -1 );
        
      } 
      else if((*v).first == n1){
        // do nothing, we've seen this one already
      }
      else{
        // we have recorded a different adjacent vertex 
        // already, but now we add this one
        CkAssert((*v).second.second==-1);
        (*v).second.second = n1;
      }
    }
    
  }
  
  
  // Iterate over all vertices on the boundary(these will include some ghost nodes adjacent to local nodes)
  for(std::set<int>::iterator iter=verticesOnBoundary.begin();iter!=verticesOnBoundary.end();iter++){
    int n = *iter;  
    
    
    // Only consider the local nodes
    if(n>=0){
      
      // Find the adjacent 2 edges on boundary
      int n1 = vertexToAdjacentBoundaryVertices[n].first;
      int n2 = vertexToAdjacentBoundaryVertices[n].second;
      
      // Get the coordinates of the 3 nodes
      double x1,y1,x2,y2,x,y;
      
      node.get_coord(n,x,y);
      node.get_coord(n1,x1,y1);
      node.get_coord(n2,x2,y2);
      
      // Compute the angle between the two edges
      
      double v1x = x1-x; // a vector from node n to n1
      double v1y = y1-y;
      
      double v2x = x2-x; // a vector from node n to n2
      double v2y = y2-y;
      
      double cos_theta = (v1x*v2x+v1y*v2y)/(sqrt(v1x*v1x+v1y*v1y) * sqrt(v2x*v2x+v2y*v2y) );
      
          
      CkPrintf("nodes (%.2lf,%.2lf)   (%.2lf,%.2lf)   (%.2lf,%.2lf)  form angle cos=%.3lf\n", x,y,x1,y1,x2,y2, cos_theta);
      
      // Apply cuttoff criterion and save 
        
      if(cos_theta > COS_CORNER_ANGLE_CUTOFF) {
        // Any angle less than the cutoff should be considered a corner
        cornersOnBoundary.insert(n);
      }
    }
    
  }
  
  CkPrintf("Found a total of %d corner nodes on boundary of mesh:\n", cornersOnBoundary.size());
  
  for(std::set<int>::iterator iter=cornersOnBoundary.begin();iter!=cornersOnBoundary.end();++iter){
    int n = (*iter);
    double x,y;
    node.get_coord(n,x,y);
    CkPrintf(" %d=(%.2lf,%.2lf)", n, x,y);
  }
  CkPrintf("\n");
  
  
}




