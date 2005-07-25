#include "fem_adapt_new.h"  

#define DEBUG_1
#define DEBUG_2

// ======================  BEGIN edge_flip  =================================
/* Perform a Delaunay flip of the edge (n1, n2) returning 1 if successful, 0 if
   not (likely due to the edge being on a boundary). The convexity of the 
   quadrilateral formed by two faces incident to edge (n1, n2) is assumed. n1 
   and n2 are assumed to be local to this chunk.  An adjacency test is 
   performed on n1 and n2 by searching for an element with edge [n1,n2].

       n3                 n3
        o                  o
       / \                /|\
      /   \              / | \
     /     \            /  |  \
    /       \          /   |   \
n1 o---------o n2  n1 o    |    o n2
    \       /          \   |   / 
     \     /            \  |  /
      \   /              \ | /
       \ /                \|/
        o                  o
       n4                 n4
*/
int FEM_Adapt::edge_flip(int n1, int n2) 
{
  int e1, e1_n1, e1_n2, e1_n3, n3;
  int e2, e2_n1, e2_n2, e2_n3, n4;
  findAdjData(n1, n2, &e1, &e2, &e1_n1, &e1_n2, &e1_n3, &e2_n1, &e2_n2, &e2_n3,
	      &n3, &n4);
  if ((e1 == -1) || (e2 == -1)) return 0; // edge on boundary are not there
  return edge_flip_help(e1, e2, n1, n2, e1_n1, e1_n2, e1_n3, n3, n4);
}
int FEM_Adapt::edge_flip_help(int e1, int e2, int n1, int n2, int e1_n1, 
			      int e1_n2, int e1_n3, int n3, int n4) 
{
  int *conn = (int*)malloc(3*sizeof(int));
  int numNodes = 4;
  int numElems = 2;
  int *locknodes = (int*)malloc(numNodes*sizeof(int));
  int *lockelems = (int*)malloc(numElems*sizeof(int));

  locknodes[0] = n1;
  locknodes[1] = n2;
  locknodes[2] = n3;
  locknodes[3] = n4;
  lockelems[0] = e1;
  lockelems[1] = e2;

  //currently we do not move chunk boundaries, so we do not flip edges in which one of the 4 nodes of the quadrilateral is a ghost node.
  if(n1 < 0 || n2 < 0 || n3 < 0 || n4 < 0) return -1;

  FEM_Modify_Lock(theMesh, locknodes, numNodes, lockelems, numElems);

#ifdef DEBUG_1
  CkPrintf("Flipping edge %d->%d on chunk %d\n", n1, n2, theMod->getfmUtil()->getIdx());
  CkPrintf("Adjacencies before flip\n");
  printAdjacencies(locknodes, numNodes, lockelems, numElems);
#endif

  FEM_remove_element(theMesh, e1, 0);
#ifdef DEBUG_2
  CkPrintf("Adjacencies after remove element %d: conn(%d,%d,%d)\n",e1,n1,n2,n3);
  lockelems[0] = -1;
  printAdjacencies(locknodes, numNodes, lockelems, numElems);
#endif
  FEM_remove_element(theMesh, e2, 0);
#ifdef DEBUG_2
  CkPrintf("Adjacencies after remove element %d: conn(%d,%d,%d)\n",e2,n1,n2,n4);
  lockelems[1] = -1;
  printAdjacencies(locknodes, numNodes, lockelems, numElems);
#endif
  // add n1, n3, n4
  conn[e1_n1] = n1;  conn[e1_n2] = n4;  conn[e1_n3] = n3;
  lockelems[0] = FEM_add_element(theMesh, conn, 3, 0);
#ifdef DEBUG_2
  CkPrintf("Adjacencies after add element %d: conn(%d,%d,%d)\n",lockelems[0],conn[0],conn[1],conn[2]);
  printAdjacencies(locknodes, numNodes, lockelems, numElems);
#endif
  // add n2, n3, n4
  conn[e1_n1] = n4;  conn[e1_n2] = n2;  conn[e1_n3] = n3;
  lockelems[1] = FEM_add_element(theMesh, conn, 3, 0);
#ifdef DEBUG_2
  CkPrintf("Adjacencies after add element %d: conn(%d,%d,%d)\n",lockelems[1],conn[0],conn[1],conn[2]);
  printAdjacencies(locknodes, numNodes, lockelems, numElems);
#endif

#ifdef DEBUG_1
  CkPrintf("Adjacencies after flip\n");
  printAdjacencies(locknodes, numNodes, lockelems, numElems);
#endif

  FEM_Modify_Unlock(theMesh);

  return 1;
}
// ======================  END edge_flip  ===================================


// ======================  BEGIN edge_bisect  ===============================
/* Given edge e:(n1, n2), remove the two elements (n1,n2,n3) and 
   (n2,n1,n4) adjacent to e, and bisect e by adding node 
   n5. Add elements (n1,n5,n3), (n5,n2,n3), (n5,n1,n4) and (n2,n5,n4); 
   returns new node n5.

       n3                 n3
        o                  o
       / \                /|\
      /   \              / | \
     /     \            /  |  \
    /       \          /   |n5 \
n1 o---------o n2  n1 o----o----o n2
    \       /          \   |   / 
     \     /            \  |  /
      \   /              \ | /
       \ /                \|/
        o                  o
       n4                 n4
*/
int FEM_Adapt::edge_bisect(int n1, int n2) 
{
  int e1, e1_n1, e1_n2, e1_n3, n3;
  int e2, e2_n1, e2_n2, e2_n3, n4;
  findAdjData(n1, n2, &e1, &e2, &e1_n1, &e1_n2, &e1_n3, &e2_n1, &e2_n2, &e2_n3,
	      &n3, &n4);
  return edge_bisect_help(e1, e2, n1, n2, e1_n1, e1_n2, e1_n3, e2_n1, e2_n2, 
			  e2_n3, n3, n4);
}
int FEM_Adapt::edge_bisect_help(int e1, int e2, int n1, int n2, int e1_n1, 
				int e1_n2, int e1_n3, int e2_n1, int e2_n2, 
				int e2_n3, int n3, int n4)
{
  int n5;
  int *conn = (int*)malloc(3*sizeof(int));
  int numNodes = 4;
  int numElems = 2;
  int numNodesNew = 5;
  int numElemsNew = 4;
  int *locknodes = (int*)malloc(numNodesNew*sizeof(int));
  int *lockelems = (int*)malloc(numElemsNew*sizeof(int));

  locknodes[0] = n1;
  locknodes[1] = n2;
  locknodes[2] = n3;
  locknodes[3] = n4;
  locknodes[4] = -1;
  lockelems[0] = e1;
  lockelems[1] = e2;
  lockelems[2] = -1;
  lockelems[3] = -1;

  FEM_Modify_Lock(theMesh, locknodes, numNodes, lockelems, numElems);

#ifdef DEBUG_1
  CkPrintf("Bisect edge %d->%d on chunk %d\n", n1, n2, theMod->getfmUtil()->getIdx());
  CkPrintf("Adjacencies before bisect\n");
  printAdjacencies(locknodes, numNodes, lockelems, numElems);
#endif

  FEM_remove_element(theMesh, e1, 0); 
#ifdef DEBUG_2
  CkPrintf("Adjacencies after remove element %d\n",e1);
  lockelems[0] = -1;
  printAdjacencies(locknodes, numNodes, lockelems, numElems);
#endif
  FEM_remove_element(theMesh, e2, 0);  // assumes intelligent behavior when no e2 exists
#ifdef DEBUG_2
  lockelems[1] = -1;
  CkPrintf("Adjacencies after remove element %d\n",e2);
  printAdjacencies(locknodes, numNodes, lockelems, numElems);
#endif
  // hmm... if e2 is a ghost and we remove it and create all the new elements
  // locally, then we don't really need to add a *shared* node
  int *adjnodes = (int*)malloc(2*sizeof(int));
  adjnodes[0] = n1;
  adjnodes[1] = n2;

  n5 = FEM_add_node(theMesh,adjnodes,2,0);
#ifdef DEBUG_2
  CkPrintf("Adjacencies after add node %d\n",n5);
  printAdjacencies(locknodes, numNodesNew, lockelems, numElemsNew);
#endif
  // add n1, n5, n3
  conn[e1_n1] = n1;  conn[e1_n2] = n5;  conn[e1_n3] = n3;
  lockelems[0] = FEM_add_element(theMesh, conn, 3, 0);
#ifdef DEBUG_2
  CkPrintf("Adjacencies after add element %d: conn(%d,%d,%d)\n",lockelems[0],conn[0],conn[1],conn[2]);
  printAdjacencies(locknodes, numNodesNew, lockelems, numElemsNew);
#endif
  // add n2, n5, n3
  conn[e1_n1] = n5;  conn[e1_n2] = n2;  conn[e1_n3] = n3;
  lockelems[1] = FEM_add_element(theMesh, conn, 3, 0);
#ifdef DEBUG_2
  CkPrintf("Adjacencies after add element %d: conn(%d,%d,%d)\n",lockelems[1],conn[0],conn[1],conn[2]);
  printAdjacencies(locknodes, numNodesNew, lockelems, numElemsNew);
#endif
  if (e2 != -1) { // e2 exists
    // add n1, n5, n4
    conn[e2_n1] = n1;  conn[e2_n2] = n5;  conn[e2_n3] = n4;
    lockelems[2] = FEM_add_element(theMesh, conn, 3, 0);
#ifdef DEBUG_2
  CkPrintf("Adjacencies after add element %d: conn(%d,%d,%d)\n",lockelems[2],conn[0],conn[1],conn[2]);
  printAdjacencies(locknodes, numNodes, lockelems, numElemsNew);
#endif
    // add n2, n5, n4
    conn[e2_n1] = n5;  conn[e2_n2] = n2;  conn[e2_n3] = n4;
    lockelems[3] = FEM_add_element(theMesh, conn, 3, 0);
#ifdef DEBUG_2
  CkPrintf("Adjacencies after add element %d: conn(%d,%d,%d)\n",lockelems[3],conn[0],conn[1],conn[2]);
  printAdjacencies(locknodes, numNodesNew, lockelems, numElemsNew);
#endif
  }

#ifdef DEBUG_2
  CkPrintf("Adjacencies after bisect\n");
  locknodes[4] = n5;
  printAdjacencies(locknodes, numNodesNew, lockelems, numElemsNew);
#endif

  FEM_Modify_Unlock(theMesh);
  return n5;
}
// ======================  END edge_bisect  ================================


// ======================  BEGIN vertex_remove  ============================
/* Inverse of edge bisect, this removes a degree 4 vertex n1 and 2 of its
   adjacent elements.  n2 indicates that the two elements removed are
   adjacent to edge [n1,n2]. This could be performed with edge_contraction,
   but this is a simpler operation. 

         n3	             n3        
          o	              o        
         /|\	             / \       
        / | \	            /   \      
       /  |  \	           /     \     
      /   |n1 \           /       \    
  n5 o----o----o n2   n5 o---------o n2
      \   |   /           \       /    
       \  |  /	           \     /     
        \ | /	            \   /      
         \|/	             \ /       
          o	              o        
         n4                  n4        
*/
int FEM_Adapt::vertex_remove(int n1, int n2)
{
  int e1, e1_n1, e1_n2, e1_n3, n3;
  int e2, e2_n1, e2_n2, e2_n3, n4;
  findAdjData(n1, n2, &e1, &e2, &e1_n1, &e1_n2, &e1_n3, &e2_n1, &e2_n2, &e2_n3,
	      &n3, &n4);
  if (e1 == -1) return 0;
  // find n5
  int *nbrNodes, nnsize, n5;
  theMesh->n2n_getAll(n1, &nbrNodes, &nnsize);
  for (int i=0; i<nnsize; i++) {
    if ((nbrNodes[i] != n2) && (nbrNodes[i] != n3) && (nbrNodes[i] != n4)) {
      n5 = nbrNodes[i];
      break;
    }
  }
  return vertex_remove_help(e1, e2, n1, n2, e1_n1, e1_n2, e1_n3, e2_n1, e2_n2, 
			    e2_n3, n3, n4, n5);
}
int FEM_Adapt::vertex_remove_help(int e1, int e2, int n1, int n2, int e1_n1, 
				  int e1_n2, int e1_n3, int e2_n1, int e2_n2, 
				  int e2_n3, int n3, int n4, int n5)
{
  int numNodes = 5;
  int numElems = 4;
  int numNodesNew = 4;
  int numElemsNew = 2;
  int *locknodes = (int*)malloc(numNodes*sizeof(int));
  int *lockelems = (int*)malloc(numElems*sizeof(int));

  locknodes[0] = n2;
  locknodes[1] = n3;
  locknodes[2] = n4;
  locknodes[3] = n5;
  locknodes[4] = n1;
  lockelems[0] = e1;
  lockelems[1] = e2;
  lockelems[2] = -1;
  lockelems[3] = -1;

  int e3 = theMesh->e2e_getNbr(e1, get_edge_index(e1_n1, e1_n3));
  int e4 = -1;
  lockelems[2] = e3;
  if (e3 != -1) {
    if (e2 != -1) {
      e4 = theMesh->e2e_getNbr(e2, get_edge_index(e2_n1, e2_n3));
      lockelems[3] = e4;
      if(e4 == -1 ) return 0;
    }
    FEM_Modify_Lock(theMesh, locknodes, numNodes, lockelems, numElems);

#ifdef DEBUG_1
  CkPrintf("Vertex Remove edge %d->%d on chunk %d\n", n1, n2, theMod->getfmUtil()->getIdx());
  CkPrintf("Adjacencies before vertex remove\n");
  printAdjacencies(locknodes, numNodes, lockelems, numElems);
#endif
    
    FEM_remove_element(theMesh, e1, 0);
#ifdef DEBUG_2
  CkPrintf("Adjacencies after remove element %d\n",e1);
  lockelems[0] = -1;
  printAdjacencies(locknodes, numNodes, lockelems, numElems);
#endif
    FEM_remove_element(theMesh, e3, 0);
#ifdef DEBUG_2
  CkPrintf("Adjacencies after remove element %d\n",e3);
  lockelems[2] = -1;
  printAdjacencies(locknodes, numNodes, lockelems, numElems);
#endif
    if (e2 != -1) {
      FEM_remove_element(theMesh, e2, 0);
#ifdef DEBUG_2
  CkPrintf("Adjacencies after remove element %d\n",e2);
  lockelems[1] = -1;
  printAdjacencies(locknodes, numNodes, lockelems, numElems);
#endif
      FEM_remove_element(theMesh, e4, 0);
#ifdef DEBUG_2
  CkPrintf("Adjacencies after remove element %d\n",e4);
  lockelems[3] = -1;
  printAdjacencies(locknodes, numNodes, lockelems, numElems);
#endif
    }
    FEM_remove_node(theMesh, n1);
#ifdef DEBUG_2
  CkPrintf("Adjacencies after remove node %d\n",n1);
  locknodes[4] = -1;
  printAdjacencies(locknodes, numNodes, lockelems, numElems);
#endif
    
    int *conn = (int*)malloc(3*sizeof(int));
    // add n2, n5, n3
    conn[e1_n1] = n2;  conn[e1_n2] = n3;  conn[e1_n3] = n5;
    lockelems[0] = FEM_add_element(theMesh, conn, 3, 0);
#ifdef DEBUG_2
  CkPrintf("Adjacencies after add element %d: conn(%d,%d,%d)\n",lockelems[0],conn[0],conn[1],conn[2]);
  printAdjacencies(locknodes, numNodes, lockelems, numElems);
#endif
    if (e2 != -1) {
      // add n2, n5, n4
      conn[e2_n1] = n5;  conn[e2_n2] = n4;  conn[e2_n3] = n2;
      lockelems[1] = FEM_add_element(theMesh, conn, 3, 0);
#ifdef DEBUG_2
  CkPrintf("Adjacencies after add element %d: conn(%d,%d,%d)\n",lockelems[1],conn[0],conn[1],conn[2]);
  printAdjacencies(locknodes, numNodes, lockelems, numElems);
#endif
    }

#ifdef DEBUG_1
  CkPrintf("Adjacencies after vertex remove\n");
  printAdjacencies(locknodes, numNodesNew, lockelems, numElemsNew);
#endif

    FEM_Modify_Unlock(theMesh);
    return 1;
  }
  return 0;
}
// ======================  END vertex_remove  ==============================
  
// ======================  BEGIN edge_contraction  ============================
/* Given and edge e:(n1, n2), determine the two adjacent elements (n1,n2,n3) 
   and (n1,n2,n4). Contract edge e by creating node n5, removing all elements 
   incident on n1 xor n2 and reinserting with incidence on n5, removing the two
   elements (n1,n2,n3) and (n1,n2,n4) adjacent to e, and finally removing nodes
   n1 and n2; return 1 if successful, 0 if not 

       n3                 n3
        o                  o
       / \                 |
      /   \                |  
 \   /     \   /         \ | / 
  \ /       \ /           \|/   
n1 o---------o n2          o n5     
  / \       / \           /|\    
 /   \     /   \         / | \ 
      \   /                |  
       \ /                 | 
        o                  o
       n4                 n4
*/
int FEM_Adapt::edge_contraction(int n1, int n2) 
{
  int e1, e1_n1, e1_n2, e1_n3, n3;
  int e2, e2_n1, e2_n2, e2_n3, n4;
  findAdjData(n1, n2, &e1, &e2, &e1_n1, &e1_n2, &e1_n3, &e2_n1, &e2_n2, &e2_n3,
	      &n3, &n4);
  if (e1 == -1) return 0;
  return edge_contraction_help(e1, e2, n1, n2, e1_n1, e1_n2, e1_n3, e2_n1, 
			       e2_n2, e2_n3, n3, n4);
}
int FEM_Adapt::edge_contraction_help(int e1, int e2, int n1, int n2, int e1_n1,
				     int e1_n2, int e1_n3, int e2_n1, 
				     int e2_n2, int e2_n3, int n3, int n4)
{
  int *conn = (int*)malloc(3*sizeof(int));
  int *adjnodes = (int*)malloc(2*sizeof(int));
  adjnodes[0] = n1;
  adjnodes[1] = n2;
  int *adjelems = (int*)malloc(2*sizeof(int));
  adjelems[0] = e1;
  adjelems[1] = e2;

  FEM_Modify_Lock(theMesh, adjnodes, 2, adjelems, 2);
#ifdef DEBUG_1
  CkPrintf("Edge Contraction, edge %d->%d on chunk %d\n", n1, n2, theMod->getfmUtil()->getIdx());
  CkPrintf("Adjacencies before edge contract\n");
  printAdjacencies(adjnodes, 2, adjelems, 2);
#endif
  FEM_remove_element(theMesh, e1, 0);
#ifdef DEBUG_2
  CkPrintf("Adjacencies after remove element %d\n",e1);
  printAdjacencies(adjnodes, 2, adjelems, 2);
#endif
  FEM_remove_element(theMesh, e2, 0);
#ifdef DEBUG_2
  CkPrintf("Adjacencies after remove element %d\n",e2);
  printAdjacencies(adjnodes, 2, adjelems, 2);
#endif
  int n5 = FEM_add_node(theMesh,adjnodes,2,0);
#ifdef DEBUG_2
  CkPrintf("Adjacencies after add node %d\n",n5);
  printAdjacencies(adjnodes, 2, adjelems, 2);
#endif
  //FEM_Modify_Unlock(theMesh);

  int *nbrElems1, nesize1, *nbrElems2, nesize2;
  theMesh->n2e_getAll(n1, &nbrElems1, &nesize1);
  theMesh->n2e_getAll(n2, &nbrElems2, &nesize2);

  //FEM_Modify_Lock(theMesh, nbrElems1, 0, nbrElems1, nesize1);
  // delete/add surrounding elements
  for (int i=0; i<nesize1; i++) {
    if ((nbrElems1[i] != e1) && (nbrElems1[i] != e2)) {
      theMesh->e2n_getAll(nbrElems1[i], conn);
      for (int j=0; j<3; j++) 
	if (conn[j] == n1) conn[j] = n5;
      FEM_remove_element(theMesh, nbrElems1[i], 0);
#ifdef DEBUG_2
      CkPrintf("Adjacencies after remove element %d\n",nbrElems1[i]);
      printAdjacencies(nbrElems1, 0, nbrElems1, nesize1);
#endif
      nbrElems1[i] = FEM_add_element(theMesh, conn, 3, 0);
#ifdef DEBUG_2
      CkPrintf("Adjacencies after add element %d: conn(%d,%d,%d)\n",nbrElems1[i],conn[0],conn[1],conn[2]);
      printAdjacencies(nbrElems1, 0, nbrElems1, nesize1);
#endif
    }
  }
  //FEM_Modify_Unlock(theMesh);

  //FEM_Modify_Lock(theMesh, nbrElems2, 0, nbrElems2, nesize2);
  for (int i=0; i<nesize2; i++) {
    if ((nbrElems2[i] != e1) && (nbrElems2[i] != e2)) {
      theMesh->e2n_getAll(nbrElems2[i], conn);
      for (int j=0; j<3; j++) 
	if (conn[j] == n2) conn[j] = n5;
      FEM_remove_element(theMesh, nbrElems2[i], 0);
#ifdef DEBUG_2
      CkPrintf("Adjacencies after remove element %d\n",nbrElems2[i]);
      printAdjacencies(nbrElems2, 0, nbrElems2, nesize2);
#endif
      nbrElems2[i] = FEM_add_element(theMesh, conn, 3, 0);
#ifdef DEBUG_2
      CkPrintf("Adjacencies after add element %d: conn(%d,%d,%d)\n",nbrElems2[i],conn[0],conn[1],conn[2]);
      printAdjacencies(nbrElems2, 0, nbrElems2, nesize2);
#endif
    }
  }
  //FEM_Modify_Unlock(theMesh);

  //FEM_Modify_Lock(theMesh, adjnodes, 2, adjelems, 0);
  FEM_remove_node(theMesh, n1);
#ifdef DEBUG_2
  CkPrintf("Adjacencies after remove node %d\n",n1);
  printAdjacencies(adjnodes, 2, adjelems, 0);
#endif
  FEM_remove_node(theMesh, n2);
#ifdef DEBUG_2
  CkPrintf("Adjacencies after remove node %d\n",n2);
  printAdjacencies(adjnodes, 2, adjelems, 0);
#endif
#ifdef DEBUG_1
  CkPrintf("Adjacencies after edge contract\n");
  adjnodes[0] = n5;
  adjnodes[1] = -1;
  adjelems[0] = -1;
  adjelems[1] = -1;
  printAdjacencies(adjnodes, 2, adjelems, 0);
#endif
  FEM_Modify_Unlock(theMesh);

  return 1;
}
// ======================  END edge_contraction  ==============================

// ======================  BEGIN vertex_split =================================
/* Given a node n and two adjacent nodes n1 and n2, split n into two nodes n 
   and np such that the edges to the neighbors n1 and n2 expand into two new 
   elements (n, np, n1) and (np, n, n2); return the id of the newly created 
   node np

    n1	            n1             
     o	             o             
     |	            / \            
     |             /   \           
   \ | /      \   /     \   /      
    \|/        \ /       \ /       
     o n      n o---------o np
    /|\        / \       / \       
   / | \      /   \     /   \      
     |             \   /           
     | 	            \ /            
     o	             o             
    n2              n2             
*/
int FEM_Adapt::vertex_split(int n, int n1, int n2) 
{
  int e1 = theMesh->getElementOnEdge(n, n1);
  if (e1 == -1) return -1;	     
  int e3 = theMesh->getElementOnEdge(n, n2);
  if (e3 == -1) return -1;	     
  return vertex_split(n, n1, n2, e1, e3);
}
int FEM_Adapt::vertex_split(int n, int n1, int n2, int e1, int e3)
{
  int e1_n = find_local_node_index(e1, n);
  int e1_n1 = find_local_node_index(e1, n1);
  int e2 = theMesh->e2e_getNbr(e1, get_edge_index(e1_n, e1_n1));
  int e3_n = find_local_node_index(e3, n);
  int e3_n2 = find_local_node_index(e3, n2);
  int e4 = theMesh->e2e_getNbr(e3, get_edge_index(e3_n, e3_n2));
  if (!check_orientation(e1, e3, n, n1, n2)) {
    int tmp = e3;
    e3 = e4;
    e4 = tmp;
    e3_n = find_local_node_index(e3, n);
    e3_n2 = find_local_node_index(e3, n2);
  }

  int *locknodes = (int*)malloc(4*sizeof(int));
  locknodes[0] = n1;
  locknodes[1] = n;
  locknodes[2] = n2;
  locknodes[3] = -1;
  int *lockelems = (int*)malloc(6*sizeof(int));
  lockelems[0] = e1;
  lockelems[1] = e2;
  lockelems[2] = e3;
  lockelems[3] = e4;
  lockelems[4] = -1;
  lockelems[5] = -1;

  FEM_Modify_Lock(theMesh, locknodes, 4, lockelems, 6);
#ifdef DEBUG_1
  CkPrintf("VErtex Split, %d-%d-%d on chunk %d\n", n1, n, n2, theMod->getfmUtil()->getIdx());
  CkPrintf("Adjacencies before vertex split\n");
  printAdjacencies(locknodes, 4, lockelems, 4);
#endif
  int *adjnodes = (int*)malloc(2*sizeof(int));
  adjnodes[0] = n; //looks like it will never be shared, since according to later code, all n1, n & n2 should be local.. appears to be not correct
  int np = FEM_add_node(theMesh,adjnodes,1,0);
  locknodes[3] = np;
#ifdef DEBUG_2
  CkPrintf("Adjacencies after add node %d\n",np);
  printAdjacencies(locknodes, 4, lockelems, 6);
#endif

  int *conn = (int*)malloc(3*sizeof(int));
  int current, next, nt, nl, eknp, eknt, eknl;
  // traverse elements on one side of n starting with e2
  current = e2;
  nt = n1;
  while ((current != e3) && (current != -1)) { 
    eknp = find_local_node_index(current, n);
    eknt = find_local_node_index(current, nt);
    eknl = 3 - eknp - eknt;
    next = theMesh->e2e_getNbr(current, get_edge_index(eknp, eknl));
    nl = theMesh->e2n_getNode(current, eknl);
    FEM_remove_element(theMesh, current, 0);
#ifdef DEBUG_2
    CkPrintf("Adjacencies after remove element %d\n",current);
    printAdjacencies(locknodes, 4, lockelems, 6);
#endif
    // add nl, nt, np
    conn[eknp] = np; conn[eknt] = nt; conn[eknl] = nl;
    int newelem = FEM_add_element(theMesh, conn, 3, 0);
#ifdef DEBUG_2
    CkPrintf("Adjacencies after add element %d: conn(%d,%d,%d)\n",newelem,conn[0],conn[1],conn[2]);
    printAdjacencies(locknodes, 4, lockelems, 6);
#endif
    nt = nl;
    current = next;
  }
  if (current == -1) { // didn't make it all the way around
    // traverse elements on one side of n starting with e4
    current = e4;
    nt = n2;
    while ((current != e1) && (current != -1)) {
      eknp = find_local_node_index(current, n);
      eknt = find_local_node_index(current, nt);
      eknl = 3 - eknp - eknt;
      next = theMesh->e2e_getNbr(current, get_edge_index(eknp, eknl));
      nl = theMesh->e2n_getNode(current, eknl);
      FEM_remove_element(theMesh, current, 0);
#ifdef DEBUG_2
      CkPrintf("Adjacencies after remove element %d\n",current);
      printAdjacencies(locknodes, 4, lockelems, 6);
#endif
      // add nl, nt, np
      conn[eknp] = np; conn[eknt] = nt; conn[eknl] = nl;
      int newelem = FEM_add_element(theMesh, conn, 3, 0);
#ifdef DEBUG_2
    CkPrintf("Adjacencies after add element %d: conn(%d,%d,%d)\n",newelem,conn[0],conn[1],conn[2]);
    printAdjacencies(locknodes, 4, lockelems, 6);
#endif
      nt = nl;
      current = next;
    }
  }

  // add n, n1, np
  conn[e1_n] = n; conn[e1_n1] = n1; conn[3 - e1_n - e1_n1] = np;
  lockelems[4] = FEM_add_element(theMesh, conn, 3, 0);
#ifdef DEBUG_2
  CkPrintf("Adjacencies after add element %d: conn(%d,%d,%d)\n",lockelems[4],conn[0],conn[1],conn[2]);
  printAdjacencies(locknodes, 4, lockelems, 6);
#endif
  // add n, n2, np
  conn[e3_n] = n; conn[e3_n2] = n2; conn[3 - e3_n - e3_n2] = np;
  lockelems[5] = FEM_add_element(theMesh, conn, 3, 0);
#ifdef DEBUG_2
  CkPrintf("Adjacencies after add element %d: conn(%d,%d,%d)\n",lockelems[5],conn[0],conn[1],conn[2]);
  printAdjacencies(locknodes, 4, lockelems, 6);
#endif
#ifdef DEBUG_1
  CkPrintf("Adjacencies after vertex split\n");
  printAdjacencies(locknodes, 4, lockelems, 4);
#endif
  FEM_Modify_Unlock(theMesh);

  return np;
}
// ======================  END vertex_split ===================

// =====================  BEGIN refine_element_leb ========================= 
/* Given an element e, if e's longest edge f is also the longest edge
   of e's neighbor across f, g, split f by adding a new node in the 
   center of f, and splitting both e and g into two elements.  If g
   does not have f as it's longest edge, recursively call refine_element_leb 
   on g, and start over. */ 
int FEM_Adapt::refine_element_leb(int e)
{
  int eConn[3], fixNode, otherNode, opNode, longEdge, nbr; 
  double eLens[3], longEdgeLen = 0.0;
  theMesh->e2n_getAll(e, eConn);
  eLens[0] = len(eConn[0], eConn[1]);
  eLens[1] = len(eConn[1], eConn[2]);
  eLens[2] = len(eConn[2], eConn[0]);
  for (int i=0; i<3; i++)
    if (eLens[i] > longEdgeLen) {
      longEdgeLen = eLens[i];
      longEdge = i;
      fixNode = eConn[i];
      otherNode = eConn[(i+1)%3];
      opNode = eConn[(i+2)%3];
    }
  nbr = theMesh->e2e_getNbr(e, longEdge);
  if (nbr == -1) // e's longEdge is on physical boundary
    return edge_bisect(fixNode, otherNode);
  int nbrOpNode = e2n_getNot(nbr, fixNode, otherNode);
  double fixEdgeLen = len(fixNode, nbrOpNode);
  double otherEdgeLen = len(otherNode, nbrOpNode);
  if ((fixEdgeLen > longEdgeLen) || (otherEdgeLen > longEdgeLen)) { 
    // longEdge is not nbr's longest edge
    int newNode = edge_bisect(fixNode, otherNode);
    int propElem, propNode; // get the element to propagate on
    if (fixEdgeLen > otherEdgeLen) {
      propElem = findElementWithNodes(newNode, fixNode, nbrOpNode);
      propNode = fixNode;
    }
    else {
      propElem = findElementWithNodes(newNode, otherNode, nbrOpNode);
      propNode = otherNode;
    }
    int localChk, nbrChk;
    localChk = theMod->getfmUtil()->getIdx();
    nbrChk = theMod->getfmUtil()->getRemoteIdx(theMesh,nbr,0);
    if (nbr >= 0) // e's neighbor on longEdge is local
      meshMod[localChk].refine_flip_element_leb(localChk, propElem, propNode, 
					     newNode, nbrOpNode, longEdgeLen);
    else { // nbr < -1 so nbr is a ghost of a non-local element
      int propNodeT = getSharedNodeIdxl(propNode, nbrChk);
      int newNodeT = getSharedNodeIdxl(newNode, nbrChk);
      int nbrOpNodeT = (nbrOpNode>=0)?(getSharedNodeIdxl(nbrOpNode, nbrChk)):(getGhostNodeIdxl(nbrOpNode, nbrChk));
      int propElemT = getGhostElementIdxl(propElem, nbrChk);
      meshMod[nbrChk].refine_flip_element_leb(localChk, propElemT, propNodeT, 
					      newNodeT,nbrOpNodeT,longEdgeLen);
    }
    return newNode;
  }
  else return edge_bisect(fixNode, otherNode); // longEdge is nbr's long edge
}
void FEM_Adapt::refine_flip_element_leb(int e, int p, int n1, int n2, double le) 
{
  int newNode = refine_element_leb(e);
  (void) edge_flip(n1, n2);
  if (len(p, newNode) > le) {
    int localChk = theMod->getfmUtil()->getIdx();
    int newElem = findElementWithNodes(newNode, n1, p);
    meshMod[localChk].refine_flip_element_leb(localChk, newElem, p, n1, 
					      newNode, le);
  }
}
// ========================  END refine_element_leb ========================

// Helpers
int FEM_Adapt::get_edge_index(int local_node1, int local_node2) 
{
  int sum = local_node1 + local_node2;
  CkAssert(local_node1 != local_node2);
  if (sum == 1) return 0;
  else if (sum == 3) return 1;
  else if (sum == 2) return 2;
  else {
    CkPrintf("ERROR: local node pair is strange: [%d,%d]\n", local_node1,
	    local_node2);
    CkAbort("ERROR: local node pair is strange\n");
    return -1;
  }
}

int FEM_Adapt::find_local_node_index(int e, int n) {
  int result = theMesh->e2n_getIndex(e, n);
  if (result < 0) {
    CkPrintf("ERROR: node %d not found on element %d\n", n, e);
    CkAbort("ERROR: node not found\n");
  }
  return result;
}

int FEM_Adapt::check_orientation(int e1, int e3, int n, int n1, int n2)
{
  int e1_n = find_local_node_index(e1, n);
  int e1_n1 = find_local_node_index(e1, n1);
  int e3_n = find_local_node_index(e3, n);
  int e3_n2 = find_local_node_index(e3, n2);
  
  if (((e1_n1 == (e1_n+1)%3) && (e3_n == (e3_n2+1)%3)) ||
      ((e1_n == (e1_n1+1)%3) && (e3_n2 == (e3_n+1)%3)))
    return 1;
  else return 0;
}

void FEM_Adapt::findAdjData(int n1, int n2, int *e1, int *e2, int *e1n1, 
			    int *e1n2, int *e1n3, int *e2n1, int *e2n2, 
			    int *e2n3, int *n3, int *n4)
{
  // Set some default values in case e1 is not there
  (*e1n1) = (*e1n2) = (*e1n3) = (*n3) = -1;
  (*e1) = theMesh->getElementOnEdge(n1, n2); // assumed to return local element
  if ((*e1) == -1) return;
  (*e1n1) = find_local_node_index((*e1), n1);
  (*e1n2) = find_local_node_index((*e1), n2);
  (*e1n3) = 3 - (*e1n1) - (*e1n2);
  (*n3) = theMesh->e2n_getNode((*e1), (*e1n3));
  (*e2) = theMesh->e2e_getNbr((*e1), get_edge_index((*e1n1), (*e1n2)));
  // Set some default values in case e2 is not there
  (*e2n1) = (*e2n2) = (*e2n3) = (*n4) = -1;
  if ((*e2) != -1) { // e2 exists
    (*e2n1) = find_local_node_index((*e2), n1);
    (*e2n2) = find_local_node_index((*e2), n2);
    (*e2n3) = 3 - (*e2n1) - (*e2n2);
    //if ((*e2) > -1) { // if e2 is a ghost, there is no e2n data
    (*n4) = theMesh->e2n_getNode((*e2), (*e2n3));
    //}
  }
}

int FEM_Adapt::getSharedNodeIdxl(int n, int chk) {
  return theMod->getfmUtil()->exists_in_IDXL(theMesh, n, chk, 0, -1);
}
int FEM_Adapt::getGhostNodeIdxl(int n, int chk) { 
  return theMod->getfmUtil()->exists_in_IDXL(theMesh, n, chk, 2, -1);
}
int FEM_Adapt::getGhostElementIdxl(int e, int chk) { 
  return theMod->getfmUtil()->exists_in_IDXL(theMesh, e, chk, 4, 0);
}

void FEM_Adapt::printAdjacencies(int *nodes, int numNodes, int *elems, int numElems) {

  for(int i=0; i<numNodes; i++) {
    if(nodes[i] == -1) continue;
    theMod->getfmUtil()->FEM_Print_n2e(theMesh, nodes[i]);
    theMod->getfmUtil()->FEM_Print_n2n(theMesh, nodes[i]);
  }
  for(int i=0; i<numElems; i++) {
    if(elems[i] == -1) continue;
    theMod->getfmUtil()->FEM_Print_e2n(theMesh, elems[i]);
    theMod->getfmUtil()->FEM_Print_e2e(theMesh, elems[i]);
  }
  return;
}
