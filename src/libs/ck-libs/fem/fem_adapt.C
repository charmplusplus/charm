/* File: fem_adapt.C
 * Authors: Nilesh Choudhury, Terry Wilmarth
 *
 */

#include "fem_adapt.h"  

// ======================  BEGIN edge_flip  =================================
/* Perform a Delaunay flip of the edge (n1, n2) returning 1 if
   successful, 0 if not (likely due to the edge being on a boundary).
   The convexity of the quadrilateral formed by two faces incident to
   edge (n1, n2) is assumed. n1 and n2 are assumed to be local to this
   chunk.  An adjacency test is performed on n1 and n2 by searching
   for an element with edge [n1,n2]. */
int FEM_Adapt::edge_flip(int n1, int n2) 
{
  int e1, e2, e1_n1, e1_n2, e1_n3, n3, edge1, e1nbr;
  findAdjData(n1, n2, &e1, &e2, &e1_n1, &e1_n2, &e1_n3, &n3, &edge1, &e1nbr);
  if (e2 == -1) return 0; // there is no neighbor; can't flip; 
  return edge_flip_help(e1, e2, n1, n2, e1_n1, e1_n2, e1_n3, n3, edge1, e1nbr);
}
int FEM_Adapt::edge_flip_help(int e1, int e2, int n1, int n2, int e1_n1, 
			      int e1_n2, int e1_n3, int n3, int edge1, 
			      int e1nbr) 
{
  int e2_n1 = find_local_node_index(e2, n1);
  int e2_n2 = find_local_node_index(e2, n2);
  int e2_n3 = 3 - e2_n1 - e2_n2;
  int mod_edge2 = get_edge_index(e2_n1, e2_n3);
  int e2nbr = theMesh->e2e_getNbr(e2, mod_edge2);
  int n4 = theMesh->e2n_getNode(e2, e2_n3);
  
  // Element-to-node updates
  theMesh->e2n_setIndex(e1, e1_n2, n4);
  theMesh->e2n_setIndex(e2, e2_n1, n3);
  // Element-to-element updates
  theMesh->e2e_replace(e1nbr, e1, e2);
  theMesh->e2e_replace(e2nbr, e2, e1);
  theMesh->e2e_replace(e1, e2, e2nbr);
  theMesh->e2e_replace(e2, e1, e1nbr);
  theMesh->e2e_setIndex(e1, edge1, e2);
  theMesh->e2e_setIndex(e2, mod_edge2, e1);
  // Node-to-node updates
  theMesh->n2n_remove(n1, n2);
  theMesh->n2n_remove(n2, n1);
  theMesh->n2n_add(n3, n4);
  theMesh->n2n_add(n4, n3);
  // Node-to-element updates
  theMesh->n2e_remove(n1, e2);
  theMesh->n2e_remove(n2, e1);
  theMesh->n2e_add(n3, e2);
  theMesh->n2e_add(n4, e1);
  return 1;
}
// ======================  END edge_flip  ===================================


// ======================  BEGIN edge_bisect  ===============================
/* Given edge e:(n1, n2), determing the two elements e1:(n1,n2,n3) and 
   e2:(n2,n1,n4) adjacent to e, and bisect e, e1 and e2 by adding a node 
   n5 to edge e. e1 becomes (n1,n5,n3) and we add e3:(n5,n2,n3), e2 becomes 
   (n5,n1,n4) and we add e4:(n2,n5,n4); returns new node n5.

       n3                 n3
        o                  o
       / \                /|\
      /   \              / | \ e1nbr
     / e1  \            /e1|e3\
    /       \          /   |n5 \
n1 o---------o n2  n1 o----o----o n2
    \       /          \   |   / 
     \ e2  /            \e2|e4/
      \   /              \ | / e2nbr
       \ /                \|/
        o                  o
       n4                 n4
*/
int FEM_Adapt::edge_bisect(int n1, int n2) 
{
  int e1, e2, e1_n1, e1_n2, e1_n3, n3, edge1, e1nbr;
  findAdjData(n1, n2, &e1, &e2, &e1_n1, &e1_n2, &e1_n3, &n3, &edge1, &e1nbr);
  return edge_bisect_help(e1, e2, n1, n2, e1_n1, e1_n2, e1_n3, n3,edge1,e1nbr);
}
int FEM_Adapt::edge_bisect_help(int e1, int e2, int n1, int n2, int e1_n1, 
				int e1_n2, int e1_n3, int n3, int edge1, 
				int e1nbr)
{
  int n5 = newNode();
  int e3 = newElement();
  theMesh->setMeshSizing(e3, getMeshSizing(e1));
  localEdgeBisect(n1, n2, e1, e2, e3, e1_n1, e1_n2, e1_n3, e1nbr, n3, n5);

  if (e2 != -1) {
    int e2_n1, e2_n2, e2_n3, n4, edge2, e2nbr;
    findAdjData(n1, n2, e2, &e2_n1, &e2_n2, &e2_n3, &n4, &edge2, &e2nbr);
    int e4 = newElement();
    theMesh->setMeshSizing(e4, getMeshSizing(e2));
    localEdgeBisect(n1, n2, e2, e1, e4, e2_n1, e2_n2, e2_n3, e2nbr, n4, n5);
    theMesh->n2n_remove(n5, n1);
    theMesh->n2n_remove(n5, n2);
    theMesh->e2e_replace(e3, -1, e4);
    theMesh->e2e_replace(e4, -1, e3);
  }
  return n5;
}
void FEM_Adapt::localEdgeBisect(int n1, int n2, int e1, int e2, int e3, 
				int e1n1, int e1n2, int e1n3, int e1nbr, 
				int n3, int n5)
{
  theMesh->e2n_setIndex(e1, e1n2, n5);
  theMesh->e2n_setIndex(e3, e1n1, n5);
  theMesh->e2n_setIndex(e3, e1n2, n2);
  theMesh->e2n_setIndex(e3, e1n3, n3);
  theMesh->e2e_replace(e1, e1n3, e3);
  theMesh->e2e_replace(e1nbr, e1, e3); // check for -1 and ghost
  int nl[3];
  nl[get_edge_index(e1n1, e1n2)] = -1;
  nl[get_edge_index(e1n2, e1n3)] = e1nbr;
  nl[get_edge_index(e1n3, e1n1)] = e1;
  theMesh->e2e_setAll(e3, nl);
  // update e3 with e4 later
  theMesh->n2n_replace(n1, n2, n5);
  theMesh->n2n_replace(n2, n1, n5);
  theMesh->n2n_add(n3, n5);
  theMesh->n2n_add(n5, n1);
  theMesh->n2n_add(n5, n2);
  theMesh->n2n_add(n5, n3);
  // update n5 with n4 later
  theMesh->n2e_replace(n2, e1, e3);
  theMesh->n2e_add(n3, e3);
  theMesh->n2e_add(n5, e1);
  theMesh->n2e_add(n5, e3);
  // update n5 with e2 & e4 later
}
// ======================  END edge_bisect  ================================


// ======================  BEGIN vertex_remove  ============================
/* Inverse of edge bisect, this removes a degree 4 vertex n1 and 2 of its
   adjacent elements.  n2 indicates that the two elements removed are
   adjacent to edge [n1,n2]. This could be performed with edge_contraction,
   but this is a simpler operation. */
int FEM_Adapt::vertex_remove(int n1, int n2)
{
  int e1, e2, e1_n1, e1_n2, e1_n3, n3, edge1, e1nbr;
  findAdjData(n1, n2, &e1, &e2, &e1_n1, &e1_n2, &e1_n3, &n3, &edge1, &e1nbr);
  return vertex_remove_help(e1, e2, n1, n2, e1_n1, e1_n2,e1_n3,n3,edge1,e1nbr);
}
int FEM_Adapt::vertex_remove_help(int e1, int e2, int n1, int n2, int e1_n1,
				  int e1_n2, int e1_n3, int n3, int edge1, 
				  int e1nbr)
{
  int e3 = theMesh->e2e_getNbr(e1, get_edge_index(e1_n1, e1_n3));
  int e4 = e1nbr;
  int e5=-1, e6=-1, n4=-1, n5=-1;
  if (e2 > -1) {
    int e2_n1, e2_n2, e2_n3;
    e2_n1 = find_local_node_index(e2, n1);
    e2_n2 = find_local_node_index(e2, n2);
    e2_n3 = 3 - e2_n1 - e2_n2;
    e5 = theMesh->e2e_getNbr(e2, get_edge_index(e2_n1, e2_n3));
    e6 = theMesh->e2e_getNbr(e2, get_edge_index(e2_n2, e2_n3));
    n4 = theMesh->e2n_getNode(e2, e2_n3);
  }
  // find n5
  int *nbrNodes, nnsize;
  theMesh->n2n_getAll(n1, &nbrNodes, &nnsize);
  for (int i=0; i<nnsize; i++) {
    if ((nbrNodes[i] != n2) && (nbrNodes[i] != n3) && (nbrNodes[i] != n4)) {
      n5 = nbrNodes[i];
      break;
    }
  }
  
  // Element-to-node updates
  theMesh->e2n_replace(e3, n1, n2);
  theMesh->e2n_replace(e5, n1, n2);
  // Element-to-element updates
  theMesh->e2e_replace(e3, e1, e4);
  theMesh->e2e_replace(e5, e2, e6);
  theMesh->e2e_replace(e4, e1, e3);
  theMesh->e2e_replace(e6, e2, e5);
  // Node-to-node updates
  theMesh->n2n_remove(n3, n1);
  theMesh->n2n_remove(n4, n1);
  theMesh->n2n_replace(n2, n1, n5);
  theMesh->n2n_replace(n5, n1, n2);
  // Node-to-element updates
  theMesh->n2e_replace(n2, e1, e3);
  theMesh->n2e_replace(n2, e2, e5);
  theMesh->n2e_remove(n3, e1);
  theMesh->n2e_remove(n4, e2);
  
  deleteNode(n1);
  deleteElement(e1);
  deleteElement(e2);
  return 1;
}
// ======================  END vertex_remove  ==============================
  
// edge_contraction and helpers
int FEM_Adapt::edge_contraction(int n1, int n2) 
{
  int e1 = theMesh->getElementOnEdge(n1, n2);
  if (e1 < 0) {
    CkPrintf("ERROR: edge_contraction: no element with edge [%d,%d]\n", n1,n2);
    return 0;	     
  }
  return edge_contraction(e1, n1, n2);
}
int FEM_Adapt::edge_contraction(int e1, int n1, int n2) 
{
  int e1_n1 = find_local_node_index(e1, n1);
  int e1_n2 = find_local_node_index(e1, n2);
  int shared_edge = get_edge_index(e1_n1, e1_n2);
  int e2 = theMesh->e2e_getNbr(e1, shared_edge); 
  return edge_contraction_help(e1, e2, n1, n2, e1_n1, e1_n2);
}
int FEM_Adapt::edge_contraction_help(int e1, int e2, int n1, int n2, int e1_n1,
				     int e1_n2)
{
  int e1_n3 = 3 - e1_n1 - e1_n2;
  int mod_edge1 = get_edge_index(e1_n1, e1_n3);
  int e1nbr1 = theMesh->e2e_getNbr(e1, mod_edge1);
  int mod_edge2 = get_edge_index(e1_n2, e1_n3);
  int e1nbr2 = theMesh->e2e_getNbr(e1, mod_edge2);
  int n3 = theMesh->e2n_getNode(e1, e1_n3);
  int e2_n1;
  int e2_n2;
  int e2_n3;
  int mod_edge3;
  int e2nbr1;
  int mod_edge4;
  int e2nbr2;
  int n4;
  if (e2 > -1) {
    e2_n1 = find_local_node_index(e2, n1);
    e2_n2 = find_local_node_index(e2, n2);
    e2_n3 = 3 - e2_n1 - e2_n2;
    mod_edge3 = get_edge_index(e2_n1, e2_n3);
    e2nbr1 = theMesh->e2e_getNbr(e2, mod_edge3);
    mod_edge4 = get_edge_index(e2_n2, e2_n3);
    e2nbr2 = theMesh->e2e_getNbr(e2, mod_edge4);
    n4 = theMesh->e2n_getNode(e2, e2_n3);
  }
  else
    {
      CkAbort("ERROR: e2 <= -1 case not handled\n");
    }
  int *n2_nbrNodes, *n2_nbrElems;
  int nnsize, nesize;
  theMesh->n2n_getAll(n2, &n2_nbrNodes, &nnsize);
  theMesh->n2e_getAll(n2, &n2_nbrElems, &nesize);
  
  // Element-to-node updates
  for (int i=0; i<nesize; i++) {
    theMesh->e2n_replace(n2_nbrElems[i], n2, n1);
    theMesh->n2e_add(n1, n2_nbrElems[i]);
  }
  theMesh->n2e_remove(n1, e1); // assume duplicated by loop add above
  theMesh->n2e_remove(n1, e2); // assume duplicated by loop add above
  // Element-to-element updates
  theMesh->e2e_replace(e1nbr1, e1, e1nbr2);
  theMesh->e2e_replace(e1nbr2, e1, e1nbr1);
  theMesh->e2e_replace(e2nbr1, e2, e2nbr2);
  theMesh->e2e_replace(e2nbr2, e2, e2nbr1);
  // Node-to-node updates
  for (int i=0; i<nnsize; i++) {
    if (n2_nbrNodes[i] != n1) {
      theMesh->n2n_remove(n2_nbrNodes[i], n1);
      theMesh->n2n_replace(n2_nbrNodes[i], n2, n1);
      theMesh->n2n_remove(n1, n2_nbrNodes[i]);
      theMesh->n2n_add(n1, n2_nbrNodes[i]);
    }
  }
  theMesh->n2n_remove(n1, n2);
  theMesh->n2n_remove(n3, n2);
  theMesh->n2n_remove(n4, n2);
  // Node-to-element updates
  theMesh->n2e_remove(n1, e1);
  theMesh->n2e_remove(n1, e2);
  theMesh->n2e_remove(n3, e1);
  theMesh->n2e_remove(n4, e2);
  
  deleteNode(n2);
  deleteElement(e1);
  deleteElement(e2);
  return 1;
}

// vertex_split and helpers
int FEM_Adapt::vertex_split(int n, int n1, int n2) 
{
  int e1 = theMesh->getElementOnEdge(n, n1);
  if (e1 < 0) {
    CkPrintf("ERROR: vertex_split: no element with edge [%d,%d]\n", n, n1);
    return -1;	     
  }
  int e3 = theMesh->getElementOnEdge(n, n2);
  if (e3 < 0) {
    CkPrintf("ERROR: vertex_split: no element with edge [%d,%d]\n", n, n2);
    return -1;	     
  }
  return vertex_split(n, n1, n2, e1, e3);
}
int FEM_Adapt::vertex_split(int n, int n1, int n2, int e1, int e3)
{
  int n_e1 = find_local_node_index(e1, n);
  int n1_e1 = find_local_node_index(e1, n1);
  int e2 = theMesh->e2e_getNbr(e1, get_edge_index(n_e1, n1_e1));
  int n_e3 = find_local_node_index(e3, n);
  int n2_e3 = find_local_node_index(e3, n2);
  int e4 = theMesh->e2e_getNbr(e3, get_edge_index(n_e3, n2_e3));
  if (!check_orientation(e1, e3, n, n1, n2)) {
    int tmp = e3;
    e3 = e4;
    e4 = tmp;
    n_e3 = find_local_node_index(e3, n);
    n2_e3 = find_local_node_index(e3, n2);
  }
  int np = newNode();
  int e5 = newElement();
  int e6 = newElement();
  int nnCount=0, neCount=0;
  int np_nodes[50], np_elems[50]; // I certainly hope the mesh is not this bad
  adj_traverse(n, n1, n2, e2, e4, &nnCount, &neCount, np_nodes, np_elems);

  // Element-to-node updates
  int nl[3];
  if ((n_e1 < n1_e1) || ((n_e1 == 2) && (n1_e1 == 0))) {
    nl[0] = n1; nl[1] = n; nl[2] = np;
    theMesh->e2n_setAll(e5, nl);
    nl[0] = e1; nl[1] = e6; nl[2] = e2;
    theMesh->e2e_setAll(e5, nl);
  }
  else {
    nl[0] = n; nl[1] = n1; nl[2] = np;
    theMesh->e2n_setAll(e5, nl);
    nl[0] = e1; nl[1] = e2; nl[2] = e6;
    theMesh->e2e_setAll(e5, nl);
  }
  if ((n_e3 < n2_e3) || ((n_e3 == 2) && (n2_e3 == 0))) {
    nl[0] = n2; nl[1] = n; nl[2] = np;
    theMesh->e2n_setAll(e6, nl);
    nl[0] = e3; nl[1] = e5; nl[2] = e4;
    theMesh->e2e_setAll(e6, nl);
  }
  else {
    nl[0] = n; nl[1] = n2; nl[2] = np;
    theMesh->e2n_setAll(e6, nl);
    nl[0] = e3; nl[1] = e4; nl[2] = e5;
    theMesh->e2e_setAll(e6, nl);
  }
  theMesh->e2n_replace(e2, n, np);
  theMesh->e2n_replace(e4, n, np);
  // Element-to-element updates
  theMesh->e2e_replace(e1, e2, e5);
  theMesh->e2e_replace(e2, e1, e5);
  theMesh->e2e_replace(e3, e4, e6);
  theMesh->e2e_replace(e4, e3, e6);
  // Node-to-node updates
  int i;
  for (i=0; i<nnCount; i++) {
    printf("np_nodes[%d] = %d\n", i, np_nodes[i]);
    theMesh->n2n_remove(n, np_nodes[i]);
    theMesh->n2n_remove(np_nodes[i], n);
    theMesh->n2n_add(np, np_nodes[i]);
    theMesh->n2n_add(np_nodes[i], np);
  }
  theMesh->n2n_add(n, np);
  theMesh->n2n_add(np, n);
  theMesh->n2n_add(n, n1);
  theMesh->n2n_add(n1, n);
  theMesh->n2n_add(n, n2);
  theMesh->n2n_add(n2, n);
  // Node-to-element updates
  for (i=0; i<neCount; i++) { 
    theMesh->n2e_remove(n, np_elems[i]);
    theMesh->e2n_replace(np_elems[i], n, np);
    theMesh->n2e_add(np, np_elems[i]);
  }
  theMesh->n2e_add(n, e5);
  theMesh->n2e_add(n, e6);
  theMesh->n2e_add(n1, e5);
  theMesh->n2e_add(n2, e6);
  theMesh->n2e_add(np, e5);
  theMesh->n2e_add(np, e6);
  return np;
}

// element_bisect and helpers
/*
void FEM_Adapt::element_bisect(int e1) 
{
  int mn = theMesh->getMarkedNode(e1);
  int mnIdx = theMesh->getMarkedNodeIdx(e1);
  int n1Idx = (mnIdx+1)%3;
  int n1 = theMesh->e2n_getNode(e1, n1Idx);
  int n2Idx = (n1Idx+1)%3;
  int n2 = theMesh->e2n_getNode(e1, n2Idx);
  int e2 = theMesh->e2e_getNbr(e1, get_edge_index(n1Idx, n2Idx));
  int mn2 = theMesh->getMarkedNode(e2);
  while (!((mn2 != n1) && (mn2 != n2))) {
    element_bisect(e2);
    mn = theMesh->getMarkedNode(e1);
    mnIdx = theMesh->getMarkedNodeIdx(e1);
    n1Idx = (mnIdx+1)%3;
    n1 = theMesh->e2n_getNode(e1, n1Idx);
    n2Idx = (n1Idx+1)%3;
    n2 = theMesh->e2n_getNode(e1, n2Idx);
    e2 = theMesh->e2e_getNbr(e1, get_edge_index(n1Idx, n2Idx));
    mn2 = theMesh->getMarkedNode(e2);
  }
  (void) edge_bisect(e1, n1, n2);
}
*/


int FEM_Adapt::newSlot(FEM_DataAttribute *validAttr){
	FEM_Entity *entity = validAttr->getEntity();
	AllocTable2d<int> *validData = &validAttr->getInt();
	int length = validAttr->getLength();
	
/*	printf("valid array before new element length %d\n",length);
	printValidArray(validAttr);*/
	for(int i=0;i<length;i++){
		if((*validData)[i][0] == 0){
		  (*validData)[i][0] = 1;
/*			printf("valid array after new element at %d \n",i);
			printValidArray(validAttr);*/
		  return i;
		}
	}
	entity->setLength(length+1);
	validData = &validAttr->getInt();
	(*validData)[length][0] = 1;
/*	printf("valid array after new element current length %d\n",validAttr->getLength());
	printValidArray(validAttr);*/
	return length;
};

void FEM_Adapt::invalidateSlot(FEM_DataAttribute *validAttr,int slotNumber){
	AllocTable2d<int> *validData = &validAttr->getInt();
	(*validData)[slotNumber][0] = 0;
};

int FEM_Adapt::newNode(){
	if(nodeValid){
		return  newSlot(nodeValid);
	}else{
		return theMesh->node.size();
	}
};

int FEM_Adapt::newElement(){
	if(elemValid){
		return newSlot(elemValid);
	}else{
		FEM_Elem *elem = (FEM_Elem *)theMesh->lookup(FEM_ELEM,"newElement");
		return elem->size();
	}
};

void FEM_Adapt::deleteNode(int n){
  theMesh->n2e_removeAll(n);
  theMesh->n2n_removeAll(n);
  invalidateSlot(nodeValid,n);
};

void FEM_Adapt::deleteElement(int e){
  theMesh->e2e_removeAll(e);
  invalidateSlot(elemValid,e);
};

void FEM_Adapt::printValidArray(FEM_DataAttribute *validAttr){
  FEM_Entity *entity = validAttr->getEntity();
  AllocTable2d<int> *validData = &validAttr->getInt();
  int length = entity->getMax();
  /*	printf("Valid array---\n");
	for(int i=0;i<length;i++){
	printf("%d %d\n",i,(*validData)[i][0]);
	}*/
};


// Helpers

FEM_DataAttribute *FEM_Adapt::validDataFor(int entityNumber){
	FEM_Entity *entity = theMesh->lookup(entityNumber,"validDataFor");
	FEM_DataAttribute *validAttribute = (FEM_DataAttribute *)entity->lookup(FEM_IS_VALID,"validDataFor");
	return validAttribute;
}

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

void FEM_Adapt::adj_traverse(int n, int startNode, int stopNode, int startElem,
			     int stopElem, int *nn, int *ne, int *nodeList,
			     int *elemList)
{
  int elm = startElem, nod = startNode;
  int nIdx;
  (*nn) = 0;
  (*ne) = 0;
  if (elm == -1) {
    nodeList[*nn] = nod; (*nn)++;
  }
  while ((elm != stopElem) && (elm > -1)) {
    nodeList[*nn] = nod; (*nn)++;
    elemList[*ne] = elm; (*ne)++;
    nIdx = 3 - find_local_node_index(elm,n) - find_local_node_index(elm,nod);
    nod = theMesh->e2n_getNode(elm, nIdx);
    elm = theMesh->e2e_getNbr(elm, get_edge_index(find_local_node_index(elm,n),
						  nIdx));
  }
  if (elm == stopElem) {
    nodeList[*nn] = nod; (*nn)++;
    elemList[*ne] = elm; (*ne)++;
    nodeList[*nn] = stopNode; (*nn)++;
  }
  else {
    nodeList[*nn] = nod; (*nn)++;
    int elm = stopElem;
    int nod = stopNode;
    while (elm > -1) {
      nodeList[*nn] = nod; (*nn)++;
      elemList[*ne] = elm; (*ne)++;
      nIdx = 3 - find_local_node_index(elm,n) - find_local_node_index(elm,nod);
      nod = theMesh->e2n_getNode(elm, nIdx);
      elm = theMesh->e2e_getNbr(elm, get_edge_index(find_local_node_index(elm,n), nIdx));
    }
    nodeList[*nn] = nod; (*nn)++;
  }
}

void FEM_Adapt::findAdjData(int n1, int n2, int *e1, int *e2, int *en1, 
			     int *en2, int *en3, int *n3, int *edge, int *nbr)
{
  (*e1) = theMesh->getElementOnEdge(n1, n2);
  CkAssert((*e1) >= 0); // e1 must exist and be local
  (*en1) = find_local_node_index((*e1), n1);
  (*en2) = find_local_node_index((*e1), n2);
  (*en3) = 3 - (*en1) - (*en2);
  (*e2) = theMesh->e2e_getNbr((*e1), get_edge_index((*en1), (*en2)));
  (*edge) = get_edge_index((*en2), (*en3));
  (*nbr) = theMesh->e2e_getNbr((*e1), (*edge));
  (*n3) = theMesh->e2n_getNode((*e1), (*en3));
}

void FEM_Adapt::findAdjData(int n1, int n2, int e2, int *en1, int *en2, 
			    int *en3, int *n4, int *edge, int *nbr)
{
  (*en1) = find_local_node_index(e2, n1);
  (*en2) = find_local_node_index(e2, n2);
  (*en3) = 3 - (*en1) - (*en2);
  (*edge) = get_edge_index((*en2), (*en3));
  (*nbr) = theMesh->e2e_getNbr(e2, (*edge));
  (*n4) = theMesh->e2n_getNode(e2, (*en3));
}

//init the validData method


