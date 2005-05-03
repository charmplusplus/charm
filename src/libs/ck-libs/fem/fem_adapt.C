#include "fem_adapt.h"  

//init the validData method
AllocTable2d<int> *FEM_Adapt::validDataFor(int entityNumber){
	FEM_Entity *entity = theMesh->lookup(entityNumber,"validDataFor");
	FEM_DataAttribute *validAttribute = (FEM_DataAttribute *)entity->lookup(FEM_VALID,"validDataFor");
	if(validAttribute==NULL){
		return NULL;
	}else{
		AllocTable2d<int> *validData = &validAttribute->getInt();
		return validData;
	}
}



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
  while ((elm != stopElem) && (elm != -1)) {
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
    int elm = stopElem;
    int nod = stopNode;
    while (elm != -1) {
      nodeList[*nn] = nod; (*nn)++;
      elemList[*ne] = elm; (*ne)++;
      nIdx = 3 - find_local_node_index(elm,n) - find_local_node_index(elm,nod);
      nod = theMesh->e2n_getNode(elm, nIdx);
      elm = theMesh->e2e_getNbr(elm, get_edge_index(find_local_node_index(elm,n), nIdx));
    }
    nodeList[*nn] = nod; (*nn)++;
  }

}

// edge_flip and helpers
int FEM_Adapt::edge_flip(int n1, int n2) 
{
  int e1 = theMesh->getElementOnEdge(n1, n2);
  if (e1 < 0) {
    CkPrintf("ERROR: edge_flip: no element with edge [%d,%d]\n", n1, n2);
    return 0;	     
  }
  return edge_flip(e1, n1, n2);
}
int FEM_Adapt::edge_flip(int e1, int n1, int n2) 
{
  int e1_n1 = find_local_node_index(e1, n1);
  int e1_n2 = find_local_node_index(e1, n2);
  int shared_edge = get_edge_index(e1_n1, e1_n2);
  int e2 = theMesh->e2e_getNbr(e1, shared_edge); 
  if (e2 == -1) { // there is no neighbor; can't flip; 
    return 1;  // this is a no-op; could return 0 if we want to flag as error
  }
  return edge_flip_help(e1, e2, n1, n2, e1_n1, e1_n2);
}
int FEM_Adapt::edge_flip_help(int e1, int e2, int n1, int n2, int e1_n1, 
			      int e1_n2) 
{
  int e1_n3 = 3 - e1_n1 - e1_n2;
  int mod_edge1 = get_edge_index(e1_n2, e1_n3);
  int e1nbr = theMesh->e2e_getNbr(e1, mod_edge1);
  int e2_n1 = find_local_node_index(e2, n1);
  int e2_n2 = find_local_node_index(e2, n2);
  int e2_n3 = 3 - e2_n1 - e2_n2;
  int mod_edge2 = get_edge_index(e2_n1, e2_n3);
  int e2nbr = theMesh->e2e_getNbr(e2, mod_edge2);
  int n3 = theMesh->e2n_getNode(e1, e1_n3);
  int n4 = theMesh->e2n_getNode(e2, e2_n3);
  
  // Element-to-node updates
  theMesh->e2n_setIndex(e1, e1_n2, n4);
  theMesh->e2n_setIndex(e2, e2_n1, n3);
  // Element-to-element updates
  theMesh->e2e_replace(e1, e1nbr, e2nbr);
  theMesh->e2e_replace(e2, e2nbr, e1nbr);
  theMesh->e2e_replace(e1nbr, e1, e2);
  theMesh->e2e_replace(e2nbr, e2, e1);
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

// edge_bisect and helpers  
int FEM_Adapt::edge_bisect(int n1, int n2) 
{
  int e1 = theMesh->getElementOnEdge(n1, n2);
  if (e1 < 0) {
    CkPrintf("ERROR: edge_bisect: no element with edge [%d,%d]\n", n1, n2);
    return -1;	     
  }
  return edge_bisect(e1, n1, n2);
}
int FEM_Adapt::edge_bisect(int e1, int n1, int n2)
{
  int e1_n1 = find_local_node_index(e1, n1);
  int e1_n2 = find_local_node_index(e1, n2);
  int shared_edge = get_edge_index(e1_n1, e1_n2);
  int e2 = theMesh->e2e_getNbr(e1, shared_edge); 
  return edge_bisect_help(e1, e2, n1, n2, e1_n1, e1_n2);
}
int FEM_Adapt::edge_bisect_help(int e1, int e2, int n1, int n2, int e1_n1, 
				int e1_n2)
{
  int e1_n3 = 3 - e1_n1 - e1_n2;
  int mod_edge1 = get_edge_index(e1_n2, e1_n3);
  int e1nbr = theMesh->e2e_getNbr(e1, mod_edge1);
  int n3 = theMesh->e2n_getNode(e1, e1_n3);
  int n5 = newNode();
  int e3 = newElement();
  int e2_n1;
  int e2_n2;
  int e2_n3;
  int mod_edge2;
  int e2nbr;
  int n4;
  int e4;
  if (e2 >= 0) { // neighbor exists
    e2_n1 = find_local_node_index(e2, n1);
    e2_n2 = find_local_node_index(e2, n2);
    e2_n3 = 3 - e2_n1 - e2_n2;
    mod_edge2 = get_edge_index(e2_n1, e2_n3);
    e2nbr = theMesh->e2e_getNbr(e2, mod_edge2);
    n4 = theMesh->e2n_getNode(e2, e2_n3);
    e4 = newElement();
  }
  
  // Element-to-node updates
  theMesh->e2n_setIndex(e1, e1_n2, n5);
  theMesh->e2n_setIndex(e3, e1_n1, n5);
  theMesh->e2n_setIndex(e3, e1_n2, n2);
  theMesh->e2n_setIndex(e3, e1_n3, n3);
  if (e2 >= 0) {
    theMesh->e2n_setIndex(e2, e2_n2, n5);
    theMesh->e2n_setIndex(e4, e2_n2, n2);
    theMesh->e2n_setIndex(e4, e2_n1, n5);
    theMesh->e2n_setIndex(e4, e2_n3, n4);
  }
  // Element-to-element updates
  theMesh->e2e_replace(e1, e1nbr, e3);
  theMesh->e2e_replace(e1nbr, e1, e3);
  int nl[3];
  if (e2 >= 0) {
    theMesh->e2e_replace(e2, e2nbr, e4);
    theMesh->e2e_replace(e2nbr, e2, e4);
    nl[0] = e1; nl[1] = e4; nl[2] = e1nbr;
    theMesh->e2e_setAll(e3, nl);
    nl[0] = e2; nl[1] = e3; nl[2] = e2nbr;
    theMesh->e2e_setAll(e4, nl);
  }
  else {
    nl[0] = e1; nl[1] = -1; nl[2] = e1nbr;
    theMesh->e2e_setAll(e3, nl);
  }
  // Node-to-node updates
  theMesh->n2n_replace(n1, n2, n5);
  theMesh->n2n_replace(n2, n1, n5);
  theMesh->n2n_add(n3, n5);
  theMesh->n2n_add(n5, n1);
  theMesh->n2n_add(n5, n2);
  theMesh->n2n_add(n5, n3);
  if (e2 >= 0) {
    theMesh->n2n_add(n4, n5);
    theMesh->n2n_add(n5, n4);
  }
  // Node-to-element updates
  theMesh->n2e_replace(n2, e1, e3);
  theMesh->n2e_add(n3, e3);
  theMesh->n2e_add(n5, e1);
  theMesh->n2e_add(n5, e3);
  if (e2 >= 0) {
    theMesh->n2e_replace(n2, e2, e4);
    theMesh->n2e_add(n4, e4);  
    theMesh->n2e_add(n5, e2);
    theMesh->n2e_add(n5, e4);
  }
  return n5;
}
  
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
  if (e2 >= 0) {
    e2_n1 = find_local_node_index(e2, n1);
    e2_n2 = find_local_node_index(e2, n2);
    e2_n3 = 3 - e2_n1 - e2_n2;
    mod_edge3 = get_edge_index(e2_n1, e2_n3);
    e2nbr1 = theMesh->e2e_getNbr(e2, mod_edge3);
    mod_edge4 = get_edge_index(e2_n2, e2_n3);
    e2nbr2 = theMesh->e2e_getNbr(e2, mod_edge4);
    n4 = theMesh->e2n_getNode(e2, e2_n3);
  }

  int *n2_nbrNodes, *n2_nbrElems;
  int nnsize, nesize;
  theMesh->n2n_getAll(n2, n2_nbrNodes, &nnsize);
  theMesh->n2e_getAll(n2, n2_nbrElems, &nesize);
  
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
    theMesh->n2n_replace(n2_nbrNodes[i], n2, n1);
    theMesh->n2n_add(n1, n2_nbrNodes[i]);
  }
  theMesh->n2n_remove(n1, n2);
  theMesh->n2n_remove(n1, n1); // assume added by loop add above; not necessary
  theMesh->n2n_remove(n1, n3); // assume duplicated by the loop add above
  theMesh->n2n_remove(n1, n4); // assume duplicated by the loop add above
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
  adj_traverse(n, n1, e2, n2, e4, &nnCount, &neCount, np_nodes, np_elems);

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
    theMesh->n2n_add(np, np_nodes[i]);
    theMesh->n2n_remove(n, np_nodes[i]);
  }
  theMesh->n2n_add(n, np);
  theMesh->n2n_add(n, n1);
  theMesh->n2n_add(n, n2);
  theMesh->n2n_add(np, n);
  // Node-to-element updates
  for (i=0; i<neCount; i++) { 
    theMesh->n2e_add(np, np_elems[i]);
    theMesh->n2e_remove(n, np_elems[i]);
  }
  theMesh->n2e_add(n, e5);
  theMesh->n2e_add(n, e6);
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


int FEM_Adapt::newSlot(AllocTable2d<int> *validData){
	int length = validData->size();
	for(int i=0;i<length;i++){
		if((*validData)[i][0] == 0){
			(*validData)[i][0] = 1;
			return i;
		}
	}
	return length;
};

void FEM_Adapt::invalidateSlot(AllocTable2d<int> *validData,int slotNumber){
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
	invalidateSlot(nodeValid,n);
};

void FEM_Adapt::deleteElement(int e){
	invalidateSlot(elemValid,e);
};
