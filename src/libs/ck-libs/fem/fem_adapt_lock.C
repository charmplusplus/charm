#include "fem_adapt_lock.h"
#include "fem_mesh_modify.h"


int FEM_AdaptL::lockNodes(int *gotlocks, int *lockrnodes, int numRNodes, int *lockwnodes, int numWNodes) {
  bool donelocks = false;
  int numNodes = numRNodes + numWNodes;
  for(int i=0; i<numNodes; i++) gotlocks[i] = 0;
  int tryCounts=0;
  while(!donelocks) {
    for(int i=0; i<numRNodes; i++) {
      if(lockrnodes[i]==-1) {
	gotlocks[i] = 1;
	continue;
      }
      if(FEM_Is_ghost_index(lockrnodes[i])) {
	if(!theMesh->node.ghost->is_valid(FEM_To_ghost_index(lockrnodes[i]))) return -1;
      } else {
	if(!theMesh->node.is_valid(lockrnodes[i])) return -1;
      }
      if(gotlocks[i]<=0) gotlocks[i] = FEM_Modify_LockN(theMesh, lockrnodes[i], 1);
    }
    for(int i=0; i<numWNodes; i++) {
      if(lockwnodes[i]==-1) {
	gotlocks[numRNodes+i] = 1;
	continue;
      }
      if(FEM_Is_ghost_index(lockwnodes[i])) {
	if(!theMesh->node.ghost->is_valid(FEM_To_ghost_index(lockwnodes[i]))) return -1;
      } else {
	if(!theMesh->node.is_valid(lockwnodes[i])) return -1;
      }
      if(gotlocks[numRNodes+i]<=0) gotlocks[numRNodes+i] = FEM_Modify_LockN(theMesh, lockwnodes[i], 0);
    }
    bool tmplocks = true;
    for(int i=0; i<numNodes; i++) {
      tmplocks = tmplocks && (gotlocks[i]>0);
    }
    if(tmplocks) {
      donelocks = true;
    }
    else {
      tryCounts++;
      if(tryCounts>=10) return -1;
      CthYield(); //spin for a while
    }
  }
  return 1;
}

int FEM_AdaptL::unlockNodes(int *gotlocks, int *lockrnodes, int numRNodes, int *lockwnodes, int numWNodes) {
  bool donelocks = false;
  int numNodes = numRNodes + numWNodes;
  int *ungetlocks = (int*)malloc(numNodes*sizeof(int));
  while(!donelocks) {
    for(int i=0; i<numRNodes; i++) {
      if(lockrnodes[i]==-1) {
	gotlocks[i] = 1;
	continue;
      }
      if(FEM_Is_ghost_index(lockrnodes[i])) {
	if(!theMesh->node.ghost->is_valid(FEM_To_ghost_index(lockrnodes[i]))) {
	  //free(ungetlocks);
	  //return -1;
	}
      } else {
	if(!theMesh->node.is_valid(lockrnodes[i])) {
	  //free(ungetlocks);
	  //return -1;
	}
      }
      if(gotlocks[i]>0) ungetlocks[i] = FEM_Modify_UnlockN(theMesh, lockrnodes[i], 1);
      else ungetlocks[i] = 1;
    }
    for(int i=0; i<numWNodes; i++) {
      if(lockwnodes[i]==-1) {
	ungetlocks[numRNodes+i] = 1;
	continue;
      }
      if(FEM_Is_ghost_index(lockwnodes[i])) {
	if(!theMesh->node.ghost->is_valid(FEM_To_ghost_index(lockwnodes[i]))) {
	  //free(ungetlocks);
	  //return -1;
	}
      } else {
	if(!theMesh->node.is_valid(lockwnodes[i])) {
	  //free(ungetlocks);
	  //return -1;
	}
      }
      if(gotlocks[numRNodes+i]>0) ungetlocks[numRNodes+i] = FEM_Modify_UnlockN(theMesh, lockwnodes[i], 0);
      else ungetlocks[numRNodes+i] = 1;
    }
    bool tmplocks = true;
    for(int i=0; i<numNodes; i++) {
      tmplocks = tmplocks && (ungetlocks[i]>0);
    }
    if(tmplocks) donelocks = true;
    else CthYield(); //block for a while
  }
  return 1;
}


int FEM_AdaptL::edge_flip(int n1, int n2) {
  int e1, e1_n1, e1_n2, e1_n3, n3;
  int e2, e2_n1, e2_n2, e2_n3, n4;
  int numNodes = 4;
  int *locknodes = (int*)malloc(numNodes*sizeof(int));
  int *gotlocks = (int*)malloc(numNodes*sizeof(int));
  bool done = false;
  int isEdge = 0;

  isEdge = findAdjData(n1, n2, &e1, &e2, &e1_n1, &e1_n2, &e1_n3, &e2_n1, &e2_n2, &e2_n3,&n3, &n4);
  if(isEdge == -1) {
    CkPrintf("Error: Flip %d->%d not done as it is no longer a valid edge\n",n1,n2);
    return -1;
  }
  locknodes[0] = n1;
  locknodes[1] = n2;
  locknodes[2] = n3;
  locknodes[3] = n4;
  while(!done) {
    int gotlock = lockNodes(gotlocks, locknodes, 0, locknodes, numNodes);
    isEdge = findAdjData(n1, n2, &e1, &e2, &e1_n1, &e1_n2, &e1_n3, &e2_n1, &e2_n2, &e2_n3,&n3, &n4);
    if(isEdge == -1) {
      unlockNodes(gotlocks, locknodes, 0, locknodes, numNodes);
      CkPrintf("Error: Flip %d->%d not done as it is no longer a valid edge\n",n1,n2);
      return -1;
    }
    if(gotlock==1 && locknodes[2]==n3 && locknodes[3]==n4) {
      done = true;
    }
    else {
      unlockNodes(gotlocks, locknodes, 0, locknodes, numNodes);
      locknodes[2] = n3;
      locknodes[3] = n4;
      CthYield();
    }
  }
  if ((e1 == -1) || (e2 == -1)) {
    unlockNodes(gotlocks, locknodes, 0, locknodes, numNodes);
    return 0;
  }
  int ret = edge_flip_help(e1, e2, n1, n2, e1_n1, e1_n2, e1_n3, n3, n4);
  unlockNodes(gotlocks, locknodes, 0, locknodes, numNodes);

  free(locknodes);
  free(gotlocks);
  return ret;
}

int FEM_AdaptL::edge_bisect(int n1, int n2) {
  int e1, e1_n1, e1_n2, e1_n3, n3;
  int e2, e2_n1, e2_n2, e2_n3, n4;
  int numNodes = 4;
  int *locknodes = (int*)malloc(numNodes*sizeof(int));
  int *gotlocks = (int*)malloc(numNodes*sizeof(int));
  bool done = false;
  int isEdge = 0;

  isEdge = findAdjData(n1, n2, &e1, &e2, &e1_n1, &e1_n2, &e1_n3, &e2_n1, &e2_n2, &e2_n3,&n3, &n4);
  if(isEdge == -1) {
    CkPrintf("Error: Bisect %d->%d not done as it is no longer a valid edge\n",n1,n2);
    return -1;
  }
  locknodes[0] = n1;
  locknodes[1] = n2;
  locknodes[2] = n3;
  locknodes[3] = n4;
  while(!done) {
    int gotlock = lockNodes(gotlocks, locknodes, 0, locknodes, numNodes);
    isEdge = findAdjData(n1, n2, &e1, &e2, &e1_n1, &e1_n2, &e1_n3, &e2_n1, &e2_n2, &e2_n3,&n3, &n4);
    if(isEdge == -1) {
      unlockNodes(gotlocks, locknodes, 0, locknodes, numNodes);
      CkPrintf("Error: Bisect %d->%d not done as it is no longer a valid edge\n",n1,n2);
      return -1;
    }
    if(gotlock==1 && locknodes[2]==n3 && locknodes[3]==n4) {
      done = true;
    }
    else {
      unlockNodes(gotlocks, locknodes, 0, locknodes, numNodes);
      locknodes[2] = n3;
      locknodes[3] = n4;
      CthYield();
    }
  }
  int ret = edge_bisect_help(e1, e2, n1, n2, e1_n1, e1_n2, e1_n3, e2_n1, e2_n2, e2_n3, n3, n4);
  unlockNodes(gotlocks, locknodes, 0, locknodes, numNodes);

  free(locknodes);
  free(gotlocks);
  return ret;
}

int FEM_AdaptL::vertex_remove(int n1, int n2) {
  int e1, e1_n1, e1_n2, e1_n3, n3;
  int e2, e2_n1, e2_n2, e2_n3, n4;
  int numNodes = 5;
  int numElems = 2;
  int *locknodes = (int*)malloc(numNodes*sizeof(int));
  int *gotlocks = (int*)malloc(numNodes*sizeof(int));
  bool done = false;
  int isEdge = 0;

  isEdge = findAdjData(n1, n2, &e1, &e2, &e1_n1, &e1_n2, &e1_n3, &e2_n1, &e2_n2, &e2_n3,&n3, &n4);
  if(isEdge == -1) {
    CkPrintf("Error: Vertex Remove %d->%d not done as it is no longer a valid edge\n",n1,n2);
    return -1;
  }
  if (e1 == -1) return 0;
  // find n5
  int *nbrNodes, nnsize, n5;
  theMesh->n2n_getAll(n1, &nbrNodes, &nnsize);
  if(!(nnsize == 4 || (nnsize==3 && e2==-1))) {
    CkPrintf("Error: Vertex Remove %d->%d on node %d with %d connections (!= 4)\n",n1,n2,n1,nnsize);
    return -1;    
  }
  for (int i=0; i<nnsize; i++) {
    if ((nbrNodes[i] != n2) && (nbrNodes[i] != n3) && (nbrNodes[i] != n4)) {
      n5 = nbrNodes[i];
      break;
    }
  }
  locknodes[0] = n1;
  locknodes[1] = n2;
  locknodes[2] = n3;
  locknodes[3] = n4;
  locknodes[4] = n5;
  while(!done) {
    int gotlock = lockNodes(gotlocks, locknodes, 0, locknodes, numNodes);
    isEdge = findAdjData(n1, n2, &e1, &e2, &e1_n1, &e1_n2, &e1_n3, &e2_n1, &e2_n2, &e2_n3,&n3, &n4);
    if(isEdge == -1) {
      unlockNodes(gotlocks, locknodes, 0, locknodes, numNodes);
      CkPrintf("Error: Vertex Remove %d->%d not done as it is no longer a valid edge\n",n1,n2);
      return -1;
    }
    if (e1 == -1) {
      unlockNodes(gotlocks, locknodes, 0, locknodes, numNodes);
      return 0;
    }
    // find n5
    int *nbrNodes, nnsize, n5;
    theMesh->n2n_getAll(n1, &nbrNodes, &nnsize);
    for (int i=0; i<nnsize; i++) {
      if ((nbrNodes[i] != n2) && (nbrNodes[i] != n3) && (nbrNodes[i] != n4)) {
	n5 = nbrNodes[i];
	break;
      }
    }
    if(!(nnsize == 4 || (nnsize==3 && e2==-1))) {
      unlockNodes(gotlocks, locknodes, 0, locknodes, numNodes);
      CkPrintf("Error: Vertex Remove %d->%d on node %d with %d connections (!= 4)\n",n1,n2,n1,nnsize);
      return -1;    
    }
    if(gotlock==1 && locknodes[2]==n3 && locknodes[3]==n4 && locknodes[4]==n5) {
      done = true;
    }
    else {
      unlockNodes(gotlocks, locknodes, 0, locknodes, numNodes);
      locknodes[2] = n3;
      locknodes[3] = n4;
      locknodes[4] = n5;
      CthYield();
    }
  }
  int ret = vertex_remove_help(e1, e2, n1, n2, e1_n1, e1_n2, e1_n3, e2_n1, e2_n2, e2_n3, n3, n4, n5);
  unlockNodes(gotlocks, locknodes, 0, locknodes, numNodes);

  free(locknodes);
  free(gotlocks);
  return ret;
}

int FEM_AdaptL::edge_contraction(int n1, int n2) {
  int e1, e1_n1, e1_n2, e1_n3, n3;
  int e2, e2_n1, e2_n2, e2_n3, n4;
  int numNodes = 4;
  int numElems = 2;
  int *locknodes = (int*)malloc(numNodes*sizeof(int));
  int *gotlocks = (int*)malloc(numNodes*sizeof(int));
  bool done = false;
  int isEdge = 0;

  if(n1<0 || n2<0) return -1; //should not contract an edge which is not local
  /*
  //if either of the nodes is on the boundary, do not contract
  int n1_bound, n2_bound;
  FEM_Mesh_dataP(theMesh, FEM_NODE, FEM_BOUNDARY, &n1_bound, n1, 1 , FEM_INT, 1);   
  FEM_Mesh_dataP(theMesh, FEM_NODE, FEM_BOUNDARY, &n2_bound, n2, 1 , FEM_INT, 1);   
  CkVec<FEM_Attribute *>*attrs = (theMesh->node).getAttrVec();
  for (int i=0; i<attrs->size(); i++) {
    FEM_Attribute *a = (FEM_Attribute *)(*attrs)[i];
    if (a->getAttr()==FEM_BOUNDARY) {
      FEM_DataAttribute *d = (FEM_DataAttribute*)a;
      if(d->getInt()[n1][0] < 0) return -1; //it is on the boundary
      if(d->getInt()[n2][0] < 0) return -1; //it is on the boundary
    }
  }
  */
  isEdge = findAdjData(n1, n2, &e1, &e2, &e1_n1, &e1_n2, &e1_n3, &e2_n1, &e2_n2, &e2_n3,&n3, &n4);
  if(isEdge == -1) {
    CkPrintf("Edge Contract %d->%d not done as it is no longer a valid edge\n",n1,n2);
    return -1;
  }
  locknodes[0] = n1;
  locknodes[1] = n2;
  locknodes[2] = n3;
  locknodes[3] = n4;
  if (e1 == -1) return 0;
  while(!done) {
    int gotlock = lockNodes(gotlocks, locknodes, 0, locknodes, numNodes);
    isEdge = findAdjData(n1, n2, &e1, &e2, &e1_n1, &e1_n2, &e1_n3, &e2_n1, &e2_n2, &e2_n3,&n3, &n4);
    if(isEdge == -1) {
      unlockNodes(gotlocks, locknodes, 0, locknodes, numNodes);
      CkPrintf("Edge contract %d->%d not done as it is no longer a valid edge\n",n1,n2);
      return -1;
    }
    if (e1 == -1) {
      unlockNodes(gotlocks, locknodes, 0, locknodes, numNodes);
      return 0;
    }
    if(gotlock==1 && locknodes[2]==n3 && locknodes[3]==n4) {
      done = true;
    }
    else {
      unlockNodes(gotlocks, locknodes, 0, locknodes, numNodes);
      locknodes[2] = n3;
      locknodes[3] = n4;
      CthYield();
    }
  }
  int ret = edge_contraction_help(e1, e2, n1, n2, e1_n1, e1_n2, e1_n3, e2_n1, e2_n2, e2_n3, n3, n4);
  unlockNodes(gotlocks, locknodes, 0, locknodes, numNodes);

  free(locknodes);
  free(gotlocks);
  return ret;
}
