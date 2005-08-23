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
  free(ungetlocks);  
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
  int numtries = 0;

  isEdge = findAdjData(n1, n2, &e1, &e2, &e1_n1, &e1_n2, &e1_n3, &e2_n1, &e2_n2, &e2_n3,&n3, &n4);
  if(isEdge == -1) {
    CkPrintf("Warning: Flip %d->%d not done as it is no longer a valid edge\n",n1,n2);
    free(locknodes);
    free(gotlocks);
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
      CkPrintf("Warning: Flip %d->%d not done as it is no longer a valid edge\n",n1,n2);
      free(locknodes);
      free(gotlocks);
      return -1;
    }
    if(gotlock==1 && locknodes[2]==n3 && locknodes[3]==n4) {
      done = true;
    }
    else {
      unlockNodes(gotlocks, locknodes, 0, locknodes, numNodes);
      locknodes[2] = n3;
      locknodes[3] = n4;
      numtries++;
      if(numtries>=10) {
	CkPrintf("Possibly a livelock\n");
	numtries = 0;
      }
      CthYield();
    }
  }
  if ((e1 == -1) || (e2 == -1)) {
    unlockNodes(gotlocks, locknodes, 0, locknodes, numNodes);
    free(locknodes);
    free(gotlocks);
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
  int numtries = 0;

  isEdge = findAdjData(n1, n2, &e1, &e2, &e1_n1, &e1_n2, &e1_n3, &e2_n1, &e2_n2, &e2_n3,&n3, &n4);
  if(isEdge == -1) {
    CkPrintf("Warning: Bisect %d->%d not done as it is no longer a valid edge\n",n1,n2);
    free(locknodes);
    free(gotlocks);
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
      CkPrintf("Warning: Bisect %d->%d not done as it is no longer a valid edge\n",n1,n2);
      free(locknodes);
      free(gotlocks);
      return -1;
    }
    if(gotlock==1 && locknodes[2]==n3 && locknodes[3]==n4) {
      done = true;
    }
    else {
      unlockNodes(gotlocks, locknodes, 0, locknodes, numNodes);
      locknodes[2] = n3;
      locknodes[3] = n4;
      numtries++;
      if(numtries>=10) {
	CkPrintf("Possibly a livelock\n");
	numtries = 0;
      }
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
  int numtries = 0;

  isEdge = findAdjData(n1, n2, &e1, &e2, &e1_n1, &e1_n2, &e1_n3, &e2_n1, &e2_n2, &e2_n3,&n3, &n4);
  if(isEdge == -1) {
    CkPrintf("Warning: Vertex Remove %d->%d not done as it is no longer a valid edge\n",n1,n2);
    free(locknodes);
    free(gotlocks);
    return -1;
  }
  if (e1 == -1) {
    free(locknodes);
    free(gotlocks);
    return 0;
  }
  // find n5
  int *nbrNodes, nnsize, n5;
  theMesh->n2n_getAll(n1, &nbrNodes, &nnsize);
  if(!(nnsize == 4 || (nnsize==3 && e2==-1))) {
    CkPrintf("Warning: Vertex Remove %d->%d on node %d with %d connections (!= 4)\n",n1,n2,n1,nnsize);
    free(locknodes);
    free(gotlocks);
    return -1;    
  }
  for (int i=0; i<nnsize; i++) {
    if ((nbrNodes[i] != n2) && (nbrNodes[i] != n3) && (nbrNodes[i] != n4)) {
      n5 = nbrNodes[i];
      break;
    }
  }
  free(nbrNodes);
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
      CkPrintf("Warning: Vertex Remove %d->%d not done as it is no longer a valid edge\n",n1,n2);
      free(locknodes);
      free(gotlocks);
      return -1;
    }
    if (e1 == -1) {
      unlockNodes(gotlocks, locknodes, 0, locknodes, numNodes);
      free(locknodes);
      free(gotlocks);
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
    free(nbrNodes);
    if(!(nnsize == 4 || (nnsize==3 && e2==-1))) {
      unlockNodes(gotlocks, locknodes, 0, locknodes, numNodes);
      CkPrintf("Warning: Vertex Remove %d->%d on node %d with %d connections (!= 4)\n",n1,n2,n1,nnsize);
      free(locknodes);
      free(gotlocks);
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
      numtries++;
      if(numtries>=10) {
	CkPrintf("Possibly a livelock\n");
	numtries = 0;
      }
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
  int numtries = 0;

  if(n1<0 || n2<0) {
    free(locknodes);
    free(gotlocks);
    return -1; //should not contract an edge which is not local
  }
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
    free(locknodes);
    free(gotlocks);
    return -1;
  }
  locknodes[0] = n1;
  locknodes[1] = n2;
  locknodes[2] = n3;
  locknodes[3] = n4;
  if (e1 == -1) {
    free(locknodes);
    free(gotlocks);
    return 0;
  }
  while(!done) {
    int gotlock = lockNodes(gotlocks, locknodes, 0, locknodes, numNodes);
    isEdge = findAdjData(n1, n2, &e1, &e2, &e1_n1, &e1_n2, &e1_n3, &e2_n1, &e2_n2, &e2_n3,&n3, &n4);
    if(isEdge == -1) {
      unlockNodes(gotlocks, locknodes, 0, locknodes, numNodes);
      CkPrintf("Edge contract %d->%d not done as it is no longer a valid edge\n",n1,n2);
      free(locknodes);
      free(gotlocks);
      return -1;
    }
    if (e1 == -1) {
      unlockNodes(gotlocks, locknodes, 0, locknodes, numNodes);
      free(locknodes);
      free(gotlocks);
      return 0;
    }
    if(gotlock==1 && locknodes[2]==n3 && locknodes[3]==n4) {
      done = true;
    }
    else {
      unlockNodes(gotlocks, locknodes, 0, locknodes, numNodes);
      locknodes[2] = n3;
      locknodes[3] = n4;
      numtries++;
      if(numtries>=10) {
	CkPrintf("Possibly a livelock\n");
	numtries = 0;
      }
      CthYield();
    }
  }
  int ret = edge_contraction_help(e1, e2, n1, n2, e1_n1, e1_n2, e1_n3, e2_n1, e2_n2, e2_n3, n3, n4);
  unlockNodes(gotlocks, locknodes, 0, locknodes, numNodes);

  free(locknodes);
  free(gotlocks);
  return ret;
}


int FEM_AdaptL::edge_contraction_help(int e1, int e2, int n1, int n2, int e1_n1,
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
  int numtries = 0;

  //FEM_Modify_Lock(theMesh, adjnodes, 2, adjelems, 2);  

  //New code for updating a node rather than deleting both
  int keepnode=0, deletenode=0, shared=0, n1_shared=0, n2_shared=0;
  n1_shared = theMod->getfmUtil()->isShared(n1);
  n2_shared = theMod->getfmUtil()->isShared(n2);
  if(n1_shared && n2_shared) {
    keepnode = n1;
    deletenode = n2;
    shared = 2;
  }
  else if(n1_shared) {
    //update n1 & delete n2
    keepnode = n1;
    deletenode = n2;
    shared = 1;
  } else if(n2_shared) {
    //update n2 & delete n1
    keepnode = n2;
    deletenode = n1;
    shared = 1;
  } else {
    //keep either
    keepnode = n1;
    deletenode = n2;
  }

  int n1_bound, n2_bound;
  FEM_Mesh_dataP(theMesh, FEM_NODE, FEM_BOUNDARY, &n1_bound, keepnode, 1 , FEM_INT, 1);
  FEM_Mesh_dataP(theMesh, FEM_NODE, FEM_BOUNDARY, &n2_bound, deletenode, 1 , FEM_INT, 1);
  //update keepnode's attributes; choose frac wisely, check if either node is on the boundary, update frac
  FEM_Interpolate *inp = theMod->getfmInp();
  FEM_Interpolate::NodalArgs nm;
  if((n1_bound < 0) && (n2_bound < 0) && (n1_bound != n2_bound)) {
    free(conn);
    free(adjnodes);
    free(adjelems);
    return -1; //they are on different boundaries
  }
  else if(n1_bound<0 && n2_bound<0) {
    nm.frac = 0.5;
    //TODO: must ensure that any of the two nodes is not a corner
    if(isCorner(keepnode)) {
      nm.frac = 1.0;
    } 
    else if(isCorner(deletenode)) {
      nm.frac = 0.0;
    }
  }
  else if(n1_bound < 0) { //keep its attributes
    nm.frac = 1.0;
  }
  else if(n2_bound < 0) {
    if(shared==2) {
      keepnode = n2;
      deletenode = n1;
      nm.frac = 1.0;
    } else {
      nm.frac = 0.0;
    }
  }
  else {
    nm.frac = 0.5;
  }
  nm.nodes[0] = keepnode;
  nm.nodes[1] = deletenode;
  nm.n = keepnode;

  //hack, if it is shared, do not change the attributes, since I am not updating them now across all chunks
  if(n1_shared || n2_shared) {
      nm.frac = 1.0;
  }

  inp->FEM_InterpolateNodeOnEdge(nm);
  if(shared) { //update the attributes of keepnode
  }

  int e1chunk=-1, e2chunk=-1;
  int index = theMod->getIdx();

#ifdef DEBUG_1
  CkPrintf("Edge Contraction, edge %d->%d on chunk %d\n", n1, n2, theMod->getfmUtil()->getIdx());
  CkPrintf("Adjacencies before edge contract\n");
  printAdjacencies(adjnodes, 2, adjelems, 2);
#endif
  e1chunk = FEM_remove_element(theMesh, e1, 0);
#ifdef DEBUG_2
  CkPrintf("Adjacencies after remove element %d\n",e1);
  printAdjacencies(adjnodes, 2, adjelems, 2);
#endif
  e2chunk = FEM_remove_element(theMesh, e2, 0);
#ifdef DEBUG_2
  CkPrintf("Adjacencies after remove element %d\n",e2);
  printAdjacencies(adjnodes, 2, adjelems, 2);
#endif


  int *nbrElems, nesize, echunk;
  theMesh->n2e_getAll(deletenode, &nbrElems, &nesize);
  for (int i=0; i<nesize; i++) {
    if ((nbrElems[i] != e1) && (nbrElems[i] != e2)) {
      theMesh->e2n_getAll(nbrElems[i], conn);
      int *gotlocks = (int*)malloc(3*sizeof(int));
      bool done = false;
      int *eConn = new int[3];
      while(!done) {
	for(int k=0; k<3; k++) {
	  if(conn[k]==n1 || conn[k]==n2 || conn[k]==n3 || conn[k]==n4) conn[k]=-1;
	}
	int gotlock = lockNodes(gotlocks, conn, 0, conn, 3);
	theMesh->e2n_getAll(nbrElems[i], eConn);
	bool valid = false;
	if(FEM_Is_ghost_index(nbrElems[i])) {
	  valid = theMesh->elem[0].ghost->is_valid(FEM_To_ghost_index(nbrElems[i]));
	}
	else {
	  valid = theMesh->elem[0].is_valid(nbrElems[i]);
	}
	if(!valid) {
	  unlockNodes(gotlocks, conn, 0, conn, 3);
	  CkPrintf("Warning: Element %d is no longer valid\n",nbrElems[i]);
	  delete [] eConn;
	  free(conn);
	  free(adjnodes);
	  free(adjelems);
	  free(gotlocks);
	  return -1;
	}
	bool isConn = true;
	for(int k=0; k<3; k++) {
	  if((eConn[k] != conn[k]) && conn[k]!=-1) {
	    isConn = false;
	    break;
	  }
	}
	if(gotlock==1 && isConn) {
	  done = true;
	}
	else {
	  unlockNodes(gotlocks, conn, 0, conn, 3);
	  theMesh->e2n_getAll(nbrElems[i], conn);
	  numtries++;
	  if(numtries>=10) {
	    CkPrintf("Possibly a livelock\n");
	    numtries = 0;
	  }
	  CthYield();
	}
      }
      for (int j=0; j<3; j++) 
	if (eConn[j] == deletenode) eConn[j] = keepnode;
      echunk = FEM_remove_element(theMesh, nbrElems[i], 0);
      nbrElems[i] = FEM_add_element(theMesh, eConn, 3, 0, echunk); //add it to the same chunk from where it was removed
      unlockNodes(gotlocks, conn, 0, conn, 3);
      delete[] eConn;
      delete [] gotlocks;
    }
  }
  FEM_remove_node(theMesh, deletenode);
  if(nesize!=0) delete[] nbrElems;
  free(conn);
  free(adjnodes);
  free(adjelems);
  return keepnode;
}
