/* File: fem_adapt_lock.C
 * Authors: Nilesh Choudhury, Terry Wilmarth
 *
 */

#include "fem_adapt_lock.h"
#include "fem_mesh_modify.h"

//#define DEBUG_1
#define ERVAL -1000000000  //might cause a problem if there are 100million nodes
#define ERVAL1 -1000000001

int FEM_AdaptL::lockNodes(int *gotlocks, int *lockrnodes, int numRNodes, int *lockwnodes, int numWNodes) {
  bool donelocks = false;
  int numNodes = numRNodes + numWNodes;
  for(int i=0; i<numNodes; i++) gotlocks[i] = 0;
  int tryCounts=0;
  int ret = 1;

  while(!donelocks) {
    for(int i=0; i<numRNodes; i++) {
      if(lockrnodes[i]==-1) {
	gotlocks[i] = 1;
	continue;
      }
      if(FEM_Is_ghost_index(lockrnodes[i])) {
	if(!theMesh->node.ghost->is_valid(FEM_To_ghost_index(lockrnodes[i]))) {
	  ret = -2;
	  break;
	}
      } else {
	if(!theMesh->node.is_valid(lockrnodes[i])) {
	  ret = -2;
	  break;
	}
      }
      if(gotlocks[i]<=0) gotlocks[i] = FEM_Modify_LockN(theMesh, lockrnodes[i], 1);
    }
    for(int i=0; i<numWNodes; i++) {
      if(lockwnodes[i]==-1) {
	gotlocks[numRNodes+i] = 1;
	continue;
      }
      if(FEM_Is_ghost_index(lockwnodes[i])) {
	if(!theMesh->node.ghost->is_valid(FEM_To_ghost_index(lockwnodes[i]))) {
#ifdef DEBUG_LOCKS
	  CkPrintf("[%d] Trying to lock invalid ghost %d\n",theMod->idx,lockwnodes[i]);
#endif
	  ret = -2;
	  break;
	}
      } else {
	if(!theMesh->node.is_valid(lockwnodes[i])) {
#ifdef DEBUG_LOCKS
	  CkPrintf("[%d] Trying to lock invalid node %d\n",theMod->idx,lockwnodes[i]);
#endif
	  ret = -2;
	  break;
	}
      }
      if(gotlocks[numRNodes+i]<=0) {
	gotlocks[numRNodes+i] = FEM_Modify_LockN(theMesh, lockwnodes[i], 0);
      }
    }
    if(ret==-2) return ret;
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
  //while(!donelocks) {
  for(int i=0; i<numRNodes; i++) {
    if(lockrnodes[i]==-1) {
      ungetlocks[i] = 1;
      continue;
    }
    if(FEM_Is_ghost_index(lockrnodes[i])) {
      if(!theMesh->node.ghost->is_valid(FEM_To_ghost_index(lockrnodes[i]))) {
	gotlocks[i] = -1;
	//free(ungetlocks);
	//return -1;
      }
    } else {
      if(!theMesh->node.is_valid(lockrnodes[i])) {
	gotlocks[i] = -1;
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
	gotlocks[i] = -1;
#ifdef DEBUG_LOCKS
	CkPrintf("[%d] Trying to unlock invalid ghost %d\n",theMod->idx,lockwnodes[i]);
#endif
	//free(ungetlocks);
	//return -1;
      }
    } else {
      if(!theMesh->node.is_valid(lockwnodes[i])) {
	gotlocks[i] = -1;
#ifdef DEBUG_LOCKS
	CkPrintf("[%d] Trying to unlock invalid node %d\n",theMod->idx,lockwnodes[i]);
#endif
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
  //else CthYield(); //block for a while
  //}
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
  bool warned = false;

  isEdge = findAdjData(n1, n2, &e1, &e2, &e1_n1, &e1_n2, &e1_n3, &e2_n1, &e2_n2, &e2_n3,&n3, &n4);
  if(isEdge == -1) {
    //CkPrintf("[%d]Warning: Flip %d->%d not done as it is no longer a valid edge\n",theMod->idx,n1,n2);
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
      //CkPrintf("[%d]Warning: Flip %d->%d not done as it is no longer a valid edge\n",theMod->idx,n1,n2);
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
      if(numtries>=1000) {
	if(!warned) {
	  //CkPrintf("[%d]Warning: Possibly a livelock in edge_flip %d & %d, supporting %d, %d\n",theMod->idx,n1,n2,n3,n4);
	  warned = true;
	}
	numtries = 0;
      }
      CthYield();
    }
  }
  if ((e1 == -1) || (e2 == -1)) {
    unlockNodes(gotlocks, locknodes, 0, locknodes, numNodes);
    free(locknodes);
    free(gotlocks);
    return -1;
  }
  int ret = edge_flip_help(e1, e2, n1, n2, e1_n1, e1_n2, e1_n3, n3, n4,locknodes);
  /*if(ret!=-1) {
    FEM_Modify_correctLockN(theMesh, locknodes[2]);
    FEM_Modify_correctLockN(theMesh, locknodes[3]);
    }*/ //should not need to do it anymore
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
  bool warned = false;

  isEdge = findAdjData(n1, n2, &e1, &e2, &e1_n1, &e1_n2, &e1_n3, &e2_n1, &e2_n2, &e2_n3,&n3, &n4);
  if(isEdge == -1) {
    //CkPrintf("[%d]Warning: Bisect %d->%d not done as it is no longer a valid edge\n",theMod->idx,n1,n2);
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
      //CkPrintf("[%d]Warning: Bisect %d->%d not done as it is no longer a valid edge\n",theMod->idx,n1,n2);
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
      if(numtries>=1000) {
	if(!warned) {
	  //CkPrintf("[%d]Warning: Possibly a livelock in edge_bisect %d & %d, supporting %d, %d\n",theMod->idx,n1,n2,n3,n4);
	  warned = true;
	}
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
  bool warned = false;

  isEdge = findAdjData(n1, n2, &e1, &e2, &e1_n1, &e1_n2, &e1_n3, &e2_n1, &e2_n2, &e2_n3,&n3, &n4);
  if(isEdge == -1) {
    //CkPrintf("[%d]Warning: Vertex Remove %d->%d not done as it is no longer a valid edge\n",theMod->idx,n1,n2);
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
    //CkPrintf("[%d]Warning: Vertex Remove %d->%d on node %d with %d connections (!= 4)\n",theMod->idx,n1,n2,n1,nnsize);
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
      //CkPrintf("[%d]Warning: Vertex Remove %d->%d not done as it is no longer a valid edge\n",theMod->idx,n1,n2);
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
      //CkPrintf("[%d]Warning: Vertex Remove %d->%d on node %d with %d connections (!= 4)\n",theMod->idx,n1,n2,n1,nnsize);
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
      if(numtries>=1000) {
	if(!warned) {
	  //CkPrintf("[%d]Warning: Possibly a livelock in vertex_remove %d & %d, supporting %d, %d and %d\n",theMod->idx,n1,n2,n3,n4,n5);
	  warned = true;
	}
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
  int ret = -1;
  bool warned = false;
  int newe1=-1, newe2=-1;
  int acquirecount=0;

  bool invalidcoarsen = false;
  if(n1<0 || n2<0) {
    invalidcoarsen = true;
  }
  if(!theMesh->node.is_valid(n1)) {
    invalidcoarsen = true;
  }
  if(!theMesh->node.is_valid(n2)) {
    invalidcoarsen = true;
  }
  if(invalidcoarsen) {
    free(locknodes);
    free(gotlocks);
    CkPrintf("Warning: Trying to coarsen invalid edge %d - %d\n",n1,n2);
    return -1; //should not contract an edge which is not local
  }
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
    return -1;
  }
  bool locked;
  int gotlock = 0;
  while(!done) {
    if(ret==ERVAL1) {
      acquirecount++;
      ret=-1; //reset ret
      //get new nodes, IT has ALL locks
      int *newnodes = new int[4];
      int newcount=0;
      if(newe1!=-1 && newe2!=-1) {
	int *e1conns = new int[3];
	int *e2conns = new int[3];
	theMesh->e2n_getAll(newe1,e1conns,0);
	theMesh->e2n_getAll(newe2,e2conns,0);
	for(int i=0; i<3; i++) {
	  for(int j=0; j<3; j++) {
	    if(e1conns[i] == e2conns[j]) {
	      newnodes[newcount++] = e1conns[i];
	      break;
	    }
	  }
	}
	if(newcount!=2) {
	  //they do not share an edge now.. check if they are both different & valid
	  //should happen only if they are both ghosts
	  CkAssert(FEM_Is_ghost_index(newe1) && FEM_Is_ghost_index(newe2));
	  if(newe1!=e1) {//updated e1 now
	    theMesh->e2n_getAll(newe1,newnodes,0);
	  }
	  if(newe2!=e2) {//updated e2 now
	    theMesh->e2n_getAll(newe2,newnodes,0);
	  }
	  newcount=3;
	  //check if the other node is valid still
	  for(int i=0; i<4; i++) {
	    bool othernodevalid = true;
	    for(int j=0; j<3; j++) {
	      if(locknodes[i] == newnodes[j]) othernodevalid = false;
	    }
	    if(othernodevalid && FEM_Is_ghost_index(locknodes[i])) {
	      newnodes[newcount++]=locknodes[i];
	    }
	  }
	}
	else {
	  for(int i=0; i<3; i++) {
	    if(e1conns[i]!=newnodes[0] && e1conns[i]!=newnodes[1]) {
	      newnodes[newcount++] = e1conns[i];
	      break;
	    }
	  }
	  CkAssert(newcount==3);
	  for(int i=0; i<3; i++) {
	    if(e2conns[i]!=newnodes[0] && e2conns[i]!=newnodes[1]) {
	      newnodes[newcount++] = e2conns[i];
	      break;
	    }
	  }
	  CkAssert(newcount==4);
	}
	delete [] e1conns;
	delete [] e2conns;
      }
      else{
	if(newe1!=-1) {
	  theMesh->e2n_getAll(newe1,newnodes,0);
	  newcount=3;
	}
	else if(newe2!=-1) {
	  theMesh->e2n_getAll(newe2,newnodes,0);
	  newcount=3;
	}
      }
      e1 = newe1;
      e2 = newe2;
      n1 = newnodes[0];
      n2 = newnodes[1];
      n3 = newnodes[2];
      n4 = newnodes[3];
      locknodes[0] = n1;
      locknodes[1] = n2;
      locknodes[2] = n3;
      locknodes[3] = n4;
      numNodes = newcount;
      delete [] newnodes;
      if(numNodes!=0) {
	isEdge = findAdjData(n1, n2, &e1, &e2, &e1_n1, &e1_n2, &e1_n3, &e2_n1, &e2_n2, &e2_n3,&n3, &n4);
      } else {
	isEdge=-1;
      }
      if(isEdge == -1 || acquirecount>=2) {
	if(locked) {
	  //if we have lost the entire region, nothing will be unlocked, as all nodes will be invalid
	  //so it needs to be handled in the aquiring step itself.
	  unlockNodes(gotlocks, locknodes, 0, locknodes, numNodes);
	  locked = false;
	}
	if(acquirecount>2) CkPrintf("Edge contract %d->%d not done as it is causes an acquire livelock\n",n1,n2);
	CkPrintf("Edge contract %d->%d not done as it is no longer a valid edge\n",n1,n2);
	free(locknodes);
	free(gotlocks);
	return -1;
      }
      if (e1==-1 || e2==-1) {
	unlockNodes(gotlocks, locknodes, 0, locknodes, numNodes);
	free(locknodes);
	free(gotlocks);
	return -1;
      }
      if(locknodes[2]==n4 && locknodes[3]==n3) {
	n3 = locknodes[2];
	n4 = locknodes[3];
      }
      gotlock = 1;
    }
    else {
      gotlock = lockNodes(gotlocks, locknodes, 0, locknodes, numNodes);
    }
    locked = true;
    isEdge = findAdjData(n1, n2, &e1, &e2, &e1_n1, &e1_n2, &e1_n3, &e2_n1, &e2_n2, &e2_n3,&n3, &n4);
    if(isEdge == -1) {
      if(locked) {
	unlockNodes(gotlocks, locknodes, 0, locknodes, numNodes);
	locked = false;
      }
      CkPrintf("Edge contract %d->%d not done as it is no longer a valid edge\n",n1,n2);
      free(locknodes);
      free(gotlocks);
      return -1;
    }
    /*if (e1 == -1 || e2==-1) {
      if(locked) {
	//locknodes[2] = n3;
	//locknodes[3] = n4;
	unlockNodes(gotlocks, locknodes, 0, locknodes, numNodes);
	locked = false;
      }
      free(locknodes);
      free(gotlocks);
      return -1;
      }*/
    if(gotlock==1 && locknodes[2]==n3 && locknodes[3]==n4) {
      int numtries1=0;
      ret = ERVAL;
      while(ret==ERVAL && numtries1 < 5) {
	newe1=e1; newe2=e2;
	ret = edge_contraction_help(&newe1, &newe2, n1, n2, e1_n1, e1_n2, e1_n3, e2_n1, e2_n2, e2_n3, n3, n4);
	if(ret == ERVAL1) {
	  done = false;
	}
	else if(ret == ERVAL) {
	  numtries1++;
	  if(numtries1 >= 5) {
	    locknodes[2] = n3;
	    locknodes[3] = n4;
	    if(locked) {
	      unlockNodes(gotlocks, locknodes, 0, locknodes, numNodes);
	      locked = false;
	    }
	    numtries+=10;
	  }
	  CthYield();
	}
	else {
	  done = true;
	}
      }
      if(numtries>=50) {
	//CkPrintf("Possibly a livelock in cloud nodes edge_contract\n");
	//it is ok to skip an edge_contract, if the lock is too difficult to get
	isEdge = findAdjData(n1, n2, &e1, &e2, &e1_n1, &e1_n2, &e1_n3, &e2_n1, &e2_n2, &e2_n3,&n3, &n4);
	if(isEdge!=-1) {
	  locknodes[2] = n3;
	  locknodes[3] = n4;
	}
	if(locked) {
	  unlockNodes(gotlocks, locknodes, 0, locknodes, numNodes);
	  locked = false;
	}
	free(locknodes);
	free(gotlocks);
	return -1;
      }
    }
    else {
      if(locked) {
	unlockNodes(gotlocks, locknodes, 0, locknodes, numNodes);
	locked = false;
      }
      locknodes[2] = n3;
      locknodes[3] = n4;
      numtries++;
      if(numtries>=50) {
	if(!warned) {
	  //CkPrintf("[%d]Warning: Possibly a livelock in edge_contract %d & %d, supporting %d, %d. Avoiding this contract operation.\n",theMod->idx,n1,n2,n3,n4);
	  warned = true;
	}
        //it is ok to skip an edge_contract, if the lock is too difficult to get
	free(locknodes);
	free(gotlocks);
	return -1;
      }
      CthYield();
    }
  }
  if(locked) {
    int deletednode = -1;
    if(ret==n1) deletednode = n2;
    else if(ret==n2) deletednode = n1;
    if(deletednode!=-1) {
      for(int i=0; i<numNodes;i++) {
	if(locknodes[i]==deletednode) locknodes[i]=-1;
      }
    }
    unlockNodes(gotlocks, locknodes, 0, locknodes, numNodes);
    locked = false;
  }
  free(locknodes);
  free(gotlocks);
  return ret;
}


int FEM_AdaptL::edge_contraction_help(int *e1P, int *e2P, int n1, int n2, int e1_n1,
				     int e1_n2, int e1_n3, int e2_n1, 
				     int e2_n2, int e2_n3, int n3, int n4)
{
  int e1=*e1P, e2=*e2P;
  
  int e1chunk=-1, e2chunk=-1;
  int index = theMod->getIdx();

  //if n1 & n2 are shared differently or are on two different boundaries return
  int n1_shared=0, n2_shared=0;
  bool same = true;
  n1_shared = theMod->getfmUtil()->isShared(n1);
  n2_shared = theMod->getfmUtil()->isShared(n2);
  if(n1_shared && n2_shared) {
    const IDXL_Rec *irec1 = theMesh->node.shared.getRec(n1);
    const IDXL_Rec *irec2 = theMesh->node.shared.getRec(n2);
    if(irec1->getShared() == irec2->getShared()) {
      for(int i=0; i<irec1->getShared(); i++) {
	same = false;
	for(int j=0; j<irec2->getShared(); j++) {
	  if(irec1->getChk(i) == irec2->getChk(j)) {
	    same = true; break;
	  }
	}
	if(!same) break;
      }
    }
    else {
      same = false;
    }
    if(!same) {
      return -1;
    }
  }
  int n1_bound, n2_bound;
  FEM_Mesh_dataP(theMesh, FEM_NODE, FEM_BOUNDARY, &n1_bound, n1, 1 , FEM_INT, 1);
  FEM_Mesh_dataP(theMesh, FEM_NODE, FEM_BOUNDARY, &n2_bound, n2, 1 , FEM_INT, 1);
  if((n1_bound != 0) && (n2_bound != 0) && (n1_bound != n2_bound)) {
    bool kcorner = isFixedNode(n1);
    bool dcorner = isFixedNode(n2);
    bool edgeb = isEdgeBoundary(n1,n2);
    if((kcorner && !dcorner && edgeb) || (dcorner && !kcorner && edgeb)) {
    }
    else {
      return -1;
    }
  }

  //if e1 or e2 has a node which is connected to a node which would become a ghost after it is deleted, then let that other chunk eat into this, before the operation
  int e1new=-1;
  int e2new=-1;
  if(e1>=0) {
    int e1conn[3];
    theMesh->e2n_getAll(e1, e1conn, 0);
    //let the eating progress, only if both n1 & n2 will not be lost
    //necessary for the locking code to know what all it has locked
    /*int *n1n2es, *n2n2es;
    int n1n2ecount, n2n2ecount;
    theMesh->n2e_getAll(n1,&n1n2es,&n1n2ecount);
    theMesh->n2e_getAll(n2,&n2n2es,&n2n2ecount);
    int n1n2elocalcount=0, n2n2elocalcount=0;
    for(int i=0; i<n1n2ecount; i++) {
      if(n1n2es[i]>=0) n1n2elocalcount++;
    }
    for(int i=0; i<n2n2ecount; i++) {
      if(n2n2es[i]>=0) n2n2elocalcount++;
    }
    if(n1n2elocalcount==1 && n2n2elocalcount==1) {
      delete[] n1n2es;
      delete[] n2n2es;
      return -1;
      }*/
    for(int i=0; i<3; i++) {
      //if(e1conn[i]!=n1 && e1conn[i]!=n2) {
	int *e1Elems, esize1;
	theMesh->n2e_getAll(e1conn[i], &e1Elems, &esize1);
	int e1count=0;
	for(int j=0; j<esize1; j++) {
	  if(e1Elems[j]>=0) e1count++;
	}
	if(e1count==1) {
	  int *elemNbrs = new int[3];
	  int e1ghostelem=-1;
	  theMesh->e2e_getAll(e1,elemNbrs,0);
	  for(int k=0; k<3; k++) {
	    if(elemNbrs[k] < -1) {
	      e1ghostelem = elemNbrs[k];
	      break;
	    }
	  }
	  delete [] elemNbrs;
	  if(e1ghostelem<-1) {
	    //the remote chunk needs to eat e1
	    int e1remoteChk = theMesh->elem[0].ghost->ghostRecv.getRec(FEM_To_ghost_index(e1ghostelem))->getChk(0);
	    int sharedIdx = theMod->fmUtil->exists_in_IDXL(theMesh,e1,e1remoteChk,3);
	    CkPrintf("[%d]Edge Contraction, edge %d->%d, chunk %d eating into chunk %d\n",theMod->idx, n1, n2, e1remoteChk, index);
	    if(FEM_Is_ghost_index(e2)) { //unlock the 4th node, if it will be lost
	      int e2conn[3];
	      theMesh->e2n_getAll(e2, e2conn, 0);
	      int gotlocksn4 = 1, lockn4=-1;
	      for(int k=0; k<3; k++) {
		if(e2conn[k]!=n1 && e2conn[k]!=n2) lockn4=e2conn[k];
	      }
	      //if my only local n2n are nodes which e1 will lose at the end of this acquire
	      //then unlock me
	      int *n4ns, n4ncount;
	      bool shouldbeunlocked=true;
	      theMesh->n2n_getAll(lockn4, &n4ns, &n4ncount);
	      for(int k=0; k<n4ncount; k++) {
		if(n4ns[k]>=0 && n4ns[k]!=n1 && n4ns[k]!=n2) shouldbeunlocked=false; 
	      }
	      if(shouldbeunlocked) {
		unlockNodes(&gotlocksn4, &lockn4, 0, &lockn4, 1);
	      }
	      if(n4ncount!=0) delete[] n4ns;
	    }
	    e1new = meshMod[e1remoteChk].eatIntoElement(index,sharedIdx)->i;
	    if(e1new!=-1) {
	      e1 = theMod->fmUtil->lookup_in_IDXL(theMesh,e1new,e1remoteChk,4);
	      //theMesh->n2e_getAll(deletenode, &nbrElems, &nesize);
	      e1 = FEM_To_ghost_index(e1);
	    }
	    else e1 = -1;
	    free(e1Elems);
	    *e1P = e1;
	    return ERVAL1;
	  }
	}
	//free(e1Elems);
	//}
    }
  }
  if(e2>=0) {
    int e2conn[3];
    int e2remoteChk=-1;
    theMesh->e2n_getAll(e2, e2conn, 0);
    //let the eating progress, only if both n1 & n2 will not be lost
    //necessary for the locking code to know what all it has locked
    /*int *n1n2es, *n2n2es;
    int n1n2ecount, n2n2ecount;
    theMesh->n2e_getAll(n1,&n1n2es,&n1n2ecount);
    theMesh->n2e_getAll(n2,&n2n2es,&n2n2ecount);
    int n1n2elocalcount=0, n2n2elocalcount=0;
    for(int i=0; i<n1n2ecount; i++) {
      if(n1n2es[i]>=0) n1n2elocalcount++;
    }
    for(int i=0; i<n2n2ecount; i++) {
      if(n2n2es[i]>=0) n2n2elocalcount++;
    }
    if(n1n2elocalcount==1 && n2n2elocalcount==1) {
      delete[] n1n2es;
      delete[] n2n2es;
      return -1;
      }*/
    for(int i=0; i<3; i++) {
      //if(e2conn[i]!=n1 && e2conn[i]!=n2) {
	int *e2Elems, esize2;
	theMesh->n2e_getAll(e2conn[i], &e2Elems, &esize2);
	int e2count=0;
	for(int j=0; j<esize2; j++) {
	  if(e2Elems[j]>=0) e2count++;
	}
	if(e2count==1) {
	  int *elemNbrs = new int[3];
	  int e2ghostelem=-1;
	  theMesh->e2e_getAll(e2,elemNbrs,0);
	  for(int k=0; k<3; k++) {
	    if(elemNbrs[k] < -1) {
	      e2ghostelem = elemNbrs[k];
	      break;
	    }
	  }
	  delete [] elemNbrs;
	  if(e2ghostelem<-1) {
	    //the remote chunk needs to eat e2
	    int e2remoteChk = theMesh->elem[0].ghost->ghostRecv.getRec(FEM_To_ghost_index(e2ghostelem))->getChk(0);
	    int sharedIdx = theMod->fmUtil->exists_in_IDXL(theMesh,e2,e2remoteChk,3);
	    CkPrintf("[%d]Edge Contraction, edge %d->%d, chunk %d eating into chunk %d\n",theMod->idx, n1, n2, e2remoteChk, index);
	    if(FEM_Is_ghost_index(e1)) {
	      int e1conn[3];
	      theMesh->e2n_getAll(e1, e1conn, 0);
	      int gotlocksn4 = 1, lockn4=-1;
	      for(int k=0; k<3; k++) {
		if(e1conn[k]!=n1 && e1conn[k]!=n2) lockn4=e1conn[k];
	      }
	      //if my only local n2n are nodes which e1 will lose at the end of this acquire
	      //then unlock me
	      int *n4ns, n4ncount;
	      bool shouldbeunlocked=true;
	      theMesh->n2n_getAll(lockn4, &n4ns, &n4ncount);
	      for(int k=0; k<n4ncount; k++) {
		if(n4ns[k]>=0 && n4ns[k]!=n1 && n4ns[k]!=n2) shouldbeunlocked=false; 
	      }
	      if(shouldbeunlocked) {
		unlockNodes(&gotlocksn4, &lockn4, 0, &lockn4, 1);
	      }
	      if(n4ncount!=0) delete[] n4ns;
	    }
	    e2new = meshMod[e2remoteChk].eatIntoElement(index,sharedIdx)->i;
	    if(e2new!=-1) {
	      e2 = theMod->fmUtil->lookup_in_IDXL(theMesh,e2new,e2remoteChk,4);
	      //theMesh->n2e_getAll(deletenode, &nbrElems, &nesize);
	      e2 = FEM_To_ghost_index(e2);
	    }
	    else e2 = -1;
	    *e2P = e2;
	    free(e2Elems);
	    return ERVAL1;
	  }
	  //free(e2Elems);
	  //}
      }
    }
  }
  if(FEM_Is_ghost_index(e1)) {
    //find out if this chunk should acquire it
    int remChk = theMesh->elem[0].ghost->ghostRecv.getRec(FEM_From_ghost_index(e1))->getChk(0);
    int shidx = theMesh->elem[0].ghost->ghostRecv.getRec(FEM_From_ghost_index(e1))->getIdx(0);
    //if this ghostelem has a ghost node connectivity, then do not acquire it
    bool shouldacquire = true;
    int e1nbrs[3];
    theMesh->e2n_getAll(e1,e1nbrs,0);
    for(int i=0; i<3;i++) {
      if(FEM_Is_ghost_index(e1nbrs[i])) {
	shouldacquire=false; break;
      }
    }
    bool shouldacquire1 = meshMod[remChk].willItLose(index,shidx)->b;
    if(shouldacquire && shouldacquire1) {
      e1new = theMod->fmUtil->eatIntoElement(e1);
      *e1P = e1new;
      return ERVAL1;
    }
    else if(shouldacquire1) return -1;
  }
  if(FEM_Is_ghost_index(e2)) {
    //find out if this chunk should acquire it
    int remChk = theMesh->elem[0].ghost->ghostRecv.getRec(FEM_From_ghost_index(e2))->getChk(0);
    int shidx = theMesh->elem[0].ghost->ghostRecv.getRec(FEM_From_ghost_index(e2))->getIdx(0);
    //if this ghostelem has a ghost node connectivity, then do not acquire it
    bool shouldacquire = true;
    int e2nbrs[3];
    theMesh->e2n_getAll(e2,e2nbrs,0);
    for(int i=0; i<3;i++) {
      if(FEM_Is_ghost_index(e2nbrs[i])) {
	shouldacquire=false; break;
      }
    }
    bool shouldacquire1 = meshMod[remChk].willItLose(index,shidx)->b;
    if(shouldacquire && shouldacquire1) {
      e2new = theMod->fmUtil->eatIntoElement(e2);
      *e2P = e2new;
      return ERVAL1;
    }
    else if(shouldacquire1) return -1;
  }


  int *conn = (int*)malloc(3*sizeof(int));
  int *adjnodes = (int*)malloc(2*sizeof(int));
  adjnodes[0] = n1;
  adjnodes[1] = n2;
  int *adjelems = (int*)malloc(2*sizeof(int));
  adjelems[0] = e1;
  adjelems[1] = e2;

  //FEM_Modify_Lock(theMesh, adjnodes, 2, adjelems, 2);  

  //New code for updating a node rather than deleting both
  int keepnode=-1, deletenode=-1, shared=0;
  //n1_shared = theMod->getfmUtil()->isShared(n1);
  //n2_shared = theMod->getfmUtil()->isShared(n2);
  bool n1fixed = isFixedNode(n1);
  bool n2fixed = isFixedNode(n2);
  if(n1fixed && n2fixed) { //both are fixed, so return
    free(conn);
    free(adjnodes);
    free(adjelems);
    return -1;
  }
  if(n1_shared && n2_shared) {
    /*same = true;
    const IDXL_Rec *irec1 = theMesh->node.shared.getRec(n1);
    const IDXL_Rec *irec2 = theMesh->node.shared.getRec(n2);
    if(irec1->getShared() == irec2->getShared()) {
      for(int i=0; i<irec1->getShared(); i++) {
	same = false;
	for(int j=0; j<irec2->getShared(); j++) {
	  if(irec1->getChk(i) == irec2->getChk(j)) {
	    same = true; break;
	  }
	}
	if(!same) break;
      }
    }
    else {
      same = false;
      }*/
    if(same) {
      if(n1fixed) {
	keepnode = n2;
	deletenode = n1;
      }
      else {
	keepnode = n1;
	deletenode = n2;
      }
      shared = 2;
    } else {
      free(conn);
      free(adjnodes);
      free(adjelems);
      return -1; //they are on different boundaries
    }
  }
  else if(n1_shared && !n2fixed) {
    //update n1 & delete n2
    keepnode = n1;
    deletenode = n2;
    shared = 1;
  } else if(n2_shared && !n1fixed) {
    //update n2 & delete n1
    keepnode = n2;
    deletenode = n1;
    shared = 1;
  } else if(!n1_shared && !n2_shared) {
    //keep either
    if(n2fixed) {
      keepnode = n2;
      deletenode = n1;
    }
    else {
      keepnode = n1;
      deletenode = n2;
    }
  }
  else {
    free(conn);
    free(adjnodes);
    free(adjelems);
    return -1;
  }
  
  FEM_Mesh_dataP(theMesh, FEM_NODE, FEM_BOUNDARY, &n1_bound, keepnode, 1 , FEM_INT, 1);
  FEM_Mesh_dataP(theMesh, FEM_NODE, FEM_BOUNDARY, &n2_bound, deletenode, 1 , FEM_INT, 1);
  //update keepnode's attributes; choose frac wisely, check if either node is on the boundary, update frac
  FEM_Interpolate *inp = theMod->getfmInp();
  FEM_Interpolate::NodalArgs nm;
  if((n1_bound != 0) && (n2_bound != 0) && (n1_bound != n2_bound)) {
    bool kcorner;
    bool dcorner;
    if(keepnode==n1) {
      kcorner = n1fixed;
      dcorner = n2fixed;
    }
    else {
      kcorner = n2fixed;
      dcorner = n1fixed;
    }
    bool edgeb = isEdgeBoundary(keepnode,deletenode);
    if(kcorner && !dcorner && edgeb) {
      nm.frac = 1.0;
    }
    else if(dcorner && !kcorner && edgeb) {
      nm.frac = 0.0;
    }
    else {//boundary attribute should be 0
      nm.frac = 0.5;
      free(conn);
      free(adjnodes);
      free(adjelems);
      return -1; //they are on different boundaries
    }
  }
  else if(n1_bound!=0 && n2_bound!=0) {
    nm.frac = 0.5;
    //TODO: must ensure that any of the two nodes is not a corner
    if(isFixedNode(keepnode)) {
      nm.frac = 1.0;
    } 
    else if(isFixedNode(deletenode)) {
      nm.frac = 0.0;
    }
  }
  else if(n1_bound != 0) { //keep its attributes
    nm.frac = 1.0;
  }
  else if(n2_bound != 0) {
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
  nm.addNode = false;

  //hack, if it is shared, do not change the attributes, since I am not updating them now across all chunks
  /*if(n1_shared || n2_shared) {
    if((n1_shared && n2_bound) ||(n2_shared && n1_bound)) {
      nm.frac = 1.0;
      //free(conn);
      //free(adjnodes);
      //free(adjelems);
      //return -1; //one edge on a shared node & the other on a boundary
    }
    else {
      nm.frac = 1.0;
    }
  }*/


  int *nbrElems, nesize, echunk;
  theMesh->n2e_getAll(deletenode, &nbrElems, &nesize);
  bool locked = false;
  /*CkVec<int> lockedNodes;
  int numtries = 0;
  for (int i=0; i<nesize; i++) {
    if ((nbrElems[i] != e1) && (nbrElems[i] != e2)) {
      theMesh->e2n_getAll(nbrElems[i], conn);
      int *gotlocks = (int*)malloc(3*sizeof(int));
      gotlocks[0] = gotlocks[1] = gotlocks[2] = -1;
      bool done = false;
      int *eConn = new int[3];
      while(!done) {
	bool valid = false;
	if(FEM_Is_ghost_index(nbrElems[i])) {
	  valid = theMesh->elem[0].ghost->is_valid(FEM_To_ghost_index(nbrElems[i]));
	}
	else {
	  valid = theMesh->elem[0].is_valid(nbrElems[i]);
	}
	if(!valid) {
	  if(locked) {
	    unlockNodes(gotlocks, conn, 0, conn, 3);
	    locked = false;
	  }
	  //CkPrintf("[%d]Warning: Element %d is no longer valid\n",theMod->idx,nbrElems[i]);
	  delete [] eConn;
	  if(nesize!=0) delete[] nbrElems;
	  free(conn);
	  free(adjnodes);
	  free(adjelems);
	  free(gotlocks);
	  return -1;
	}
	for(int k=0; k<3; k++) {
	  if(conn[k]==n1 || conn[k]==n2 || conn[k]==n3 || conn[k]==n4) conn[k]=-1;
	  for(int l=0; l<lockedNodes.size(); l++) {
	    if(conn[k] == lockedNodes[l]) conn[k]=-1;
	  }
	}
#ifdef DEBUG_LOCKS
	CkPrintf("[%d]Trying to get locks %d,%d & %d\n",theMod->idx,conn[0],conn[1],conn[2]);
#endif
	int gotlock = lockNodes(gotlocks, conn, 0, conn, 3);
	locked = true;
	if(gotlock==1) {
	  theMesh->e2n_getAll(nbrElems[i], eConn);
	  bool isConn = true;
	  for(int k=0; k<3; k++) {
	    if((eConn[k] != conn[k]) && conn[k]!=-1) {
	      isConn = false;
	      break;
	    }
	  }
	  if(isConn) {
	    for(int k=0; k<3; k++) {
	      if(conn[k] != -1) {
		lockedNodes.push_back(conn[k]);
	      }
	    }
	    done = true;
#ifdef DEBUG_LOCKS
	    CkPrintf("[%d]Got locks\n",theMod->idx);
#endif
	  }
	  else { //conn has changed, try again
	    int size = lockedNodes.size();
	    int *gotlocks1 = (int*)malloc(size*sizeof(int));
	    int *lockw = (int*)malloc(size*sizeof(int));
#ifdef DEBUG_LOCKS
	    CkPrintf("[%d]Conn changed, trying to unlock\n",theMod->idx);
#endif
	    for(int k=0; k<size; k++) {
	      gotlocks1[k] = 1;
	      lockw[k] = lockedNodes[k];
	      //CkPrintf(" %d",lockw[k]);
	    }
	    //CkPrintf("\n");
	    unlockNodes(gotlocks1, lockw, 0, lockw, size);
	    free(gotlocks1);
	    free(lockw);
	    if(locked) {
#ifdef DEBUG_LOCKS
	      CkPrintf("[%d]Trying to unlock %d, %d, & %d\n",theMod->idx,conn[0],conn[1],conn[2]);
#endif
	      unlockNodes(gotlocks, conn, 0, conn, 3);
	      locked = false;
	    }
	    theMesh->e2n_getAll(nbrElems[i], conn);
	    numtries++;
	    delete [] eConn;
	    if(nesize!=0) delete[] nbrElems;
	    free(conn);
	    free(adjnodes);
	    free(adjelems);
	    free(gotlocks);
	    return ERVAL;
	  }
	}
	else {
	  numtries++;
	  if(locked) {
#ifdef DEBUG_LOCKS
	    CkPrintf("[%d]Trying to unlock %d(%d), %d(%d), & %d(%d)\n",theMod->idx,conn[0],gotlocks[0],conn[1],gotlocks[1],conn[2],gotlocks[2]);
#endif
	    unlockNodes(gotlocks, conn, 0, conn, 3);
	    locked = false;
	  }
	  if(numtries>=10) {
	    int size = lockedNodes.size();
	    int *gotlocks1 = (int*)malloc(size*sizeof(int));
	    int *lockw = (int*)malloc(size*sizeof(int));
#ifdef DEBUG_LOCKS
	    CkPrintf("[%d]Couldn't get locks, trying to unlock\n",theMod->idx);
#endif
	    for(int k=0; k<size; k++) {
	      gotlocks1[k] = 1;
	      lockw[k] = lockedNodes[k];
	      //CkPrintf(" %d",lockw[k]);
	    }
	    //CkPrintf("\n");
	    unlockNodes(gotlocks1, lockw, 0, lockw, size);
	    free(gotlocks1);
	    free(lockw);
	    //CkPrintf("Possibly a livelock in edge_contract_help\n");
	    delete [] eConn;
	    if(nesize!=0) delete[] nbrElems;
	    free(conn);
	    free(adjnodes);
	    free(adjelems);
	    free(gotlocks);
	    return ERVAL;
	  }
	  theMesh->e2n_getAll(nbrElems[i], conn);
	}
      }
      delete[] eConn;
      free(gotlocks);
    }
  }*/

  //lock the adjacent nodes
  CkVec<int> lockedNodes;
  int *nnNodes;
  int nnsize;
  int nncount=0;
  int numtries=0;
  bool done = false;
  while(!done) {
    lockedNodes.removeAll();
    nncount=0;
    theMesh->n2n_getAll(deletenode, &nnNodes, &nnsize);
    for(int i=0; i<nnsize; i++) {
      if(nnNodes[i]!=n1 && nnNodes[i]!=n2 && nnNodes[i]!=n3 && nnNodes[i]!=n4) {
	lockedNodes.push_back(nnNodes[i]);
	nncount++;
      }
    }
    int *gotlocks = new int[nncount];
    int *lockw = new int[nncount];
    for(int i=0; i<nncount; i++) {
      gotlocks[i]=-1;
      lockw[i] = lockedNodes[i];
    }
    int gotlock = lockNodes(gotlocks,lockw,0,lockw,nncount);
    locked = true;
    numtries++;
    if(gotlock==1) {
      bool isConn = true;
      int *nnNodes1;
      int nnsize1;
      theMesh->n2n_getAll(deletenode, &nnNodes1, &nnsize1);
      if(nnsize!=nnsize1) isConn=false;
      else {
	for(int i=0; i<nnsize; i++) {
	  isConn = false;
	  for(int j=0; j<nnsize1; j++) {
	    if(nnNodes1[i] == nnNodes[j]) {
	      isConn=true; break;
	    }
	  }
	  if(!isConn) break;
	}
      }
      if(!isConn) { //connectivity has changed, try acquiring the locks again
	unlockNodes(gotlocks,lockw,0,lockw,nncount);
	if(numtries>=3) {
	  if(nnsize1!=0) free(nnNodes1);
	  if(nnsize!=0) free(nnNodes);
	  if(nncount!=0) delete [] lockw;
	  if(nncount!=0) delete [] gotlocks;
	  free(conn);
	  free(adjnodes);
	  free(adjelems);
	  return ERVAL;
	}
	CthYield();
      }
      else done = true;
      if(nnsize1!=0) free(nnNodes1);
    }
    else unlockNodes(gotlocks,lockw,0,lockw,nncount);
    if(nnsize!=0) free(nnNodes);
    if(nncount!=0) delete [] lockw;
    if(nncount!=0) delete [] gotlocks;
    if(numtries>=3 && !done) return ERVAL;
    free(conn);
    free(adjnodes);
    free(adjelems);
  }


  //verify if it is causing a flip/sliver
  double *n1_coord = new double[2];
  double *n2_coord = new double[2];
  double *new_coord = new double[2];
  FEM_Mesh_dataP(theMesh, FEM_NODE, theMod->fmAdaptAlgs->coord_attr, (void *)n1_coord, nm.nodes[0], 1, FEM_DOUBLE, 2);
  FEM_Mesh_dataP(theMesh, FEM_NODE, theMod->fmAdaptAlgs->coord_attr, (void *)n2_coord, nm.nodes[1], 1, FEM_DOUBLE, 2);
  new_coord[0] = nm.frac*n1_coord[0] + (1-nm.frac)*n2_coord[0];
  new_coord[1] = nm.frac*n1_coord[1] + (1-nm.frac)*n2_coord[1];
  int flipSliver = false;
  int *nbr1Elems, nesize1;
  int *conn1 = new int[3];
  theMesh->n2e_getAll(keepnode, &nbr1Elems, &nesize1);
  for (int i=0; i<nesize; i++) {
    if ((nbrElems[i] != e1) && (nbrElems[i] != e2)) {
      theMesh->e2n_getAll(nbrElems[i], conn);
      for(int j=0; j<2; j++) {
	if (conn[j] == deletenode) {
	  conn[j] = conn[2];
	  conn[2] = deletenode;
	}
      }

      if(theMod->fmAdaptAlgs->didItFlip(conn[0],conn[1],conn[2],new_coord)) {
	flipSliver = true;
	//CkPrintf("[%d]Warning: Elem %d(%d,%d,%d) would become a sliver if %d->%d is contracted\n",theMod->idx,nbrElems[i],conn[0],conn[1],conn[2],n1,n2);
	break;
      }
    }
  }
  if(!flipSliver) {
    for (int i=0; i<nesize1; i++) {
      if ((nbr1Elems[i] != e1) && (nbr1Elems[i] != e2)) {
	theMesh->e2n_getAll(nbr1Elems[i], conn1);
	for(int j=0; j<2; j++) {
	  if (conn1[j] == keepnode) {
	    conn1[j] = conn1[2];
	    conn1[2] = keepnode;
	  }
	}
	if(theMod->fmAdaptAlgs->didItFlip(conn1[0],conn1[1],conn1[2],new_coord)) {
	  flipSliver = true;
	  //CkPrintf("[%d]Warning: Elem %d(%d,%d,%d) would become a sliver if %d->%d is contracted\n",theMod->idx,nbr1Elems[i],conn1[0],conn1[1],conn1[2],n1,n2);
	  break;
	}
      }
    }
  }
  if(nesize1 != 0) delete [] nbr1Elems;
  delete [] conn1;
  delete [] n1_coord;
  delete [] n2_coord;
  delete [] new_coord;
  if(flipSliver) {
    int size = lockedNodes.size();
    int *gotlocks = (int*)malloc(size*sizeof(int));
    int *lockw = (int*)malloc(size*sizeof(int));
    for(int k=0; k<size; k++) {
      gotlocks[k] = 1;
      lockw[k] = lockedNodes[k];
    }
    if(locked) {
#ifdef DEBUG_LOCKS
      CkPrintf("[%d]Trying to unlock %d, %d, & %d\n",theMod->idx,conn[0],conn[1],conn[2]);
#endif
      unlockNodes(gotlocks, lockw, 0, lockw, size);
      locked = false;
    }
    free(gotlocks);
    free(lockw);

    if(nesize!=0) delete[] nbrElems;
    free(conn);
    free(adjnodes);
    free(adjelems);
    return -1;
  }

  inp->FEM_InterpolateNodeOnEdge(nm); //update the attributes of keepnode across shared


#ifdef DEBUG_1
  CkPrintf("[%d]Edge Contraction, edge %d(%d:%d)->%d(%d:%d) on chunk %d:: deleted %d\n",theMod->idx, n1,n1_bound,n1_shared, n2,n2_bound,n2_shared, theMod->getfmUtil()->getIdx(),deletenode);
#endif
  e1chunk = FEM_remove_element(theMesh, e1, 0);
  if(e2!=-1) e2chunk = FEM_remove_element(theMesh, e2, 0);
  FEM_purge_element(theMesh,e1,0);
  if(e2!=-1) FEM_purge_element(theMesh,e2,0);

  for (int i=0; i<nesize; i++) {
    if ((nbrElems[i] != e1) && (nbrElems[i] != e2)) {
      theMesh->e2n_getAll(nbrElems[i], conn);
      for (int j=0; j<3; j++) {
	if (conn[j] == deletenode) conn[j] = keepnode;
      }
      int eTopurge = nbrElems[i];
      echunk = FEM_remove_element(theMesh, nbrElems[i], 0);
      nbrElems[i] = FEM_add_element(theMesh, conn, 3, 0, echunk); //add it to the same chunk from where it was removed
      theMod->fmUtil->copyElemData(0,eTopurge,nbrElems[i]);
      FEM_purge_element(theMesh,eTopurge,0);
    }
  }

  //unlock
  int size = lockedNodes.size();
  int *gotlocks = (int*)malloc(size*sizeof(int));
  int *lockw = (int*)malloc(size*sizeof(int));
  for(int k=0; k<size; k++) {
    lockw[k] = lockedNodes[k];
    gotlocks[k] = 1;
  }
  if(locked) {
#ifdef DEBUG_LOCKS
    CkPrintf("[%d]Done contraction, Trying to unlock %d, %d, & %d\n",theMod->idx,conn[0],conn[1],conn[2]);
#endif
    //unlock all other than the one that was deleted. Its lock was reset in remove
    unlockNodes(gotlocks, lockw, 0, lockw, size);
    locked = false;
  }
  free(gotlocks);
  free(lockw);

  FEM_remove_node(theMesh, deletenode);
  if(nesize!=0) delete[] nbrElems;
  free(conn);
  free(adjnodes);
  free(adjelems);
  return keepnode;
}
// ======================  END edge_contraction  ==============================


#undef DEBUG_1

