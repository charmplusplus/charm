/* File: fem_interpolate.C
 * Authors: Terry Wilmarth, Nilesh Choudhury
 * 
 */

// FEM_Interpolate default implementation - TLW
#include "ParFUM.h"
#include "ParFUM_internals.h"

/** A node is added on an edge; interpolate from neighboring nodes; this
    uses n, nodes[2], dim and frac. Frac is between 0.0 and 1.0 and weights 
    nodes[0]; i.e. if frac=1.0, n gets a copy of nodes[0]'s data, and
    nodes[0]'s coords 
*/
void FEM_Interpolate::FEM_InterpolateNodeOnEdge(NodalArgs args)
{
  if (nodeEdgeFnPtr) { // default is overridden
    nodeEdgeFnPtr(args,theMesh);
    return;
  }
  // do default interpolation
  // DEFAULT BEHAVIOR: Interpolate whatever user data is available, weighted by
  // frac.
  
  CkVec<FEM_Attribute *>*attrs = (theMesh->node).getAttrVec();
  for (int i=0; i<attrs->size(); i++) {
    FEM_Attribute *a = (FEM_Attribute *)(*attrs)[i];
    if (a->getAttr() < FEM_ATTRIB_TAG_MAX) {
      FEM_DataAttribute *d = (FEM_DataAttribute *)a;
      d->interpolate(args.nodes[0], args.nodes[1], args.n, args.frac);
    } 
    else if(a->getAttr()==FEM_BOUNDARY) {
      int n1_bound, n2_bound;
      FEM_Mesh_dataP(theMesh, FEM_NODE, FEM_BOUNDARY, &n1_bound, args.nodes[0], 1 , FEM_INT, 1);
      FEM_Mesh_dataP(theMesh, FEM_NODE, FEM_BOUNDARY, &n2_bound, args.nodes[1], 1 , FEM_INT, 1);
      if(args.frac==1.0) {
	a->copyEntity(args.n,*a,args.nodes[0]);
      } else if(args.frac==0.0) {
	a->copyEntity(args.n,*a,args.nodes[1]);
      } else {
	//here, all requests from addnode will come here, from contract_edge only 
	//some requests will end up in this case
	//both cases need to handled differently
	if(args.addNode) {
	  if(n1_bound!=0 && n2_bound!=0 && n1_bound!=n2_bound) {
	    //figure out if one of them is a fixed node (which is treated as a corner),
	    bool n1corner = theMod->fmAdaptL->isFixedNode(args.nodes[0]);
	    bool n2corner = theMod->fmAdaptL->isFixedNode(args.nodes[1]);
	    bool edgeb = theMod->fmAdaptL->isEdgeBoundary(args.nodes[0],args.nodes[1]);
	    if(!edgeb) { //the edge is not on the boundary, so the node is internal
	      int nbound = 0;
	      FEM_DataAttribute *d = (FEM_DataAttribute *)a;
	      d->getInt().setRow(args.n,nbound);
	    }
	    else if(n1corner && !n2corner && edgeb) {
	      a->copyEntity(args.n,*a,args.nodes[1]);
	    }
	    else if(n2corner && !n1corner && edgeb) {
	      a->copyEntity(args.n,*a,args.nodes[0]);
	    }
	    else if(n2corner && n1corner && edgeb) {
	      //assign it a new number other than the two
	      //add a new boundary number less than n1 & n2 and which is not used for any other boundary
	      //can not do this yet, need some global data structure to do this
	      //FIXME!!!
	      //if(abs(n1_bound - n2_bound) == 2) {
	      //nbound = (abs(n1_bound)<abs(n2_bound)) ? n1_bound-1 : n2_bound-1;
	      //} else {
	      int nbound = (abs(n1_bound)<abs(n2_bound)) ? n1_bound : n2_bound;
	      //}
	      FEM_DataAttribute *d = (FEM_DataAttribute *)a;
	      d->getInt().setRow(args.n,nbound);
	    }
	    else {//boundary attribute should be 0
	      int nbound = 0;
	      FEM_DataAttribute *d = (FEM_DataAttribute *)a;
	      d->getInt().setRow(args.n,nbound);
	    }
	  }
	  else if(n1_bound!=0 && n2_bound!=0) {
	    //both nodes on same boundary, copy from any one
	    a->copyEntity(args.n,*a,args.nodes[0]);
	  }
	  else if(n1_bound!=0) {
	    a->copyEntity(args.n,*a,args.nodes[1]);
	  }
	  else if(n2_bound!=0) {
	    a->copyEntity(args.n,*a,args.nodes[0]);
	  }
	  else {
	    //both nodes are internal, copy any one
	    a->copyEntity(args.n,*a,args.nodes[0]);
	  }
	}
	else {
	  //the contract operation, one of the existing nodes should get the final attrs
	  if(n1_bound!=0 && n2_bound==0) {
	    a->copyEntity(args.n,*a,args.nodes[0]);
	  }
	  else if(n2_bound!=0 && n1_bound==0) {
	    a->copyEntity(args.n,*a,args.nodes[1]);
	  }
	  else if(n1_bound!=0 && n2_bound!=0 && n1_bound==n2_bound) {
	    //copy any one, both are on the same boundary
	    a->copyEntity(args.n,*a,args.nodes[1]);
	  }
	  else if(n1_bound==0 && n2_bound==0) {
	    //copy any one, both are internal
	    a->copyEntity(args.n,*a,args.nodes[1]);
	  }
	  else {
	    //shouldn't be here, we have a problem earlier
	    CkAssert(false);
	  }
	}
      }
    }
  }
  //if the node is shared, then update these values on the other chunks as well
  if(theMod->fmUtil->isShared(args.n)) {
    //build up the list of chunks it is shared/local to and update the coords & boundary falgs, others should be updated by the user
    int numchunks;
    IDXL_Share **chunks1;
    theMod->fmUtil->getChunkNos(0,args.n,&numchunks,&chunks1);
    char *data;
    int size, count;
    theMod->fmUtil->packEntData(&data, &size, &count, args.n, true, 0);

    for(int j=0; j<numchunks; j++) {
      int chk = chunks1[j]->chk;
      if(chk==theMod->getIdx()) continue;
      updateAttrsMsg *umsg = new (size,0)updateAttrsMsg(count);
      umsg->fromChk = theMod->idx;
      umsg->isnode = true;
      memcpy(umsg->data, data, size);
      umsg->sharedIdx = theMod->fmUtil->exists_in_IDXL(theMesh,args.n,chk,0);
      meshMod[chk].updateAttrs(umsg);
    }
    for(int j=0; j<numchunks; j++) {
      delete chunks1[j];
    }
    free(chunks1);
    free(data);
  }

  return;
}

/** A node is added on an face; interpolate from nodes of face; this uses
    n, nodes[3] or nodes[4] depending on element type, dim and coord 
*/
void FEM_Interpolate::FEM_InterpolateNodeOnFace(NodalArgs args)
{
  if (nodeFaceFnPtr) { // default is overridden
    nodeFaceFnPtr(args, theMesh);
    return;
  }
  // do default interpolation
  // DEFAULT BEHAVIOR: Interpolate whatever user data is available, positioning
  // new point at coordinate average of surrounding face nodes

  CkVec<FEM_Attribute *>*attrs = (theMesh->node).getAttrVec();
  for (int i=0; i<attrs->size(); i++) {
    FEM_Attribute *a = (FEM_Attribute *)(*attrs)[i];
    if (a->getAttr() < FEM_ATTRIB_TAG_MAX) {
      FEM_DataAttribute *d = (FEM_DataAttribute *)a;
      if ((args.nNbrs == 3) || (args.nNbrs == 4)){
	d->interpolate(args.nodes, args.n, args.nNbrs);
      }
      else {
	CkPrintf("ERROR: %d node faces not supported for node data interpolation.\n", args.nNbrs);
	CkAbort("ERROR: FEM_InterpolateNodeOnFace\n");
      }
    }
  }
  return;
}

/** A node is added inside a volume; interpolate from nodes of element; this
   uses n, nodes[4] or more, and coord 
*/
void FEM_Interpolate::FEM_InterpolateNodeInElement(NodalArgs args)
{
  if (nodeElementFnPtr) { // default is overridden
    nodeElementFnPtr(args, theMesh);
    return;
  }
  // do default interpolation
  // DEFAULT BEHAVIOR: Interpolate whatever user data is available, positioning
  // new point at coordinate average of surrounding element nodes

  CkVec<FEM_Attribute *>*attrs = (theMesh->node).getAttrVec();
  for (int i=0; i<attrs->size(); i++) {
    FEM_Attribute *a = (FEM_Attribute *)(*attrs)[i];
    if (a->getAttr() < FEM_ATTRIB_TAG_MAX) {
      FEM_DataAttribute *d = (FEM_DataAttribute *)a;
      if ((args.nNbrs >= 4) && (args.nNbrs <= 8)) {
	d->interpolate(args.nodes, args.n, args.nNbrs);
      }
      else {
	CkPrintf("ERROR: %d node elements not supported for node data interpolation.\n", args.nNbrs);
	CkAbort("ERROR: FEM_InterpolateNodeInElement\n");
      }
    }
  }
  return;
}



/** An element added is completely encapsulated by element to be removed; for 
   example, edge bisect two elements replace the old element, and we simply 
   copy the data of the old element to both new elements 
*/
void FEM_Interpolate::FEM_InterpolateElementCopy(ElementArgs args)
{
  if (elemCopyFnPtr) { // default is overridden
    elemCopyFnPtr(args);
    return;
  }
  // do default interpolation
  // DEFAULT BEHAVIOR: COPY ALL ELEMENT DATA
  if(args.e>=0 && args.oldElement>=0) {
    CkVec<FEM_Attribute *>*elemattrs = (theMesh->elem[0]).getAttrVec();
    for(int j=0;j<elemattrs->size();j++){
      FEM_Attribute *elattr = (FEM_Attribute *)(*elemattrs)[j];
      if(elattr->getAttr() < FEM_ATTRIB_FIRST){ 
	elattr->copyEntity(args.e,*elattr,args.oldElement);
      }
      else if(elattr->getAttr()==FEM_MESH_SIZING) {
	elattr->copyEntity(args.e,*elattr,args.oldElement);
      }
    }
  }
  else if(FEM_Is_ghost_index(args.e) && FEM_Is_ghost_index(args.oldElement)) {
    const IDXL_Rec *irec1 = theMesh->elem[args.elType].ghost->ghostRecv.getRec(FEM_To_ghost_index(args.oldElement));
    int remotechk = irec1->getChk(0);
    int sharedIdx1 = irec1->getIdx(0);
    int sharedIdx2 = theMod->fmUtil->exists_in_IDXL(theMesh,args.e,remotechk,4,0);
    meshMod[remotechk].interpolateElemCopy(theMod->idx, sharedIdx1, sharedIdx2);
  }
  else {
    CkAssert(!FEM_Is_ghost_index(args.e) && FEM_Is_ghost_index(args.oldElement));
    const IDXL_Rec *irec1 = theMesh->elem[args.elType].ghost->ghostRecv.getRec(FEM_To_ghost_index(args.oldElement));
    int remotechk = irec1->getChk(0);
    int sharedIdx = irec1->getIdx(0);
    entDataMsg *em = meshMod[remotechk].packEntData(theMod->idx,sharedIdx,false,0);
    theMod->fmUtil->updateAttrs(em->data, em->datasize, args.e, false, 0);
    delete em;
  }
  return;
}

/** An element is added and derives data from its nodes; assumes relevant data 
    was copied to the appropriate nodes prior to this operation; see the utility
    function below 
*/
void FEM_Interpolate::FEM_InterpolateElementFromNodes(ElementArgs args)
{
  if (elemNodeFnPtr) { // default is overridden
    elemNodeFnPtr(args);
    return;
  }
  // do default interpolation
  // DEFAULT BEHAVIOR: NO ELEMENT DATA
}

/** Store data of an element temporarily on all nodes; this data is used later 
    to derive an element's data 
*/
void FEM_Interpolate::FEM_InterpolateElementToNodes(int e)
{
  // DEFAULT BEHAVIOR: NO ELEMENT DATA
  // This function intentionally left blank; derived classes may define it, or
  // user is responsible for transferring element data to nodes on their own
  return;
}



/** This is to be used to copy the attributes of oldnode to newnode
    oldnode or newnode can be ghost or local
    currently it will only copy coordinates & boundary
*/
void FEM_Interpolate::FEM_InterpolateCopyAttributes(int oldnode, int newnode) {
  int numchunks;
  IDXL_Share **chunks1;
  int index = theMod->idx;

#ifndef FEM_SILENT
  theMod->fmUtil->FEM_Print_coords(theMesh,oldnode);
#endif
  if(FEM_Is_ghost_index(oldnode) || FEM_Is_ghost_index(newnode)) {
    theMod->fmUtil->getChunkNos(0,oldnode,&numchunks,&chunks1);
  }

  char *data;
  int size, count;
  entDataMsg *em;
  if(!FEM_Is_ghost_index(oldnode)) {
    theMod->fmUtil->packEntData(&data, &size, &count, oldnode, true, 0);
  }
  else {
    for(int j=0; j<numchunks; j++) {
      int chk = chunks1[j]->chk;
      if(chk==index) continue;
      int ghostIdx = theMod->fmUtil->exists_in_IDXL(theMesh,oldnode,chk,2);
      em = meshMod[chk].packEntData(theMod->idx,ghostIdx,true,0);
      data = em->data;
      size = em->memsize;
      count = em->datasize;
      break;
    }
  }

  //update it on all chunks from which it receives that ghost
  if(FEM_Is_ghost_index(newnode)) {
    for(int j=0; j<numchunks; j++) {
      int chk = chunks1[j]->chk;
      if(chk==index) continue;
      int sharedIdx = theMod->fmUtil->exists_in_IDXL(theMesh,newnode,chk,2);
      updateAttrsMsg *umsg = new (size, 0)updateAttrsMsg(count);
      umsg->fromChk = theMod->idx;
      umsg->sharedIdx = sharedIdx;
      umsg->isnode = true;
      umsg->isGhost = true;
      memcpy(umsg->data,data,size);
      meshMod[chk].updateAttrs(umsg);
    }
  }
  else {
    theMod->fmUtil->updateAttrs(data, count, newnode, true, 0);
  }
  delete em;

  if(FEM_Is_ghost_index(oldnode) || FEM_Is_ghost_index(newnode)) {
    for(int j=0; j<numchunks; j++) {
      delete chunks1[j];
    }
    if(numchunks != 0) free(chunks1);
  }
#ifndef FEM_SILENT
  theMod->fmUtil->FEM_Print_coords(theMesh,newnode);
#endif
  return;
}
