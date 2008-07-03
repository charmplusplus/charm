/* File: bulk_adapt.C
 * Authors: Terry Wilmarth
 */

/** This module implements high level mesh adaptivity algorithms that make use 
 *  of the bulk mesh adaptivity operations provided by bulk_adapt_ops.
 */
#include "ParFUM.h"
#include "ParFUM_internals.h"
#include "ParFUM_SA.h"
#include "bulk_adapt.h"
#include "bulk_adapt_ops.h"

#define MINAREA 1.0e-18
#define MAXAREA 1.0e12
#define GRADATION 1.2
#define ADAPT_VERBOSE

CtvDeclare(Bulk_Adapt *, _bulkAdapt);

/** Perform refinements on a mesh.  Tries to maintain/improve element quality
 * as specified by a quality measure qm;
 * if method = 0, refine areas with size larger than factor down to factor
 * if method = 1, refine elements down to sizes specified in sizes array
 * Negative entries in size array indicate no refinement. 
 */
void Bulk_Adapt::ParFUM_Refine(int qm, int method, double factor, double *sizes)
{
  SetMeshSize(method, factor, sizes);
  ParFUM_GradateMesh(GRADATION);
  (void)Refine_h(qm, method, factor, sizes);
}

/** The actual refine in the previous operation */
int Bulk_Adapt::Refine_h(int qm, int method, double factor, double *sizes)
{
  int numNodes, numElements;
  int elId, mods, iter_mods;
  int elemWidth = theMesh->elem[0].getConn().width();
  Element_Bucket elemsToRefine;
  int elemConn[3];
  int count = 0;
  int myId = FEM_My_partition();

  mods=0;
  iter_mods=1;

  while ((iter_mods != 0) || (count < 20)) {
    if (iter_mods == 0)
      count++;
    else 
      count = 0;
    iter_mods=0;
    numNodes = theMesh->node.size();
    numElements = theMesh->elem[0].size();
    // sort elements to be refined by quality into elemsToRefine
    elemsToRefine.Clear();
    elemsToRefine.Alloc(numElements);
    for (int i=0; i<numElements; i++) { 
      if (theMesh->elem[0].is_valid(i)) {
	// find avg and max edge length of element i
	double avgEdgeLength, maxEdgeLength, minEdgeLength;
	int maxEdge, minEdge;
	findEdgeLengths(i, &avgEdgeLength, &maxEdgeLength, &maxEdge, 
			&minEdgeLength, &minEdge);
	//double qFactor=getAreaQuality(i);
	if (theMesh->elem[0].getMeshSizing(i) <= 0.0) 
	  CkPrintf("WARNING: mesh element %d has no sizing!\n", i);
	if ((theMesh->elem[0].getMeshSizing(i) > 0.0) &&
	  (avgEdgeLength > (theMesh->elem[0].getMeshSizing(i)*REFINE_TOL))){
	    //|| (qFactor < QUALITY_MIN)) {
	  //elemsToRefine.Insert(i, qFactor*(1.0/(avgEdgeLength+maxEdgeLength)), 0);
	  elemsToRefine.Insert(i, 1.0/maxEdgeLength);
#ifdef ADAPT_VERBOSE
	  CkPrintf("[%d]ParFUM_Refine: Added element %d to refine list: targetSize=%6.4f maxEdgeLength=%6.4f maxEdge=%d\n", myId, i, theMesh->elem[0].getMeshSizing(i), maxEdgeLength, maxEdge);
#endif	  
	}
      }
    }

    while (!elemsToRefine.IsBucketEmpty()) { // loop through the elements
      int n1, n2;
      double len;

      elId = elemsToRefine.Remove(&len);

      if ((elId != -1) && (theMesh->elem[0].is_valid(elId))) {
	int n1, n2;
	double avgEdgeLength, maxEdgeLength, minEdgeLength;
	int maxEdge, minEdge;
	findEdgeLengths(elId, &avgEdgeLength, &maxEdgeLength, &maxEdge, 
			&minEdgeLength, &minEdge);
	//double qFactor=getAreaQuality(elId);

	//if (len == qFactor*(1.0/(avgEdgeLength+maxEdgeLength))) { // this elem does not appear to be modified; refine it
	if (len == (1.0/maxEdgeLength)) { // this elem does not appear to be modified; refine it
	  if ((theMesh->elem[0].getMeshSizing(elId) > 0.0) &&
	      (avgEdgeLength>(theMesh->elem[0].getMeshSizing(elId)*REFINE_TOL))){
	    //|| (qFactor < QUALITY_MIN)) {
	    //decide if we should do a flip or bisect
#ifdef ADAPT_VERBOSE
	    CkPrintf("[%d]ParFUM_Refine: Refining element %d: targetSize=%6.4f maxEdgeLength=%6.4f maxEdge=%d\n", myId, elId, theMesh->elem[0].getMeshSizing(elId), maxEdgeLength, maxEdge);
#endif	  
	    RegionID lockRegionID;
	    int success;
	    lockRegionID.localID = -1;
	    success = theMesh->parfumSA->bulkAdapt->lock_3D_region(elId, 0, maxEdge, maxEdgeLength, 
								   &lockRegionID);
	    if (success == 2) { // lock obtained straight away
#ifdef ADAPT_VERBOSE
	      CkPrintf("[%d]ParFUM_Refine: Refining element %d: GOT THE LOCK\n", myId, elId);
#endif	  
	      iter_mods = iter_mods + 1;
	      CkAssert(theMesh->parfumSA->holdingLock == lockRegionID);
	      (void) theMesh->parfumSA->bulkAdapt->edge_bisect(elId, 0, maxEdge, dim, lockRegionID);
	      theMesh->parfumSA->bulkAdapt->unlock_3D_region(lockRegionID);
	    }
 	    else if (success == 1) { // lock is pending
#ifdef ADAPT_VERBOSE
	      CkPrintf("[%d]ParFUM_Refine: Refining element %d: LOCK PENDING\n", myId, elId);
#endif	  
	      while (success == 1) {
		CthYield();
		double DavgEdgeLength, DmaxEdgeLength, DminEdgeLength;
		int DmaxEdge, DminEdge;
		findEdgeLengths(elId, &DavgEdgeLength, &DmaxEdgeLength, &DmaxEdge, 
				&DminEdgeLength, &DminEdge);
		if (len == (1.0/DmaxEdgeLength)) { // element is unchanged: i.e. still needs refinement
		  success = theMesh->parfumSA->bulkAdapt->lock_3D_region(elId, 0, DmaxEdge, DmaxEdgeLength, 
									 &lockRegionID);
		}
		else {
		  theMesh->parfumSA->bulkAdapt->unpend_3D_region(lockRegionID);
#ifdef ADAPT_VERBOSE
		  CkPrintf("[%d]ParFUM_Refine: Refining element %d: FAILED TO GET LOCK\n", myId, elId);
#endif	  
		  elemsToRefine.Insert(elId, 1.0/DmaxEdgeLength);	  
		  break; // element was modified by a different operation
		}
	      }
	      if (success==2) { // broke out of loop with a successful lock
#ifdef ADAPT_VERBOSE
		CkPrintf("[%d]ParFUM_Refine: Refining element %d: GOT THE LOCK\n", myId, elId);
#endif	  
		iter_mods = iter_mods + 1;
		CkAssert(theMesh->parfumSA->holdingLock == lockRegionID);
		(void) theMesh->parfumSA->bulkAdapt->edge_bisect(elId, 0, maxEdge, dim, lockRegionID);
		theMesh->parfumSA->bulkAdapt->unlock_3D_region(lockRegionID);
	      }
	      else if (success==0) {
		theMesh->parfumSA->bulkAdapt->unpend_3D_region(lockRegionID);
#ifdef ADAPT_VERBOSE
		CkPrintf("[%d]ParFUM_Refine: Refining element %d: FAILED TO GET LOCK\n", myId, elId);
#endif	  
		elemsToRefine.Insert(elId, 1.0/maxEdgeLength);	  
	      }
	    }
	    else if (success == 0) { // lock failed immediately
#ifdef ADAPT_VERBOSE
	      CkPrintf("[%d]ParFUM_Refine: Refining element %d: FAILED TO GET LOCK\n", myId, elId);
#endif	  
	      elemsToRefine.Insert(elId, 1.0/maxEdgeLength);	  
	    }
	  }
	}
	else { // elem was modified; return to bucket
	  elemsToRefine.Insert(elId, 1.0/maxEdgeLength);	  
	}
      }
      CthYield(); // give other chunks on the same PE a chance
    }
    mods += iter_mods;
#ifdef ADAPT_VERBOSE
    CkPrintf("[%d]ParFUM_Refine: %d modifications in last pass.\n",myId,iter_mods);
#endif
  }
#ifdef ADAPT_VERBOSE
  CkPrintf("[%d]ParFUM_Refine: %d total modifications.\n",myId,mods);
#endif
  elemsToRefine.Clear();
  return mods;
}

/** Perform coarsening on a mesh.  Tries to maintain/improve element quality
 * as specified by a quality measure qm;
 * if method = 0, coarsen areas with size smaller than factor up to factor
 * if method = 1, coarsen elements up to sizes specified in sizes array
 * Negative entries in size array indicate no coarsening. 
 */
void Bulk_Adapt::ParFUM_Coarsen(int qm, int method, double factor, 
				 double *sizes)
{
  SetMeshSize(method, factor, sizes);
  ParFUM_GradateMesh(GRADATION);
  //  (void)Coarsen_h(qm, method, factor, sizes);
}

/** The actual coarsen in the previous operation
*/
int Bulk_Adapt::Coarsen_h(int qm, int method, double factor, double *sizes)
{
  return 0;
  /*
  // loop through elemsToRefine
  int elId, mods=0, iter_mods=1, pass=0;
  int elemWidth = theMesh->elem[0].getConn().width();
  double qFactor;
  coarsenElements = NULL;
  coarsenHeapSize = 0;
  int elemConn[3];
  while (iter_mods != 0) {
    iter_mods=0;
    pass++;
    numNodes = theMesh->node.size();
    numElements = theMesh->elem[0].size();
    // sort elements to be refined by quality into elemsToRefine
    if (coarsenElements) delete [] coarsenElements;
    coarsenElements = new elemHeap[numElements+1];
    coarsenElements[0].len=-2.0;
    coarsenElements[0].elID=-1;
    for (int i=0; i<numElements; i++) { 
      if (theMesh->elem[0].is_valid(i)) {
	// find minEdgeLength of i
	theMesh->e2n_getAll(i, elemConn);
	bool notclear = false;
	for(int j=0; j<elemWidth; j++) {
	  int nd = elemConn[j];
	  if(nd>=0) notclear = theMod->fmLockN[nd].haslocks();
	  //else cannot look up because connectivity might be screwed up,
	  //I am hoping that at least one of the nodes will be local
	}
	if(notclear) continue;
	double tmpLen, avgEdgeLength=0.0, 
	  minEdgeLength = length(elemConn[0], elemConn[1]);
	for (int j=0; j<elemWidth-1; j++) {
	  for (int k=j+1; k<elemWidth; k++) {
	    tmpLen = length(elemConn[j], elemConn[k]);
	    if(tmpLen < -1.0) {notclear = true;}
	    avgEdgeLength += tmpLen;
	    if (tmpLen < minEdgeLength) minEdgeLength = tmpLen;
	  }
	}
	if(notclear) continue;
	avgEdgeLength /= 3.0;
	qFactor=getAreaQuality(i);
	if (((theMesh->elem[0].getMeshSizing(i) > 0.0) &&
	     (avgEdgeLength < (theMesh->elem[0].getMeshSizing(i)*COARSEN_TOL)))
	     || (qFactor < QUALITY_MIN)) {
	    //){
	  //CkPrintf("Marking elem %d for coarsening\n", i);
	  Insert(i, qFactor*minEdgeLength, 1);
	}
      }
    }
    while (coarsenHeapSize>0) { // loop through the elements
      elId=Delete_Min(1);
      if ((elId != -1) && (theMesh->elem[0].is_valid(elId))) {
	theMesh->e2n_getAll(elId, elemConn);
	bool notclear = false;
	for(int j=0; j<elemWidth; j++) {
	  int nd = elemConn[j];
	  if(nd>=0) notclear = theMod->fmLockN[nd].haslocks();
	  //else cannot look up because connectivity might be screwed up,
	  //I am hoping that at least one of the nodes will be local
	}
	if(notclear) continue;
	int n1=elemConn[0], n2=elemConn[1];
	double tmpLen, avgEdgeLength=0.0, 
	  minEdgeLength = length(n1, n2);
	for (int j=0; j<elemWidth-1; j++) {
	  for (int k=j+1; k<elemWidth; k++) {
	    tmpLen = length(elemConn[j], elemConn[k]);
	    if(tmpLen < -1.0) {notclear = true;}
	    avgEdgeLength += tmpLen;
	    if (tmpLen < minEdgeLength) {
	      minEdgeLength = tmpLen;
	      n1 = elemConn[j]; n2 = elemConn[k];
	    }
	  }
	}
	if(notclear) continue;
	CkAssert(n1!=-1 && n2!=-1);
	avgEdgeLength /= 3.0;
	qFactor=getAreaQuality(elId);
	// coarsen element's short edge
	if (((theMesh->elem[0].getMeshSizing(elId) > 0.0) &&
	     (avgEdgeLength < (theMesh->elem[0].getMeshSizing(elId)*COARSEN_TOL)))
	      || (qFactor < QUALITY_MIN)) {
	  //){

	  int eNbr = theMesh->e2e_getNbr(elId, theMesh->e2n_getIndex(elId,n1)+
					 theMesh->e2n_getIndex(elId,n2)-1, 0);
	  // determine if eNbr should also be coarsened
	  if ((eNbr >= 0) && (theMesh->elem[0].is_valid(eNbr))) {
	    theMesh->e2n_getAll(eNbr, elemConn);
	    avgEdgeLength=0.0;
	    for (int j=0; j<elemWidth-1; j++) {
	      for (int k=j+1; k<elemWidth; k++) {
		avgEdgeLength += length(elemConn[j], elemConn[k]);
	      }
	    }
	    avgEdgeLength /= 3.0;
	    qFactor=getAreaQuality(eNbr);
	    if (((theMesh->elem[0].getMeshSizing(eNbr) > 0.0) &&
		 (avgEdgeLength < (theMesh->elem[0].getMeshSizing(eNbr))))
		|| (qFactor < QUALITY_MIN)) {
	      //CkPrintf("Coarsening elem %d which has desired edge length %6.6e and average edge length %6.6e.\n", elId, theMesh->elem[0].getMeshSizing(elId), avgEdgeLength);
	      if (theAdaptor->edge_contraction(n1, n2) > 0)  iter_mods++;
	    }
	  }
	  else {
	    //CkPrintf("Coarsening elem %d which has desired edge length %6.6e and average edge length %6.6e.\n", elId, theMesh->elem[0].getMeshSizing(elId), avgEdgeLength);
	    if (theAdaptor->edge_contraction(n1, n2) > 0)  iter_mods++;
	  }
	}
      }
      CthYield(); // give other chunks on the same PE a chance
    }
    mods += iter_mods;
#ifdef ADAPT_VERBOSE
    CkPrintf("[%d]ParFUM_Coarsen: %d modifications in pass %d.\n",theMod->idx,iter_mods,pass);
#endif
#ifdef DEBUG_QUALITY
    tests(false);
#endif
  }
#ifdef ADAPT_VERBOSE
  CkPrintf("[%d]ParFUM_Coarsen: %d total modifications over %d passes.\n",theMod->idx,mods,pass);
#endif
  delete[] coarsenElements;
  return mods;
  */
}

/** Performs a sequence of refinements or coarsening as is needed
 * to achieve the target areas for elements
 */
void Bulk_Adapt::ParFUM_AdaptMesh(int qm, int method, double factor, 
				   double *sizes)
{
  /*
  MPI_Comm comm=(MPI_Comm)FEM_chunk::get("FEM_Update_mesh")->defaultComm;
  MPI_Barrier(comm);
#ifdef ADAPT_VERBOSE
  CkPrintf("[%d]BEGIN: FEM_AdaptMesh...\n",theMod->idx);
#endif
#ifdef DEBUG_QUALITY
  tests(true);
#endif
  SetMeshSize(method, factor, sizes);
  MPI_Barrier(comm);
  GradateMesh(GRADATION);
  MPI_Barrier(comm);
  (void)Refine(qm, method, factor, sizes);
  MPI_Barrier(comm);
  GradateMesh(GRADATION);
  MPI_Barrier(comm);
  (void)Coarsen(qm, method, factor, sizes);
#ifdef DEBUG_QUALITY
  MPI_Barrier(comm);
#endif
  FEM_Repair(qm);
  MPI_Barrier(comm);
#ifdef DEBUG_QUALITY
  tests(true);
  MPI_Barrier(comm);
#endif
#ifdef ADAPT_VERBOSE
  CkPrintf("[%d]...END: FEM_AdaptMesh.\n",theMod->idx);
#endif
  */
}

/** Smooth the mesh using method according to some quality measure qm
 */
void Bulk_Adapt::ParFUM_Smooth(int qm, int method)
{
  CkPrintf("WARNING: Bulk_Adapt::Smooth: Not yet implemented.\n");
}

/** FEM_Mesh_smooth
 *	Inputs	: meshP - a pointer to the FEM_Mesh object to smooth
 *		: nodes - an array of local node numbers to be smoothed.  Send
 *			  NULL pointer to smooth all nodes.
 *		: nNodes - the size of the nodes array
 *		: attrNo - the attribute number where the coords are registered
 *	Shifts nodes around to improve mesh quality.  FEM_BOUNDARY attribute
 *	and interpolator function must be registered by user to maintain 
 *	boundary information.
 */
/*
void  FEM_Adapt_Algs::FEM_mesh_smooth(FEM_Mesh *meshP, int *nodes, int nNodes, int attrNo) {
  vector2d newPos, *coords, *ghostCoords;
  int idx, nNod, nGn, gIdxN, *boundVals, nodesInChunk, mesh;
  int *adjnodes;

  mesh=FEM_Mesh_default_read();
  nodesInChunk = FEM_Mesh_get_length(mesh,FEM_NODE);
  nGn = FEM_Mesh_get_length(mesh, FEM_GHOST + FEM_NODE);
  
  boundVals = new int[nodesInChunk];
  coords = new vector2d[nodesInChunk+nGn];

  FEM_Mesh_data(mesh, FEM_NODE, FEM_BOUNDARY, (int*) boundVals, 0, nodesInChunk, FEM_INT, 1);    

  FEM_Mesh_data(mesh, FEM_NODE, attrNo, (double*)coords, 0, nodesInChunk, FEM_DOUBLE, 2);

  IDXL_Layout_t coord_layout = IDXL_Layout_create(IDXL_DOUBLE, 2);
  FEM_Update_ghost_field(coord_layout,-1, coords); 
  ghostCoords = &(coords[nodesInChunk]);
  for (int i=0; i<nNodes; i++)
  {
    if (nodes==NULL) idx=i;
    else idx=nodes[i];
    newPos.x=0;
    newPos.y=0;
    CkAssert(idx<nodesInChunk);  
    if (FEM_is_valid(mesh, FEM_NODE, idx) && boundVals[idx]==0) //node must be internal
    {
      meshP->n2n_getAll(idx, adjnodes, nNod);
      for (int j=0; j<nNod; j++) { //for all adjacent nodes, find coords
	if (adjnodes[j]<-1) {
	  gIdxN = FEM_From_ghost_index(adjnodes[j]);
	  newPos.x += ghostCoords[gIdxN].x;
	  newPos.y += ghostCoords[gIdxN].y;
	}
	else {
	  newPos.x += coords[adjnodes[j]].x;
	  newPos.y += coords[adjnodes[j]].y;
	}     
      }
      newPos.x/=nNod;
      newPos.y/=nNod;
      FEM_set_entity_coord2(mesh, FEM_NODE, idx, newPos.x, newPos.y);
      delete [] adjnodes;
    }
  }

  delete [] coords;
  delete [] boundVals;
}
*/

/** Elements with a bad quality metric are either flipped or coarsened to 
 * newer elements which are of better quality
 */
void Bulk_Adapt::ParFUM_Repair(int qm)
{
  /*
  double avgQual = 0.0, minQual = getAreaQuality(0);
  int numBadElems = 0;
  int elemConn[3];
#ifdef ADAPT_VERBOSE
  CkPrintf("[%d]WARNING: ParFUM_Repair: Under construction.\n",theMod->idx);
  numElements = theMesh->elem[0].size();
  for (int i=0; i<numElements; i++) { 
    if (theMesh->elem[0].is_valid(i)) {
      double qFactor=getAreaQuality(i);
      avgQual += qFactor;
      if (qFactor <  QUALITY_MIN) {
	numBadElems++;
	if (qFactor < minQual) minQual = qFactor;
      }
    }
  }
  avgQual /= numElements;
  CkPrintf("BEFORE FEM_Repair: Average Element Quality = %2.6f, Min = %2.6f (1.0 is perfect)\n", avgQual, minQual);
  //CkPrintf("BEFORE FEM_Repair: Average Element Quality = %2.6f, Min = %2.6f (1.0 is perfect)\n  %d out of %d elements were below the minimum quality tolerance of %2.6f\n", avgQual, minQual, numBadElems, numElements, QUALITY_MIN);
#endif

  int elemWidth = theMesh->elem[0].getConn().width();
  int changes=1, totalChanges=0;
  int count=0;
  while (changes!=0 && count<4) {
    count++;
    changes = 0;
    numElements = theMesh->elem[0].size();
    for (int i=0; i<numElements; i++) { 
      if (theMesh->elem[0].is_valid(i)) {
	double qFactor=getAreaQuality(i);
	if (qFactor <  0.75*QUALITY_MIN) {
	  int elId = i;
	  theMesh->e2n_getAll(elId, elemConn);
	  bool notclear = false;
	  for(int j=0; j<elemWidth; j++) {
	    int nd = elemConn[j];
	    if(nd>=0) notclear = theMod->fmLockN[nd].haslocks();
	    //else cannot look up because connectivity might be screwed up,
	    //I am hoping that at least one of the nodes will be local
	  }
	  if(notclear) continue;
	  int n1=elemConn[0], n2=elemConn[1];
	  //too bad.. should not decide without locking!!
	  //these values might change by the time lock is acquired
	  double len1 = length(elemConn[0],elemConn[1]);
	  double len2 = length(elemConn[1],elemConn[2]);
	  double len3 = length(elemConn[2],elemConn[0]);
	  if(len1<-1.0 || len2<-1.0 || len3<-1.0) continue;
	  double avglen=(len1+len2+len3)/3.0;
	  int maxn1=0, maxn2=1;
	  int minn1=0, minn2=1;
	  double maxlen=len1;
	  int maxed=0;
	  if(len2>maxlen) {
	    maxlen = len2;
	    maxn1 = 1;
	    maxn2 = 2;
	    maxed = 1;
	  }
	  if(len3>maxlen) {
	    maxlen = len3;
	    maxn1 = 2;
	    maxn2 = 0;
	    maxed = 2;
	  }
	  double minlen = len1;
	  if(len2<minlen) {
	    minlen = len2;
	    minn1 = 1;
	    minn2 = 2;
	  }
	  if(len3<minlen) {
	    minlen = len3;
	    minn1 = 2;
	    minn2 = 0;
	  }
	  double otherlen = 3*avglen - maxlen - minlen;
	  if (maxlen > 0.95*(minlen+otherlen)) { // refine
	    //decide if this should be a bisect or flip, 
	    //depends on if the longest edge on this element is also the longest on the
	    //element sharing this edge, if not, flip it
	    int nbrEl = theMesh->e2e_getNbr(i,maxed);
	    double len4=0.0, len5=0.0;
	    if(nbrEl!=-1) {
	      int con1[3];
	      theMesh->e2n_getAll(nbrEl,con1);
	      int nbrnode=-1;
	      for(int j=0; j<3; j++) {
		if(con1[j]!=elemConn[maxn1] && con1[j]!=elemConn[maxn2]) {
		  nbrnode = con1[j];
		  break;
		}
	      }
	      len4 = length(elemConn[maxn1],nbrnode);
	      len5 = length(elemConn[maxn2],nbrnode);
	    }
	    int success = -1;
	    if(len4>maxlen || len5>maxlen) {
#ifdef DEBUG_FLIP
	      success = theAdaptor->edge_flip(elemConn[maxn1], elemConn[maxn2]);
#endif
	    }
	    else {
	      success = theAdaptor->edge_bisect(elemConn[maxn1], elemConn[maxn2]);
	    }
	    if (success >= 0) {
	      //CkPrintf("Refined bad element!\n");
	      changes++;
	    }
	  }
	  else if (minlen < 0.10*(maxlen+otherlen)) { // coarsen
	    int success = theAdaptor->edge_contraction(elemConn[minn1], elemConn[minn2]);
	    if (success >= 0) { 
	      //CkPrintf("Coarsened bad element!\n");
	      changes++;
	    }
	  }
	  else {
	    //CkPrintf("Leaving one bad element alone...\n");
	  }
	}
      }
    }
    totalChanges += changes;
  }

#ifdef ADAPT_VERBOSE
  numElements = theMesh->elem[0].size();
  numBadElems = 0;
  avgQual = 0.0;
  minQual = getAreaQuality(0);
  for (int i=0; i<numElements; i++) { 
    if (theMesh->elem[0].is_valid(i)) {
      double qFactor=getAreaQuality(i);
      avgQual += qFactor;
      if (qFactor <  QUALITY_MIN) {
	numBadElems++;
	if (qFactor < minQual) minQual = qFactor;
      }
    }
  }
  avgQual /= numElements;
  CkPrintf("[%d]AFTER FEM_Repair: Average Element Quality = %2.6f, Min = %2.6f (1.0 is perfect) No. of repairs %d\n",theMod->idx,avgQual, minQual,totalChanges);
#ifdef DEBUG_QUALITY
    tests(false);
#endif
  //  CkPrintf("AFTER FEM_Repair: Average Element Quality = %2.6f, Min = %2.6f (1.0 is perfect)\n  %d out of %d elements were below the minimum quality tolerance of %2.6f\n", avgQual, minQual, numBadElems, numElements, QUALITY_MIN);
#endif
  */
}

/** Remesh entire mesh according to quality measure qm
 * if method = 0, set entire mesh size to factor
 * if method = 1, keep regional mesh sizes, and scale by factor
 * if method = 2, uses sizes to size mesh by regions 
 */
void Bulk_Adapt::ParFUM_Remesh(int qm, int method, double factor, double *sizes)
{
  CkPrintf("WARNING: ParFUM_Remesh: Under construction.\n");
}


/** For each element, set its size to its average edge length
 */
void Bulk_Adapt::ParFUM_SetReferenceMesh()
{
  // TODO: do we need to run this loop for element types other than 0?
  double avgLength = 0.0;
  int width = theMesh->elem[0].getConn().width();
  int numElements = theMesh->elem[0].size();
  int elemConn[3];
  
  for (int i=0; i<numElements; ++i, avgLength=0) {
    if(theMesh->elem[0].is_valid(i)) {
      theMesh->e2n_getAll(i, elemConn);
      for (int j=0; j<width-1; ++j) {
	avgLength += length(elemConn[j], elemConn[j+1]);
      }
      avgLength += length(elemConn[0], elemConn[width-1]);
      avgLength /= width;
      theMesh->elem[0].setMeshSizing(i, avgLength);      
    }
  }
}

/** Resize mesh elements to avoid jumps in element size
    Algorithm based on h-shock correction, described in
    Mesh Gradation Control, Borouchaki et al
    IJNME43 1998 www.ann.jussieu.fr/~frey/publications/ijnme4398.pdf 
*/
void Bulk_Adapt::ParFUM_GradateMesh(double smoothness)
{
  const double beta = smoothness;
  double maxShock, minShock;
  int iteration = 0, updates = 0;
  int *adjNodes, *boundNodes;
  int nadjNodes, nnodes;
  int meshNum = FEM_Mesh_default_read();
  //if (smoothness < 1.0) {
  //printf("");
  //}
  nnodes = theMesh->node.size();
  boundNodes = new int[nnodes];
  FEM_Mesh_data(meshNum, FEM_NODE, FEM_BOUNDARY, 
		boundNodes, 0, nnodes, FEM_INT, 1);
  
  //printf("Running h-shock mesh gradation with beta=%.3f\n", beta);
  fflush(NULL);
  
#ifndef GRADATION_ITER_LIMIT
#define GRADATION_ITER_LIMIT    10
#endif
  
  do {
    maxShock = 0;
    minShock = 1e10;
    
    for (int node=0; node<nnodes; ++node) {
      if (boundNodes[node]!= 0 || !FEM_is_valid(meshNum, FEM_NODE, node))
	continue;
      //if (!FEM_is_valid(meshNum, FEM_NODE, node))
      //    continue;
      
      theMesh->n2n_getAll(node, adjNodes, nadjNodes);
      for (int adjNode=0; adjNode<nadjNodes; ++adjNode) {
	double edgelen = length(node, adjNodes[adjNode]);
	
	// get adjacent elemnents and their sizes
	int e1, e2;
	theMesh->get2ElementsOnEdge(node, adjNodes[adjNode], &e1, &e2);
	if (e1 <= -1 || e2 <= -1) continue; //this will not smooth across boundaries
	
	double s1, s2;
	s1 = theMesh->elem[0].getMeshSizing(e1);
	s2 = theMesh->elem[0].getMeshSizing(e2);
	if (s1 <= 0 || s2 <= 0) continue;
	
	// h-shock=max(size ratio)^(1/edge length)
	CkAssert(s1 >= 0 && s2 >= 0 && "Bad size");
	//CkAssert(edgelen > 1e-6 && "Length 0 edge");
	CkAssert(edgelen == edgelen && "Length inf edge");
	
	double ratio = (s1 > s2) ? s1/s2 : s2/s1;
	CkAssert (ratio >= 1.0 && ratio == ratio && "Bad ratio");
	
	// WARNING WARNING WARNING
	// TEST ONLY, THIS IS NOT CORRECT
	if (ratio > beta) {
	  if (s1 > s2) {
	    theMesh->elem[0].setMeshSizing(e1, s1 - (s1-s2)/3);
	    theMesh->elem[0].setMeshSizing(e2, s2 + (s1-s2)/3);
	  } else {
	    theMesh->elem[0].setMeshSizing(e2, s2 - (s2-s1)/3);
	    theMesh->elem[0].setMeshSizing(e1, s1 + (s2-s1)/3);
	  }
	  updates++;
	}
	if (ratio > maxShock) maxShock = ratio;
	if (ratio < minShock) minShock = ratio;
	
	
	////double hs = ratio;
	//double hs = pow(ratio, 1.0/edgelen);
	//
	//if (hs > maxShock) maxShock = hs;
	//if (hs < minShock) minShock = hs;
	
	//// if hs > beta, resize the larger elt:
	//// new size = old size / eta^2
	//// eta = (beta / h-shock)^(edge length)
	////     = beta^(edge length) / size ratio
	//
	//if (hs > beta) {
	//    double etasq = pow(beta, edgelen) / ratio;
	//    etasq *= etasq;
	
	//    //if (hs > 100) {
	//    //    printf("hs: %8.5f\ns1: %8.5f\ns2: %8.5f\nedgelen: %8.5f\nratio: %8.5f\netasq: %8.5f", hs, s1, s2, edgelen, ratio, etasq);
	//    //    abort();
	//    //}
	//    
	//    if (s1 > s2) {
	//        theMesh->elem[0].setMeshSizing(e1, s1 / etasq);
	//    } else {
	//        theMesh->elem[0].setMeshSizing(e2, s2 / etasq);
	//    }
	//    updates++;
	//}
	
      }
      if (nadjNodes > 0)
	delete[] adjNodes;
    } 
    
    //printf("Finished iteration %d\n", iteration);
    //printf("Max shock:%8.3f\n", maxShock);
    //printf("Min shock:%8.3f\n", minShock);
    //printf("Target:%8.3f\n", beta);
    
  } while (maxShock > beta && ++iteration < GRADATION_ITER_LIMIT);
  
  //printf("%d total updates in %d iterations in GradateMesh\n", updates, iteration);
  fflush(NULL);
  delete[] boundNodes;
  return;
}

/** Set sizes on elements throughout the mesh; note: size is edge length */
void Bulk_Adapt::SetMeshSize(int method, double factor, double *sizes)
{
  int numNodes = theMesh->node.size();
  int numElements = theMesh->elem[0].size();
  int elemConn[3];

  if (method == 0) { // set uniform sizing specified in factor

    for (int i=0; i<numElements; i++) {
      theMesh->elem[0].setMeshSizing(i, factor);
    }
    //CkPrintf("ParFUM_SetMeshSize: UNIFORM %4.6e\n", factor);
  }
  else if (method == 1) { // copy sizing from array
    for (int i=0; i<numElements; i++) {
      if (sizes[i] > 0.0) {
	theMesh->elem[0].setMeshSizing(i, sizes[i]);
      }
    }
    //CkPrintf("ParFUM_SetMeshSize: SIZES input\n");
  }
  else if (method == 2) { // calculate current sizing and scale by factor
    double avgEdgeLength = 0.0;
    int width = theMesh->elem[0].getConn().width();
    int numEdges=3;
    if (dim==3) numEdges=6;
    for (int i=0; i<numElements; i++) {
      if(theMesh->elem[0].is_valid(i)) {
	theMesh->e2n_getAll(i, elemConn);
	for (int j=0; j<width-1; j++) {
	  for (int k=j+1; k<width; k++) {
	    avgEdgeLength += length(elemConn[j], elemConn[k]);
	  }
	}
	avgEdgeLength += length(elemConn[0], elemConn[width-1]);
	avgEdgeLength /= (double)numEdges;
	theMesh->elem[0].setMeshSizing(i, factor*avgEdgeLength);
      }
    }
    //CkPrintf("ParFUM_SetMeshSize: CALCULATED & SCALED \n");
  }
  else if (method == 3) { // scale existing sizes by array sizes
    for (int i=0; i<numElements; i++) {
      if (sizes[i] > 0.0) {
	theMesh->elem[0].setMeshSizing(i, sizes[i]*theMesh->elem[0].getMeshSizing(i));
      }
    }
  }
  else if (method == 4) { // scale existing sizes by factor
    for (int i=0; i<numElements; i++) {
      theMesh->elem[0].setMeshSizing(i, factor*theMesh->elem[0].getMeshSizing(i));
    }
  }
  else if (method == 5) { // mesh sizing has been set independently; use as is
    //CkPrintf("ParFUM_SetMeshSize: USE EXISTING SIZES \n");
  }
  //  CkPrintf("Current mesh sizing: ");
  //for (int i=0; i<numElements; i++) {
  //CkPrintf("%4.6e ", theMesh->elem[0].getMeshSizing(i));
  //}
}


//HELPER functions
double Bulk_Adapt::length(int n1, int n2)
{
  double coordsn1[2], coordsn2[2];
  getCoord(n1, coordsn1);
  getCoord(n2, coordsn2);

  double ret = length(coordsn1, coordsn2);
  return ret;
}

double Bulk_Adapt::length3D(int n1, int n2) {
  double coordsn1[3], coordsn2[3];
  getCoord(n1, coordsn1);
  getCoord(n2, coordsn2);

  double ret = length3D(coordsn1, coordsn2);
  return ret;
}

double Bulk_Adapt::length(double *n1_coord, double *n2_coord) { 
  double d, ds_sum=0.0;

  for (int i=0; i<dim; i++) {
    if(n1_coord[i]<-1.0 || n2_coord[i]<-1.0) return -2.0;
    d = n1_coord[i] - n2_coord[i];
    ds_sum += d*d;
  }
  return (sqrt(ds_sum));
}

double Bulk_Adapt::length3D(double *n1_coord, double *n2_coord) { 
  double d, ds_sum=0.0;

  for (int i=0; i<3; i++) {
    if(n1_coord[i]<-1.0 || n2_coord[i]<-1.0) return -2.0;
    d = n1_coord[i] - n2_coord[i];
    ds_sum += d*d;
  }
  return (sqrt(ds_sum));
}

void Bulk_Adapt::findEdgeLengths(int elemID, double *avgEdgeLength, double *maxEdgeLength, int *maxEdge, 
				 double *minEdgeLength, int *minEdge)
{
  FEM_Elem &elem = theMesh->elem[elemType]; // elem is local elements
  int *conn = elem.connFor(elemID);
  //CkPrintf("conn[%d]:[%d,%d,%d,%d]\n", elemID, conn[0], conn[1], conn[2], conn[3]);
  FEM_DataAttribute *coord = theMesh->node.getCoord(); // all local coords
  double *n0co = (coord->getDouble()).getRow(conn[0]);
  double *n1co = (coord->getDouble()).getRow(conn[1]);
  double *n2co = (coord->getDouble()).getRow(conn[2]);
  double *n3co = (coord->getDouble()).getRow(conn[3]);
  //CkPrintf(" node %d = (%6.4f, %6.4f, %6.4f)\n", conn[0], n0co[0], n0co[1], n0co[2]);
  //CkPrintf(" node %d = (%6.4f, %6.4f, %6.4f)\n", conn[1], n1co[0], n1co[1], n1co[2]);
  //CkPrintf(" node %d = (%6.4f, %6.4f, %6.4f)\n", conn[2], n2co[0], n2co[1], n2co[2]);
  //CkPrintf(" node %d = (%6.4f, %6.4f, %6.4f)\n", conn[3], n3co[0], n3co[1], n3co[2]);

  // assuming tets for now
  double edgeLengths[6];
  edgeLengths[0] = length3D(conn[0], conn[1]);
  edgeLengths[1] = length3D(conn[0], conn[2]);
  edgeLengths[2] = length3D(conn[0], conn[3]);
  edgeLengths[3] = length3D(conn[1], conn[2]);
  edgeLengths[4] = length3D(conn[1], conn[3]);
  edgeLengths[5] = length3D(conn[2], conn[3]);
  //CkPrintf("edge 0 has length %6.4f\n", edgeLengths[0]);
  (*maxEdgeLength) = (*minEdgeLength) = (*avgEdgeLength) = edgeLengths[0];
  (*maxEdge) = (*minEdge) = 0;
  for (int i=1; i<6; i++) {
    //CkPrintf("edge %d has length %6.4f\n", i, edgeLengths[i]);
    (*avgEdgeLength) += edgeLengths[i];
    if (edgeLengths[i] > (*maxEdgeLength)) {
      (*maxEdgeLength) = edgeLengths[i]; 
      (*maxEdge) = i;
    }
    else if (edgeLengths[i] < (*minEdgeLength)) {
      (*minEdgeLength) = edgeLengths[i];
      (*minEdge) = i;
    }
  }
  (*avgEdgeLength) /= 6;
  CkAssert((*minEdgeLength) > 0.0);
}


double Bulk_Adapt::getArea(int n1, int n2, int n3)
{
  double coordsn1[2], coordsn2[2], coordsn3[2];
  getCoord(n1, coordsn1);
  getCoord(n2, coordsn2);
  getCoord(n3, coordsn3);

  double ret = getArea(coordsn1, coordsn2, coordsn3);
  return ret;
}

double Bulk_Adapt::getArea(double *n1_coord, double *n2_coord, double *n3_coord) {
  double area=0.0;
  double aLen, bLen, cLen, sLen, d, ds_sum;

  ds_sum = 0.0;
  for (int i=0; i<2; i++) {
    d = n1_coord[i] - n2_coord[i];
    ds_sum += d*d;
  }
  aLen = sqrt(ds_sum);
  ds_sum = 0.0;
  for (int i=0; i<2; i++) {
    d = n2_coord[i] - n3_coord[i];
    ds_sum += d*d;
  }
  bLen = sqrt(ds_sum);
  ds_sum = 0.0;
  for (int i=0; i<2; i++) {
    d = n3_coord[i] - n1_coord[i];
    ds_sum += d*d;
  }
  cLen = sqrt(ds_sum);
  sLen = (aLen+bLen+cLen)/2;
  if(sLen-aLen < 0) return 0.0;
  else if(sLen-bLen < 0) return 0.0;
  else if(sLen-cLen < 0) return 0.0; //area too small to note
  return (sqrt(sLen*(sLen-aLen)*(sLen-bLen)*(sLen-cLen)));
}

double Bulk_Adapt::getSignedArea(int n1, int n2, int n3) {
  double coordsn1[2], coordsn2[2], coordsn3[2];
  getCoord(n1, coordsn1);
  getCoord(n2, coordsn2);
  getCoord(n3, coordsn3);

  double ret = getSignedArea(coordsn1, coordsn2, coordsn3);
  return ret;
}

double Bulk_Adapt::getSignedArea(double *n1_coord, double *n2_coord, double *n3_coord) {
  double area=0.0;
  double vec1_x, vec1_y, vec2_x, vec2_y;

  vec1_x = n1_coord[0] - n2_coord[0];
  vec1_y = n1_coord[1] - n2_coord[1];
  vec2_x = n3_coord[0] - n2_coord[0];
  vec2_y = n3_coord[1] - n2_coord[1];

  area = vec1_x*vec2_y - vec2_x*vec1_y;
  return area;
}

int Bulk_Adapt::getCoord(int n1, double *crds) {
  FEM_DataAttribute *coord = theMesh->node.getCoord(); // entire local coords
  double *nodeCoords = (coord->getDouble()).getRow(n1); // ptrs to ACTUAL coords!
  crds[0] = nodeCoords[0];
  crds[1] = nodeCoords[1];
  if (dim == 3) 
    crds[2] = nodeCoords[2];
  return 1;
}

int Bulk_Adapt::getShortestEdge(int n1, int n2, int n3, int* shortestEdge) {
  //note that getCoord might be a remote call, which means 
  //it might not have the same value in the mem
  //location if the memory is reused by someone else meanwhile!
  double coordsn1[2], coordsn2[2], coordsn3[2];
  getCoord(n1, coordsn1);
  getCoord(n2, coordsn2);
  getCoord(n3, coordsn3);

  double aLen = length(coordsn1, coordsn2);
  int shortest = 0;

  double bLen = length(coordsn2, coordsn3);
  if(bLen < aLen) shortest = 1;

  double cLen = length(coordsn3, coordsn1);
  if((cLen < aLen) && (cLen < bLen)) shortest = 2;

  if(shortest==0) {
    shortestEdge[0] = n1;
    shortestEdge[1] = n2;
  }
  else if (shortest==1) {
    shortestEdge[0] = n2;
    shortestEdge[1] = n3;
  }
  else {
    shortestEdge[0] = n3;
    shortestEdge[1] = n1;
  }
  return 1;
}



/** The quality metric is proportional to the area of the triangle upon
    the sum of squares of the lengths of each of the 3 sides
*/
double Bulk_Adapt::getAreaQuality(int elem)
{
  double f, q, len[3];
  int n[3];
  double currentArea;
  double coordsn1[2], coordsn2[2], coordsn3[2];

  theMesh->e2n_getAll(elem, n);
  getCoord(n[0], coordsn1);
  getCoord(n[1], coordsn2);
  getCoord(n[2], coordsn3);

  currentArea = getArea(coordsn1, coordsn2, coordsn3);

  len[0] = length(coordsn1, coordsn2);
  len[1] = length(coordsn2, coordsn3);
  len[2] = length(coordsn3, coordsn1);
  f = 4.0*sqrt(3.0); //proportionality constant
  q = (f*currentArea)/(len[0]*len[0]+len[1]*len[1]+len[2]*len[2]);
  return q;
}

/** The quality metric used is the maxEdgeLength / shortestAltitude of the triangle
    If this metric is less than 100 and the area is greater than sliverArea, 
    the quality is supposed to be good
*/
void Bulk_Adapt::ensureQuality(int n1, int n2, int n3) {
  double coordsn1[2], coordsn2[2], coordsn3[2];
  getCoord(n1, coordsn1);
  getCoord(n2, coordsn2);
  getCoord(n3, coordsn3);
  double area = getSignedArea(coordsn1, coordsn2, coordsn3);
  double len1 = length(coordsn1, coordsn2);
  double len2 = length(coordsn2, coordsn3);
  double len3 = length(coordsn3, coordsn1);
  double max = len1;
  if(len2>max) max = len2;
  if(len3>max) max = len3;
  //shortest edge
  double min = len1;
  if(len2<min) min = len2;
  if(len3<min) min = len3;
  double shortest_al = fabs(area/max);
  double largestR = max/shortest_al;
  CkAssert(largestR<=100.0 && -area > SLIVERAREA);
}

/** Verify the quality of the two new elements that will be created by flip
 */
bool Bulk_Adapt::controlQualityF(int n1, int n2, int n3, int n4) {
  //n1 or n2 will be replaced by n4
  double coordsn1[2], coordsn2[2], coordsn3[2], coordsn4[2];
  if(n4==-1) return false;
  getCoord(n1, coordsn1);
  getCoord(n2, coordsn2);
  getCoord(n3, coordsn3);
  getCoord(n4, coordsn4);
  bool flag = false;
  if(!flag) flag = controlQualityC(coordsn3,coordsn1,coordsn2,coordsn4);
  if(!flag) flag = controlQualityC(coordsn4,coordsn2,coordsn1,coordsn3);
  return flag;
}

bool Bulk_Adapt::controlQualityR(double *coordsn1, double *coordsn2, double *coordsn3) {
  double area = getArea(coordsn1, coordsn2, coordsn3);
  //do some quality preservation
  double len1 = length(coordsn1, coordsn2);
  double len2 = length(coordsn2, coordsn3);
  double len3 = length(coordsn3, coordsn1);
  //longest edge
  double max = len1;
  if(len2>max) max = len2;
  if(len3>max) max = len3;
  double shortest_al = area/max;
  double largestR = max/shortest_al;
  if(largestR>50.0) return true;
  else return false;
}

/** Computes the longestLength, shotestLength, shortestAltitude, and other
    quality metrics for the new element (n1,n2,n4)
    If the new element forms a sliver or bad quality, return true
*/
bool Bulk_Adapt::controlQualityC(int n1, int n2, int n3, double *n4_coord) {
  //n3 is the node to be deleted, n4 is the new node to be added
  double coordsn1[2], coordsn2[2], coordsn3[2];
  getCoord(n1, coordsn1);
  getCoord(n2, coordsn2);
  getCoord(n3, coordsn3);
  return controlQualityC(coordsn1,coordsn2,coordsn3,n4_coord);
}

bool Bulk_Adapt::controlQualityC(double *coordsn1, double *coordsn2, double *coordsn3, double *n4_coord) {
  double ret_old = getSignedArea(coordsn1, coordsn2, coordsn3);
  double ret_new = getSignedArea(coordsn1, coordsn2, n4_coord);
  //do some quality preservation
  double len1 = length(coordsn1, coordsn2);
  double len2 = length(coordsn2, n4_coord);
  double len3 = length(n4_coord, coordsn1);
  //longest edge
  double max = len1;
  if(len2>max) max = len2;
  if(len3>max) max = len3;
  //shortest edge
  double min = len1;
  if(len2<min) min = len2;
  if(len3<min) min = len3;
  double shortest_al = ret_new/max;
  double largestR = max/shortest_al;
  if(ret_old > SLIVERAREA && ret_new < -SLIVERAREA) return true; //it is a flip
  else if(ret_old < -SLIVERAREA && ret_new > SLIVERAREA) return true; //it is a flip
  else if(fabs(ret_new) < SLIVERAREA) return true; // it is a sliver
  //else if(fabs(shortest_al) < 1e-5) return true; //shortest altitude is too small
  //else if(fabs(min) < 1e-5) return true; //shortest edge is too small
  else if(fabs(largestR)>50.0) return true;
  else return false;
}



/** Used from FEM_Repair, when poor quality elements are encountered, depending
    on their maxEdgeLength, it could be better to flip it or bisect it to get
    better quality elements.
    Returns true if it should flip.
*/
bool Bulk_Adapt::flipOrBisect(int elId, int n1, int n2, int maxEdgeIdx, double maxlen) {
  //return true if it should flip
  int nbrEl = theMesh->e2e_getNbr(elId,maxEdgeIdx);
  double len4=0.0, len5=0.0;
  if(nbrEl!=-1) {
    int con1[3];
    theMesh->e2n_getAll(nbrEl,con1);
    int nbrnode=-1;
    for(int j=0; j<3; j++) {
      if(con1[j]!=n1 && con1[j]!=n2) {
	nbrnode = con1[j];
	break;
      }
    }
    len4 = length(n1,nbrnode);
    len5 = length(n2,nbrnode);
  }
  if(len4>1.2*maxlen || len5>1.2*maxlen) {
    return true;
  }
  else return false;
}

