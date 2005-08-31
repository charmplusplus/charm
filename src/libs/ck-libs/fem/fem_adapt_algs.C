// This module implements high level mesh adaptivity algorithms that make use 
// of the primitive mesh adaptivity operations provided by fem_adapt(_new).
// Ask: TLW
#include "fem_adapt_algs.h"
#include "fem_mesh_modify.h"

#define MINAREA 1.0e-15
#define MAXAREA 1.0e15

FEM_Adapt_Algs::FEM_Adapt_Algs(FEM_Mesh *m, femMeshModify *fm, int dimension) 
{ 
  theMesh = m; 
  theMod = fm; 
  dim = dimension; 
  //theAdaptor = theMod->fmAdapt;
  theAdaptor = theMod->fmAdaptL;
}

/* Perform refinements on a mesh.  Tries to maintain/improve element quality as
   specified by a quality measure qm; if method = 0, refine areas with size 
   larger than factor down to factor; if method = 1, refine elements down to 
   sizes specified in sizes array; Negative entries in size array indicate no 
   refinement. */
void FEM_Adapt_Algs::FEM_Refine(int qm, int method, double factor, 
				double *sizes)
{
  CkPrintf("WARNING: FEM_Refine: Under construction.\n");
  numNodes = theMesh->node.size();
  numElements = theMesh->elem[0].size();
  (void)Refine(qm, method, factor, sizes);
}

/* Performs refinement; returns number of modifications */
int FEM_Adapt_Algs::Refine(int qm, int method, double factor, double *sizes)
{
  // loop through elemsToRefine
  int elId, mods=0, iter_mods=1, orig_elems = -1;
  SetMeshSize(method, factor, sizes);
  while (iter_mods != 0) {
    iter_mods=0;
    numNodes = theMesh->node.size();
    numElements = theMesh->elem[0].size();
    if (orig_elems == -1) orig_elems = numElements;
    // sort elements to be refined by quality into elemsToRefine
    if (refineStack) delete [] refineStack;
    refineStack = new elemHeap[numElements];
    if (refineElements) delete [] refineElements;
    refineElements = new elemHeap[numElements+1];
    for (int i=0; i<orig_elems; i++) { 
      if (theMesh->elem[0].is_valid(i)) {
	// find maxEdgeLength of i
	int *eConn = (int*)malloc(3*sizeof(int));
	double tmpLen, maxEdgeLength;
	theMesh->e2n_getAll(i, eConn);
	maxEdgeLength = length(eConn[0], eConn[1]);
	tmpLen = length(eConn[1], eConn[2]);
	if (tmpLen > maxEdgeLength) maxEdgeLength = tmpLen;
	tmpLen = length(eConn[2], eConn[0]);
	if (tmpLen > maxEdgeLength) maxEdgeLength = tmpLen;
	if (maxEdgeLength > (regional_sizes[i]*REFINE_TOL)) {
	  double qFactor=1.0;//getAreaQuality(i);
	  Insert(i, qFactor, 0);
	}
      }
    }
    while (refineHeapSize>0 || refineTop > 0) { // loop through the elements
      if (refineTop>0) {
	refineTop--;
	elId=refineStack[refineTop].elID;
      }
      else  elId=Delete_Min(0);
      if ((elId != -1) && (theMesh->elem[0].is_valid(elId))) {
	(void)refine_element_leb(elId); // refine the element
	iter_mods++;
      }
      CthYield(); // give other chunks on the same PE a chance
    }
    mods += iter_mods;
    CkPrintf("Refine: %d modifications in last pass.\n", iter_mods);
  }
  CkPrintf("Refine: %d total modifications.\n", mods);
  return mods;
}

/* Perform coarsening on a mesh.  Tries to maintain/improve element quality as 
   specified by a quality measure qm; if method = 0, coarsen areas with size 
   smaller than factor up to factor; if method = 1, coarsen elements up to 
   sizes specified in sizes array; Negative entries in size array indicate no 
   coarsening. */
void FEM_Adapt_Algs::FEM_Coarsen(int qm, int method, double factor, 
				 double *sizes)
{
  CkPrintf("WARNING: FEM_Coarsen: Under construction.\n");
  SetMeshSize(method, factor, sizes);
  (void)Coarsen(qm);
}

/* Performs coarsening; returns number of modifications */
int FEM_Adapt_Algs::Coarsen(int qm)
{
  return 0;
}

/* Smooth the mesh using method according to some quality measure qm */
void FEM_Adapt_Algs::FEM_Smooth(int qm, int method)
{
  CkPrintf("WARNING: FEM_Smooth: Not yet implemented.\n");
}

/* Repair the mesh according to some quality measure qm */
void FEM_Adapt_Algs::FEM_Repair(int qm)
{
  CkPrintf("WARNING: FEM_Repair: Not yet implemented.\n");
}

/* Remesh entire mesh according to quality measure qm. If method = 0, set 
   entire mesh size to factor; if method = 1, use sizes array; if method = 2, 
   uses existing regional sizes and scale by factor*/
void FEM_Adapt_Algs::FEM_Remesh(int qm, int method, double factor, 
				double *sizes)
{
  CkPrintf("WARNING: FEM_Remesh: Under construction.\n");
}

/* Set sizes on elements throughout the mesh; note: size is edge length */
void FEM_Adapt_Algs::SetMeshSize(int method, double factor, double *sizes)
{
  if (method == 0) {
    regional_sizes = (double *)malloc(numElements*sizeof(double));
    for (int i=0; i<numElements; i++) {
      regional_sizes[i] = factor;
    }
  }
  else if (method == 1) {
    regional_sizes = (double *)malloc(numElements*sizeof(double));
    for (int i=0; i<numElements; i++) {
      regional_sizes[i] = sizes[i];
    }
  }
  else if (method == 2) {
    regional_sizes = (double *)malloc(numElements*sizeof(double));
    double avgEdgeLength = 0.0;
    for (int i=0; i<numElements; i++) {
      int eConn[4];
      theMesh->e2n_getAll(i, eConn);
      avgEdgeLength = length(eConn[0], eConn[1]) + length(eConn[1], eConn[2]) +
	length(eConn[2], eConn[0]);
      if (dim == 3) {
	avgEdgeLength += length(eConn[0], eConn[3]) + length(eConn[1], eConn[3]) +
	  length(eConn[2], eConn[3]);
	avgEdgeLength /= 6.0;
      }
      else {
	avgEdgeLength /= 3.0;
      }
      regional_sizes[i] = factor * avgEdgeLength;
    }
  }
}

int FEM_Adapt_Algs::simple_refine(double targetA) {
  int noEle = theMesh->elem[0].size();
  int *con = (int*)malloc(3*sizeof(int));
  double *areas = (double*)malloc(noEle*sizeof(double));
  int *map1 = (int*)malloc(noEle*sizeof(int));
  double *n1_coord = (double*)malloc(2*sizeof(double));
  double *n2_coord = (double*)malloc(2*sizeof(double));
  double *n3_coord = (double*)malloc(2*sizeof(double));

  for(int i=0; i<noEle; i++) {
    if(theMesh->elem[0].is_valid(i)) {
      theMesh->e2n_getAll(i,con,0);
      getCoord(con[0], n1_coord);
      getCoord(con[1], n2_coord);
      getCoord(con[2], n3_coord);
      //do a refinement only if it has any node within x coords 0.087 to 0.063
      /*if(!((n1_coord[0]<0.0087 && n1_coord[0]>0.0063) || (n2_coord[0]<0.0087 && n2_coord[0]>0.0063) || (n3_coord[0]<0.0087 && n3_coord[0]>0.0063))) {
	areas[i] = MINAREA; //make it believe that this triangle does not need refinement
	} else */{
	areas[i] = getArea(n1_coord, n2_coord, n3_coord);
      }
    } else {
      areas[i] = MINAREA;
    }
    map1[i] = i;
  }

  for(int i=0; i<noEle; i++) {
    for(int j=i+1; j<noEle; j++) {
      if(areas[j] > areas[i]) {
	double tmp = areas[j];
	areas[j] = areas[i];
	areas[i] = tmp;
	int t = map1[j];
	map1[j] = map1[i];
	map1[i] = t;
      }
    }
  }

  for(int i=0; i<noEle; i++) {
    if(theMesh->elem[0].is_valid(map1[i])) {
      if(areas[i] > targetA) {
	refine_element_leb(map1[i]);
      }
    }
  }
  free(con);
  free(areas);
  free(map1);
  free(n1_coord);
  free(n2_coord);
  free(n3_coord);
  return 1;
}

int FEM_Adapt_Algs::simple_coarsen(double targetA) {
  int noEle = theMesh->elem[0].size();
  int *con = (int*)malloc(3*sizeof(int));
  double *areas = (double*)malloc(noEle*sizeof(double));
  int *map1 = (int*)malloc(noEle*sizeof(int));
  double *n1_coord = (double*)malloc(2*sizeof(double));
  double *n2_coord = (double*)malloc(2*sizeof(double));
  double *n3_coord = (double*)malloc(2*sizeof(double));
  int *shortestEdge = (int *)malloc(2*sizeof(int));

  for(int i=0; i<noEle; i++) {
    if(theMesh->elem[0].is_valid(i)) {
      theMesh->e2n_getAll(i,con,0);
      getCoord(con[0], n1_coord);
      getCoord(con[1], n2_coord);
      getCoord(con[2], n3_coord);
      //do a coarsening only if it has any node within y coords less than 0.04
      /*if(!((n1_coord[1]<0.04) || (n2_coord[1]<0.04) || (n3_coord[1]<0.04))) {
	areas[i] = MAXAREA; //make it believe that this triangle is big enough
	} else */{
	areas[i] = getArea(n1_coord, n2_coord, n3_coord);
      }
    } else {
      areas[i] = MAXAREA;
    }
    map1[i] = i;
  }

  for(int i=0; i<noEle; i++) {
    for(int j=i+1; j<noEle; j++) {
      if(areas[j] < areas[i]) {
	double tmp = areas[j];
	areas[j] = areas[i];
	areas[i] = tmp;
	int t = map1[j];
	map1[j] = map1[i];
	map1[i] = t;
      }
    }
  }

  for(int i=0; i<noEle; i++) {
    if(theMesh->elem[0].is_valid(map1[i])) {
      if(areas[i] < targetA) {
	//find the nodes along the smallest edge & coarsen the edge
	theMesh->e2n_getAll(map1[i],con,0);
	getShortestEdge(con[0], con[1], con[2], shortestEdge);
	theAdaptor->edge_contraction(shortestEdge[0], shortestEdge[1]);
      }
    }
  }
  free(con);
  free(areas);
  free(map1);
  free(n1_coord);
  free(n2_coord);
  free(n3_coord);
  free(shortestEdge);
  return 1;
}

// =====================  BEGIN refine_element_leb ========================= 
/* Given an element e, if e's longest edge f is also the longest edge
   of e's neighbor across f, g, split f by adding a new node in the 
   center of f, and splitting both e and g into two elements.  If g
   does not have f as it's longest edge, recursively call refine_element_leb 
   on g, and start over. */ 
int FEM_Adapt_Algs::refine_element_leb(int e) {
  int *eConn = (int*)malloc(3*sizeof(int));
  int fixNode, otherNode, opNode, longEdge, nbr; 
  double eLens[3], longEdgeLen = 0.0;

  if(e==-1) {
    free(eConn);
    return -1;
  }

  theMesh->e2n_getAll(e, eConn);
  eLens[0] = length(eConn[0], eConn[1]);
  eLens[1] = length(eConn[1], eConn[2]);
  eLens[2] = length(eConn[2], eConn[0]);
  for (int i=0; i<3; i++) {
    if (eLens[i] > longEdgeLen) {
      longEdgeLen = eLens[i];
      longEdge = i;
      fixNode = eConn[i];
      otherNode = eConn[(i+1)%3];
      opNode = eConn[(i+2)%3];
    }
  }
  free(eConn);
  nbr = theMesh->e2e_getNbr(e, longEdge);
  if (nbr == -1) // e's longEdge is on physical boundary
    return theAdaptor->edge_bisect(fixNode, otherNode);
  int nbrOpNode = theAdaptor->e2n_getNot(nbr, fixNode, otherNode);
  double fixEdgeLen = length(fixNode, nbrOpNode);
  double otherEdgeLen = length(otherNode, nbrOpNode);
  if ((fixEdgeLen > longEdgeLen) || (otherEdgeLen > longEdgeLen)) { 
    // longEdge is not nbr's longest edge
    int newNode = theAdaptor->edge_bisect(fixNode, otherNode);
    if(newNode==-1) return -1;
    int propElem, propNode; // get the element to propagate on
    if (fixEdgeLen > otherEdgeLen) {
      propElem = theAdaptor->findElementWithNodes(newNode, fixNode, nbrOpNode);
      propNode = fixNode;
    }
    else {
      propElem = theAdaptor->findElementWithNodes(newNode,otherNode,nbrOpNode);
      propNode = otherNode;
    }

    //if propElem is ghost, then it's propagating to neighbor, otherwise not
    if(!FEM_Is_ghost_index(propElem)) {
      refine_flip_element_leb(propElem,propNode,newNode,nbrOpNode,longEdgeLen);
    }
    else {
      int localChk, nbrChk;
      localChk = theMod->getfmUtil()->getIdx();
      nbrChk = theMod->getfmUtil()->getRemoteIdx(theMesh,propElem,0);
      int propNodeT = theAdaptor->getSharedNodeIdxl(propNode, nbrChk);
      int newNodeT = theAdaptor->getSharedNodeIdxl(newNode, nbrChk);
      int nbrghost = (nbrOpNode>=0)?0:1;
      int nbrOpNodeT = (nbrOpNode>=0)?(theAdaptor->getSharedNodeIdxl(nbrOpNode, nbrChk)):(theAdaptor->getGhostNodeIdxl(nbrOpNode, nbrChk));
      int propElemT = theAdaptor->getGhostElementIdxl(propElem, nbrChk);
      meshMod[nbrChk].refine_flip_element_leb(localChk, propElemT, propNodeT, newNodeT,nbrOpNodeT,nbrghost,longEdgeLen);
    }
    return newNode;
  }
  else return theAdaptor->edge_bisect(fixNode, otherNode); // longEdge on nbr
}
void FEM_Adapt_Algs::refine_flip_element_leb(int e, int p, int n1, int n2, 
					     double le) 
{
  int newNode = refine_element_leb(e);
  if(newNode == -1) return;
  (void) theAdaptor->edge_flip(n1, n2);
  if (length(p, newNode) > le) {
    int localChk = theMod->getfmUtil()->getIdx();
    int newElem = theAdaptor->findElementWithNodes(newNode, n1, p);
    refine_flip_element_leb(newElem, p, n1, newNode, le);
  }
}
// ========================  END refine_element_leb ========================

//HELPER functions

double FEM_Adapt_Algs::length(int n1, int n2)
{
  double *n1_coord = (double*)malloc(dim*sizeof(double));
  double *n2_coord = (double*)malloc(dim*sizeof(double));

  getCoord(n1, n1_coord);
  getCoord(n2, n2_coord);

  double ret = length(n1_coord, n2_coord);

  free(n1_coord);
  free(n2_coord);
  return ret;
}

double FEM_Adapt_Algs::length(double *n1_coord, double *n2_coord) { 
  double d, ds_sum=0.0;

  for (int i=0; i<dim; i++) {
    d = n1_coord[i] - n2_coord[i];
    ds_sum += d*d;
  }
  return (sqrt(ds_sum));
}


double FEM_Adapt_Algs::getArea(int n1, int n2, int n3)
{
  double *n1_coord = (double*)malloc(dim*sizeof(double));
  double *n2_coord = (double*)malloc(dim*sizeof(double));
  double *n3_coord = (double*)malloc(dim*sizeof(double));

  getCoord(n1, n1_coord);
  getCoord(n2, n2_coord);
  getCoord(n3, n3_coord);

  double ret = getArea(n1_coord, n2_coord, n3_coord);

  free(n1_coord);
  free(n2_coord);
  free(n3_coord);
  return ret;
}

double FEM_Adapt_Algs::getArea(double *n1_coord, double *n2_coord, double *n3_coord) {
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

bool FEM_Adapt_Algs::didItFlip(int n1, int n2, int n3, double *n4_coord)
{
  //n3 is the node to be deleted, n4 is the new node to be added
  double *n1_coord = (double*)malloc(dim*sizeof(double));
  double *n2_coord = (double*)malloc(dim*sizeof(double));
  double *n3_coord = (double*)malloc(dim*sizeof(double));

  getCoord(n1, n1_coord);
  getCoord(n2, n2_coord);
  getCoord(n3, n3_coord);

  double ret_old = getSignedArea(n1_coord, n2_coord, n3_coord);
  double ret_new = getSignedArea(n1_coord, n2_coord, n4_coord);

  free(n1_coord);
  free(n2_coord);
  free(n3_coord);

  if(ret_old > MINAREA && ret_new < -MINAREA) return true;
  else if(ret_old < -MINAREA && ret_new > MINAREA) return true;
  else return false;
}


bool FEM_Adapt_Algs::didItFlip(double *n1_coord, double *n2_coord, double *n3_coord, double *n4_coord)
{
  double ret_old = getSignedArea(n1_coord, n2_coord, n3_coord);
  double ret_new = getSignedArea(n1_coord, n2_coord, n4_coord);
  if(ret_old > MINAREA && ret_new < -MINAREA) return true;
  else if(ret_old < -MINAREA && ret_new > MINAREA) return true;
  else return false;
}

double FEM_Adapt_Algs::getSignedArea(double *n1_coord, double *n2_coord, double *n3_coord) {
  double area=0.0;
  double vec1_x, vec1_y, vec2_x, vec2_y;

  vec1_x = n1_coord[0] - n2_coord[0];
  vec1_y = n1_coord[1] - n2_coord[1];
  vec2_x = n3_coord[0] - n2_coord[0];
  vec2_y = n3_coord[1] - n2_coord[1];

  area = vec1_x*vec2_y - vec2_x*vec1_y;
  return area;
}

int FEM_Adapt_Algs::getCoord(int n1, double *crds) {
  if(!FEM_Is_ghost_index(n1)) {
    FEM_Mesh_dataP(theMesh, FEM_NODE, coord_attr, (void *)crds, n1, 1, FEM_DOUBLE, dim);
  }
  else {
    int ghostidx = FEM_To_ghost_index(n1);
    FEM_Mesh_dataP(theMesh, FEM_NODE + FEM_GHOST, coord_attr, (void *)crds, ghostidx, 1, FEM_DOUBLE, dim);
  }
  return 1;
}

int FEM_Adapt_Algs::getShortestEdge(int n1, int n2, int n3, int* shortestEdge) {
  double *n1_coord = (double*)malloc(dim*sizeof(double));
  double *n2_coord = (double*)malloc(dim*sizeof(double));
  double *n3_coord = (double*)malloc(dim*sizeof(double));

  getCoord(n1, n1_coord);
  getCoord(n2, n2_coord);
  getCoord(n3, n3_coord);

  double aLen = length(n1_coord, n2_coord);
  int shortest = 0;

  double bLen = length(n2_coord, n3_coord);
  if(bLen < aLen) shortest = 1;

  double cLen = length(n3_coord, n1_coord);
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
  free(n1_coord);
  free(n2_coord);
  free(n3_coord);
  return 1;
}


void FEM_Adapt_Algs::Insert(int eIdx, double len, int cFlag)
{
  int i;
  if (cFlag) {
    i = ++coarsenHeapSize; 
    while ((coarsenElements[i/2].len>=len) && (i != 1)) {
      coarsenElements[i].len=coarsenElements[i/2].len;
      coarsenElements[i].elID=coarsenElements[i/2].elID;
      i/=2;                     
    }
    coarsenElements[i].elID=eIdx;
    coarsenElements[i].len=len; 
  }
  else {
    i = ++refineHeapSize; 
    while ((refineElements[i/2].len>=len) && (i != 1)) {
      refineElements[i].len=refineElements[i/2].len;
      refineElements[i].elID=refineElements[i/2].elID;
      i/=2;                     
    }
    refineElements[i].elID=eIdx;
    refineElements[i].len=len; 
  }
}

// removes and returns the minimum element of the heap 
int FEM_Adapt_Algs::Delete_Min(int cflag)
{
  int Child, i, Min_ID; 
  if (cflag) {
    Min_ID=coarsenElements[1].elID;
    for (i=1; i*2 <= coarsenHeapSize-1; i=Child) { // Find smaller child
      Child = i*2; // child is left child  
      if (Child != coarsenHeapSize)  // right child exists
	if (coarsenElements[Child+1].len < coarsenElements[Child].len)
	  Child++; 
      // Percolate one level
      if (coarsenElements[coarsenHeapSize].len >= coarsenElements[Child].len) {
	coarsenElements[i].elID = coarsenElements[Child].elID;
	coarsenElements[i].len = coarsenElements[Child].len;
      }
      else break; 
    }
    coarsenElements[i].elID = coarsenElements[coarsenHeapSize].elID;
    coarsenElements[i].len = coarsenElements[coarsenHeapSize].len; 
    coarsenHeapSize--;
    return Min_ID; 
  }
  else {
    Min_ID=refineElements[1].elID;
    for (i=1; i*2 <= refineHeapSize-1; i=Child) { // Find smaller child
      Child = i*2;       // child is left child  
      if (Child !=refineHeapSize)  // right child exists
	if (refineElements[Child+1].len < refineElements[Child].len)
	  Child++; 
      // Percolate one level
      if (refineElements[refineHeapSize].len >= refineElements[Child].len){  
	refineElements[i].elID = refineElements[Child].elID;   
	refineElements[i].len = refineElements[Child].len;
      }
      else break; 
    }
    refineElements[i].elID = refineElements[refineHeapSize].elID;
    refineElements[i].len = refineElements[refineHeapSize].len; 
    refineHeapSize--;
    return Min_ID; 
  }
}

double FEM_Adapt_Algs::getAreaQuality(int elem)
{
  double f, q, len[3];
  int n[3];
  double currentArea;
  double *n1_coord = (double*)malloc(dim*sizeof(double));
  double *n2_coord = (double*)malloc(dim*sizeof(double));
  double *n3_coord = (double*)malloc(dim*sizeof(double));
  theMesh->e2n_getAll(elem, n);
  getCoord(n[0], n1_coord);
  getCoord(n[1], n2_coord);
  getCoord(n[2], n3_coord);

  currentArea = getArea(n1_coord, n2_coord, n3_coord);

  len[0] = length(n1_coord, n2_coord);
  len[1] = length(n2_coord, n3_coord);
  len[2] = length(n3_coord, n1_coord);
  f = 4.0*sqrt(3.0); //proportionality constant
  q = (f*currentArea)/(len[0]*len[0]+len[1]*len[1]+len[2]*len[2]);
  return q;
}

// FEM_Mesh_mooth
//  Inputs  : meshP - a pointer to the FEM_Mesh object to smooth
//	    : nodes - an array of local node numbers to be smoothed.  Send
//		  NULL pointer to smooth all nodes.
//	    : nNodes - the number of nodes to be smoothed.  This must 
//		      be the total number of nodes on this chunk if the nodes
//		      array is NULL
//	    : attrNo - the attribute number where the coords are registered
//  Shifts nodes around to improve mesh quality.  FEM_BOUNDARY attribute
//  and interpolator function must be registered by user to maintain 
//  boundary information.
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
    if (FEM_is_valid(mesh, FEM_NODE, idx) && boundVals[idx]>-1) //node must be internal
    {
      meshP->n2n_getAll(idx, &adjnodes, &nNod);
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

