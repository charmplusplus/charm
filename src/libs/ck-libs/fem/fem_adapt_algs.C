/* File: fem_adapt_algs.C
 * Authors: Terry Wilmarth, Nilesh Choudhury
 * 
 */


// This module implements high level mesh adaptivity algorithms that make use 
// of the primitive mesh adaptivity operations provided by fem_adapt(_new).
// Ask: TLW
#include "fem_adapt_algs.h"
#include "fem_mesh_modify.h"
#include "fem_adapt_if.h"

#define MINAREA 1.0e-18
#define MAXAREA 1.0e12

#define GRADATION 1.3

CtvDeclare(FEM_Adapt_Algs *, _adaptAlgs);

FEM_Adapt_Algs::FEM_Adapt_Algs(FEM_Mesh *m, femMeshModify *fm, int dimension) 
{ 
  theMesh = m; 
  theMod = fm; 
  dim = dimension; 
  //theAdaptor = theMod->fmAdapt;
  theAdaptor = theMod->fmAdaptL;
}

void FEM_Adapt_Algs::FEM_AdaptMesh(int qm, int method, double factor, 
				   double *sizes)
{
  SetMeshSize(method, factor, sizes);
  GradateMesh(GRADATION);
  (void)Refine(qm, method, factor, sizes);
  GradateMesh(GRADATION);
  (void)Coarsen(qm, method, factor, sizes);
}

/* Perform refinements on a mesh.  Tries to maintain/improve element quality as
   specified by a quality measure qm; if method = 0, refine areas with size 
   larger than factor down to factor; if method = 1, refine elements down to 
   sizes specified in sizes array; Negative entries in size array indicate no 
   refinement. */
void FEM_Adapt_Algs::FEM_Refine(int qm, int method, double factor, 
				double *sizes)
{
  SetMeshSize(method, factor, sizes);
  GradateMesh(GRADATION);
  (void)Refine(qm, method, factor, sizes);
}

/* Performs refinement; returns number of modifications */
int FEM_Adapt_Algs::Refine(int qm, int method, double factor, double *sizes)
{
  // loop through elemsToRefine
  int elId, mods=0, iter_mods=1;
  int elemWidth = theMesh->elem[0].getConn().width();
  refineElements = refineStack = NULL;
  refineTop = refineHeapSize = 0;
  while (iter_mods != 0) {
    iter_mods=0;
    numNodes = theMesh->node.size();
    numElements = theMesh->elem[0].size();
    // sort elements to be refined by quality into elemsToRefine
    if (refineStack) delete [] refineStack;
    refineStack = new elemHeap[numElements];
    if (refineElements) delete [] refineElements;
    refineElements = new elemHeap[numElements+1];
    for (int i=0; i<numElements; i++) { 
      if (theMesh->elem[0].is_valid(i)) {
	// find maxEdgeLength of i
	int *eConn = (int*)malloc(elemWidth*sizeof(int));
	double tmpLen, avgEdgeLength=0.0, maxEdgeLength = 0.0;
	theMesh->e2n_getAll(i, eConn);
	for (int j=0; j<elemWidth-1; j++) {
	  for (int k=j+1; k<elemWidth; k++) {
	    tmpLen = length(eConn[j], eConn[k]);
	    avgEdgeLength += tmpLen;
	    if (tmpLen > maxEdgeLength) maxEdgeLength = tmpLen;
	  }
	}
	avgEdgeLength /= 3.0;
	double qFactor=getAreaQuality(i);
	if (theMesh->elem[0].getMeshSizing(i) <= 0.0) 
	  CkPrintf("WARNING: mesh element %d has no sizing!\n", i);
	if ((theMesh->elem[0].getMeshSizing(i) > 0.0) &&
	    (avgEdgeLength > (theMesh->elem[0].getMeshSizing(i)*REFINE_TOL))){
	  Insert(i, qFactor*(1.0/maxEdgeLength), 0);
	}
      }
    }
    while (refineHeapSize>0 || refineTop > 0) { // loop through the elements
      int n1, n2;
      if (refineTop>0) {
	refineTop--;
	elId=refineStack[refineTop].elID;
      }
      else  elId=Delete_Min(0);
      if ((elId != -1) && (theMesh->elem[0].is_valid(elId))) {
	int *eConn = (int*)malloc(elemWidth*sizeof(int));
	int n1, n2;
	double tmpLen, avgEdgeLength=0.0, maxEdgeLength = 0.0;
	theMesh->e2n_getAll(elId, eConn);
	for (int j=0; j<elemWidth-1; j++) {
	  for (int k=j+1; k<elemWidth; k++) {
	    tmpLen = length(eConn[j], eConn[k]);
	    avgEdgeLength += tmpLen;
	    if (tmpLen > maxEdgeLength) { 
	      maxEdgeLength = tmpLen;
	      n1 = eConn[j]; n2 = eConn[k];
	    }
	  }
	}
	avgEdgeLength /= 3.0;
	if ((theMesh->elem[0].getMeshSizing(elId) > 0.0) &&
	    (avgEdgeLength>(theMesh->elem[0].getMeshSizing(elId)*REFINE_TOL))){
	  if (theAdaptor->edge_bisect(n1, n2) > 0)  iter_mods++;
	}
      }
      CthYield(); // give other chunks on the same PE a chance
    }
    mods += iter_mods;
    CkPrintf("ParFUM_Refine: %d modifications in last pass.\n", iter_mods);
  }
  CkPrintf("ParFUM_Refine: %d total modifications.\n", mods);
  delete[] refineStack;
  delete[] refineElements;
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
  CkPrintf("WARNING: ParFUM_Coarsen: Under construction.\n");
  SetMeshSize(method, factor, sizes);
  GradateMesh(GRADATION);
  (void)Coarsen(qm, method, factor, sizes);
}

/* Performs coarsening; returns number of modifications */
int FEM_Adapt_Algs::Coarsen(int qm, int method, double factor, double *sizes)
{
  // loop through elemsToRefine
  int elId, mods=0, iter_mods=1, pass=0;
  int elemWidth = theMesh->elem[0].getConn().width();
  double qFactor;
  coarsenElements = NULL;
  coarsenHeapSize = 0;
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
	int *eConn = (int*)malloc(elemWidth*sizeof(int));
	theMesh->e2n_getAll(i, eConn);
	double tmpLen, avgEdgeLength=0.0, 
	  minEdgeLength = length(eConn[0], eConn[1]);
	for (int j=0; j<elemWidth-1; j++) {
	  for (int k=j+1; k<elemWidth; k++) {
	    tmpLen = length(eConn[j], eConn[k]);
	    avgEdgeLength += tmpLen;
	    if (tmpLen < minEdgeLength) minEdgeLength = tmpLen;
	  }
	}
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
	int *eConn = (int*)malloc(elemWidth*sizeof(int));
	theMesh->e2n_getAll(elId, eConn);
	int n1=eConn[0], n2=eConn[1];
	double tmpLen, avgEdgeLength=0.0, 
	  minEdgeLength = length(n1, n2);
	for (int j=0; j<elemWidth-1; j++) {
	  for (int k=j+1; k<elemWidth; k++) {
	    tmpLen = length(eConn[j], eConn[k]);
	    avgEdgeLength += tmpLen;
	    if (tmpLen < minEdgeLength) {
	      minEdgeLength = tmpLen;
	      n1 = eConn[j]; n2 = eConn[k];
	    }
	  }
	}
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
	    eConn = (int*)malloc(elemWidth*sizeof(int));
	    theMesh->e2n_getAll(eNbr, eConn);
	    avgEdgeLength=0.0;
	    for (int j=0; j<elemWidth-1; j++) {
	      for (int k=j+1; k<elemWidth; k++) {
		avgEdgeLength += length(eConn[j], eConn[k]);
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
    CkPrintf("ParFUM_Coarsen: %d modifications in pass %d.\n", iter_mods, pass);
  }
  CkPrintf("ParFUM_Coarsen: %d total modifications over %d passes.\n", mods, pass);
  delete[] coarsenElements;
  return mods;
}

/* Smooth the mesh using method according to some quality measure qm */
void FEM_Adapt_Algs::FEM_Smooth(int qm, int method)
{
  CkPrintf("WARNING: ParFUM_Smooth: Not yet implemented.\n");
}

/* Repair the mesh according to some quality measure qm */
void FEM_Adapt_Algs::FEM_Repair(int qm)
{
  CkPrintf("WARNING: ParFUM_Repair: Not yet implemented.\n");
}

/* Remesh entire mesh according to quality measure qm. If method = 0, set 
   entire mesh size to factor; if method = 1, use sizes array; if method = 2, 
   uses existing regional sizes and scale by factor*/
void FEM_Adapt_Algs::FEM_Remesh(int qm, int method, double factor, 
				double *sizes)
{
  CkPrintf("WARNING: ParFUM_Remesh: Under construction.\n");
}

/* Set sizes on elements throughout the mesh; note: size is edge length */
void FEM_Adapt_Algs::SetMeshSize(int method, double factor, double *sizes)
{
  numNodes = theMesh->node.size();
  numElements = theMesh->elem[0].size();

  if (method == 0) { // set uniform sizing specified in factor

    for (int i=0; i<numElements; i++) {
      theMesh->elem[0].setMeshSizing(i, factor);
    }
    CkPrintf("ParFUM_SetMeshSize: UNIFORM %4.6e\n", factor);
  }
  else if (method == 1) { // copy sizing from array
    for (int i=0; i<numElements; i++) {
      if (sizes[i] > 0.0) {
	theMesh->elem[0].setMeshSizing(i, sizes[i]);
      }
    }
    CkPrintf("ParFUM_SetMeshSize: SIZES input\n");
  }
  else if (method == 2) { // calculate current sizing and scale by factor
    double avgEdgeLength = 0.0;
    int width = theMesh->elem[0].getConn().width();
    int* eConn = (int*)malloc(width*sizeof(int));
    int numEdges=3;
    if (dim==3) numEdges=6;
    for (int i=0; i<numElements; i++) {
      theMesh->e2n_getAll(i, eConn);
      for (int j=0; j<width-1; j++) {
	for (int k=j+1; k<width; k++) {
	  avgEdgeLength += length(eConn[j], eConn[k]);
	}
      }
      avgEdgeLength += length(eConn[0], eConn[width-1]);
      avgEdgeLength /= (double)numEdges;
      theMesh->elem[0].setMeshSizing(i, factor*avgEdgeLength);
    }
    CkPrintf("ParFUM_SetMeshSize: CALCULATED & SCALED \n");
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
    CkPrintf("ParFUM_SetMeshSize: USE EXISTING SIZES \n");
  }
  //  CkPrintf("Current mesh sizing: ");
  //for (int i=0; i<numElements; i++) {
  //CkPrintf("%4.6e ", theMesh->elem[0].getMeshSizing(i));
  //}
}


void FEM_Adapt_Algs::SetReferenceMesh()
{
  // for each element, set its size to its average edge length
  // TODO: do we need to run this loop for element types other than 0?
  double avgLength = 0.0;
  int width = theMesh->elem[0].getConn().width();
  int* eConn = (int*)malloc(width*sizeof(int));
  int numElements = theMesh->elem[0].size();
  
  for (int i=0; i<numElements; ++i, avgLength=0) {
    theMesh->e2n_getAll(i, eConn);
    for (int j=0; j<width-1; ++j) {
      avgLength += length(eConn[j], eConn[j+1]);
    }
    avgLength += length(eConn[0], eConn[width-1]);
    avgLength /= width;
    theMesh->elem[0].setMeshSizing(i, avgLength);      
  }
  free(eConn);
}


void FEM_Adapt_Algs::GradateMesh(double smoothness)
{
    // Resize mesh elements to avoid jumps in element size
    // Algorithm based on h-shock correction, described in
    // Mesh Gradation Control, Borouchaki et al
    // IJNME43 1998 www.ann.jussieu.fr/~frey/publications/ijnme4398.pdf

    const double beta = smoothness;

    double maxShock, minShock;
    int iteration = 0, updates = 0;

    int* adjNodes, *boundNodes;
    int nadjNodes, nnodes;
    int meshNum = FEM_Mesh_default_read();

    if (smoothness < 1.0) {
        printf("");
    }

    nnodes = theMesh->node.size();
    boundNodes = new int[nnodes];
    FEM_Mesh_data(meshNum, FEM_NODE, FEM_BOUNDARY, 
            boundNodes, 0, nnodes, FEM_INT, 1);


    printf("Running h-shock mesh gradation with beta=%.3f\n", beta);
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
            
            theMesh->n2n_getAll(node, &adjNodes, &nadjNodes);
            for (int adjNode=0; adjNode<nadjNodes; ++adjNode) {
                double edgelen = length(node, adjNodes[adjNode]);
                
                // get adjacent elemnents and their sizes
                // TODO: are we skipping boundary elts here?
                int e1, e2;
                theMesh->get2ElementsOnEdge(node, adjNodes[adjNode], &e1, &e2);
                
                double s1, s2;
                s1 = theMesh->elem[0].getMeshSizing(e1);
                s2 = theMesh->elem[0].getMeshSizing(e2);

                if (s1 <= 0 || s2 <= 0) continue;
                
                // h-shock=max(size ratio)^(1/edge length)
                CkAssert(s1 >= 0 && s2 >= 0 && "Bad size");
                CkAssert(edgelen > 1e-6 && "Length 0 edge");
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
            delete[] adjNodes;
        } 
        
        printf("Finished iteration %d\n", iteration);
        printf("Max shock:%8.3f\n", maxShock);
        printf("Min shock:%8.3f\n", minShock);
        printf("Target:%8.3f\n", beta);
        
    } while (maxShock > beta && ++iteration < GRADATION_ITER_LIMIT);
    tests();

    printf("%d total updates in %d iterations in GradateMesh\n", 
            updates, iteration);
    fflush(NULL);

    delete[] boundNodes;
}


int FEM_Adapt_Algs::simple_refine(double targetA, double xmin, double ymin, double xmax, double ymax) {
  int *con = (int*)malloc(3*sizeof(int));
  double *n1_coord = (double*)malloc(2*sizeof(double));
  double *n2_coord = (double*)malloc(2*sizeof(double));
  double *n3_coord = (double*)malloc(2*sizeof(double));
  bool adapted = true;

  while(adapted) {
    adapted = false;
    int noEle = theMesh->elem[0].size();
    double *areas = (double*)malloc(noEle*sizeof(double));
    int *map1 = (int*)malloc(noEle*sizeof(int));
    for(int i=0; i<noEle; i++) {
      if(theMesh->elem[0].is_valid(i)) {
	theMesh->e2n_getAll(i,con,0);
	getCoord(con[0], n1_coord);
	getCoord(con[1], n2_coord);
	getCoord(con[2], n3_coord);
	//do a refinement only if it has any node within the refinement box
	if((n1_coord[0]<xmax && n1_coord[0]>xmin && n1_coord[1]<ymax && n1_coord[1]>ymin) || (n2_coord[0]<xmax && n2_coord[0]>xmin && n2_coord[1]<ymax && n2_coord[1]>ymin) || (n3_coord[0]<xmax && n3_coord[0]>xmin && n3_coord[1]<ymax && n3_coord[1]>ymin)) {
	  areas[i] = getArea(n1_coord, n2_coord, n3_coord);
	} 
	else {
	  areas[i] = MINAREA; //make it believe that this triangle does not need refinement
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
	  adapted = true;
	}
      }
      //if(adapted) break;
    }
    free(areas);
    free(map1);
    //if(adapted) break;
  }

  free(con);
  free(n1_coord);
  free(n2_coord);
  free(n3_coord);

  if(adapted) return -1;
  else return 1;
}

int FEM_Adapt_Algs::simple_coarsen(double targetA, double xmin, double ymin, double xmax, double ymax) {
  int noEle = theMesh->elem[0].size();
  int *con = (int*)malloc(3*sizeof(int));
  double *areas = (double*)malloc(noEle*sizeof(double));
  int *map1 = (int*)malloc(noEle*sizeof(int));
  double *n1_coord = (double*)malloc(2*sizeof(double));
  double *n2_coord = (double*)malloc(2*sizeof(double));
  double *n3_coord = (double*)malloc(2*sizeof(double));
  int *shortestEdge = (int *)malloc(2*sizeof(int));
  bool adapted = true;

  while(adapted) {
    adapted = false;
    for(int i=0; i<noEle; i++) {
      if(theMesh->elem[0].is_valid(i)) {
	theMesh->e2n_getAll(i,con,0);
	getCoord(con[0], n1_coord);
	getCoord(con[1], n2_coord);
	getCoord(con[2], n3_coord);
	//do a coarsening only if it has any node within the coarsen box
	if((n1_coord[0]<xmax && n1_coord[0]>xmin && n1_coord[1]<ymax && n1_coord[1]>ymin) || (n2_coord[0]<xmax && n2_coord[0]>xmin && n2_coord[1]<ymax && n2_coord[1]>ymin) || (n3_coord[0]<xmax && n3_coord[0]>xmin && n3_coord[1]<ymax && n3_coord[1]>ymin)) {
	  areas[i] = getArea(n1_coord, n2_coord, n3_coord);
	} 
	else {
	  areas[i] = MAXAREA; //make it believe that this triangle is big enough
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
	  int ret = theAdaptor->edge_contraction(shortestEdge[0], shortestEdge[1]);
	  if(ret != -1) adapted = true;
	}
      }
      //if(adapted) break;
    }
    //if(adapted) break;
  }

  free(con);
  free(areas);
  free(map1);
  free(n1_coord);
  free(n2_coord);
  free(n3_coord);
  free(shortestEdge);

  if(adapted) return -1;
  else return 1;
}

void FEM_Adapt_Algs::tests() {
  //test the mesh for slivered triangles

  theMod->fmUtil->StructureTest(theMesh);
  theMod->fmUtil->IdxlListTest(theMesh);
  theMod->fmUtil->residualLockTest(theMesh);
  /*for(int i=0; i<theMesh->node.size(); i++) {
    if(theMesh->node.is_valid(i)) CkPrintf("Valid -- ");
    else  CkPrintf("Invalid -- ");
    theMod->fmUtil->FEM_Print_coords(theMesh,i);
    }*/

  return;
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

  if(ret_old > SLIVERAREA && ret_new < -SLIVERAREA) return true; //it is a flip
  else if(ret_old < -SLIVERAREA && ret_new > SLIVERAREA) return true; //it is a flip
  else if(fabs(ret_new) < SLIVERAREA) return true; // it is a sliver
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
    int numchunks;
    IDXL_Share **chunks1;
    theMod->fmUtil->getChunkNos(0,n1,&numchunks,&chunks1);
    int index = theMod->idx;
    for(int j=0; j<numchunks; j++) {
      int chk = chunks1[j]->chk;
      if(chk==index) continue;
      int ghostidx = theMod->fmUtil->exists_in_IDXL(theMesh,n1,chk,2);
      double2Msg *d = meshMod[chk].getRemoteCoord(index,ghostidx);
      crds[0] = d->i;
      crds[1] = d->j;
      for(int j=0; j<numchunks; j++) {
	delete chunks1[j];
      }
      if(numchunks != 0) free(chunks1);
      break;
    }
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
    if (FEM_is_valid(mesh, FEM_NODE, idx) && boundVals[idx]==0) //node must be internal
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

