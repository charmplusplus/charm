#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "tri.h"
#include "pgm.h"
#include "Pgm.def.h"

main::main(CkArgMsg *m)
{
  mesh = CProxy_chunk::ckNew();

  CkGetChareID(&mainhandle);
  CProxy_main M(mainhandle);

  if (m->argc != 2)
    CmiAbort("Usage: tri <meshfile>\n");
  CkPrintf("Opening file %s...\n", m->argv[1]);
  
  readMesh(m->argv[1]);
  M.performRefinements();
}

void main::readMesh(char *filename)
{
  FILE *fp;
  int numC, cid, numNodes, numEdges, numElements, i, j, k;
  intMsg *im;
  nodeMsg *nm;
  edgeMsg *edm;
  elementMsg *elm;

  CkPrintf("Reading mesh from file %s... \n", filename);
  fp = fopen(filename, "r");
  fscanf(fp, "%d", &numC);
  CkPrintf("%d chunk(s)...", numC);
  
  CkPrintf("\n ...Reading chunks...\n");
  for (i=0; i<numC; i++) {
    fscanf(fp, "%d", &cid);
    fscanf(fp, "%d%d%d", &numNodes, &numEdges, &numElements);
    CkPrintf("Chunk %d has %d nodes, %d edges, %d elements.\n", cid, 
	     numNodes, numEdges, numElements);

    im = new intMsg;
    im->anInt = numC;
    mesh[cid].insert(im);
    for (j=0; j<numNodes; j++) {
      nm = new nodeMsg;
      fscanf(fp, "%lf%lf", &nm->x, &nm->y);
      mesh[cid].addNode(nm);
    }
    for (j=0; j<numEdges; j++) {
      edm = new edgeMsg;
      for (k=0; k<2; k++)
	fscanf(fp, "%d%d", &edm->nodes[k].idx, &edm->nodes[k].cid);
      for (k=0; k<2; k++)
	fscanf(fp, "%d%d", &edm->elements[k].idx, &edm->elements[k].cid);
      mesh[cid].addEdge(edm);
    }
    for (j=0; j<numElements; j++) {
      elm = new elementMsg;
      for (k=0; k<3; k++)
	fscanf(fp, "%d%d", &elm->nodes[k].idx, &elm->nodes[k].cid);
      for (k=0; k<3; k++)
	fscanf(fp, "%d%d", &elm->edges[k].idx, &elm->edges[k].cid);
      mesh[cid].addElement(elm);
    }
  }
  mesh.doneInserting();
  
  fclose(fp);
  CkPrintf("\nDone.\n");
}

void main::performRefinements()
{
  refineMsg *rm = new refineMsg;
  coarsenMsg *cm = new coarsenMsg;
  doubleMsg *dm;
  int i;

  CkPrintf("Awaiting mesh construction completion...\n");

  CkWaitQD();
  mesh.deriveBorderNodes();
  CkWaitQD();

  CkPrintf("Printing start mesh... \n");
  for (i=0; i<5; i++) 
    mesh[i].print();
  CkWaitQD();
  CkPrintf("Beginning meshing around... \n");

  /*
  rm->idx = 0;
  rm->area = 0.1;
  mesh[0].refineElement(rm);

  CkWaitQD();
  CkPrintf("Printing mesh after 1st refine... \n");
  for (i=0; i<5; i++) 
    mesh[i].print();
  CkWaitQD();
  */

  mesh[0].improve();
  CkWaitQD();
  CkPrintf("Printing mesh after improvement... \n");
  for (i=0; i<5; i++) 
    mesh[i].print();
  CkWaitQD();

  /*
  rm = new refineMsg;
  rm->idx = 145;
  rm->area = 0.01;
  mesh[0].refineElement(rm);

  CkWaitQD();
  CkPrintf("Printing mesh after 2nd refine... \n");
  for (i=0; i<1; i++) 
    mesh[i].print();
  CkWaitQD();

  rm = new refineMsg;
  rm->idx = 6;
  rm->area = 0.1;
  mesh[0].refineElement(rm);

  CkWaitQD();
  CkPrintf("Printing mesh after 3rd refine... \n");
  for (i=0; i<5; i++) 
    mesh[i].print();
  CkWaitQD();

  dm = new doubleMsg;
  dm->idx = 36;
  dm->aDouble = 0.001;
  mesh[0].setTargetArea(dm);
  dm = new doubleMsg;
  dm->idx = 6;
  dm->aDouble = 0.001;
  mesh[0].setTargetArea(dm);

  rm = new refineMsg;
  rm->idx = 53;
  rm->area = 0.001;
  mesh[0].refineElement(rm);

  CkWaitQD();
  CkPrintf("Printing mesh after 4th refine... \n");
  for (i=0; i<5; i++) 
    mesh[i].print();
  CkWaitQD();

  dm = new doubleMsg;
  dm->idx = 557;
  dm->aDouble = 0.001;
  mesh[0].setTargetArea(dm);
  dm = new doubleMsg;
  dm->idx = 575;
  dm->aDouble = 0.001;
  mesh[0].setTargetArea(dm);
  dm = new doubleMsg;
  dm->idx = 417;
  dm->aDouble = 0.001;
  mesh[0].setTargetArea(dm);
  dm = new doubleMsg;
  dm->idx = 58;
  dm->aDouble = 0.001;
  mesh[0].setTargetArea(dm);
  dm = new doubleMsg;
  dm->idx = 377;
  dm->aDouble = 0.001;
  mesh[0].setTargetArea(dm);

  rm = new refineMsg;
  rm->idx = 378;
  rm->area = 0.001;
  mesh[0].refineElement(rm);

  CkWaitQD();
  CkPrintf("Printing mesh after 5th refine... \n");
  for (i=0; i<5; i++) 
    mesh[i].print();
  CkWaitQD();
  */

  // Freshen the mesh first!
  mesh.freshen();
  CkWaitQD();

  for (int p=0; p<200; p++) {
    dm = new doubleMsg;
    dm->idx = p;
    dm->aDouble = 0.07;
    mesh[0].resetTargetArea(dm);
  }

  cm->idx = 208;
  cm->area = 0.07;
  mesh[0].coarsenElement(cm);

  CkWaitQD();
  CkPrintf("Printing mesh after 1st coarsen... \n");
  for (i=0; i<5; i++) 
    mesh[i].print();
  CkWaitQD();

  /*
  cm = new coarsenMsg;
  cm->idx = 31;
  cm->area = 1.99;
  mesh[3].coarsenElement(cm);

  CkWaitQD();
  CkPrintf("Printing mesh after 2nd coarsen... \n");
  for (i=0; i<5; i++) 
    mesh[i].print();
  CkWaitQD();

  cm = new coarsenMsg;
  cm->idx = 102;
  cm->area = 5.0;
  mesh[0].coarsenElement(cm);

  CkWaitQD();
  CkPrintf("Printing mesh after 6th coarsen... \n");
  for (i=0; i<5; i++) 
    mesh[i].print();
  CkWaitQD();

  cm = new coarsenMsg;
  cm->idx = 104;
  cm->area = 5.0;
  mesh[0].coarsenElement(cm);

  CkWaitQD();
  CkPrintf("Printing mesh after 6th coarsen... \n");
  for (i=0; i<5; i++) 
    mesh[i].print();
  CkWaitQD();
  */
  CkPrintf("Done meshing around... exiting.\n");
  CkExit();
}
