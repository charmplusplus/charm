#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "chunk.h"
#include "pgm.h"
#include "Pgm.def.h"

main::main(CkArgMsg *m)
{
  mesh = CProxy_chunk::ckNew();

  CkGetChareID(&mainhandle);
  CProxy_main M(mainhandle);

  if (m->argc != 2)
    CmiAbort("Usage: pmaf <meshfile>\n");
  strcpy(filename, m->argv[1]);
  CkPrintf("Opening file %s...\n", filename);

  M.readMesh();
}

void main::readMesh()
{
  // file format:
  // #chunks 
  // The rest are per chunk:
  // cid
  // #elements #ghostElements #idxOffset
  // <the 4 nodes of element 1>
  // ...
  // <the 4 nodes of element #ghostElements
  // <cid & idx of ghostElement 1>
  // ...
  // <cid & idx of ghostElement #ghostElements>
  // <the 3 double coords of node 1>
  // ...
  // <the 3 double coords of node n (quantity determined from elements)>
  FILE *fp;
  int numElements, numGhosts, numSurfaces, idxOffset, numNodes, numFixed, 
    cid, i, j, k;
  int *conn, *gid, *surface, *fixedNodes;
  double *nodeCoords;
  meshMsg *mm;
  coordMsg *cm;

  CkPrintf("Reading mesh from file %s... \n", filename);
  fp = fopen(filename, "r");
  fscanf(fp, "%d", &numChunks);
  CkPrintf("%d chunk(s)", numChunks);
  // create the chunks
  
  CkPrintf("\n ...Reading chunks...\n");
  for (i=0; i<numChunks; i++) {
    // insert a chunk at index i in the mesh
    mesh[i].insert(numChunks);

    fscanf(fp, "%d", &cid);
    fscanf(fp, "%d%d%d%d", &numElements, &numGhosts, &numSurfaces, &idxOffset);
    CkPrintf("Chunk %d has %d elements, %d ghosts, %d surfaces.\n", cid, 
        numElements, numGhosts, numSurfaces);

    // read elements
    numNodes = 0;
    conn = new int[numGhosts*4];
    gid = new int[numGhosts*2];
    surface = new int[numSurfaces*3];
    CkPrintf("Reading elements...\n");
    for (j=0; j<numGhosts; j++) {
      for (k=0; k<4; k++) {
   fscanf(fp, "%d", &conn[j*4+k]);
   CkPrintf("Node is %d!\n", conn[j*4+k]);
   if (conn[j*4+k]+1 > numNodes)
     numNodes = conn[j*4+k]+1;
      }
    }
    CkPrintf("Detected %d nodes on chunk %d...\n", numNodes, i);
    for (j=0; j<numGhosts; j++)
      for (k=0; k<2; k++)
   fscanf(fp, "%d", &gid[j*2+k]);

    for (j=0; j<numSurfaces; j++)
      for (k=0; k<3; k++)
   fscanf(fp, "%d", &surface[j*3+k]);

    // call newMesh to put data on chunk
    // make this into a message
    CkPrintf("Sending elements to chunk...\n");
    mm = new(numGhosts*4, numGhosts*2, numSurfaces*3, 0) meshMsg;
    mm->numElements = numElements;
    mm->numGhosts = numGhosts;
    mm->numSurFaces = numSurfaces;
    mm->idxOffset = idxOffset;
    for (j=0; j<numGhosts*4; j++)  mm->conn[j] = conn[j];
    for (j=0; j<numGhosts*2; j++)  mm->gid[j] = gid[j];
    for (j=0; j<numSurfaces*3; j++)  mm->surface[j] = surface[j];
    mesh[i].newMesh(mm);

    // read nodes
    CkPrintf("Reading node values...\n");
    nodeCoords = new double[numNodes*3];
    for (j=0; j<numNodes; j++) {
      for (k=0; k<3; k++)
   fscanf(fp, "%lf", &nodeCoords[j*3+k]);
    }
    fscanf(fp, "%d", &numFixed);
    fixedNodes = new int[numFixed];
    for (j=0; j<numFixed; j++)
      fscanf(fp, "%d", &fixedNodes[j]);

    // initialize node data
    // make this into a message
    CkPrintf("Sending node values to chunk...\n");
    cm = new(numNodes*3, numFixed, 0) coordMsg;
    cm->numNodes = numNodes;
    cm->numElements = numElements;
    cm->numFixed = numFixed;
    for (j=0; j<numNodes*3; j++)  {
      cm->coords[j] = nodeCoords[j];
      CkPrintf(" %lf ", nodeCoords[j]); 
    }
    for (j=0; j<numFixed; j++)
      cm->fixedNodes[j] = fixedNodes[j];
    CkPrintf("\n");
    CkPrintf("\n");
    mesh[i].updateNodeCoords(cm);

    // delete the arrays
    delete[] conn;
    delete[] gid;
    delete[] nodeCoords;
    delete[] surface;
    delete[] fixedNodes;
    CkPrintf("Done reading chunk %d.\n", i);
  }
  mesh.doneInserting();
  
  fclose(fp);
  CkPrintf("\nDone reading 3D mesh file.\n");
  CkWaitQD();
  mesh.deriveFaces();
  CkWaitQD();

  CkPrintf("Printing start mesh... \n");
  for (i=0; i<numChunks; i++) 
    mesh[i].print();
  CkWaitQD();

  CkPrintf("Beginning meshing around... \n");
  for (i=0; i<numChunks; i++) {
    mesh[i].refine();
  }
  for (i=0; i<numChunks; i++)
    mesh[i].start();
  CkWaitQD();

  CkPrintf("Done refining.\n");
  CkPrintf("Printing refined mesh... \n");
  for (i=0; i<numChunks; i++) 
    mesh[i].print();
  CkWaitQD();


  //  CkPrintf("Checking refined mesh... \n");
  //  for (i=0; i<numChunks; i++) 
  //    mesh[i].checkRefine();
  //  CkWaitQD();
  

  /*
  for (j=0; j<5; j++) {
    for (i=0; i<numChunks; i++) 
      mesh[i].improve();
    CkWaitQD();
    for (i=0; i<numChunks; i++) 
      mesh[i].finalizeImprovements();
    CkWaitQD();
  }

  CkPrintf("Done improving.\n");
  CkPrintf("Printing improved mesh... \n");
  for (i=0; i<numChunks; i++) 
    mesh[i].print();
  CkWaitQD();
  */

  CkExit();
}
