/** \file ZoltanLB.C
 *
 * Load balancer using Zoltan hypergraph partitioner. This is a multicast aware
 * load balancer
 * Harshitha, 2012/02/21
 */

/**
 * \addtogroup CkLdb
*/
/*@{*/


#include "ZoltanLB.h"
#include "mpi.h"
#include "ckgraph.h"
#include "zoltan.h"

typedef struct {
  int numMyVertices;
  // Ids of the vertices (chares)
  ZOLTAN_ID_TYPE *vtxGID;
  // Weight/load of each chare
  int *vtxWgt;

  int numMyHEdges;
  // Ids of the edges
  ZOLTAN_ID_TYPE *edgeGID;
  int *edgWgt;

  // For compressed hyperedge storage. nborIndex indexes into nborGID.
  int *nborIndex;
  // nborGID contains ids of the chares that constitute an edge
  ZOLTAN_ID_TYPE *nborGID;

  int numAllNbors;

  ProcArray *parr;
  ObjGraph *ogr;
} HGRAPH_DATA;

static int get_number_of_vertices(void *data, int *ierr);
static void get_vertex_list(void *data, int sizeGID, int sizeLID,
            ZOLTAN_ID_PTR globalID, ZOLTAN_ID_PTR localID,
                  int wgt_dim, float *obj_wgts, int *ierr);
static void get_hypergraph_size(void *data, int *num_lists, int *num_nonzeroes,
                                int *format, int *ierr);
static void get_hypergraph(void *data, int sizeGID, int num_edges, int num_nonzeroes,
                           int format, ZOLTAN_ID_PTR edgeGID, int *vtxPtr,
                           ZOLTAN_ID_PTR vtxGID, int *ierr);
static void get_hypergraph_edge_size(void *data, int *num_edges, int *ierr);

static void get_hypergraph_edge_wgts(void *data, int numGID, int numLID, int num_edges,
                                     int edge_weight_dim, ZOLTAN_ID_PTR edgeGID, ZOLTAN_ID_PTR edgeLID,
                                     float *edge_wgts, int *ierr);

CreateLBFunc_Def(ZoltanLB, "Use Zoltan(tm) to partition object graph")

ZoltanLB::ZoltanLB(const CkLBOptions &opt): CBase_ZoltanLB(opt)
{
  lbname = "ZoltanLB";
  if (CkMyPe() == 0)
    CkPrintf("[%d] ZoltanLB created\n",CkMyPe());
}

void ZoltanLB::work(LDStats* stats)
{
  /** ========================== INITIALIZATION ============================= */
  ProcArray *parr = new ProcArray(stats);
  ObjGraph *ogr = new ObjGraph(stats);

  /** ============================= STRATEGY ================================ */
  if (_lb_args.debug() >= 2) {
    CkPrintf("[%d] In ZoltanLB Strategy...\n", CkMyPe());
  }

  int rc;
  float ver;
  struct Zoltan_Struct *zz;
  int changes, numGidEntries, numLidEntries, numImport, numExport;
  int myRank, numProcs;
  ZOLTAN_ID_PTR importGlobalGids, importLocalGids, exportGlobalGids, exportLocalGids;
  int *importProcs, *importToPart, *exportProcs, *exportToPart;
  int *parts;

  HGRAPH_DATA hg;

  hg.parr = parr;
  hg.ogr = ogr;

  int numPes = parr->procs.size();
  // convert ObjGraph to the adjacency structure
  int numVertices = ogr->vertices.size();	// number of vertices
  int numEdges = 0;				// number of edges
  int numAllNbors = 0;
  hg.numMyVertices = numVertices;
  hg.vtxGID = (ZOLTAN_ID_TYPE *)malloc(sizeof(ZOLTAN_ID_TYPE) * numVertices);
  hg.vtxWgt = (int *)malloc(sizeof(int) * numVertices);

  double maxLoad = 0.0;
  double maxBytes = 0.0;
  int i, j, k, vert;

  /** remove duplicate edges from recvFrom */
  for(i = 0; i < numVertices; i++) {
    hg.vtxGID[i] = (ZOLTAN_ID_TYPE) i;

    for(j = 0; j < ogr->vertices[i].sendToList.size(); j++) {
      vert = ogr->vertices[i].sendToList[j].getNeighborId();
      for(k = 0; k < ogr->vertices[i].recvFromList.size(); k++) {
        if(ogr->vertices[i].recvFromList[k].getNeighborId() == vert) {
          ogr->vertices[i].sendToList[j].setNumBytes(ogr->vertices[i].sendToList[j].getNumBytes() + ogr->vertices[i].recvFromList[k].getNumBytes());
          ogr->vertices[i].recvFromList.erase(ogr->vertices[i].recvFromList.begin() + k);
        }
      }
    }
  }

  /** the object load is normalized to an integer between 0 and 256 */
  for(i = 0; i < numVertices; i++) {
    if(ogr->vertices[i].getVertexLoad() > maxLoad)
      maxLoad = ogr->vertices[i].getVertexLoad();
    numEdges += ogr->vertices[i].sendToList.size();
    numEdges += ogr->vertices[i].mcastToList.size();

    numAllNbors += 2 * ogr->vertices[i].sendToList.size(); // For each point to point edge, add 2 numAllNbors 
    for (k = 0; k < ogr->vertices[i].mcastToList.size(); k++) {
      McastSrc mcast_src = ogr->vertices[i].mcastToList[k];
      numAllNbors += mcast_src.destList.size();
      numAllNbors++;
    }
  }

  for(i = 0; i < numVertices; i++) {
    for(j = 0; j < ogr->vertices[i].sendToList.size(); j++) {
      if (ogr->vertices[i].sendToList[j].getNumBytes() > maxBytes) {
        maxBytes = ogr->vertices[i].sendToList[j].getNumBytes();
      }
    }
  }

  hg.numMyHEdges = numEdges;
  hg.numAllNbors = numAllNbors;
  hg.edgeGID = (ZOLTAN_ID_TYPE *)malloc(sizeof(ZOLTAN_ID_TYPE) * numEdges);
  hg.edgWgt = (int *)malloc(sizeof(int) * numEdges);
  hg.nborIndex = (int *)malloc(sizeof(int) * numEdges);
  hg.nborGID = (ZOLTAN_ID_TYPE *)malloc(sizeof(ZOLTAN_ID_TYPE) * numAllNbors);

  double ratio = 256.0/maxLoad;
  double byteRatio = 1024.0 / maxBytes;
  int edgeNum = 0;
  int index = 0;

  for (i = 0; i < numVertices; i++) {
    hg.vtxWgt[i] = (int)ceil(ogr->vertices[i].getVertexLoad() * ratio);

    for (j = 0; j < ogr->vertices[i].sendToList.size(); j++) {
        hg.edgeGID[edgeNum] = edgeNum;
        hg.edgWgt[edgeNum] = (int) ceil(ogr->vertices[i].sendToList[j].getNumBytes() * byteRatio);
        hg.nborIndex[edgeNum++] = index;
        hg.nborGID[index++] = i; 
        hg.nborGID[index++] = ogr->vertices[i].sendToList[j].getNeighborId(); 
    }

    for (j = 0; j < ogr->vertices[i].mcastToList.size(); j++) {
      hg.edgeGID[edgeNum] = edgeNum;
      hg.edgWgt[edgeNum] = (int) ceil(ogr->vertices[i].mcastToList[j].getNumBytes() * byteRatio);
      hg.nborIndex[edgeNum++] = index;
      McastSrc mcast_src = ogr->vertices[i].mcastToList[j];

      hg.nborGID[index++] = i; 
      for (k = 0; k < mcast_src.destList.size(); k++) {
        // For all the vertices in the multicast edge, add it
        hg.nborGID[index++] = mcast_src.destList[k]; 
      }
    }
  }

  CkAssert(edgeNum == numEdges);

  rc = Zoltan_Initialize(0, NULL, &ver);
  zz = Zoltan_Create(MPI_COMM_WORLD);
  char global_parts[10];
  sprintf(global_parts, "%d", numPes);

  Zoltan_Set_Param(zz, "DEBUG_LEVEL", "0");
  Zoltan_Set_Param(zz, "LB_METHOD", "HYPERGRAPH");   /* partitioning method */
  Zoltan_Set_Param(zz, "HYPERGRAPH_PACKAGE", "PHG"); /* version of method */
  Zoltan_Set_Param(zz, "NUM_GID_ENTRIES", "1");/* global IDs are integers */
  Zoltan_Set_Param(zz, "NUM_LID_ENTRIES", "1");/* local IDs are integers */
  Zoltan_Set_Param(zz, "RETURN_LISTS", "PART"); /* export AND import lists */
  Zoltan_Set_Param(zz, "OBJ_WEIGHT_DIM", "1"); /* use Zoltan default vertex weights */
  Zoltan_Set_Param(zz, "EDGE_WEIGHT_DIM", "1");/* use Zoltan default hyperedge weights */
  Zoltan_Set_Param(zz, "NUM_GLOBAL_PARTS", global_parts);
  Zoltan_Set_Param(zz, "LB_APPROACH", "PARTITION");

  /* Application defined query functions */

  Zoltan_Set_Num_Obj_Fn(zz, get_number_of_vertices, &hg);
  Zoltan_Set_Obj_List_Fn(zz, get_vertex_list, &hg);
  Zoltan_Set_HG_Size_CS_Fn(zz, get_hypergraph_size, &hg);
  Zoltan_Set_HG_CS_Fn(zz, get_hypergraph, &hg);
  Zoltan_Set_HG_Size_Edge_Wts_Fn(zz, get_hypergraph_edge_size, &hg);
  Zoltan_Set_HG_Edge_Wts_Fn(zz, get_hypergraph_edge_wgts, &hg);


  rc = Zoltan_LB_Partition(zz, /* input (all remaining fields are output) */
        &changes,        /* 1 if partitioning was changed, 0 otherwise */ 
        &numGidEntries,  /* Number of integers used for a global ID */
        &numLidEntries,  /* Number of integers used for a local ID */
        &numImport,      /* Number of vertices to be sent to me */
        &importGlobalGids,  /* Global IDs of vertices to be sent to me */
        &importLocalGids,   /* Local IDs of vertices to be sent to me */
        &importProcs,    /* Process rank for source of each incoming vertex */
        &importToPart,   /* New partition for each incoming vertex */
        &numExport,      /* Number of vertices I must send to other processes*/
        &exportGlobalGids,  /* Global IDs of the vertices I must send */
        &exportLocalGids,   /* Local IDs of the vertices I must send */
        &exportProcs,    /* Process to which I send each of the vertices */
        &exportToPart);  /* Partition to which each vertex will belong */

  if (rc != ZOLTAN_OK){
    CkPrintf("Zoltan exiting\n");
    Zoltan_Destroy(&zz);
    exit(0);
  }


  for(i = 0; i < numVertices; i++) {
    if(exportToPart[i] != ogr->vertices[i].getCurrentPe())
      ogr->vertices[i].setNewPe(exportToPart[i]);
  }

  /******************************************************************
  ** Free the arrays allocated by Zoltan_LB_Partition, and free
  ** the storage allocated for the Zoltan structure.
  ******************************************************************/

  Zoltan_LB_Free_Part(&importGlobalGids, &importLocalGids, 
                      &importProcs, &importToPart);
  Zoltan_LB_Free_Part(&exportGlobalGids, &exportLocalGids, 
                      &exportProcs, &exportToPart);

  Zoltan_Destroy(&zz);

  /**********************
  ** all done ***********
  **********************/

  if (hg.numMyVertices > 0){
    free(hg.vtxGID);
  }
  if (hg.numMyHEdges > 0){
    free(hg.edgeGID);
    free(hg.nborIndex);
    if (hg.numAllNbors > 0){
      free(hg.nborGID);
    }
  }

  ogr->convertDecisions(stats);

  delete parr;
  delete ogr;
}

/* Application defined query functions */

static int get_number_of_vertices(void *data, int *ierr)
{
  HGRAPH_DATA *hg = (HGRAPH_DATA *)data;
  *ierr = ZOLTAN_OK;
  return hg->numMyVertices;
}

static void get_vertex_list(void *data, int sizeGID, int sizeLID,
            ZOLTAN_ID_PTR globalID, ZOLTAN_ID_PTR localID,
                  int wgt_dim, float *obj_wgts, int *ierr)
{
int i;

  HGRAPH_DATA *hg= (HGRAPH_DATA *)data;
  *ierr = ZOLTAN_OK;

  /* In this example, return the IDs of our vertices, but no weights.
   * Zoltan will assume equally weighted vertices.
   */

  for (i=0; i<hg->numMyVertices; i++){
    globalID[i] = hg->vtxGID[i];
    localID[i] = i;
    obj_wgts[i] = hg->vtxWgt[i];
  }
}

static void get_hypergraph_size(void *data, int *num_lists, int *num_nonzeroes,
                                int *format, int *ierr)
{
  HGRAPH_DATA *hg = (HGRAPH_DATA *)data;
  *ierr = ZOLTAN_OK;

  *num_lists = hg->numMyHEdges;
  *num_nonzeroes = hg->numAllNbors;

  /* We will provide compressed hyperedge (row) format.  The alternative is
   * is compressed vertex (column) format: ZOLTAN_COMPRESSED_VERTEX.
   */

  *format = ZOLTAN_COMPRESSED_EDGE;

  return;
}

static void get_hypergraph(void *data, int sizeGID, int num_edges, int num_nonzeroes,
                           int format, ZOLTAN_ID_PTR edgeGID, int *vtxPtr,
                           ZOLTAN_ID_PTR vtxGID, int *ierr)
{
int i;

  HGRAPH_DATA *hg = (HGRAPH_DATA *)data;
  *ierr = ZOLTAN_OK;

  if ( (num_edges != hg->numMyHEdges) || (num_nonzeroes != hg->numAllNbors) ||
       (format != ZOLTAN_COMPRESSED_EDGE)) {
    *ierr = ZOLTAN_FATAL;
    return;
  }

  for (i=0; i < num_edges; i++){
    edgeGID[i] = hg->edgeGID[i];
    vtxPtr[i] = hg->nborIndex[i];
  }

  for (i=0; i < num_nonzeroes; i++){
    vtxGID[i] = hg->nborGID[i];
  }

  return;
}

static void get_hypergraph_edge_size(void *data, int *num_edges, int *ierr) {
  HGRAPH_DATA *hg = (HGRAPH_DATA *) data;
  *ierr = ZOLTAN_OK;
  *num_edges = hg->numMyHEdges;
}

static void get_hypergraph_edge_wgts(void *data, int numGID, int numLID, int num_edges,
                                     int edge_weight_dim, ZOLTAN_ID_PTR edgeGID, ZOLTAN_ID_PTR edgeLID,
                                     float *edge_wgts, int *ierr) {
  int i;
  HGRAPH_DATA *hg = (HGRAPH_DATA *) data;
  *ierr = ZOLTAN_OK;
  for (i = 0; i < num_edges; i++) {
    edgeGID[i] = hg->edgeGID[i];
    edgeLID[i] = hg->edgeGID[i];
    edge_wgts[i] = hg->edgWgt[i];
  }
}

#include "ZoltanLB.def.h"

/*@}*/
