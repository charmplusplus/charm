/*
 * Copyright 1997, Regents of the University of Minnesota
 *
 * proto.h
 *
 * This file contains header files
 *
 * Started 10/19/95
 * George
 *
 * $Id$
 *
 */

/* balance.c */
void Balance2Way(CtrlType *, GraphType *, int *, floattype);
void Bnd2WayBalance(CtrlType *, GraphType *, int *);
void General2WayBalance(CtrlType *, GraphType *, int *);

/* bucketsort.c */
void BucketSortKeysInc(int, int, idxtype *, idxtype *, idxtype *);

/* ccgraph.c */
void CreateCoarseGraph(CtrlType *, GraphType *, int, idxtype *, idxtype *);
void CreateCoarseGraphNoMask(CtrlType *, GraphType *, int, idxtype *, idxtype *);
void CreateCoarseGraph_NVW(CtrlType *, GraphType *, int, idxtype *, idxtype *);
GraphType *SetUpCoarseGraph(GraphType *, int, int);
void ReAdjustMemory(GraphType *, GraphType *, int);

/* checkgraph.c */
int CheckGraph(GraphType *);

/* coarsen.c */
GraphType *Coarsen2Way(CtrlType *, GraphType *);

/* compress.c */
void CompressGraph(CtrlType *, GraphType *, int, idxtype *, idxtype *, idxtype *, idxtype *);
void PruneGraph(CtrlType *, GraphType *, int, idxtype *, idxtype *, idxtype *, floattype);

/* debug.c */
int ComputeCut(GraphType *, idxtype *);
int CheckBnd(GraphType *);
int CheckBnd2(GraphType *);
int CheckNodeBnd(GraphType *, int);
int CheckRInfo(RInfoType *);
int CheckNodePartitionParams(GraphType *);
int IsSeparable(GraphType *);

/* estmem.c */
void METIS_EstimateMemory(int *, idxtype *, idxtype *, int *, int *, int *);
void EstimateCFraction(int, idxtype *, idxtype *, floattype *, floattype *);
int ComputeCoarseGraphSize(int, idxtype *, idxtype *, int, idxtype *, idxtype *, idxtype *);

/* fm.c */
void FM_2WayEdgeRefine(CtrlType *, GraphType *, int *, int);

/* fortran.c */
void Change2CNumbering(int, idxtype *, idxtype *);
void Change2FNumbering(int, idxtype *, idxtype *, idxtype *);
void Change2FNumbering2(int, idxtype *, idxtype *);
void Change2FNumberingOrder(int, idxtype *, idxtype *, idxtype *, idxtype *);
void ChangeMesh2CNumbering(int, idxtype *);
void ChangeMesh2FNumbering(int, idxtype *, int, idxtype *, idxtype *);
void ChangeMesh2FNumbering2(int, idxtype *, int, int, idxtype *, idxtype *);

/* frename.c */
void METIS_PARTGRAPHRECURSIVE(int *, idxtype *, idxtype *, idxtype *, idxtype *, int *, int *, int *, int *, int *, idxtype *); 
void metis_partgraphrecursive(int *, idxtype *, idxtype *, idxtype *, idxtype *, int *, int *, int *, int *, int *, idxtype *); 
void metis_partgraphrecursive_(int *, idxtype *, idxtype *, idxtype *, idxtype *, int *, int *, int *, int *, int *, idxtype *); 
void metis_partgraphrecursive__(int *, idxtype *, idxtype *, idxtype *, idxtype *, int *, int *, int *, int *, int *, idxtype *); 
void METIS_WPARTGRAPHRECURSIVE(int *, idxtype *, idxtype *, idxtype *, idxtype *, int *, int *, int *, floattype *, int *, int *, idxtype *); 
void metis_wpartgraphrecursive(int *, idxtype *, idxtype *, idxtype *, idxtype *, int *, int *, int *, floattype *, int *, int *, idxtype *); 
void metis_wpartgraphrecursive_(int *, idxtype *, idxtype *, idxtype *, idxtype *, int *, int *, int *, floattype *, int *, int *, idxtype *); 
void metis_wpartgraphrecursive__(int *, idxtype *, idxtype *, idxtype *, idxtype *, int *, int *, int *, floattype *, int *, int *, idxtype *); 
void METIS_PARTGRAPHKWAY(int *, idxtype *, idxtype *, idxtype *, idxtype *, int *, int *, int *, int *, int *, idxtype *); 
void metis_partgraphkway(int *, idxtype *, idxtype *, idxtype *, idxtype *, int *, int *, int *, int *, int *, idxtype *); 
void metis_partgraphkway_(int *, idxtype *, idxtype *, idxtype *, idxtype *, int *, int *, int *, int *, int *, idxtype *); 
void metis_partgraphkway__(int *, idxtype *, idxtype *, idxtype *, idxtype *, int *, int *, int *, int *, int *, idxtype *); 
void METIS_WPARTGRAPHKWAY(int *, idxtype *, idxtype *, idxtype *, idxtype *, int *, int *, int *, floattype *, int *, int *, idxtype *); 
void metis_wpartgraphkway(int *, idxtype *, idxtype *, idxtype *, idxtype *, int *, int *, int *, floattype *, int *, int *, idxtype *); 
void metis_wpartgraphkway_(int *, idxtype *, idxtype *, idxtype *, idxtype *, int *, int *, int *, floattype *, int *, int *, idxtype *); 
void metis_wpartgraphkway__(int *, idxtype *, idxtype *, idxtype *, idxtype *, int *, int *, int *, floattype *, int *, int *, idxtype *); 
void METIS_EDGEND(int *, idxtype *, idxtype *, int *, int *, idxtype *, idxtype *); 
void metis_edgend(int *, idxtype *, idxtype *, int *, int *, idxtype *, idxtype *); 
void metis_edgend_(int *, idxtype *, idxtype *, int *, int *, idxtype *, idxtype *); 
void metis_edgend__(int *, idxtype *, idxtype *, int *, int *, idxtype *, idxtype *); 
void METIS_NODEND(int *, idxtype *, idxtype *, int *, int *, idxtype *, idxtype *); 
void metis_nodend(int *, idxtype *, idxtype *, int *, int *, idxtype *, idxtype *); 
void metis_nodend_(int *, idxtype *, idxtype *, int *, int *, idxtype *, idxtype *); 
void metis_nodend__(int *, idxtype *, idxtype *, int *, int *, idxtype *, idxtype *); 
void METIS_NODEWND(int *, idxtype *, idxtype *, idxtype *, int *, int *, idxtype *, idxtype *); 
void metis_nodewnd(int *, idxtype *, idxtype *, idxtype *, int *, int *, idxtype *, idxtype *); 
void metis_nodewnd_(int *, idxtype *, idxtype *, idxtype *, int *, int *, idxtype *, idxtype *); 
void metis_nodewnd__(int *, idxtype *, idxtype *, idxtype *, int *, int *, idxtype *, idxtype *); 
void METIS_PARTMESHNODAL(int *, int *, idxtype *, int *, int *, int *, int *, idxtype *, idxtype *);
void metis_partmeshnodal(int *, int *, idxtype *, int *, int *, int *, int *, idxtype *, idxtype *);
void metis_partmeshnodal_(int *, int *, idxtype *, int *, int *, int *, int *, idxtype *, idxtype *);
void metis_partmeshnodal__(int *, int *, idxtype *, int *, int *, int *, int *, idxtype *, idxtype *);
void METIS_PARTMESHDUAL(int *, int *, idxtype *, int *, int *, int *, int *, idxtype *, idxtype *);
void metis_partmeshdual(int *, int *, idxtype *, int *, int *, int *, int *, idxtype *, idxtype *);
void metis_partmeshdual_(int *, int *, idxtype *, int *, int *, int *, int *, idxtype *, idxtype *);
void metis_partmeshdual__(int *, int *, idxtype *, int *, int *, int *, int *, idxtype *, idxtype *);
void METIS_MESHTONODAL(int *, int *, idxtype *, int *, int *, idxtype *, idxtype *);
void metis_meshtonodal(int *, int *, idxtype *, int *, int *, idxtype *, idxtype *);
void metis_meshtonodal_(int *, int *, idxtype *, int *, int *, idxtype *, idxtype *);
void metis_meshtonodal__(int *, int *, idxtype *, int *, int *, idxtype *, idxtype *);
void METIS_MESHTODUAL(int *, int *, idxtype *, int *, int *, idxtype *, idxtype *);
void metis_meshtodual(int *, int *, idxtype *, int *, int *, idxtype *, idxtype *);
void metis_meshtodual_(int *, int *, idxtype *, int *, int *, idxtype *, idxtype *);
void metis_meshtodual__(int *, int *, idxtype *, int *, int *, idxtype *, idxtype *);
void METIS_ESTIMATEMEMORY(int *, idxtype *, idxtype *, int *, int *, int *);
void metis_estimatememory(int *, idxtype *, idxtype *, int *, int *, int *);
void metis_estimatememory_(int *, idxtype *, idxtype *, int *, int *, int *);
void metis_estimatememory__(int *, idxtype *, idxtype *, int *, int *, int *);
void METIS_MCPARTGRAPHRECURSIVE(int *, int *, idxtype *, idxtype *, idxtype *, idxtype *, int *, int *, int *, int *, int *, idxtype *);
void metis_mcpartgraphrecursive(int *, int *, idxtype *, idxtype *, idxtype *, idxtype *, int *, int *, int *, int *, int *, idxtype *);
void metis_mcpartgraphrecursive_(int *, int *, idxtype *, idxtype *, idxtype *, idxtype *, int *, int *, int *, int *, int *, idxtype *);
void metis_mcpartgraphrecursive__(int *, int *, idxtype *, idxtype *, idxtype *, idxtype *, int *, int *, int *, int *, int *, idxtype *);
void METIS_MCPARTGRAPHKWAY(int *, int *, idxtype *, idxtype *, idxtype *, idxtype *, int *, int *, int *, floattype *, int *, int *, idxtype *);
void metis_mcpartgraphkway(int *, int *, idxtype *, idxtype *, idxtype *, idxtype *, int *, int *, int *, floattype *, int *, int *, idxtype *);
void metis_mcpartgraphkway_(int *, int *, idxtype *, idxtype *, idxtype *, idxtype *, int *, int *, int *, floattype *, int *, int *, idxtype *);
void metis_mcpartgraphkway__(int *, int *, idxtype *, idxtype *, idxtype *, idxtype *, int *, int *, int *, floattype *, int *, int *, idxtype *);
void METIS_PARTGRAPHVKWAY(int *, idxtype *, idxtype *, idxtype *, idxtype *, int *, int *, int *, int *, int *, idxtype *);
void metis_partgraphvkway(int *, idxtype *, idxtype *, idxtype *, idxtype *, int *, int *, int *, int *, int *, idxtype *);
void metis_partgraphvkway_(int *, idxtype *, idxtype *, idxtype *, idxtype *, int *, int *, int *, int *, int *, idxtype *);
void metis_partgraphvkway__(int *, idxtype *, idxtype *, idxtype *, idxtype *, int *, int *, int *, int *, int *, idxtype *);
void METIS_WPARTGRAPHVKWAY(int *, idxtype *, idxtype *, idxtype *, idxtype *, int *, int *, int *, floattype *, int *, int *, idxtype *);
void metis_wpartgraphvkway(int *, idxtype *, idxtype *, idxtype *, idxtype *, int *, int *, int *, floattype *, int *, int *, idxtype *);
void metis_wpartgraphvkway_(int *, idxtype *, idxtype *, idxtype *, idxtype *, int *, int *, int *, floattype *, int *, int *, idxtype *);
void metis_wpartgraphvkway__(int *, idxtype *, idxtype *, idxtype *, idxtype *, int *, int *, int *, floattype *, int *, int *, idxtype *);

/* graph.c */
void SetUpGraph(GraphType *, int, int, int, idxtype *, idxtype *, idxtype *, idxtype *, int);
void SetUpGraphKway(GraphType *, int, idxtype *, idxtype *);
void SetUpGraph2(GraphType *, int, int, idxtype *, idxtype *, floattype *, idxtype *);
void VolSetUpGraph(GraphType *, int, int, int, idxtype *, idxtype *, idxtype *, idxtype *, int);
void RandomizeGraph(GraphType *);
int IsConnectedSubdomain(CtrlType *, GraphType *, int, int);
int IsConnected(CtrlType *, GraphType *, int);
int IsConnected2(GraphType *, int);
int FindComponents(CtrlType *, GraphType *, idxtype *, idxtype *);

/* initpart.c */
void Init2WayPartition(CtrlType *, GraphType *, int *, floattype);
void InitSeparator(CtrlType *, GraphType *, floattype);
void GrowBisection(CtrlType *, GraphType *, int *, floattype);
void GrowBisectionNode(CtrlType *, GraphType *, floattype);
void RandomBisection(CtrlType *, GraphType *, int *, floattype);

/* kmetis.c */
void METIS_PartGraphKway(int *, idxtype *, idxtype *, idxtype *, idxtype *, int *, int *, int *, int *, int *, idxtype *); 
void METIS_WPartGraphKway(int *, idxtype *, idxtype *, idxtype *, idxtype *, int *, int *, int *, floattype *, int *, int *, idxtype *); 
int MlevelKWayPartitioning(CtrlType *, GraphType *, int, idxtype *, floattype *, floattype);

/* kvmetis.c */
void METIS_PartGraphVKway(int *, idxtype *, idxtype *, idxtype *, idxtype *, int *, int *, int *, int *, int *, idxtype *);
void METIS_WPartGraphVKway(int *, idxtype *, idxtype *, idxtype *, idxtype *, int *, int *, int *, floattype *, int *, int *, idxtype *);
int MlevelVolKWayPartitioning(CtrlType *, GraphType *, int, idxtype *, floattype *, floattype);

/* kwayfm.c */
void Random_KWayEdgeRefine(CtrlType *, GraphType *, int, floattype *, floattype, int, int);
void Greedy_KWayEdgeRefine(CtrlType *, GraphType *, int, floattype *, floattype, int);
void Greedy_KWayEdgeBalance(CtrlType *, GraphType *, int, floattype *, floattype, int);

/* kwayrefine.c */
void RefineKWay(CtrlType *, GraphType *, GraphType *, int, floattype *, floattype);
void AllocateKWayPartitionMemory(CtrlType *, GraphType *, int);
void ComputeKWayPartitionParams(CtrlType *, GraphType *, int);
void ProjectKWayPartition(CtrlType *, GraphType *, int);
int IsBalanced(idxtype *, int, floattype *, floattype);
void ComputeKWayBoundary(CtrlType *, GraphType *, int);
void ComputeKWayBalanceBoundary(CtrlType *, GraphType *, int);

/* kwayvolfm.c */
void Random_KWayVolRefine(CtrlType *, GraphType *, int, floattype *, floattype, int, int);
void Random_KWayVolRefineMConn(CtrlType *, GraphType *, int, floattype *, floattype, int, int);
void Greedy_KWayVolBalance(CtrlType *, GraphType *, int, floattype *, floattype, int);
void Greedy_KWayVolBalanceMConn(CtrlType *, GraphType *, int, floattype *, floattype, int);
void KWayVolUpdate(CtrlType *, GraphType *, int, int, int, idxtype *, idxtype *, idxtype *);
void ComputeKWayVolume(GraphType *, int, idxtype *, idxtype *, idxtype *);
int ComputeVolume(GraphType *, idxtype *);
void CheckVolKWayPartitionParams(CtrlType *, GraphType *, int);
void ComputeVolSubDomainGraph(GraphType *, int, idxtype *, idxtype *);
void EliminateVolSubDomainEdges(CtrlType *, GraphType *, int, floattype *);
void EliminateVolComponents(CtrlType *, GraphType *, int, floattype *, floattype);

/* kwayvolrefine.c */
void RefineVolKWay(CtrlType *, GraphType *, GraphType *, int, floattype *, floattype);
void AllocateVolKWayPartitionMemory(CtrlType *, GraphType *, int);
void ComputeVolKWayPartitionParams(CtrlType *, GraphType *, int);
void ComputeKWayVolGains(CtrlType *, GraphType *, int);
void ProjectVolKWayPartition(CtrlType *, GraphType *, int);
void ComputeVolKWayBoundary(CtrlType *, GraphType *, int);
void ComputeVolKWayBalanceBoundary(CtrlType *, GraphType *, int);

/* match.c */
void Match_RM(CtrlType *, GraphType *);
void Match_RM_NVW(CtrlType *, GraphType *);
void Match_HEM(CtrlType *, GraphType *);
void Match_SHEM(CtrlType *, GraphType *);

/* mbalance.c */
void MocBalance2Way(CtrlType *, GraphType *, floattype *, floattype);
void MocGeneral2WayBalance(CtrlType *, GraphType *, floattype *, floattype);

/* mbalance2.c */
void MocBalance2Way2(CtrlType *, GraphType *, floattype *, floattype *);
void MocGeneral2WayBalance2(CtrlType *, GraphType *, floattype *, floattype *);
void SelectQueue3(int, floattype *, floattype *, int *, int *, PQueueType [MAXNCON][2], floattype *);

/* mcoarsen.c */
GraphType *MCCoarsen2Way(CtrlType *, GraphType *);

/* memory.c */
void AllocateWorkSpace(CtrlType *, GraphType *, int);
void FreeWorkSpace(CtrlType *, GraphType *);
int WspaceAvail(CtrlType *);
idxtype *idxwspacemalloc(CtrlType *, memsize_t);
void idxwspacefree(CtrlType *, int);
floattype *fwspacemalloc(CtrlType *, memsize_t);
void fwspacefree(CtrlType *, memsize_t);
GraphType *CreateGraph(void);
void InitGraph(GraphType *);
void FreeGraph(GraphType *);

/* mesh.c */
void METIS_MeshToDual(int *, int *, idxtype *, int *, int *, idxtype *, idxtype *);
void METIS_MeshToNodal(int *, int *, idxtype *, int *, int *, idxtype *, idxtype *);
void GENDUALMETIS(int, int, int, idxtype *, idxtype *, idxtype *adjncy);
void TRINODALMETIS(int, int, idxtype *, idxtype *, idxtype *adjncy);
void TETNODALMETIS(int, int, idxtype *, idxtype *, idxtype *adjncy);
void HEXNODALMETIS(int, int, idxtype *, idxtype *, idxtype *adjncy);
void QUADNODALMETIS(int, int, idxtype *, idxtype *, idxtype *adjncy);

/* meshpart.c */
void METIS_PartMeshNodal(int *, int *, idxtype *, int *, int *, int *, int *, idxtype *, idxtype *);
void METIS_PartMeshDual(int *, int *, idxtype *, int *, int *, int *, int *, idxtype *, idxtype *);

/* mfm.c */
void MocFM_2WayEdgeRefine(CtrlType *, GraphType *, floattype *, int);
void SelectQueue(int, floattype *, floattype *, int *, int *, PQueueType [MAXNCON][2]);
int BetterBalance(int, floattype *, floattype *, floattype *);
floattype Compute2WayHLoadImbalance(int, floattype *, floattype *);
void Compute2WayHLoadImbalanceVec(int, floattype *, floattype *, floattype *);

/* mfm2.c */
void MocFM_2WayEdgeRefine2(CtrlType *, GraphType *, floattype *, floattype *, int);
void SelectQueue2(int, floattype *, floattype *, int *, int *, PQueueType [MAXNCON][2], floattype *);
int IsBetter2wayBalance(int, floattype *, floattype *, floattype *);

/* mincover.o */
void MinCover(idxtype *, idxtype *, int, int, idxtype *, int *);
int MinCover_Augment(idxtype *, idxtype *, int, idxtype *, idxtype *, idxtype *, int);
void MinCover_Decompose(idxtype *, idxtype *, int, int, idxtype *, idxtype *, int *);
void MinCover_ColDFS(idxtype *, idxtype *, int, idxtype *, idxtype *, int);
void MinCover_RowDFS(idxtype *, idxtype *, int, idxtype *, idxtype *, int);

/* minitpart.c */
void MocInit2WayPartition(CtrlType *, GraphType *, floattype *, floattype);
void MocGrowBisection(CtrlType *, GraphType *, floattype *, floattype);
void MocRandomBisection(CtrlType *, GraphType *, floattype *, floattype);
void MocInit2WayBalance(CtrlType *, GraphType *, floattype *);
int SelectQueueoneWay(int, floattype *, floattype *, int, PQueueType [MAXNCON][2]);

/* minitpart2.c */
void MocInit2WayPartition2(CtrlType *, GraphType *, floattype *, floattype *);
void MocGrowBisection2(CtrlType *, GraphType *, floattype *, floattype *);
void MocGrowBisectionNew2(CtrlType *, GraphType *, floattype *, floattype *);
void MocInit2WayBalance2(CtrlType *, GraphType *, floattype *, floattype *);
int SelectQueueOneWay2(int, floattype *, PQueueType [MAXNCON][2], floattype *);

/* mkmetis.c */
void METIS_mCPartGraphKway(int *, int *, idxtype *, idxtype *, idxtype *, idxtype *, int *, int *, int *, floattype *, int *, int *, idxtype *);
int MCMlevelKWayPartitioning(CtrlType *, GraphType *, int, idxtype *, floattype *);

/* mkwayfmh.c */
void MCRandom_KWayEdgeRefineHorizontal(CtrlType *, GraphType *, int, floattype *, int);
void MCGreedy_KWayEdgeBalanceHorizontal(CtrlType *, GraphType *, int, floattype *, int);
int AreAllHVwgtsBelow(int, floattype, floattype *, floattype, floattype *, floattype *);
int AreAllHVwgtsAbove(int, floattype, floattype *, floattype, floattype *, floattype *);
void ComputeHKWayLoadImbalance(int, int, floattype *, floattype *);
int MocIsHBalanced(int, int, floattype *, floattype *);
int IsHBalanceBetterFT(int, int, floattype *, floattype *, floattype *, floattype *);
int IsHBalanceBetterTT(int, int, floattype *, floattype *, floattype *, floattype *);

/* mkwayrefine.c */
void MocRefineKWayHorizontal(CtrlType *, GraphType *, GraphType *, int, floattype *);
void MocAllocateKWayPartitionMemory(CtrlType *, GraphType *, int);
void MocComputeKWayPartitionParams(CtrlType *, GraphType *, int);
void MocProjectKWayPartition(CtrlType *, GraphType *, int);
void MocComputeKWayBalanceBoundary(CtrlType *, GraphType *, int);

/* mmatch.c */
void MCMatch_RM(CtrlType *, GraphType *);
void MCMatch_HEM(CtrlType *, GraphType *);
void MCMatch_SHEM(CtrlType *, GraphType *);
void MCMatch_SHEBM(CtrlType *, GraphType *, int);
void MCMatch_SBHEM(CtrlType *, GraphType *, int);
floattype BetterVBalance(int, int, floattype *, floattype *, floattype *);
int AreAllVwgtsBelowFast(int, floattype *, floattype *, floattype);

/* mmd.c */
void genmmd(int, idxtype *, idxtype *, idxtype *, idxtype *, int , idxtype *, idxtype *, idxtype *, idxtype *, int, int *);
void mmdelm(int, idxtype *xadj, idxtype *, idxtype *, idxtype *, idxtype *, idxtype *, idxtype *, idxtype *, int, int);
int  mmdint(int, idxtype *xadj, idxtype *, idxtype *, idxtype *, idxtype *, idxtype *, idxtype *, idxtype *);
void mmdnum(int, idxtype *, idxtype *, idxtype *);
void mmdupd(int, int, idxtype *, idxtype *, int, int *, idxtype *, idxtype *, idxtype *, idxtype *, idxtype *, idxtype *, int, int *tag);

/* mpmetis.c */
void METIS_mCPartGraphRecursive(int *, int *, idxtype *, idxtype *, idxtype *, idxtype *, int *, int *, int *, int *, int *, idxtype *);
void METIS_mCHPartGraphRecursive(int *, int *, idxtype *, idxtype *, idxtype *, idxtype *, int *, int *, int *, floattype *, int *, int *, idxtype *);
void METIS_mCPartGraphRecursiveInternal(int *, int *, idxtype *, idxtype *, floattype *, idxtype *, int *, int *, int *, idxtype *);
void METIS_mCHPartGraphRecursiveInternal(int *, int *, idxtype *, idxtype *, floattype *, idxtype *, int *, floattype *, int *, int *, idxtype *);
int MCMlevelRecursiveBisection(CtrlType *, GraphType *, int, idxtype *, floattype, int);
int MCHMlevelRecursiveBisection(CtrlType *, GraphType *, int, idxtype *, floattype *, int);
void MCMlevelEdgeBisection(CtrlType *, GraphType *, floattype *, floattype);
void MCHMlevelEdgeBisection(CtrlType *, GraphType *, floattype *, floattype *);

/* mrefine.c */
void MocRefine2Way(CtrlType *, GraphType *, GraphType *, floattype *, floattype);
void MocAllocate2WayPartitionMemory(CtrlType *, GraphType *);
void MocCompute2WayPartitionParams(CtrlType *, GraphType *);
void MocProject2WayPartition(CtrlType *, GraphType *);

/* mrefine2.c */
void MocRefine2Way2(CtrlType *, GraphType *, GraphType *, floattype *, floattype *);

/* mutil.c */
int AreAllVwgtsBelow(int, floattype, floattype *, floattype, floattype *, floattype);
int AreAnyVwgtsBelow(int, floattype, floattype *, floattype, floattype *, floattype);
int AreAllVwgtsAbove(int, floattype, floattype *, floattype, floattype *, floattype);
floattype ComputeLoadImbalance(int, int, floattype *, floattype *);
int AreAllBelow(int, floattype *, floattype *);

/* myqsort.c */
void iidxsort(int, idxtype *);
void iintsort(int, int *);
void ikeysort(int, KeyValueType *);
void ikeyvalsort(int, KeyValueType *);

/* ometis.c */
void METIS_EdgeND(int *, idxtype *, idxtype *, int *, int *, idxtype *, idxtype *); 
void METIS_NodeND(int *, idxtype *, idxtype *, int *, int *, idxtype *, idxtype *); 
void METIS_NodeWND(int *, idxtype *, idxtype *, idxtype *, int *, int *, idxtype *, idxtype *); 
void MlevelNestedDissection(CtrlType *, GraphType *, idxtype *, floattype, int);
void MlevelNestedDissectionCC(CtrlType *, GraphType *, idxtype *, floattype, int);
void MlevelNodeBisectionMultiple(CtrlType *, GraphType *, int *, floattype);
void MlevelNodeBisection(CtrlType *, GraphType *, int *, floattype);
void SplitGraphOrder(CtrlType *, GraphType *, GraphType *, GraphType *);
void MMDOrder(CtrlType *, GraphType *, idxtype *, int);
int SplitGraphOrderCC(CtrlType *, GraphType *, GraphType *, int, idxtype *, idxtype *);

/* parmetis.c */
void METIS_PartGraphKway2(int *, idxtype *, idxtype *, idxtype *, idxtype *, int *, int *, int *, int *, int *, idxtype *); 
void METIS_WPartGraphKway2(int *, idxtype *, idxtype *, idxtype *, idxtype *, int *, int *, int *, floattype *, int *, int *, idxtype *); 
void METIS_NodeNDP(int, idxtype *, idxtype *, int, int *, idxtype *, idxtype *, idxtype *);
void MlevelNestedDissectionP(CtrlType *, GraphType *, idxtype *, int, int, int, idxtype *);
void METIS_NodeComputeSeparator(int *, idxtype *, idxtype *, idxtype *, idxtype *, int *, int *, idxtype *); 
void METIS_EdgeComputeSeparator(int *, idxtype *, idxtype *, idxtype *, idxtype *, int *, int *, idxtype *); 
void METIS_mCPartGraphRecursive2(int *nvtxs, int *ncon, idxtype *xadj, idxtype *adjncy, idxtype *vwgt, idxtype *adjwgt, int *wgtflag, int *numflag, int *nparts, floattype *tpwgts, int *options, int *edgecut, idxtype *part);
int MCMlevelRecursiveBisection2(CtrlType *ctrl, GraphType *graph, int nparts, floattype *tpwgts, idxtype *part, floattype ubfactor, int fpart);



/* pmetis.c */
void METIS_PartGraphRecursive(int *, idxtype *, idxtype *, idxtype *, idxtype *, int *, int *, int *, int *, int *, idxtype *); 
void METIS_WPartGraphRecursive(int *, idxtype *, idxtype *, idxtype *, idxtype *, int *, int *, int *, floattype *, int *, int *, idxtype *); 
int MlevelRecursiveBisection(CtrlType *, GraphType *, int, idxtype *, floattype *, floattype, int);
void MlevelEdgeBisection(CtrlType *, GraphType *, int *, floattype);
void SplitGraphPart(CtrlType *, GraphType *, GraphType *, GraphType *);
void SetUpSplitGraph(GraphType *, GraphType *, int, int);

/* pqueue.c */
void PQueueInit(CtrlType *ctrl, PQueueType *, int, int);
void PQueueReset(PQueueType *);
void PQueueFree(CtrlType *ctrl, PQueueType *);
int PQueueGetSize(PQueueType *);
int PQueueInsert(PQueueType *, int, int);
int PQueueDelete(PQueueType *, int, int);
int PQueueUpdate(PQueueType *, int, int, int);
void PQueueUpdateUp(PQueueType *, int, int, int);
int PQueueGetMax(PQueueType *);
int PQueueSeeMax(PQueueType *);
int PQueueGetKey(PQueueType *);
int CheckHeap(PQueueType *);

/* refine.c */
void Refine2Way(CtrlType *, GraphType *, GraphType *, int *, floattype ubfactor);
void Allocate2WayPartitionMemory(CtrlType *, GraphType *);
void Compute2WayPartitionParams(CtrlType *, GraphType *);
void Project2WayPartition(CtrlType *, GraphType *);

/* separator.c */
void ConstructSeparator(CtrlType *, GraphType *, floattype);
void ConstructMinCoverSeparator0(CtrlType *, GraphType *, floattype);
void ConstructMinCoverSeparator(CtrlType *, GraphType *, floattype);

/* sfm.c */
void FM_2WayNodeRefine(CtrlType *, GraphType *, floattype, int);
void FM_2WayNodeRefineEqWgt(CtrlType *, GraphType *, int);
void FM_2WayNodeRefine_OneSided(CtrlType *, GraphType *, floattype, int);
void FM_2WayNodeBalance(CtrlType *, GraphType *, floattype);
int ComputeMaxNodeGain(int, idxtype *, idxtype *, idxtype *);

/* srefine.c */
void Refine2WayNode(CtrlType *, GraphType *, GraphType *, floattype);
void Allocate2WayNodePartitionMemory(CtrlType *, GraphType *);
void Compute2WayNodePartitionParams(CtrlType *, GraphType *);
void Project2WayNodePartition(CtrlType *, GraphType *);

/* stat.c */
void ComputePartitionInfo(GraphType *, int, idxtype *);
void ComputePartitionInfoBipartite(GraphType *, int, idxtype *);
void ComputePartitionBalance(GraphType *, int, idxtype *, floattype *);
floattype ComputeElementBalance(int, int, idxtype *);
void Moc_ComputePartitionBalance(GraphType *graph, int nparts, idxtype *where, floattype *ubvec);

/* subdomains.c */
void Random_KWayEdgeRefineMConn(CtrlType *, GraphType *, int, floattype *, floattype, int, int);
void Greedy_KWayEdgeBalanceMConn(CtrlType *, GraphType *, int, floattype *, floattype, int);
void PrintSubDomainGraph(GraphType *, int, idxtype *);
void ComputeSubDomainGraph(GraphType *, int, idxtype *, idxtype *);
void EliminateSubDomainEdges(CtrlType *, GraphType *, int, floattype *);
void MoveGroupMConn(CtrlType *, GraphType *, idxtype *, idxtype *, int, int, int, idxtype *);
void EliminateComponents(CtrlType *, GraphType *, int, floattype *, floattype);
void MoveGroup(CtrlType *, GraphType *, int, int, int, idxtype *, idxtype *);

/* timing.c */
void InitTimers(CtrlType *);
void PrintTimers(CtrlType *);
double seconds(void);

/* util.c */
void errexit(char *,...);
#ifndef DMALLOC
int *imalloc(size_t, char *);
idxtype *idxmalloc(size_t, char *);
floattype *fmalloc(size_t, char *);
int *ismalloc(size_t, int, char *);
idxtype *idxsmalloc(size_t, idxtype, char *);
void *GKmalloc(size_t, char *);
#endif
/*void GKfree(void **,...); */
int *iset(int n, int val, int *x);
idxtype *idxset(int n, idxtype val, idxtype *x);
floattype *sset(int n, floattype val, floattype *x);
int iamax(int, int *);
int idxamax(int, idxtype *);
int idxamax_strd(int, idxtype *, int);
int samax(int, floattype *);
int samax2(int, floattype *);
int idxamin(int, idxtype *);
int samin(int, floattype *);
int idxsum(int, idxtype *);
int idxsum_strd(int, idxtype *, int);
void idxadd(int, idxtype *, idxtype *);
int charsum(int, char *);
int isum(int, int *);
floattype ssum(int, floattype *);
floattype ssum_strd(int n, floattype *x, int);
void sscale(int n, floattype, floattype *x);
floattype snorm2(int, floattype *);
floattype sdot(int n, floattype *, floattype *);
void saxpy(int, floattype, floattype *, int, floattype *, int);
void RandomPermute(int, idxtype *, int);
int ispow2(int);
void InitRandom(int);
int log2Int(int);










/***************************************************************
* Programs Directory
****************************************************************/

/* io.c */
void ReadGraph(GraphType *, char *, int *);
void WritePartition(char *, idxtype *, int, int);
void WriteMeshPartition(char *, int, int, idxtype *, int, idxtype *);
void WritePermutation(char *, idxtype *, int);
int CheckGraph(GraphType *);
idxtype *ReadMesh(char *, int *, int *, int *);
void WriteGraph(char *, int, idxtype *, idxtype *);

/* smbfactor.c */
void ComputeFillIn(GraphType *, idxtype *);
idxtype ComputeFillIn2(GraphType *, idxtype *);
int smbfct(int, idxtype *, idxtype *, idxtype *, idxtype *, idxtype *, int *, idxtype *, idxtype *, int *);


/***************************************************************
* Test Directory
****************************************************************/
void Test_PartGraph(int, idxtype *, idxtype *);
int VerifyPart(int, idxtype *, idxtype *, idxtype *, idxtype *, int, int, idxtype *);
int VerifyWPart(int, idxtype *, idxtype *, idxtype *, idxtype *, int, floattype *, int, idxtype *);
void Test_PartGraphV(int, idxtype *, idxtype *);
int VerifyPartV(int, idxtype *, idxtype *, idxtype *, idxtype *, int, int, idxtype *);
int VerifyWPartV(int, idxtype *, idxtype *, idxtype *, idxtype *, int, floattype *, int, idxtype *);
void Test_PartGraphmC(int, idxtype *, idxtype *);
int VerifyPartmC(int, int, idxtype *, idxtype *, idxtype *, idxtype *, int, floattype *, int, idxtype *);
void Test_ND(int, idxtype *, idxtype *);
int VerifyND(int, idxtype *, idxtype *);

