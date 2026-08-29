/** \file DiffusionLB.C
 *  Authors: Monika G
 *           Kavitha C
 *
 */

/**
 *  1. Each node has a list of neighbors (bi-directional) (either topology-based
 *     or other mechanisms like k highest communicating nodes)
 *  2. Over multiple iterations, each node diffuses load to neighbor nodes
 *     by only passing load tokens (not actual objects)
 *  3. Once the diffusion iterations converge (load imbalance threshold is reached),
 *     actual load balancing is done by taking object communication into account
 */

#include "DiffusionLB.h"
#include "LBSimulation.h"


#include "ck.h"
#include "ckgraph.h"
#include "envelope.h"
// #include "LBDBManager.h"
// #include "LBSimulation.h"
#include "DiffusionHelper.C"
#include "elements.h"

#define DEBUGF(x) CmiPrintf x;
#define DEBUGR(x)  // CmiPrintf x;
#define DEBUGL(x) /*CmiPrintf x*/;
// Rounds of the pseudo-load diffusion loop. Fixed, with no convergence check:
// every round costs two neighbour exchanges and the SDAG waits between them, so
// the strategy pays all 40 even when the load is already even. On a GPU-bound
// run where diffusion wants to move ~1% of the load, that is the single largest
// cost of load balancing -- larger than the migration it decides on.
// CHARM_LB_DIFFUSION_ITERS overrides it so the trade can be measured.
static int diffusionIterations() {
  static const int n = []() {
    const char* s = getenv("CHARM_LB_DIFFUSION_ITERS");
    const int v = s ? atoi(s) : 40;
    return v > 0 ? v : 40;
  }();
  return n;
}
#define ITERATIONS (diffusionIterations())

#include "DiffusionMetric.C"
#include "DiffusionNeighbors.C"
#include "DiffusionPseudo.C"
#include "DiffusionCore.C"

// Percentage of error acceptable.
#define THRESHOLD 2

// Diffusion rounds stop once no node wants to shift more than this fraction of
// its own load. Measured on a GPU-bound run, diffusion asks to move ~1.6% on the
// first round and converges immediately after, so the fixed 40 rounds were
// almost entirely wasted.
static double pseudoConvergeRatio() {
  static const double r = []() {
    const char* s = getenv("CHARM_LB_DIFFUSION_CONVERGE");
    const double v = s ? atof(s) : (THRESHOLD / 100.0);
    return v > 0.0 ? v : (THRESHOLD / 100.0);
  }();
  return r;
}
#define PSEUDO_CONVERGE_RATIO (pseudoConvergeRatio())

// Initialize static Diffusion timing variables
double DiffusionLB::totalNeighborTime = 0.0;
double DiffusionLB::totalLBTime = 0.0;
double DiffusionLB::totalStartTime = 0.0;
double DiffusionLB::totalPseudoLBTime = 0.0;
  double DiffusionLB::totalAcrossTime = 0.0;
double DiffusionLB::totalWithinTime = 0.0;
  double DiffusionLB::phaseStartTime = 0.0;

static bool diffusionTimingExitRegistered = false;

static void printDiffusionTimingAtExit() {
  DiffusionLB::printDiffusionTiming();
  CkContinueExit();
}

// CreateLBFunc_Def(DiffusionLB, "The distributed graph refinement load balancer")
static void lbinit()
{
  LBRegisterBalancer<DiffusionLB>("DiffusionLB",
                                  "The distributed graph refine load balancer");

  numPes = CkNumPes();
  
  // Register exit function to print Diffusion timing
  if (!diffusionTimingExitRegistered) {
    registerExitFn(printDiffusionTimingAtExit);
    diffusionTimingExitRegistered = true;
  }
}

using std::vector;


DiffusionLB::DiffusionLB(const CkLBOptions& opt) : CBase_DiffusionLB(opt)
{
  nodeSize = CkNodeSize(0);
  myNodeId = CkMyPe() / nodeSize;
  acks = 0;
  max = 0;
  round = 0;
  statsReceived = 0;
  rank0_barrier_counter = 0;

  myNodeInternalBytes = 0.0;
  myNodeExternalBytes = 0.0;

  num_migrations = 0;
  pseudoSectionBuilt = false;

#if CMK_LBDB_ON
  lbname = "DiffusionLB";
  if (_lb_args.statsOn())
    lbmgr->CollectStatsOn();
  // Every comm-aware balancer in this tree enables communication instrumentation
  // from its own constructor rather than requiring the user to pass +LBCommOn --
  // MetisLB.C:23, ScotchLB.C:23, ScotchTopoLB.C:25, ScotchRefineLB.C:20,
  // RecBipartLB.C:119, ZoltanLB.C:62. Without this the comm graph is empty and
  // MetricComm scores every object identically.
  LBTurnCommOn();
  thisProxy = CProxy_DiffusionLB(thisgroup);
  numNodes = CkNumPes() / nodeSize;  // CkNumNodes();
  myStats = new DistBaseLB::LDStats;

  rank0PE = myNodeId * nodeSize;  // CkNodeFirst(CkMyNode());
  if (CkMyPe() == rank0PE)
  {
    statsList = new CLBStatsMsg*[nodeSize];
    nodeStats = new BaseLB::LDStats(nodeSize);
    numObjects.resize(nodeSize);
    prefixObjects.resize(nodeSize);
    pe_load.resize(nodeSize);
  }
  if (CkMyPe() == 0)
  {
    fullStats = new BaseLB::LDStats(CkNumPes());
  }
#endif
}

DiffusionLB::DiffusionLB(CkMigrateMessage* m) : CBase_DiffusionLB(m) {}

DiffusionLB::~DiffusionLB()
{
#if CMK_LBDB_ON
  delete[] statsList;
  delete nodeStats;
  delete myStats;
  delete[] gain_val;
  lbmgr = CProxy_LBManager(_lbmgr).ckLocalBranch();
  if (lbmgr)
    lbmgr->RemoveStartLBFn(startLbFnHdl);
#endif
}

// Main entry point for the load balancer
void DiffusionLB::Strategy(const DistBaseLB::LDStats* const stats)
{
  startOverallTiming();
  total_migrates = 0;
  total_crossnode_migrates = 0;

  if (CkMyPe() == 0 && _lb_args.debug() >= 1)
  {
    double start_time = CmiWallTimer();
  }
  statsmsg = AssembleStats();
  if (statsmsg == NULL)
    CkAbort("Error: statsmsg is NULL\n");

  // start stats assembly on rank0PE
  marshmsg = new CkMarshalledCLBStatsMessage(statsmsg);

  // reset variables (necessary for mutliple LB rounds)
  acks = 0;
  max = 0;
  round = 0;
  rank0_barrier_counter = 0;
  pseudo_done = true;

  num_migrations = 0;

  mig_id_map.clear();
  objectHandles.clear();
  objectSrcIds.clear();
  objSenderPEs.clear();
  objectLoads.clear();

  

  thisProxy[rank0PE].ReceiveStats(*marshmsg);

  if (CkMyPe() != rank0PE)
  {
    CkCallback cb(CkReductionTarget(DiffusionLB, statsAssembled), thisProxy);
    contribute(cb);
  }

  if (CkMyPe() == rank0PE) {
      for (int i = 0; i < nodeSize; i++) pe_load[i] = 0;
      myNodeInternalBytes = 0.0;
      myNodeExternalBytes = 0.0;
  }
}

/*Entry method called on each rank0PE to collect all node-relevant stats. On completion,
 * all PEs call statsAssembled().*/
void DiffusionLB::ReceiveStats(CkMarshalledCLBStatsMessage&& data)
{
  // TODO: why is this in CMK_LBDB_ON? needs to be done always?
#if CMK_LBDB_ON
  CLBStatsMsg* m = data.getMessage();
  CmiAssert(CkMyPe() == rank0PE);

  // store the message
  int fromRank = m->from_pe - rank0PE;
  statsReceived++;

  // Clear nodeStats at the start of each new round to prevent accumulation
  if (statsReceived == 1) {
    nodeStats->objData.clear();
    nodeStats->from_proc.clear();
    nodeStats->to_proc.clear();
    nodeStats->commData.clear();
    nodeStats->n_migrateobjs = 0;
    nodeStats->deleteCommHash();
  }

  AddToList(m, fromRank);

  if (statsReceived == nodeSize)
  {
    // build LDStats
    BuildStats();
    CkCallback cb(CkReductionTarget(DiffusionLB, statsAssembled), thisProxy);
    contribute(cb);
    statsReceived = 0;
  }
#endif
}

/*Once stats are assembled on rank0PEs, can begin finding Nbors*/
void DiffusionLB::statsAssembled()
{
  if (CkMyPe() == rank0PE)
  {
    findNBors(1);
  }
}

void DiffusionLB::InitializeObjHeap(int n)
{
  obj_heap.resize(n);
  heap_pos.resize(n);
  for (int i = 0; i < n; i++)
  {
    obj_heap[i] = i;
    heap_pos[i] = i;
  }
  heapify(obj_heap, ObjCompareOperator(&objects, gain_val), heap_pos);
}

// Create a migrate message for this obj from resident PE to rank0PE
// objId should be PE local id of the object
void DiffusionLB::LoadReceived(int objId, int destPE)
{
  int sourcePE = CkMyPe();
  
  if (objId < 0 || objId >= myStats->objData.size()) {
    CkAbort("Error: objId %d out of bounds for size %d on PE %d\n", objId, (int)myStats->objData.size(), CkMyPe());
  }
  // load is received, hence create a migrate message for the object with id objId.
  auto it = mig_id_map.find(objId);
  if(it!=mig_id_map.end()) {
    MigrateInfo* migrateMe = it->second;
    migrateMe->to_pe = destPE;
  } else {
    MigrateInfo* migrateMe = new MigrateInfo;
    migrateMe->obj = myStats->objData[objId].handle;
    migrateMe->from_pe = CkMyPe();
    migrateMe->to_pe = destPE;
    // migrateMe->async_arrival = myStats->objData[objId].asyncArrival;
    migrateInfo.push_back(migrateMe);
    mig_id_map.emplace(objId, migrateMe);
    total_migrates++;

    if (CkMyPe() / nodeSize != destPE / nodeSize)
      total_crossnode_migrates++;
  }

  if (_lb_args.debug() > 2) CkPrintf("[%d] Completing LoadReceived for objId %d to %d\n", CkMyPe(), objId, destPE);
}

void DiffusionLB::update_peload(int rank, double load) {
  pe_load[rank] -= load;
}

/* Load has been logically sent from overloaded to underloaded nodes in LoadBalance().
 * Now we should load balance the PE's within the node. This function should only be
 * called by rank0PE.
 *
 * At a high level, this does the following:
 * - find overloaded and underloaded PEs on my node
 * - create minheap of PEs sorted by load
 * - create maxheap of objects (using ckheap) sorted by load
 * - iterate through objects in maxheap and offload based on minheap (via LoadReceived)
 *
 * MEMORY CONTRACT (LBMemoryContract.h) -- integration design, not yet wired:
 * DiffusionLB does not pass through CentralLB::Strategy, so the contract
 * verifier does not cover it. The decentralized form of the contract is
 * receiver-side acceptance: (1) each PE advertises its device's free memory
 * on the neighbor load-exchange messages it already sends; (2) a transfer
 * becomes an offer that the RECEIVER accepts or refuses against a local
 * ledger of headroom minus committed arrivals (receiver-side serialization
 * resolves concurrent senders; a refusal is ordinary diffusion back-pressure
 * -- the object stays and later rounds retry); (3) I-batch is local: each PE
 * bounds its own round's outgoing staged bytes by its staging reserve. The
 * offer/refusal round-trip touches this file, DiffusionNeighbors, and the
 * LoadMetaInfo/LoadReceived protocol, and must be validated on a multi-device
 * run -- deliberately not implemented blind on a single-GPU machine.
 * */
void DiffusionLB::WithinNodeLB()
{

   endAcrossTiming();
      startWithinTiming();
  if (thisIndex == 0)
    if (_lb_args.debug() == 3) CkPrintf("--------STARTING WITHIN NODE LB--------\n");

  if (CkMyPe() == 0)
    {
      if (step() == LBSimulation::dumpStep)
      {
        CkCallback cb(CkIndex_DiffusionLB::ProcessFinalStats(), thisProxy);
         CkStartQD(cb);
      }
      else if (_lb_args.debug() > 0) {
        CkCallback cb(CkIndex_DiffusionLB::CollectStats(), thisProxy);
         CkStartQD(cb);
      }
      else
      {
        CkCallback cb(CkIndex_DiffusionLB::ProcessMigrations(), thisProxy);
        CkStartQD(cb);
      }
    }

  if( nodeSize == 1) {
      if (_lb_args.debug() == 3) CkPrintf("--------Node size is 1--------\n");
    return;
  }
  if (CkMyPe() == rank0PE)
  {
   
    // CkPrintf("[%d] GRD: DoneNodeLB \n", CkMyPe());
    double avgPE = averagePE();

    // Create a max heap and min heap for pe loads
    std::vector<double> objectSizes;
    std::vector<int> objectIds;
    std::vector<int> objectPEs;
    std::vector<LDObjHandle> objectHdl;
    std::vector<int> isToken;
    minHeap minPes(nodeSize);
    double threshold = THRESHOLD * avgPE / 100.0;

    // for each pe... find overload, something with prefix sum?
    // and store the underloaded pes
    for (int rank = 0; rank < nodeSize; rank++)
    {
      if (_lb_args.debug() == 3) CkPrintf("\nOrig PE load with node LB [%d] = %lf", rank+rank0PE, pe_load[rank]);
      if (pe_load[rank] > avgPE + threshold)
      {
        double overLoad = pe_load[rank] - avgPE;
        int start = 0;
        if (rank != 0)
        {
          start = prefixObjects[rank - 1];
        }
        for (int j = start; j < prefixObjects[rank]; j++)
        {
          // getCompLoad(), not getVertexLoad(): this weighs an object's load against
          // a budget in seconds (overLoad), and getVertexLoad()'s MAX(compLoad, 0.1)
          // floor reports every object as 0.1s whenever real per-object load is
          // smaller -- the common case. With a typical overLoad well under 0.1s the
          // test then fails for every object on every step, and within-node balancing
          // silently does nothing while reporting that it ran.
          if (objs[j].isMigratable() && objs[j].getCurrPe() != -1 && objs[j].getCompLoad() <= overLoad)
          {
            objectSizes.push_back(objs[j].getCompLoad());

            int pe_local_id = j;
            if (rank != 0) {
              pe_local_id = j - prefixObjects[rank - 1];
            }
            objectIds.push_back(pe_local_id);
            objectPEs.push_back(rank+rank0PE);
            objectHdl.push_back(nodeStats->objData[j].handle);
            isToken.push_back(0);
            overLoad -= objs[j].getCompLoad();
          }
        }
        if(rank==0) {
          //Objects migrating in
          for(int i=0;i<objectLoads.size();i++) {
            if(objectLoads[i] <= overLoad) {
              objectSizes.push_back(objectLoads[i]);
              objectIds.push_back(objectSrcIds[i]); // this is pe local id
              objectHdl.push_back(objectHandles[i]);
              objectPEs.push_back(objSenderPEs[i]);
              isToken.push_back(1);
              overLoad -= objectLoads[i];
            }
          }
        }
      }
      else if (pe_load[rank] < avgPE - threshold)
      {
        InfoRecord* itemMin = new InfoRecord;
        itemMin->load = pe_load[rank];
        itemMin->Id = rank;
        minPes.insert(itemMin);
      }
    }

    // build heap of objects
    maxHeap objects(objectIds.size());
    for (int i = 0; i < objectIds.size(); i++)
    {
      InfoRecord* item = new InfoRecord;
      item->load = objectSizes[i];  // sorting factor in maxheap
      item->Id = objectIds[i];
      item->pe = objectPEs[i]; // sending pe
      item->handle = objectHdl[i];
      item->token = false;
      if(isToken[i])
        item->token = true;
      objects.insert(item);
    }

    // pop object from priority queue and migrate to most underloaded PE
    // TODO: this needs a strategy update
    InfoRecord* minPE = NULL;
    while (objects.numElements() > 0 &&
           ((minPE == NULL && minPes.numElements() > 0) || minPE != NULL))
    {
      InfoRecord* maxObj = objects.deleteMax();
      if (minPE == NULL)
        minPE = minPes.deleteMin();
      double diff = avgPE - minPE->load;
      if(diff < 0) {
        minPE = minPes.deleteMin();
        continue;
      }
      int objId = maxObj->Id; // this is the pe local id of the object
      int nodeObjId = objId; // node local id (to be cmoputed below)

      // TODO!!!! objID must be the pe_local one the whole time!! otherwise, might tyr to compute pe index here for non local object!!
      int rank = maxObj->pe % nodeSize; // donor PE rank (original)
      bool is_local = !maxObj->token;

      if (!is_local) rank = 0; // coming from rank0PE (as a token)

      if (rank > 0)
        nodeObjId += prefixObjects[rank - 1];

      if (maxObj->load > diff || pe_load[rank] < avgPE - threshold)
      {
        delete maxObj;
        continue;
      }

      int destPE = rank0PE + minPE->Id;
      int donorPE = maxObj->pe;

      if (is_local && donorPE != rank + rank0PE) {
        CkAbort("Error: donorPE %d does not match object PE %d = %d + %d\n", donorPE, rank+rank0PE, rank, rank0PE);
      }

      LDObjHandle objHandle = maxObj->handle;
      double currLoad = maxObj->load;     
 
      if(!is_local) {
        migrates_expected--;
        //subtract from intermediate PE (rank0, i.e. me)
      } else {
        nodeStats->to_proc[nodeObjId] = destPE;
      }
    
      if (objId < 0) {
        CkAbort("Error: objId %d is negative for objId %d on donorPE %d (rank0PE %d)\n",
                objId, objId, donorPE, rank0PE);
      }
      thisProxy[destPE].LoadMetaInfo(objHandle, objId, currLoad, donorPE, 1); // to the receiving PE (mig++)
      thisProxy[donorPE].LoadReceived(objId, destPE);
      pe_load[minPE->Id] += maxObj->load;
      pe_load[rank] -= maxObj->load;
      if (pe_load[minPE->Id] < avgPE)
      {
        minPE->load += maxObj->load;//= pe_load[minPE->Id];
        minPes.insert(minPE);
      }
      else
        delete minPE;
      minPE = NULL;
    }
    //may be clearing heaps for next LB step, after intra-node LB is done above
    // TODO: clear the heaps? why?
    while (minPes.numElements() > 0)
    {
      InfoRecord* minPE = minPes.deleteMin();
      delete minPE;
    }
    while (objects.numElements() > 0)
    {
      InfoRecord* maxObj = objects.deleteMax();
      delete maxObj;
    }

    // TODO: submit to print stats
    // This QD is essential because, before the actual migration starts, load should be
    // divided amongs intra node PE's.


    endWithinTiming();
  }
}

void DiffusionLB::ProcessMigrations()
{
  if (CkMyPe() == 0)
  BaseLB::endLBStrategyTiming();

  // SAME AS IN PACKANDSENDMIGRATEMSGS
  LBMigrateMsg* msg = new (total_migrates, CkNumPes(), CkNumPes(), 0) LBMigrateMsg;
  msg->n_moves = total_migrates;
  if (_lb_args.debug() > 1) CkPrintf("PE-%d with %d migrates and %d cross-node migrates\n", CkMyPe(), total_migrates, total_crossnode_migrates);
  for (int i = 0; i < total_migrates; i++)
  {
    MigrateInfo* item = (MigrateInfo*)migrateInfo[i];
    msg->moves[i] = *item;
    delete item;
    migrateInfo[i] = 0;
  }
  migrateInfo.clear();

  // if we don't do the barrier here, must be done with LBSyncResume so that it is done in
  // MigrationDone
  if (!_lb_args.syncResume())
  {
  // SAME AS IN PROCESSMIGRATIONDECISION
  const int me = CkMyPe();
  for (int i = 0; i < msg->n_moves; i++)
  {
    MigrateInfo& move = msg->moves[i];
     if (_lb_args.debug() == 3) CkPrintf("\n[PE-%d] Migrating obj from %d to %d", CkMyPe(), move.from_pe,
               move.to_pe);
    if (move.from_pe == me)
    {
      if (move.to_pe == me)
      {
          CkAbort("[%i] Error, attempting to migrate object myself to myself\n",
                  CkMyPe());
      }
     
      lbmgr->Migrate(move.obj, move.to_pe);
    }
    else if (move.from_pe != me)
    {
      CkAbort("Trying to move objs not on my PE\n");
    }
  }

#if CMK_GLOBAL_LOCATION_UPDATE
  // SAME AS IN PROCESSMIGRATIONDECISION
  if (!_lb_args.lbPeerDecision()) BroadcastLocationUpdate(msg);
#endif

    CkCallback cb(CkIndex_DiffusionLB::MigrationDoneWrapper(), thisProxy);
    contribute(cb);
    // Nothing holds the message past this point: the moves have been issued and
    // BroadcastLocationUpdate copies. The syncResume branch below hands
    // ownership to ProcessMigrationDecision, which frees it.
    delete msg;
  }
  else
    ProcessMigrationDecision(msg);

}

void DiffusionLB::CascadingMigration(LDObjHandle h, double load)
{
#if 0
  CkAbort("CASCADING: we don't understand this implementation yet\n");
  double threshold = THRESHOLD * avgLoadNeighbor / 100.0;
  int minNode = 0;
  int myPos = 0;  // neighborPos[CkNodeOf(rank0PE)];

  if (loadReceivers > 0)
  {
    double minLoad;
    // Send to max underloaded node
    for (int i = 0; i < neighborCount; i++)
    {
      if (toSendLoad[i] >= threshold && load <= toSendLoad[i] &&
          (minNode == -1 || minLoad < toSendLoad[i]))
      {
        minNode = i;
        minLoad = toSendLoad[i];
      }
    }
    if (minNode != -1 && minNode != myPos)
    {
      // Send load info to receiving load
      toSendLoad[minNode] -= load;
      if (toSendLoad[minNode] < threshold)
      {
        loadReceivers--;
      }
      thisProxy[sendToNeighbors[minNode] *
                nodeSize /*CkNodeFirst(sendToNeighbors[minNode])*/]
          .LoadMetaInfo(h, 0, load, CkMyPe(), 0);
      const int acrossNodeToPe = sendToNeighbors[minNode] *
                            nodeSize /*CkNodeFirst(sendToNeighbors[minNode])*/;
      lbmgr->Migrate(h, acrossNodeToPe);
#if CMK_GLOBAL_LOCATION_UPDATE
      // This is the across-node move, so the object lands in a different
      // process. Bystanders have to learn the new location, or a GPU-direct
      // sender keeps picking its transfer mode for the process the object just
      // left -- and a MEMCPY chosen that way hands the receiver a pointer into
      // an address space it cannot read. Migrating one object at a time means
      // the move-list broadcast never sees these.
      //
      // +LBPeerDecision drops the broadcast: CkLocMgr's process residency table
      // answers the "is it on my GPU" question exactly, without anyone having
      // to be told, so the mode decision no longer depends on every PE holding
      // a fresh cache entry. Ordinary messages route by home as always.
      if (!_lb_args.lbPeerDecision())
        BroadcastSingleLocationUpdate(h, acrossNodeToPe);
#endif
    }
  }
  if (loadReceivers <= 0 || minNode == myPos || minNode == -1)
  {
    int minRank = -1;
    double minLoad = 0;
    for (int i = 0; i < nodeSize; i++)
    {
      if (minRank == -1 || pe_load[i] < minLoad)
      {
        minRank = i;
        minLoad = pe_load[i];
      }
    }

    pe_load[minRank] += load;
    if (minRank > 0)
    {
      lbmgr->Migrate(h, rank0PE + minRank);
#if CMK_GLOBAL_LOCATION_UPDATE
      // Within-node move: same process, so no transfer mode changes meaning,
      // but other PEs still cache a location that is now wrong.
      if (!_lb_args.lbPeerDecision())
        BroadcastSingleLocationUpdate(h, rank0PE + minRank);
#endif
    }
  }
#endif
}


void DiffusionLB::MigrationDoneWrapper()
{
  int balancing = 1;
  MigrationDone(balancing);  // call DistBaseLB version
  
  // End LB timing instrumentation
}

void DiffusionLB::printDiffusionTiming()
{
  if (CkMyPe() == 0 && (totalNeighborTime > 0 || totalPseudoLBTime > 0 || totalAcrossTime > 0 || totalWithinTime > 0))
  {
    CkPrintf("\n[DiffusionLB Timing] Neighbor Selection: %.6f seconds\n", totalNeighborTime);
    CkPrintf("[DiffusionLB Timing] Pseudo LB: %.6f seconds\n", totalPseudoLBTime);
    CkPrintf("[DiffusionLB Timing] Across Node: %.6f seconds\n", totalAcrossTime);
    CkPrintf("[DiffusionLB Timing] Within Node: %.6f seconds\n", totalWithinTime);
    CkPrintf("[DiffusionLB Timing] Total: %.6f seconds (sum of phases %.6f)\n", 
             totalLBTime, 
             totalNeighborTime + totalPseudoLBTime + totalAcrossTime + totalWithinTime);
  }
}

#include "DiffusionLB.def.h"
