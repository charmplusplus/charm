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
#define ITERATIONS 40

#include "DiffusionMetric.C"
#include "DiffusionNeighbors.C"
#include "DiffusionPseudo.C"
#include "DiffusionCore.C"

// Percentage of error acceptable.
#define THRESHOLD 2

// CreateLBFunc_Def(DiffusionLB, "The distributed graph refinement load balancer")
static void lbinit()
{
  LBRegisterBalancer<DiffusionLB>("DiffusionLB",
                                  "The distributed graph refine load balancer");

  numPes = CkNumPes();
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

#if CMK_LBDB_ON
  lbname = "DiffusionLB";
  if (_lb_args.statsOn())
    lbmgr->CollectStatsOn();
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
  if (_lb_args.debug())CkPrintf("\n[PE-%d] In Strategy", CkMyPe());
  total_migrates = 0;

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
    if (_lb_args.debug()) CkPrintf("\nPE-%d, calling findNNeighbors", CkMyPe());
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
void DiffusionLB::LoadReceived(int objId, int from0PE)
{
  // load is received, hence create a migrate message for the object with id objId.
  auto it = mig_id_map.find(objId);
  if(it!=mig_id_map.end()) {
    MigrateInfo* migrateMe = it->second;
    if (_lb_args.debug())CkPrintf("\nUpdating to PE from %d to %d", migrateMe->to_pe, from0PE);
    migrateMe->to_pe = from0PE;
  } else {
    MigrateInfo* migrateMe = new MigrateInfo;
    migrateMe->obj = myStats->objData[objId].handle;
    migrateMe->from_pe = CkMyPe();
    migrateMe->to_pe = from0PE;
    if(CkMyPe()==rank0PE)
      pe_load[CkMyRank()] -= myStats->objData[objId].wallTime;
    else
      thisProxy[rank0PE].update_peload(CkMyRank(), myStats->objData[objId].wallTime);
    // migrateMe->async_arrival = myStats->objData[objId].asyncArrival;
    migrateInfo.push_back(migrateMe);
    mig_id_map.emplace(objId, migrateMe);
    total_migrates++;
  }
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
 * */
void DiffusionLB::WithinNodeLB()
{

  if (thisIndex == 0)
    if (_lb_args.debug()) CkPrintf("--------STARTING WITHIN NODE LB--------\n");

  if(nodeSize==1) {
      if (_lb_args.debug()) CkPrintf("--------Node size is 1--------\n");

    if (CkMyPe() == 0)
    {
      if (step() == LBSimulation::dumpStep)
      {
        CkCallback cb(CkIndex_DiffusionLB::ProcessFinalStats(), thisProxy);
         CkStartQD(cb);
      }
      else
      {
        CkCallback cb(CkIndex_DiffusionLB::ProcessMigrations(), thisProxy);
        CkStartQD(cb);
      }
    }
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
      if (_lb_args.debug()) CkPrintf("\nOrig PE load with node LB [%d] = %lf", rank+rank0PE, pe_load[rank]);
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
          if (objs[j].isMigratable() && objs[j].getCurrPe() != -1 && objs[j].getVertexLoad() <= overLoad)
          {
            objectSizes.push_back(objs[j].getVertexLoad());

            objectIds.push_back(j );
            objectPEs.push_back(/*GetPENumber(j)*/rank+rank0PE);
            objectHdl.push_back(nodeStats->objData[j].handle);
            isToken.push_back(0);
            overLoad -= objs[j].getVertexLoad();
          }
        }
        if(rank==0) {
          //Objects migrating in
          for(int i=0;i<objectLoads.size();i++) {
            if(objectLoads[i] <= overLoad) {
              objectSizes.push_back(objectLoads[i]);
              //CkPrintf("Adding object with objectSrcId %d with load %lf\n", objectSrcIds[i], objectLoads[i]);
              objectIds.push_back(objectSrcIds[i]/*objHandle*/);
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
        if (_lb_args.debug())CkPrintf("\nAdding PE-%d to minPEs on node %d", CkMyPe(), myNodeId);
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
      int objId = maxObj->Id;
      int pe = GetPENumber(objId);
      if (maxObj->load > diff || pe_load[pe] < avgPE - threshold)
      {
        delete maxObj;
        continue;
      }

      int destPE = rank0PE + minPE->Id;
      int donorPE = maxObj->pe;
      LDObjHandle objHandle = maxObj->handle;
      double currLoad = maxObj->load;

      // update nodestats for within node migration as well
      nodeStats->to_proc[objId] = destPE;

      thisProxy[destPE].LoadMetaInfo(objHandle, objId, currLoad, donorPE, 1); // to the receiving PE (mig++)
 
      if(maxObj->token) {
        migrates_expected--;
        //subtract from intermediate PE (rank0, i.e. me)
      }
      
      thisProxy[donorPE].LoadReceived(objId, destPE);

      pe_load[minPE->Id] += maxObj->load;
      pe_load[pe] -= maxObj->load;
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

    if (CkMyPe() == 0)
    { if (step() == LBSimulation::dumpStep)
      {
        CkCallback cb(CkIndex_DiffusionLB::ProcessFinalStats(), thisProxy);
         CkStartQD(cb);
      }
      else {

      CkCallback cb(CkIndex_DiffusionLB::ProcessMigrations(), thisProxy);
      CkStartQD(cb);
      }
    }
  }
}

void DiffusionLB::ProcessMigrations()
{
  

  // SAME AS IN PACKANDSENDMIGRATEMSGS
  LBMigrateMsg* msg = new (total_migrates, CkNumPes(), CkNumPes(), 0) LBMigrateMsg;
  msg->n_moves = total_migrates;
  if (_lb_args.debug()) CkPrintf("PE-%d with %d migrates\n", CkMyPe(), total_migrates);
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
     if (_lb_args.debug()) CkPrintf("\n[PE-%d] Migrating obj from %d to %d", CkMyPe(), move.from_pe,
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

    CkCallback cb(CkIndex_DiffusionLB::MigrationDoneWrapper(), thisProxy);
    contribute(cb);
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
      lbmgr->Migrate(h, sendToNeighbors[minNode] *
                            nodeSize /*CkNodeFirst(sendToNeighbors[minNode])*/);
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
    }
  }
#endif
}


void DiffusionLB::MigrationDoneWrapper()
{
  int balancing = 1;
  MigrationDone(balancing);  // call DistBaseLB version
}

#include "DiffusionLB.def.h"
