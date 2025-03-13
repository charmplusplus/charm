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

#include "ck.h"
#include "ckgraph.h"
#include "envelope.h"
// #include "LBDBManager.h"
// #include "LBSimulation.h"
#include "DiffusionHelper.C"
#include "Heap_helper.C"
#include "elements.h"

#define DEBUGF(x) CmiPrintf x;
#define DEBUGR(x)  // CmiPrintf x;
#define DEBUGL(x) CmiPrintf x;
#define ITERATIONS 40

#define NUM_NEIGHBORS 2

#include "DiffusionNeighbors.C"

// Percentage of error acceptable.
#define THRESHOLD 2

// CreateLBFunc_Def(DiffusionLB, "The distributed graph refinement load balancer")
static void lbinit()
{
  LBRegisterBalancer<DiffusionLB>("DiffusionLB",
                                  "The distributed graph refine load balancer");
}

using std::vector;

DiffusionLB::DiffusionLB(const CkLBOptions& opt) : CBase_DiffusionLB(opt)
{
  nodeSize = CkNodeSize(0);
  myNodeId = CkMyPe() / nodeSize;
  acks = 0;
  max = 0;
  edgeCount = 0;
  edge_indices.reserve(100);
  round = 0;
  statsReceived = 0;
  rank0_barrier_counter = 0;

#if CMK_LBDB_ON
  lbname = "DiffusionLB";
  if (CkMyPe() == 0)
    CkPrintf("[%d] Diffusion created\n", CkMyPe());
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
  delete[] obj_arr;
  lbmgr = CProxy_LBManager(_lbmgr).ckLocalBranch();
  if (lbmgr)
    lbmgr->RemoveStartLBFn(startLbFnHdl);
#endif
}

// Main entry point for the load balancer
void DiffusionLB::Strategy(const DistBaseLB::LDStats* const stats)
{
  total_migrates = 0;

  if (CkMyPe() == 0 && _lb_args.debug() >= 1)
  {
    double start_time = CmiWallTimer();
    CkPrintf("In DiffusionLB strategy at %lf\n", start_time);
  }
  if (CkMyPe() == 0)
  {
    CkCallback cb(CkIndex_DiffusionLB::AcrossNodeLB(), thisProxy);
    CkStartQD(cb);
  }
  statsmsg = AssembleStats();
  if (statsmsg == NULL)
    CkAbort("Error: statsmsg is NULL\n");

  // start stats assembly on rank0PE
  marshmsg = new CkMarshalledCLBStatsMessage(statsmsg);
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
    findNBors(1);
  }
}

void DiffusionLB::startStrategy()
{
  if (++rank0_barrier_counter < numNodes)
    return;

  rank0_barrier_counter = 0;
  CkPrintf("--------NEIGHBOR SELECTION COMPLETE--------\n");
  for (int i = 0; i < numNodes; i++) thisProxy[i * nodeSize].pseudolb_rounds();
}

void DiffusionLB::InitializeObjHeap(BaseLB::LDStats* stats, int* obj_arr, int n,
                                    int* gain_val)
{
  for (int i = 0; i < n; i++)
  {
    obj_heap[i] = obj_arr[i];
    heap_pos[obj_arr[i]] = i;
  }
  heapify(obj_heap, ObjCompareOperator(&objs, gain_val), heap_pos);
}

void DiffusionLB::PseudoLoadBalancing()
{
  double threshold = THRESHOLD * avgLoadNeighbor / 100.0;

  avgLoadNeighbor = (avgLoadNeighbor + my_pseudo_load) / 2;
  double totalOverload = my_pseudo_load - avgLoadNeighbor;
  double totalUnderLoad = 0.0;
  double thisIterToSend[neighborCount];
  for (int i = 0; i < neighborCount; i++) thisIterToSend[i] = 0.0;
  if (totalOverload > 0)
    for (int i = 0; i < neighborCount; i++)
    {
      if (loadNeighbors[i] < (avgLoadNeighbor - threshold))
      {
        thisIterToSend[i] = avgLoadNeighbor - loadNeighbors[i];
        totalUnderLoad += avgLoadNeighbor - loadNeighbors[i];
        //        DEBUGL2(("[PE-%d] iteration %d thisIterToSend %f avgLoadNeighbor %f
        //        loadNeighbors[%d] %f to node %d\n",
        //                thisIndex, itr, thisIterToSend[i], avgLoadNeighbor, i,
        //                loadNeighbors[i], sendToNeighbors[i]));
      }
    }
  if (totalUnderLoad > 0 && totalOverload > 0 && totalUnderLoad > totalOverload)
    totalOverload += threshold;
  else
    totalOverload = totalUnderLoad;

  for (int i = 0; i < neighborCount; i++)
  {
#if 1
    if (totalOverload > 0 && totalUnderLoad > 0 && thisIterToSend[i] > 0)
    {
      //      DEBUGL2(("[%d] GRD: Pseudo Load Balancing Sending, iteration %d node
      //      %d(pe-%d) toSend %lf totalToSend %lf\n", CkMyPe(), itr, sendToNeighbors[i],
      //      CkNodeFirst(sendToNeighbors[i]), thisIterToSend[i],
      //      (thisIterToSend[i]*totalOverload)/totalUnderLoad));
      thisIterToSend[i] *= totalOverload / totalUnderLoad;
      toSendLoad[i] += thisIterToSend[i];
      if (my_pseudo_load - thisIterToSend[i] < 0)
        CkAbort("Error: my_pseudo_load (%f) - thisIterToSend[i] (%f) < 0\n",
                my_pseudo_load, thisIterToSend[i]);
      my_pseudo_load -= thisIterToSend[i];
    }
    if (thisIterToSend[i] < 0.0)
      thisIterToSend[i] = 0.0;
#endif
    int nbor_node = sendToNeighbors[i];
    thisProxy[nbor_node * nodeSize].PseudoLoad(itr, thisIterToSend[i], myNodeId);
  }
}

#define SELF_IDX NUM_NEIGHBORS
#define EXT_IDX NUM_NEIGHBORS + 1
/* On completion, waits for QD then calls WITHINNODELB*/
void DiffusionLB::AcrossNodeLB()
{
  if (CkMyPe() != rank0PE)
    return;
  if (CkMyPe() == 0)
  {
    CkPrintf("--------STARTING ACROSS NODE LB--------\n");
    CkCallback cb(CkIndex_DiffusionLB::WithinNodeLB(), thisProxy);
    CkStartQD(cb);
  }
  balanced.resize(neighborCount);
  for (int i = 0; i < neighborCount; i++)
  {
    balanced[i] = false;
    if (toSendLoad[i] > 0)
    {
      balanced[i] = true;
      loadReceivers++;
    }
  }

  int n_objs = nodeStats->objData.size();

  objectComms.resize(n_objs);
  gain_val = new int[n_objs];
  memset(gain_val, 100, n_objs);

  for (int i = 0; i < n_objs; i++)
  {
    objectComms[i].resize(NUM_NEIGHBORS + 2);
    for (int j = 0; j < NUM_NEIGHBORS + 2; j++) objectComms[i][j] = 0;
  }

#if 1
  // build object comms
  int obj = 0;
  for (int edge = 0; edge < nodeStats->commData.size(); edge++)
  {
    LDCommData& commData = nodeStats->commData[edge];
    // ensure that the message is not from a processor but from an object
    // and that the type is an object to object message
    if ((!commData.from_proc()) && (commData.recv_type() == LD_OBJ_MSG))
    {
      LDObjKey from = commData.sender;
      LDObjKey to = commData.receiver.get_destObj();
      int fromNode = myNodeId;

      // Check the possible values of lastKnown.
      int toPE = commData.receiver.lastKnown();
      int toNode = toPE / nodeSize;
      // store internal bytes in the last index pos ? -q
      if (fromNode == toNode)
      {
        //        int pos = neighborPos[toNode];
        int nborIdx = SELF_IDX;  // why self id?
        int fromObj = nodeStats->getHash(from);
        int toObj = nodeStats->getHash(to);
        // DEBUGR(("[%d] GRD Load Balancing from obj %d and to obj %d and total objects
        // %d\n", CkMyPe(), fromObj, toObj, nodeStats->n_objs));
        objectComms[fromObj][nborIdx] += commData.bytes;
        // lastKnown PE value can be wrong.
        if (toObj != -1)
        {
          objectComms[toObj][nborIdx] += commData.bytes;
          internalBefore += commData.bytes;
        }
        else
          externalBefore += commData.bytes;
      }
      else
      {  // External communication? - q
        externalBefore += commData.bytes;
        int nborIdx = findNborIdx(toNode);
        if (nborIdx == -1)
          nborIdx = EXT_IDX;  // Store in last index if it is external bytes going to
        //        non-immediate neighbors? -q
        if (fromNode == myNodeId /*peNodes[rank0PE]*/)
        {  // ensure bytes are going from my node? -q
          int fromObj = nodeStats->getHash(from);

          objectComms[fromObj][nborIdx] += commData.bytes;
          obj++;
        }
      }
    }
  }  // end for
#endif

  loadReceivers = 0;
  balanced.resize(toSendLoad.size());
  for (int i = 0; i < toSendLoad.size(); i++)
  {
    balanced[i] = false;
    if (toSendLoad[i] > 0)
    {
      balanced[i] = true;
      loadReceivers++;
    }
  }

  // build obj heap from gain values
  if (loadReceivers > 0)
  {
    if (obj_arr != NULL)
      delete[] obj_arr;

    obj_arr = new int[n_objs];

    // compute gain vals
    for (int i = 0; i < n_objs; i++)
    {
      int sum_bytes = 0;
      // comm bytes with all neighbors
      vector<int> comm_w_nbors = objectComms[i];
      obj_arr[i] = i;
      // compute the sume of bytes of all comms for this obj
      for (int j = 0; j < comm_w_nbors.size(); j++) sum_bytes += comm_w_nbors[j];

      // This gives higher gain value to objects that have within node communication
      gain_val[i] = 2 * objectComms[i][SELF_IDX] - sum_bytes;
    }

    // T1: create a heap based on gain values, and its position also.
    obj_heap.clear();
    heap_pos.clear();
    //      objs.clear();

    obj_heap.resize(n_objs);
    heap_pos.resize(n_objs);
    //     objs.resize(n_objs);
    std::vector<CkVertex> objs_cpy = objs;
    InitializeObjHeap(nodeStats, obj_arr, n_objs, gain_val);

    // T2: Actual load balancingDecide which node it should go, based on object comm data
    // structure. Let node be n
    int v_id;
    double totalSent = 0;
    int counter = 0;
  }

  fflush(stdout);
  int i = 0;

  while (my_loadAfterTransfer > 0)
  {
    double currLoad = objs[i].getVertexLoad();
    // CkPrintf("\n[Node-%d] Can offload object with load %lf", CkMyNode(),
    //          objs[i].getVertexLoad());
    int k = 0;
    for (; k < neighborCount; k++)
    {
      if (toSendLoad[k] > 0)
        break;
    }
    if (k == neighborCount)
      break;
    toSendLoad[k] -= currLoad;
    my_loadAfterTransfer -= currLoad;
    //    int v_id = i;
    int v_id = heap_pop(obj_heap, ObjCompareOperator(&objs, gain_val), heap_pos);
    objs[v_id].setCurrPe(-1);
    int objId = objs[v_id].getVertexId();
    int rank = GetPENumber(objId);
    int node = sendToNeighbors[k];
    int donorPE = rank0PE + rank;

    thisProxy[myNodeId * nodeSize].LoadMetaInfo(nodeStats->objData[v_id].handle,
                                                currLoad);
    thisProxy[donorPE].LoadReceived(objId, node * nodeSize /*CkNodeFirst(node)*/);
    i++;
  }
}

/* Load has been logically sent from overloaded to underloaded nodes in LoadBalance(). Now
 * we should load balance the PE's within the node. This function should only be called by
 * rank0PE.
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
    CkPrintf("--------STARTING WITHIN NODE LB--------\n");

  if (CkMyPe() == rank0PE)
  {
    // CkPrintf("[%d] GRD: DoneNodeLB \n", CkMyPe());
    double avgPE = averagePE();

    // Create a max heap and min heap for pe loads
    vector<double> objectSizes;
    vector<int> objectIds;
    minHeap minPes(nodeSize);
    double threshold = THRESHOLD * avgPE / 100.0;

    // for each pe... find overload, something with prefix sum?
    // and store the underloaded pes
    for (int i = 0; i < nodeSize; i++)
    {
      if (pe_load[i] > avgPE + threshold)
      {
        double overLoad = pe_load[i] - avgPE;
        int start = 0;
        if (i != 0)
        {
          start = prefixObjects[i - 1];
        }
        for (int j = start; j < prefixObjects[i]; j++)
        {
          if (objs[j].getCurrPe() != -1 && objs[j].getVertexLoad() <= overLoad)
          {
            objectSizes.push_back(objs[j].getVertexLoad());
            objectIds.push_back(j);
          }
        }
      }
      else if (pe_load[i] < avgPE - threshold)
      {
        InfoRecord* itemMin = new InfoRecord;
        itemMin->load = pe_load[i];
        itemMin->Id = i;
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
      int objId = maxObj->Id;
      int pe = GetPENumber(objId);
      if (maxObj->load > diff || pe_load[pe] < avgPE - threshold)
      {
        delete maxObj;
        continue;
      }

      thisProxy[pe + rank0PE].LoadReceived(objId, rank0PE + minPE->Id);

      pe_load[minPE->Id] += maxObj->load;
      pe_load[pe] -= maxObj->load;
      if (pe_load[minPE->Id] < avgPE)
      {
        minPE->load = pe_load[minPE->Id];
        minPes.insert(minPE);
      }
      else
        delete minPE;
      minPE = NULL;
    }

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

    // This QD is essential because, before the actual migration starts, load should be
    // divided amongs intra node PE's.
    if (CkMyPe() == 0)
    {
      CkCallback cb(CkIndex_DiffusionLB::ProcessMigrations(), thisProxy);
      CkStartQD(cb);
    }
  }
}

// Create a migrate message for this obj from resident PE to rank0PE
void DiffusionLB::LoadReceived(int objId, int from0PE)
{
  // load is received, hence create a migrate message for the object with id objId.

  MigrateInfo* migrateMe = new MigrateInfo;
  migrateMe->obj = myStats->objData[objId].handle;
  migrateMe->from_pe = CkMyPe();
  migrateMe->to_pe = from0PE;
  // migrateMe->async_arrival = myStats->objData[objId].asyncArrival;
  migrateInfo.push_back(migrateMe);
  total_migrates++;
}

void DiffusionLB::ProcessMigrations()
{
  // SAME AS IN PACKANDSENDMIGRATEMSGS
  LBMigrateMsg* msg = new (total_migrates, CkNumPes(), CkNumPes(), 0) LBMigrateMsg;
  msg->n_moves = total_migrates;
  for (int i = 0; i < total_migrates; i++)
  {
    MigrateInfo* item = (MigrateInfo*)migrateInfo[i];
    msg->moves[i] = *item;
    delete item;
    migrateInfo[i] = 0;
  }
  migrateInfo.clear();

  CkPrintf("PE %d doing %d migrates\n", CkMyPe(), total_migrates);

  // SAME AS IN PROCESSMIGRATIONDECISION
  const int me = CkMyPe();
  for (int i = 0; i < msg->n_moves; i++)
  {
    MigrateInfo& move = msg->moves[i];
    if (move.from_pe == me)
    {
      if (move.to_pe == me)
      {
        CkAbort("[%i] Error, attempting to migrate object myself to myself\n", CkMyPe());
      }
      lbmgr->Migrate(move.obj, move.to_pe);
    }
    else if (move.from_pe != me)
    {
      CkPrintf("[%d] Error, strategy wants to move from %d to  %d\n", me, move.from_pe,
               move.to_pe);
      CkAbort("Trying to move objs not on my PE\n");
    }
  }

  if (CkMyPe() == 0)
  {
    CkCallback cb(CkIndex_DiffusionLB::MigrationDoneWrapper(), thisProxy);
    CkStartQD(cb);
  }
}

void DiffusionLB::CascadingMigration(LDObjHandle h, double load)
{
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
      if (balanced[i] == true && load <= toSendLoad[i] &&
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
      if (toSendLoad[minNode] < threshold && balanced[minNode] == true)
      {
        balanced[minNode] = false;
        loadReceivers--;
      }
      thisProxy[sendToNeighbors[minNode] *
                nodeSize /*CkNodeFirst(sendToNeighbors[minNode])*/]
          .LoadMetaInfo(h, load);
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
}

// When load balancing, remove object handle from your list, since it is about to be
// migrated
/* LoadMetaInfo is called on the receiver with the object that will be migrated to it (via
 * a MigrateMe in  LoadReceived). It is only called when migrating at the node level. Not
 * sure why the receiver would already have this handle though...*/
void DiffusionLB::LoadMetaInfo(LDObjHandle h, double load)
{
  int idx = FindObjectHandle(h);  // if object is in my handles
  if (idx == -1)
  {
    objectHandles.push_back(h);
    objectLoads.push_back(load);
  }
  else
  {
    CascadingMigration(h, load);
    objectHandles[idx] = objectHandles[objectHandles.size() - 1];
    objectLoads[idx] = objectLoads[objectLoads.size() - 1];
    objectHandles.pop_back();
    objectLoads.pop_back();
  }
}

// void DiffusionLB::Migrated(LDObjHandle h, int waitBarrier)
// {
//   if (CkMyPe() == rank0PE)
//   {
//     thisProxy[CkMyPe()].MigratedHelper(h, waitBarrier);
//   }
// }

// void DiffusionLB::MigratedHelper(LDObjHandle h, int waitBarrier)
// {
//   // CkPrintf("[%d] GRD Migrated migrates_completed %d migrates_expected %d \n",
//   CkMyPe(),
//   //          migrates_completed, migrates_expected);
//   int idx = FindObjectHandle(h);
//   if (idx == -1)
//   {
//     objectHandles.push_back(h);
//     objectLoads.push_back(-1);
//   }
//   else
//   {
//     CascadingMigration(h, objectLoads[idx]);
//     objectHandles[idx] = objectHandles[objectHandles.size() - 1];
//     objectLoads[idx] = objectLoads[objectLoads.size() - 1];
//     objectHandles.pop_back();
//     objectLoads.pop_back();
//   }
// }

// void DiffusionLB::PrintDebugMessage(int len, double* result)
// {
//   avgB += result[2];
//   if (result[0] > maxB)
//   {
//     maxB = result[0];
//     maxPEB = (int)result[12];
//   }
//   if (minB == -1 || result[1] < minB)
//   {
//     minB = result[1];
//     minPEB = (int)result[11];
//   }
//   avgA += result[5];
//   if (result[3] > maxA)
//   {
//     maxA = result[3];
//     maxPEA = (int)result[14];
//   }
//   if (minA == -1 || result[4] < minA)
//   {
//     minA = result[4];
//     minPEA = (int)result[13];
//   }
//   internalBeforeFinal += result[6];
//   externalBeforeFinal += result[7];
//   internalAfterFinal += result[8];
//   externalAfterFinal += result[9];
//   migrates += result[10];

//   receivedStats++;
//   CkPrintf("\nPrints on PE %d, receivedStats = %d\n", CkMyPe(), receivedStats);
//   if (receivedStats == numNodes)
//   {
//     receivedStats = 0;
//     avgB = avgB / CkNumPes();
//     avgA = avgA / CkNumPes();
//     CkPrintf("Max PE load before %f(%d), after %f(%d) \n", maxB, maxPEB, maxA, maxPEA);
//     CkPrintf("Min PE load before %f(%d), after %f(%d) \n", minB, minPEB, minA, minPEA);
//     CkPrintf("Avg PE load before %f, after %f \n", avgB, avgA);
//     CkPrintf("Internal Communication before %f, after %f \n", internalBeforeFinal,
//              internalAfterFinal);
//     CkPrintf("External communication before %f, after %f \n", externalBeforeFinal,
//              externalAfterFinal);
//     CkPrintf("Number of migrations across nodes %d \n", migrates);
//     for (int i = 0; i < numNodes; i++)
//     {
//       CkPrintf("\nnodes[%d] = %d", i, i);
//       thisProxy[i /*nodes[i]*/].CallResumeClients();
//     }
//   }
//   fflush(stdout);
// }

// TODO: same as DistBaseLB::MigrationDone
void DiffusionLB::MigrationDoneWrapper()
{
  // ProcessMigrationDecision(msg);
  int balancing = 1;
  MigrationDone(balancing);  // call DistBaseLB version
}

// void DiffusionLB::MigrationDone()
// {
//   lb_started = false;  // TODO: this should remain private to DistBaseLB
// #if CMK_LBDB_ON
//   migrates_completed = 0;
//   total_migrates = 0;
//   migrates_expected = -1;
//   total_migratesActual = -1;
//   avgLoadNeighbor = 0;
//   // myStats->objData = NULL;
//   //  myStats->objData.erase();
//   //  delete[] myStats->objData;
//   //  myStats->commData.erase();
//   //  delete[] myStats->commData;
//   //  myStats->commData = NULL;
//   if (CkMyPe() == 0)
//   {
//     end_lb_time = CkWallTimer();
//     CkPrintf("Strategy Time %f \n", end_lb_time - start_lb_time);
//   }
//   // nodeStats->objData.clear();
//   // nodeStats->commData.clear();
//   /*
//       for(int i = 0; i < nodeSize; i++) {
//           pe_load[i] = 0;
//           numObjects[i] = 0;
//           migratedTo[i] = 0;
//           migratedFrom[i] = 0;
//       }
//   */
//   // Increment to next step
//   finalBalancing = 1;
//   lbmgr->incStep();
//   if (finalBalancing)
//     lbmgr->ClearLoads();

//   // if sync resume invoke a barrier
//   //  if(!_lb_args.debug() || CkMyPe() != rank0PE) {
//   if (finalBalancing && _lb_args.syncResume())
//   {
//     CkPrintf("Resume clients reduction\n");
//     CkCallback cb(CkIndex_DiffusionLB::ResumeClients((CkReductionMsg*)(NULL)),
//     thisProxy); contribute(0, NULL, CkReduction::sum_int, cb);
//   }
//   else
//   {
//     CkPrintf("Calling resume clients directly with finalBalancing %d\n",
//     finalBalancing); thisProxy[CkMyPe()].ResumeClients(finalBalancing);
//   }
// //  }
// #endif
// }

// void DiffusionLB::ResumeClients(CkReductionMsg* msg)
// {
//   ResumeClients(1);
//   delete msg;
// }

// void DiffusionLB::CallResumeClients()
// {
//   CmiAssert(_lb_args.debug());
//   CkPrintf("[%d] GRD: Call Resume clients \n", CkMyPe());
//   thisProxy[CkMyPe()].ResumeClients(finalBalancing);
// }

// // TODO: this is the same function as DistBaseLB::ResumeClients
// void DiffusionLB::ResumeClients(int balancing)
// {
// #if CMK_LBDB_ON

//   if (CkMyPe() == 0 && balancing)
//   {
//     double end_lb_time = CkWallTimer();
//     if (_lb_args.debug())
//       CkPrintf("%s> step %d finished at %f duration %f memory usage: %f\n", lbName(),
//                step() - 1, end_lb_time, end_lb_time /*- strat_start_time*/,
//                CmiMemoryUsage() / (1024.0 * 1024.0));
//   }

//   lbmgr->ResumeClients();
// #endif
// }
#include "DiffusionLB.def.h"
