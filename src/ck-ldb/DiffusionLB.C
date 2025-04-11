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
#include "elements.h"

#define DEBUGF(x) CmiPrintf x;
#define DEBUGR(x)  // CmiPrintf x;
#define DEBUGL(x) /*CmiPrintf x*/;
#define ITERATIONS 40

#include "DiffusionMetric.C"
#include "DiffusionNeighbors.C"

// Percentage of error acceptable.
#define THRESHOLD 2

#define COMM

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
//  CkPrintf("\n[PE-%d] In Strategy", CkMyPe());
//  fflush(stdout);
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
  statsReceived = 0;
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
//    CkPrintf("\nPE-%d, calling findNBors", CkMyPe());
//    fflush(stdout);
    findNBors(1);
  }
}

void DiffusionLB::startStrategy()
{
  if (++rank0_barrier_counter < numNodes)
    return;

  if (CkMyPe() == 0)
  {
    CkCallback cb(CkIndex_DiffusionLB::AcrossNodeLB(), thisProxy);
    CkStartQD(cb);
  }

  rank0_barrier_counter = 0;
  CkPrintf("--------NEIGHBOR SELECTION COMPLETE--------\n"); fflush(stdout);
  for (int i = 0; i < numNodes; i++) thisProxy[i * nodeSize].pseudolb_rounds();
}

void DiffusionLB::pseudolb_barrier(int allZero)
{
  if (!allZero)
  {
    pseudo_done = false;
  }

  if (++rank0_barrier_counter < numNodes)
    return;

  for (int node = 0; node < numNodes; node++)
  {
    thisProxy[node * nodeSize].pseudoDone(pseudo_done);
  }
  pseudo_done = true;  // set up for next round
  rank0_barrier_counter = 0;
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

/* In combination with the pseudolb_rounds SDAG code, this builds the toReceiveLoad and
 * toSendLoad vectors for each node. It is onlyl called on rank0PEs*/
void DiffusionLB::PseudoLoadBalancing()
{
  std::vector<double> thisRoundToSend(sendToNeighbors.size(), 0.0);

  // create pairs for sorting
  std::vector<std::pair<int, double>> nborPairs;
    for (int i = 0; i < neighborCount; i++)
    {
    nborPairs.push_back(std::make_pair(i, loadNeighbors[i]));
  }

  // sort by load
  std::sort(nborPairs.begin(), nborPairs.end(),
            [](const std::pair<int, double>& a, const std::pair<int, double>& b)
            { return a.second < b.second; });

  // find the neighbors that I should balance with (set such that I am the only one with
  // more load than set average)
  std::vector<std::pair<int, double>> nborsToBalance;

  double currAverage = my_pseudo_load;
  for (std::pair<int, double> p : nborPairs)
  {
    int id = p.first;
    double load = p.second;

    if (load >= currAverage)
    {
      break;
    }

    nborsToBalance.push_back(p);
    currAverage =
        (currAverage * nborsToBalance.size() + load) / (nborsToBalance.size() + 1);
  }

  // balance with neighborstobalance
  double myOverload = my_pseudo_load - currAverage;

  // adjust my overload for what I've already sent out
  double alreadySent = std::accumulate(toSendLoad.begin(), toSendLoad.end(), 0.0,
                                       [](double sum, double value)
                                       { return value > 0 ? sum + value : sum; });

  double leftToSend = my_load - alreadySent;  // my_load is original load
  myOverload = std::min(myOverload, leftToSend);

  for (std::pair<int, double> p : nborsToBalance)
  {
    int id = p.first;
    double load = p.second;

    double trySend = currAverage - load;
    double toSend = 0;

    // exhaust a negative edge first
    if (toSendLoad[id] < 0)
    {
      toSend += std::min(-toSendLoad[id], trySend);
      trySend -= toSend;
    }

    // either edge was enough (trySend == 0)
    // or we need to send more

    if (trySend > 0)
    {
      toSend += std::min(myOverload, trySend);
      trySend -= toSend;
      myOverload -= toSend;
    }

    toSendLoad[id] += toSend;
    thisRoundToSend[id] = toSend;
  }

  bool allZero = true;

  for (int i = 0; i < neighborCount; i++)
  {
    int nbor_node = sendToNeighbors[i];

    if (thisRoundToSend[i] > 0)
    {
      allZero = false;
    }

    my_pseudo_load -= thisRoundToSend[i];
    thisProxy[nbor_node * nodeSize].PseudoLoad(pseudo_itr, thisRoundToSend[i], myNodeId);
  }

  // contribute to reduction to check if round is over
  thisProxy[0].pseudolb_barrier(allZero);

  // double threshold = THRESHOLD * avgLoadNeighbor / 100.0;

  // avgLoadNeighbor = (avgLoadNeighbor + my_pseudo_load) / 2;
  // double totalOverload = my_pseudo_load - avgLoadNeighbor;
  // double totalUnderLoad = 0.0;
  // double thisIterToSend[neighborCount];
  // for (int i = 0; i < neighborCount; i++) thisIterToSend[i] = 0.0;
  // if (totalOverload > 0)
  //   for (int i = 0; i < neighborCount; i++)
  //   {
  //     if (loadNeighbors[i] < (avgLoadNeighbor - threshold))
  //     {
  //       thisIterToSend[i] = avgLoadNeighbor - loadNeighbors[i];
  //       totalUnderLoad += avgLoadNeighbor - loadNeighbors[i];
  //       //        DEBUGL2(("[PE-%d] iteration %d thisIterToSend %f avgLoadNeighbor %f
  //       //        loadNeighbors[%d] %f to node %d\n",
  //       //                thisIndex, itr, thisIterToSend[i], avgLoadNeighbor, i,
  //       //                loadNeighbors[i], sendToNeighbors[i]));
  //     }
  //   }
  // if (totalUnderLoad > 0 && totalOverload > 0 && totalUnderLoad > totalOverload)
  //   totalOverload += threshold;
  // else
  //   totalOverload = totalUnderLoad;

  // for (int i = 0; i < neighborCount; i++)
  // {
  //   if (totalOverload > 0 && totalUnderLoad > 0 && thisIterToSend[i] > 0)
  //   {
  //     //      DEBUGL2(("[%d] GRD: Pseudo Load Balancing Sending, iteration %d node
  //     //      %d(pe-%d) toSend %lf totalToSend %lf\n", CkMyPe(), itr,
  //     //      sendToNeighbors[i], CkNodeFirst(sendToNeighbors[i]), thisIterToSend[i],
  //     //      (thisIterToSend[i]*totalOverload)/totalUnderLoad));
  //     thisIterToSend[i] *= totalOverload / totalUnderLoad;
  //     toSendLoad[i] += thisIterToSend[i];
  //     if (my_pseudo_load - thisIterToSend[i] < 0)
  //       CkAbort("Error: my_pseudo_load (%f) - thisIterToSend[i] (%f) < 0\n",
  //               my_pseudo_load, thisIterToSend[i]);
  //     my_pseudo_load -= thisIterToSend[i];
  //   }
  //   if (thisIterToSend[i] < 0.0)
  //     thisIterToSend[i] = 0.0;
  //   int nbor_node = sendToNeighbors[i];
  //   thisProxy[nbor_node * nodeSize].PseudoLoad(pseudo_itr, thisIterToSend[i],
  //   myNodeId);
  // }
}

/* At the highest level:
  - for each object compute the gain value (for comm, based on communication OUTWARD
    - this changes in new impl
  - while I have neighbors to send to, pick best object

  On completion, waits for QD then calls WITHINNODELB.
*/
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

  int n_objs = nodeStats->objData.size();

  gain_val = new int[n_objs];
  memset(gain_val, 100, n_objs);

  // build object comms
  // DiffusionMetric* metric =
  //     new MetricCommEI(nodeStats, myNodeId, nodeSize, neighborCount, toSendLoad);

#ifdef COMM
  DiffusionMetric* metric = new MetricComm(nodeStats, myNodeId, nodeSize, neighborCount,
                                           toSendLoad, sendToNeighbors);
#else
  DiffusionMetric* metric =
      new MetricCentroid(nborCentroids, nborDistances, myCentroid, nodeStats, myNodeId,
                         toSendLoad, sendToNeighbors, nborObjCount);
#endif

  loadReceivers = std::count_if(toSendLoad.begin(), toSendLoad.end(),
                                [](double load) { return load > 0; });

  // iterate through objects and set from_pe and to_pe correctly
  for (int i = 0; i < n_objs; i++)
  {
    int from = nodeStats->from_proc[i];
    CkAssert(from < CkNumPes() && from >= 0);
    // todo also assert from is on this node?
    nodeStats->to_proc[i] = -1;  // negative one if not migrated
  }

  // build obj heap from gain values
  if (loadReceivers > 0)
  {
    // compute gain vals
    // buildGainValues(n_objs);

    // // T1: create a heap based on gain values, and its position also.
    // InitializeObjHeap(n_objs);
    int tries[neighborCount];
    for (int i = 0; i < neighborCount; i++)
      tries[i] = 0;

    int nid = 0; 
    while (my_loadAfterTransfer > 0)
    {
      nid = (nid + 1)%neighborCount; //change to round robin for now
      int nborId = nid;//metric->getBestNeighbor();  // this is causing cascading???
      if (tries[nborId]==0 && /*nborId == -1 || */toSendLoad[nborId] <= 0)
      {
        tries[nborId] = 1;
        continue;//break;  // no more neighbors to send to
      }

      int v_id = metric->popBestObject(nborId);

      if (v_id == -1)// && nborId==-1)
      {
        tries[nborId] = 1;
        bool not_done = false;
        for(int i = 0; i < neighborCount; i++)
          if(tries[i] == 0)
            not_done = true;
        if(!not_done)
          break;  // no more objects to send
      }

      double currLoad = objs[v_id].getVertexLoad();
      objs[v_id].setCurrPe(-1);
      int objId = objs[v_id].getVertexId();

      int rank = GetPENumber(objId);
      int node = sendToNeighbors[nborId];
      int donorPE = rank0PE + rank;
      int destPE = node * nodeSize;  // send to rank0PE of dest node
      CkAssert(destPE != donorPE);   // if this is hit, our neighbor choice is not working

      if (nodeStats->from_proc[v_id] != donorPE) {
        continue;
        CkPrintf(
            "ERROR: not sure if this is supposed to work, but from_proc[%d] = %d, "
            "donorPE = %d\n",
            v_id, nodeStats->from_proc[v_id], donorPE);
      }

      toSendLoad[nborId] -= currLoad;
      my_loadAfterTransfer -= currLoad;

      metric->updateState(v_id, nborId);  // update state to keep track of migrations

      LDObjHandle objHandle = nodeStats->objData[v_id].handle;
      thisProxy[destPE].LoadMetaInfo(objHandle, objId, currLoad, donorPE);
      thisProxy[donorPE].LoadReceived(objId, destPE);

      nodeStats->to_proc[v_id] = destPE;
    }
  }

  std::vector<bool> isMigratable(n_objs);
  for (int i = 0; i < n_objs; i++)
  {
    isMigratable[i] = nodeStats->objData[i].migratable;
  }

  std::vector<std::vector<LBRealType>> positions(n_objs);
  std::vector<double> load(n_objs);
  for (int i = 0; i < n_objs; i++)
  {
    load[i] = nodeStats->objData[i].wallTime;

    int size = nodeStats->objData[i].position.size();
    positions[i].resize(size);
    for (int j = 0; j < size; j++)
    {
      positions[i][j] = nodeStats->objData[i].position[j];
    }
  }

  thisProxy[0].ReceiveFinalStats(isMigratable, nodeStats->from_proc, nodeStats->to_proc,
                                 nodeStats->n_migrateobjs, positions, load);
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
    CkPrintf("--------STARTING WITHIN NODE LB--------\n");
  
  if(nodeSize==1) {
    if (CkMyPe() == 0)
    {
      CkCallback cb(CkIndex_DiffusionLB::ProcessMigrations(), thisProxy);
      CkStartQD(cb);
    }
    return;
  }
  if (CkMyPe() == rank0PE)
  {
    // CkPrintf("[%d] GRD: DoneNodeLB \n", CkMyPe());
    double avgPE = averagePE();

    // Create a max heap and min heap for pe loads
    vector<double> objectSizes;
    vector<int> objectIds;
    vector<int> objectPEs;
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
          if (objs[j].isMigratable() && objs[j].getCurrPe() != -1 && objs[j].getVertexLoad() <= overLoad)
          {
            objectSizes.push_back(objs[j].getVertexLoad());
            objectIds.push_back(j);
            objectPEs.push_back(GetPENumber(j)+rank0PE);
            overLoad -= objs[j].getVertexLoad();
          }
        }
        if(i==0) {
          //Objects migrating in
          for(int i=0;i<objectLoads.size();i++) {
            if(objectLoads[i] <= overLoad) {
              objectSizes.push_back(objectLoads[i]);
              objectIds.push_back(objectSrcIds[i]/*objHandle*/);
              objectPEs.push_back(objSenderPEs[i]);
            }
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
      item->pe = objectPEs[i]; // sending pe
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

      thisProxy[maxObj->pe/*pe + rank0PE*/].LoadReceived(objId, rank0PE + minPE->Id);

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

    // TODO: submit to print stats
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
  auto it = mig_id_map.find(objId);
  if(it!=mig_id_map.end()) {
    MigrateInfo* migrateMe = it->second;
    CkPrintf("\nUpdating to PE from %d to %d", migrateMe->to_pe, from0PE);
    migrateMe->to_pe = from0PE;
  } else{
    MigrateInfo* migrateMe = new MigrateInfo;
    migrateMe->obj = myStats->objData[objId].handle;
    migrateMe->from_pe = CkMyPe();
    migrateMe->to_pe = from0PE;
    // migrateMe->async_arrival = myStats->objData[objId].asyncArrival;
    migrateInfo.push_back(migrateMe);
    mig_id_map.emplace(objId, migrateMe);
    total_migrates++;
  }
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

  // if we don't do the barrier here, must be done with LBSyncResume so that it is done in
  // MigrationDone
  if (!_lb_args.syncResume())
  {
  // SAME AS IN PROCESSMIGRATIONDECISION
  const int me = CkMyPe();

  for (int i = 0; i < msg->n_moves; i++)
  {
    MigrateInfo& move = msg->moves[i];
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
      CkPrintf("[%d] Error, strategy wants to move from %d to  %d\n", me, move.from_pe,
               move.to_pe);
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
          .LoadMetaInfo(h, 0, load, CkMyPe());
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
/* LoadMetaInfo is called on the receiver with the object that will be migrated to it
 * (via a MigrateMe in  LoadReceived). It is only called when migrating at the node
 * level. Not sure why the receiver would already have this handle though...*/
void DiffusionLB::LoadMetaInfo(LDObjHandle h, int objId, double load, int senderPE)
{
  migrates_expected++;
  pe_load[0] += load;
  int idx = FindObjectHandle(h);  // if object is in my handles
  if (idx == -1)
  {
    objectHandles.push_back(h);
    objectSrcIds.push_back(objId);
    objectLoads.push_back(load);
    objSenderPEs.push_back(senderPE);
  }
  else
  {
#if 0
    CascadingMigration(h, load);
    objectHandles[idx] = objectHandles[objectHandles.size() - 1];
    objectLoads[idx] = objectLoads[objectLoads.size() - 1];
    objectSrcIds[idx] = objectSrcIds[objectSrcIds.size()-1];
    objSenderPEs[idx] = objSenderPEs[objSenderPEs.size()-1];
    objectHandles.pop_back();
    objectLoads.pop_back();
    objectSrcIds.pop_back();
    objSenderPEs.pop_back();
#endif
  }
}

void DiffusionLB::MigrationDoneWrapper()
{
  int balancing = 1;
  MigrationDone(balancing);  // call DistBaseLB version
}

#include "DiffusionLB.def.h"
