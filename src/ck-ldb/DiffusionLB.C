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
//#include "LBDBManager.h"
//#include "LBSimulation.h"
#include "elements.h"
#include "Heap_helper.C"
#include "Helper.C"

#define DEBUGF(x) CmiPrintf x;
#define DEBUGR(x) CmiPrintf x;
#define DEBUGL(x) CmiPrintf x;
#define ITERATIONS 40

#define NUM_NEIGHBORS 4

#include "Neighbor_list.C"

// Percentage of error acceptable.
#define THRESHOLD 2

//CreateLBFunc_Def(DiffusionLB, "The distributed graph refinement load balancer")
static void lbinit()
{
  LBRegisterBalancer<DiffusionLB>("DiffusionLB", "The distributed graph refine load balancer");
}

using std::vector;

DiffusionLB::DiffusionLB(const CkLBOptions &opt) : CBase_DiffusionLB(opt) {
  nodeSize = CkNodeSize(0);
  myNodeId = CkMyPe()/nodeSize;
  acks = 0;
  max = 0;
  edgeCount = 0;
  edge_indices.reserve(100);
  round = 0;
  statsReceived = 0;
  rank0_acks = 0;
#if CMK_LBDB_ON
  lbname = "DiffusionLB";
  if (CkMyPe() == 0)
      CkPrintf("[%d] Diffusion created\n",CkMyPe());
  if (_lb_args.statsOn()) lbmgr->CollectStatsOn();
  thisProxy = CProxy_DiffusionLB(thisgroup);
  numNodes = CkNumPes()/nodeSize;//CkNumNodes();
  myStats = new DistBaseLB::LDStats;

  rank0PE = myNodeId*nodeSize;//CkNodeFirst(CkMyNode());
  if(CkMyPe() == rank0PE) {
    statsList = new CLBStatsMsg*[nodeSize];
    nodeStats = new BaseLB::LDStats(nodeSize);
    numObjects.resize(nodeSize);
    prefixObjects.resize(nodeSize);
    migratedTo.resize(nodeSize);
    migratedFrom.resize(nodeSize);
    pe_load.resize(nodeSize);
  }
#endif
}

DiffusionLB::DiffusionLB(CkMigrateMessage *m) : CBase_DiffusionLB(m) {}


DiffusionLB::~DiffusionLB()
{
#if CMK_LBDB_ON
  delete [] statsList;
  delete nodeStats;
  delete myStats;
  delete[] gain_val;
  delete[] obj_arr;
  lbmgr = CProxy_LBManager(_lbmgr).ckLocalBranch();
  if (lbmgr)
    lbmgr->RemoveStartLBFn(startLbFnHdl); 
#endif
}

void DiffusionLB::Strategy(const DistBaseLB::LDStats* const stats) {
  if (CkMyPe() == 0 && _lb_args.debug() >= 1) {
    double start_time = CmiWallTimer();
    CkPrintf("In DiffusionLB strategy at %lf\n", start_time);
  }
  if (CkMyPe() == 0) {
    CkCallback cb(CkIndex_DiffusionLB::LoadBalancing(), thisProxy);
    CkStartQD(cb);
  }
  statsmsg = AssembleStats();
  if(statsmsg == NULL)
    CkPrintf("!!!Its null!!!\n");

  marshmsg = new CkMarshalledCLBStatsMessage(statsmsg);
  thisProxy[rank0PE].ReceiveStats(*marshmsg);
  if(CkMyPe() != rank0PE) {
    CkCallback cb(CkReductionTarget(DiffusionLB, statsAssembled), thisProxy);
    contribute(cb);
  }
}

void DiffusionLB::statsAssembled() {
  if(CkMyPe() == rank0PE)
    findNBors(1);
}

void DiffusionLB::ReceiveStats(CkMarshalledCLBStatsMessage &&data)
{
#if CMK_LBDB_ON
  CLBStatsMsg *m = data.getMessage();

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

void DiffusionLB::startStrategy(){
  if(++rank0_acks < numNodes) return;
  CkPrintf("\nIn startStrategy()");
  for(int i=0;i<numNodes;i++)
    thisProxy[i*nodeSize].diffuse_scalar();
}

double DiffusionLB::avgNborLoad() {
  double sum = 0.0;
  DEBUGL(("\n[PE-%d load = %lf] n[0]=%lf, n[1]=%lf, ncount=%d\n", CkMyPe(), my_load, loadNeighbors[0], loadNeighbors[1], neighborCount));
  for(int i = 0; i < neighborCount; i++)
    sum += loadNeighbors[i];
  return sum/neighborCount;
}

int DiffusionLB::GetPENumber(int& obj_id) {
  int i = 0;
  for(i = 0;i < nodeSize; i++) {
    if(obj_id < prefixObjects[i]) {
      int prevAgg = 0;
      if(i != 0)
          prevAgg = prefixObjects[i-1];
      obj_id = obj_id - prevAgg;
      break;
    }
  }
  return i;
}

bool DiffusionLB::AggregateToSend() {
  bool res = false;
  for(int i = 0; i < neighborCount; i++)
      toSendLoad[i] -= toReceiveLoad[i];
  return res;
}

void DiffusionLB::InitializeObjHeap(BaseLB::LDStats *stats, int* obj_arr,int n,
  int* gain_val) {
  for(int i = 0; i < n; i++) {
    obj_heap[i]=obj_arr[i];
    heap_pos[obj_arr[i]]=i;
  }
  heapify(obj_heap, ObjCompareOperator(&objs, gain_val), heap_pos);
}

void DiffusionLB::PseudoLoadBalancing() {
  double threshold = THRESHOLD * avgLoadNeighbor / 100.0;

  avgLoadNeighbor = (avgLoadNeighbor + my_load) / 2;
  double totalOverload = my_load - avgLoadNeighbor;
  double totalUnderLoad = 0.0;
  double thisIterToSend[neighborCount];
  for (int i = 0; i < neighborCount; i++)
    thisIterToSend[i] = 0.0;
  if (totalOverload > 0)
    for (int i = 0; i < neighborCount; i++)
    {
      if (loadNeighbors[i] < (avgLoadNeighbor - threshold))
      {
        thisIterToSend[i] = avgLoadNeighbor - loadNeighbors[i];
        totalUnderLoad += avgLoadNeighbor - loadNeighbors[i];
        //        DEBUGL2(("[PE-%d] iteration %d thisIterToSend %f avgLoadNeighbor %f loadNeighbors[%d] %f to node %d\n",
        //                thisIndex, itr, thisIterToSend[i], avgLoadNeighbor, i, loadNeighbors[i], sendToNeighbors[i]));
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
      //      DEBUGL2(("[%d] GRD: Pseudo Load Balancing Sending, iteration %d node %d(pe-%d) toSend %lf totalToSend %lf\n", CkMyPe(), itr, sendToNeighbors[i], CkNodeFirst(sendToNeighbors[i]), thisIterToSend[i], (thisIterToSend[i]*totalOverload)/totalUnderLoad));
      thisIterToSend[i] *= totalOverload / totalUnderLoad;
      toSendLoad[i] += thisIterToSend[i];
      if (my_load - thisIterToSend[i] < 0)
        CkAbort("Error: my_load (%f) - thisIterToSend[i] (%f) < 0\n", my_load, thisIterToSend[i]);
      my_load -= thisIterToSend[i];
    }
    if (thisIterToSend[i] < 0.0)
      thisIterToSend[i] = 0.0;
#endif
    int nbor_node = sendToNeighbors[i];
    thisProxy[nbor_node*nodeSize].PseudoLoad(itr, thisIterToSend[i], myNodeId);
  }
}

int DiffusionLB::findNborIdx(int node) {
  for(int i=0;i<sendToNeighbors.size();i++)
    if(sendToNeighbors[i] == node)
      return i;
  return -1;
}

void DiffusionLB::LoadBalancing() {
  if(CkMyPe() != rank0PE) return;
  if (CkMyPe() == 0) {
    CkCallback cb(CkIndex_DiffusionLB::DoneNodeLB(), thisProxy);
    CkStartQD(cb);
  }
  balanced.resize(neighborCount);
  for(int i = 0; i < neighborCount; i++) {
    balanced[i] = false;
    if(toSendLoad[i] > 0) {
      balanced[i] = true;
      actualSend++;
    }
  }
  int n_objs = nodeStats->objData.size();
  CkPrintf("[%d] GRD: Load Balancing w objects size = %d \n", CkMyPe(), n_objs);
  fflush(stdout);
  int i = 0;
  while(my_loadAfterTransfer > 0) {
    double currLoad = objs[i].getVertexLoad();
    CkPrintf("\n[Node-%d] Can offload object with load %lf", CkMyNode(), objs[i].getVertexLoad());
    int k = 0;
    for(;k<neighborCount;k++) {
      if(toSendLoad[k] > 0) break;
    }
    if(k == neighborCount) break;
    toSendLoad[k] -= currLoad;
    my_loadAfterTransfer -= currLoad;
    int v_id = i;
    objs[v_id].setCurrPe(-1);
    int objId = objs[v_id].getVertexId();
    int rank = GetPENumber(objId);
    int node = sendToNeighbors[k];
    int donorPE = rank0PE + rank;
    thisProxy[myNodeId*nodeSize].LoadMetaInfo(nodeStats->objData[v_id].handle, currLoad);
    thisProxy[donorPE].LoadReceived(objId, node*nodeSize/*CkNodeFirst(node)*/);
    i++;
  }
  if (CkMyPe() == 0) {
//    CkCallback cb(CkIndex_DiffusionLB::MigrationEnded(), thisProxy);
//    CkStartQD(cb);
  }
}

// Load is sent from overloaded to underloaded nodes, now we should load balance the PE's within the node
void DiffusionLB::DoneNodeLB() {
  entered = false;
  if(CkMyPe() == rank0PE) {
    CkPrintf("[%d] GRD: DoneNodeLB \n", CkMyPe());
    double avgPE = averagePE();

    // Create a max heap and min heap for pe loads
    vector<double> objectSizes;
    vector<int> objectIds;
    minHeap minPes(nodeSize);
    double threshold = THRESHOLD*avgPE/100.0;
    
    for(int i = 0; i < nodeSize; i++) {
      if(pe_load[i] > avgPE + threshold) {
        DEBUGR(("[%d] GRD: DoneNodeLB rank %d is overloaded with load %f\n", CkMyPe(), i, pe_load[i]));
        double overLoad = pe_load[i] - avgPE;
        int start = 0;
        if(i != 0) {
          start = prefixObjects[i-1];
        }
        for(int j = start; j < prefixObjects[i]; j++) {
          if(objs[j].getCurrPe() != -1 && objs[j].getVertexLoad() <= overLoad) {
            objectSizes.push_back(objs[j].getVertexLoad());
            objectIds.push_back(j);
          }
        } 
      }
      else if(pe_load[i] < avgPE - threshold) {
        DEBUGR(("[%d] GRD: DoneNodeLB rank %d is underloaded with load %f\n", CkMyPe(), i, pe_load[i]));
        InfoRecord* itemMin = new InfoRecord;
        itemMin->load = pe_load[i];
        itemMin->Id = i;
        minPes.insert(itemMin);
      }
    }

    maxHeap objects(objectIds.size());
    for(int i = 0; i < objectIds.size(); i++) {
        InfoRecord* item = new InfoRecord;
        item->load = objectSizes[i];
        item->Id = objectIds[i];
        objects.insert(item); 
    }
    DEBUGR(("[%d] GRD DoneNodeLB: underloaded PE's %d objects which might shift %d \n", CkMyPe(), minPes.numElements(), objects.numElements()));

    InfoRecord* minPE = NULL;
    while(objects.numElements() > 0 && ((minPE == NULL && minPes.numElements() > 0) || minPE != NULL)) {
      InfoRecord* maxObj = objects.deleteMax();
      if(minPE == NULL)
          minPE = minPes.deleteMin();
      double diff = avgPE - minPE->load;
      int objId = maxObj->Id;
      int pe = GetPENumber(objId);
      if(maxObj->load > diff || pe_load[pe] < avgPE - threshold) {
          delete maxObj;
          continue;
      }
      migratedFrom[pe]++;
      CkPrintf("[%d] GRD Intranode: Transfer obj %d (handle %d) from %d of load %f to %d of load %f avg %f and threshold %f \n", CkMyPe(), maxObj->Id, myStats->objData[objId].handle, pe, pe_load[pe], minPE->Id, minPE->load, avgPE, threshold);
      thisProxy[pe + rank0PE].LoadReceived(objId, rank0PE+minPE->Id);

      pe_load[minPE->Id] += maxObj->load;
      migratedTo[minPE->Id]++;
      pe_load[pe] -= maxObj->load;
      if(pe_load[minPE->Id] < avgPE) {
          minPE->load = pe_load[minPE->Id];
          minPes.insert(minPE);
      }
      else
          delete minPE;
      minPE = NULL;
    }
    while(minPes.numElements() > 0) {
      InfoRecord* minPE = minPes.deleteMin();
      delete minPE;
    }
    while(objects.numElements() > 0) {
      InfoRecord* maxObj = objects.deleteMax();
      delete maxObj;
    }

  // This QD is essential because, before the actual migration starts, load should be divided amongs intra node PE's.
    if (CkMyPe() == 0) {
      CkCallback cb(CkIndex_DiffusionLB::MigrationEnded(), thisProxy);
      CkStartQD(cb);
    }
    /*for(int i = 0; i < nodeSize; i++) {
      thisProxy[rank0PE + i].MigrationInfo(migratedTo[i], migratedFrom[i]);
    }*/
  }
}

double DiffusionLB::averagePE() {
  double avg = 0.0;
  for(int i = 0; i < nodeSize; i++)
    avg += pe_load[i];
  avg /= nodeSize;
  return avg;
}

int DiffusionLB::FindObjectHandle(LDObjHandle h) {
  for(int i = 0; i < objectHandles.size(); i++)
    if(objectHandles[i].id == h.id)
      return i;
  return -1;  
}

//Create a migrate message for this obj from resident PE to rank0PE
void DiffusionLB::LoadReceived(int objId, int from0PE) {
  // load is received, hence create a migrate message for the object with id objId.
  if(objId > myStats->objData.size()) {
    DEBUGR(("[%d] GRD: objId %d total objects %d \n", objId, myStats->objData.size()));
    CmiAbort("this object does not exist \n");
  }
  MigrateInfo* migrateMe = new MigrateInfo;
  migrateMe->obj = myStats->objData[objId].handle;
  migrateMe->from_pe = CkMyPe();
  migrateMe->to_pe = from0PE;
  //migrateMe->async_arrival = myStats->objData[objId].asyncArrival;
  migrateInfo.push_back(migrateMe);
  total_migrates++;
  entered = false;
  CkPrintf("[%d] GRD Load Received objId %d (handle %d)  with load %f and toPE %d total_migrates %d total_migratesActual %d migrates_expected %d migrates_completed %d\n", CkMyPe(), objId, myStats->objData[objId].handle,  myStats->objData[objId].wallTime, from0PE, total_migrates, total_migratesActual, migrates_expected, migrates_completed);
}

void DiffusionLB::MigrationEnded() {
    if(CkMyPe()!=rank0PE) return;
    // TODO: not deleted
    entered = true;
    DEBUGR(("[%d] GRD Migration Ended total_migrates %d total_migratesActual %d \n", CkMyPe(), total_migrates, total_migratesActual));
    msg = new(migrateInfo.size()/*total_migrates*/,CkNumPes(),CkNumPes(),0) LBMigrateMsg;
    msg->n_moves = migrateInfo.size();//total_migrates;
    msg->moves = new  MigrateInfo[migrateInfo.size()];
    total_migrates = msg->n_moves;

    for(int i=0; i < total_migrates; i++) {
      MigrateInfo* item = (MigrateInfo*) migrateInfo[i];
      msg->moves[i] = *item;
//      delete item;
//      migrateInfo[i] = 0;
    }
//    migrateInfo.clear();
//    return;
    // Migrate messages from me to elsewhere
    for(int i=0; i < msg->n_moves; i++) {
        MigrateInfo& move = *((MigrateInfo*) migrateInfo[i]);//msg->moves[i];
        const int me = CkMyPe();
        if (move.from_pe == me && move.to_pe != me) {
          
          CkPrintf("\n[PE-%d] Moving obj%d from PE %d to PE %d", me, move.obj.id, move.from_pe, move.to_pe);
	        lbmgr->Migrate(move.obj,move.to_pe);
        } else if (move.from_pe != me) {
	        CkPrintf("[%d] error, strategy wants to move from %d to  %d\n",
		    me,move.from_pe,move.to_pe); fflush(stdout);
        }
    }
    if (CkMyPe() == 0) {
        CkCallback cb(CkIndex_DiffusionLB::MigrationDone(), thisProxy);
        CkStartQD(cb);
    }
}

void DiffusionLB::CascadingMigration(LDObjHandle h, double load) {
    double threshold = THRESHOLD*avgLoadNeighbor/100.0;
    int minNode = 0;
    int myPos = 0;//neighborPos[CkNodeOf(rank0PE)];
    CkPrintf("[%d] GRD Cascading Migration actualSend %d to node %d\n", CkMyPe(), actualSend, nbors[minNode]);
    if(actualSend > 0) {
      double minLoad;
      // Send to max underloaded node
      for(int i = 0; i < neighborCount; i++) {
        if(balanced[i] == true && load <= toSendLoad[i] && (minNode == -1 || minLoad < toSendLoad[i])) {
          minNode = i;
          minLoad = toSendLoad[i];
        }
      }
      if(minNode != -1 && minNode != myPos) {
          // Send load info to receiving load
        toSendLoad[minNode] -= load;
        if(toSendLoad[minNode] < threshold && balanced[minNode] == true) {
          balanced[minNode] = false;
          actualSend--;
        }
        thisProxy[sendToNeighbors[minNode]*nodeSize/*CkNodeFirst(sendToNeighbors[minNode])*/].LoadMetaInfo(h, load);
        lbmgr->Migrate(h, sendToNeighbors[minNode]*nodeSize/*CkNodeFirst(sendToNeighbors[minNode])*/);
      }
    }
    if(actualSend <= 0 || minNode == myPos || minNode == -1) {
      int minRank = -1;
      double minLoad = 0;
      for(int i = 0; i < nodeSize; i++) {
        if(minRank == -1 || pe_load[i] < minLoad) {
          minRank = i;
          minLoad = pe_load[i];
        }
      }
      DEBUGR(("[%d] GRD Cascading Migration actualSend %d sending to rank %d \n", CkMyPe(), actualSend, minRank));
      pe_load[minRank] += load;
      if(minRank > 0) {
        lbmgr->Migrate(h, rank0PE+minRank);
      }
    }
}

//When load balancing, remove object handle from your list, since it is about to be migrated
void DiffusionLB::LoadMetaInfo(LDObjHandle h, double load) {
    int idx = FindObjectHandle(h);
    if(idx == -1) {
        objectHandles.push_back(h);
        objectLoads.push_back(load);
    }
    else {
        CascadingMigration(h, load);
        objectHandles[idx] = objectHandles[objectHandles.size()-1];
        objectLoads[idx] = objectLoads[objectLoads.size()-1];
        objectHandles.pop_back();
        objectLoads.pop_back();
    }
}

void DiffusionLB::Migrated(LDObjHandle h, int waitBarrier)
{
    if(CkMyPe() == rank0PE) {
        thisProxy[CkMyPe()].MigratedHelper(h, waitBarrier);
    }
}

void DiffusionLB::MigratedHelper(LDObjHandle h, int waitBarrier) {
    CkPrintf("[%d] GRD Migrated migrates_completed %d migrates_expected %d \n", CkMyPe(), migrates_completed, migrates_expected);
    int idx = FindObjectHandle(h);
    if(idx == -1) {
        objectHandles.push_back(h);
        objectLoads.push_back(-1);
    }
    else {
        CascadingMigration(h, objectLoads[idx]);
        objectHandles[idx] = objectHandles[objectHandles.size()-1];
        objectLoads[idx] = objectLoads[objectLoads.size()-1];
        objectHandles.pop_back();
        objectLoads.pop_back();
    }
}

void DiffusionLB::PrintDebugMessage(int len, double* result) {
    avgB += result[2];
    if(result[0] > maxB) {
        maxB=result[0];
        maxPEB = (int)result[12];
    }
    if(minB == -1 || result[1] < minB) {
        minB = result[1];
        minPEB = (int)result[11];
    }
    avgA += result[5];
    if(result[3] > maxA) {
        maxA=result[3];
        maxPEA = (int)result[14];
    }
    if(minA == -1 || result[4] < minA) {
        minA = result[4];
        minPEA = (int)result[13];
    }
    internalBeforeFinal += result[6];
    externalBeforeFinal += result[7];
    internalAfterFinal += result[8];
    externalAfterFinal += result[9];
    migrates += result[10];
    
    receivedStats++;
    CkPrintf("\nPrints on PE %d, receivedStats = %d\n", CkMyPe(), receivedStats);
    if(receivedStats == numNodes) {
        receivedStats = 0;
        avgB = avgB /CkNumPes();
        avgA = avgA / CkNumPes();
        CkPrintf("Max PE load before %f(%d), after %f(%d) \n", maxB, maxPEB, maxA, maxPEA);
        CkPrintf("Min PE load before %f(%d), after %f(%d) \n", minB, minPEB, minA, minPEA);
        CkPrintf("Avg PE load before %f, after %f \n", avgB, avgA);
        CkPrintf("Internal Communication before %f, after %f \n", internalBeforeFinal, internalAfterFinal);
        CkPrintf("External communication before %f, after %f \n", externalBeforeFinal, externalAfterFinal);
        CkPrintf("Number of migrations across nodes %d \n", migrates);
        for(int i = 0; i < numNodes; i++) {
          CkPrintf("\nnodes[%d] = %d",i, i);
          thisProxy[i/*nodes[i]*/].CallResumeClients();
        }
    }
    fflush(stdout);
}

void DiffusionLB::MigrationDone() {
    DEBUGR(("[%d] GRD Migration Done \n", CkMyPe()));
#if CMK_LBDB_ON
  migrates_completed = 0;
  total_migrates = 0;
  migrates_expected = -1;
  total_migratesActual = -1;
  avgLoadNeighbor = 0;
  //myStats->objData = NULL;
//  myStats->objData.erase();
//  delete[] myStats->objData;
//  myStats->commData.erase();
//  delete[] myStats->commData;
//  myStats->commData = NULL; 
    if(CkMyPe() == 0) {
        end_lb_time = CkWallTimer();
        CkPrintf("Strategy Time %f \n", end_lb_time - start_lb_time);
    }
    //nodeStats->objData.clear();
    //nodeStats->commData.clear();
/*
    for(int i = 0; i < nodeSize; i++) {
        pe_load[i] = 0;
        numObjects[i] = 0;
        migratedTo[i] = 0;
        migratedFrom[i] = 0;
    }
*/
  // Increment to next step
  lbmgr->incStep();
  if(finalBalancing)
    lbmgr->ClearLoads();

  // if sync resume invoke a barrier
  if(!_lb_args.debug() || CkMyPe() != rank0PE) {
  if (finalBalancing && _lb_args.syncResume()) {
    CkCallback cb(CkIndex_DiffusionLB::ResumeClients((CkReductionMsg*)(NULL)), 
        thisProxy);
    contribute(0, NULL, CkReduction::sum_int, cb);
  }
  else 
    thisProxy [CkMyPe()].ResumeClients(finalBalancing);
  }
#endif
}

void DiffusionLB::ResumeClients(CkReductionMsg *msg) {
  ResumeClients(1);
  delete msg;
}

void DiffusionLB::CallResumeClients() {
    CmiAssert(_lb_args.debug());
    CkPrintf("[%d] GRD: Call Resume clients \n", CkMyPe());
    thisProxy[CkMyPe()].ResumeClients(finalBalancing);
}

void DiffusionLB::ResumeClients(int balancing) {
#if CMK_LBDB_ON

  if (CkMyPe() == 0 && balancing) {
    double end_lb_time = CkWallTimer();
    if (_lb_args.debug())
      CkPrintf("%s> step %d finished at %f duration %f memory usage: %f\n",
          lbName(), step() - 1, end_lb_time, end_lb_time /*- strat_start_time*/,
          CmiMemoryUsage() / (1024.0 * 1024.0));
  }

  lbmgr->ResumeClients();
#endif
}
#include "DiffusionLB.def.h"

