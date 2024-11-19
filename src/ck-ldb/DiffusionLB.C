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
#define DEBUGR(x) /*CmiPrintf x*/;
#define DEBUGL(x) CmiPrintf x;
#define ITERATIONS 4

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
  thisNode = CkMyNode();
  nodeSize = CkNodeSize(thisNode);
  edgeCount = 0;
  edge_indices.reserve(100);
  round = 0;
#if CMK_LBDB_ON
  lbname = "DiffusionLB";
  if (CkMyPe() == 0)
      CkPrintf("[%d] Diffusion created\n",CkMyPe());
  if (_lb_args.statsOn()) lbmgr->CollectStatsOn();
  thisProxy = CProxy_DiffusionLB(thisgroup);
  numNodes = CkNumNodes();
  myStats = new DistBaseLB::LDStats;

  rank0PE = CkNodeFirst(CkMyNode());
  if(CkMyPe() == rank0PE) {
    statsList = new CLBStatsMsg*[nodeSize];
    nodeStats = new BaseLB::LDStats(nodeSize);
    numObjects.resize(nodeSize);
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
  statsmsg = AssembleStats();
  if(statsmsg == NULL)
    CkPrintf("!!!Its null!!!\n");

  marshmsg = new CkMarshalledCLBStatsMessage(statsmsg);
  thisProxy[rank0PE].ReceiveStats(*marshmsg);
  if(CkMyPe() != rank0PE) {
    CkCallback cb(CkReductionTarget(DiffusionLB, findNeighbors), thisProxy);
    contribute(sizeof(int), &do_again, CkReduction::max_int, cb);
  }
}

void DiffusionLB::AddNeighbor(int node) {
  toSend++;
#if DEBUG_K
  CkPrintf("[%d] Send to neighbors node %d\n", CkMyPe(), node);//, nodes[node]);
#endif
  sendToNeighbors.push_back(node);
}


void DiffusionLB::doneNborExng() {
}

void DiffusionLB::ComputeNeighbors() {
}

void DiffusionLB::sortArr(long arr[], int n, int *nbors)
{
 
  vector<std::pair<long, int> > vp;

  // Inserting element in pair vector
  // to keep track of previous indexes
  for (int i = 0; i < n; ++i) {
      vp.push_back(std::make_pair(arr[i], i));
  }

  // Sorting pair vector
  sort(vp.begin(), vp.end());
  int found = 0;
  for(int i=0;i<CkNumNodes();i++)
    if(CkMyNode()!=vp[i].second) //Ideally we shouldn't need to check this
      nbors[found++] = vp[i].second;
  if(found == 0)
    CkPrintf("\nPE-%d Error!!!!!", CkMyPe());
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
    CkCallback cb(CkReductionTarget(DiffusionLB, findNeighbors), thisProxy);
    contribute(sizeof(int), &do_again, CkReduction::max_int, cb);
    statsReceived = 0;
  }
#endif  
}

void DiffusionLB::startStrategy(){
  thisProxy[CkMyPe()].diffuse_scalar();
}

double DiffusionLB::avgNborLoad() {
  double sum = 0.0;
  DEBUGL(("\n[PE-%d load = %lf] n[0]=%lf, n[1]=%lf, ncount=%d\n", CkMyPe(), my_load, loadNeighbors[0], loadNeighbors[1], neighborCount));
  for(int i = 0; i < neighborCount; i++)
    sum += loadNeighbors[i];
  return sum/neighborCount;
}
/*
void Diffusion::ReceiveLoadInfo(double load, int node) {
    DEBUGR(("[%d] GRD Receive load info, load %f node %d loadReceived %d neighborCount %d\n", CkMyPe(), load, node, loadReceived, neighborCount));
    int pos = neighborPos[node];
    loadNeighbors[pos] = load;
    loadReceived++;

    if(loadReceived == neighborCount) {
        loadReceived = 0;     
        avgLoadNeighbor = average();
        DEBUGR(("[%d] GRD Received all loads of node, avg is %f and my_load %f \n", CkMyPe(), avgLoadNeighbor, my_load));
        double threshold = THRESHOLD*avgLoadNeighbor/100.0;
        if(my_load > avgLoadNeighbor + threshold) {
            LoadBalancing();
        }
        else if (CkMyPe() == 0) {
            CkCallback cb(CkIndex_Diffusion::DoneNodeLB(), thisProxy);
            CkStartQD(cb);
        }
    }
}
*/
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
#if 0
  for(int i = 0; i < neighborCount; i++) {
    int node = nbors[i];
    if(neighborPosReceive.find(node) != neighborPosReceive.end()) {
      // One of them will become negative
      int pos = neighborPosReceive[node];
      toSendLoad[i] -= toReceiveLoad[pos];
      if(toSendLoad[i] > 0)
          res= true;
      toReceiveLoad[pos] -= toSendLoad[i];
    }
    CkPrintf("[%d] Diff: To Send load to node %d load %f res %d\n", CkMyPe(), node, toSendLoad[i], res);
    CkPrintf("[%d] Diff: To Send load to node %d load %f res %d myLoadB %f\n", CkMyPe(), node, toSendLoad[i], res, my_loadAfterTransfer);
  }
#endif
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
#if 0
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
    thisProxy[nbor_node].PseudoLoad(itr, 0.0/*thisIterToSend[i]*/, thisIndex);
  }
}

int DiffusionLB::findNborIdx(int node) {
  for(int i=0;i<sendToNeighbors.size();i++)
    if(sendToNeighbors[i] == node)
      return i;
  return -1;
}

#define SELF_IDX NUM_NEIGHBORS
#define EXT_IDX NUM_NEIGHBORS+1
void DiffusionLB::LoadBalancing() {
  int n_objs = nodeStats->objData.size();
  CkPrintf("[%d] GRD: Load Balancing w objects size = %d \n", CkMyPe(), n_objs);

//  Iterate over the comm data and for each object, store its comm bytes
//  to other neighbor nodes and own node.

  //objectComms maintains the comm bytes for each object on this node
  //with the neighboring node
  //we also maintain comm within this node and comm bytes outside
  //(of this node and neighboring nodes)
  vector<vector<int>> objectComms(n_objs);
//  objectComms.resize(n_objs);

  if(gain_val != NULL)
      delete[] gain_val;
  gain_val = new int[n_objs];
  memset(gain_val, -1, n_objs);

  CkPrintf("\n[PE-%d] n_objs=%d", CkMyPe(), n_objs);

  for(int i = 0; i < n_objs; i++) {
//    CkPrintf("\nPE-%d objid= %" PRIu64 ", vrtx id=%d", CkMyPe(), nodeStats->objData[i].objID(), objs[i].getVertexId());
    objectComms[i].resize(neighborCount+1);
    for(int j = 0; j < neighborCount+1; j++)
      objectComms[i][j] = 0;
  }

  // TODO: Set objectComms to zero initially
  int obj = 0;
  for(int edge = 0; edge < nodeStats->commData.size(); edge++) {
    LDCommData &commData = nodeStats->commData[edge];
    // ensure that the message is not from a processor but from an object
    // and that the type is an object to object message
    if( (!commData.from_proc()) && (commData.recv_type()==LD_OBJ_MSG) ) {
      LDObjKey from = commData.sender;
      LDObjKey to = commData.receiver.get_destObj();
      int fromNode = CkMyNode();//peNodes[rank0PE]; //Originating from my node? - q

      // Check the possible values of lastKnown.
      int toPE = commData.receiver.lastKnown();
      int toNode = CkNodeOf(toPE);
      //store internal bytes in the last index pos ? -q
      if(fromNode == toNode) {
//        int pos = neighborPos[toNode];
        int nborIdx = SELF_IDX;// why self id?
        int fromObj = nodeStats->getHash(from);
        int toObj = nodeStats->getHash(to);
        //DEBUGR(("[%d] GRD Load Balancing from obj %d and to obj %d and total objects %d\n", CkMyPe(), fromObj, toObj, nodeStats->n_objs));
        objectComms[fromObj][nborIdx] += commData.bytes;
        // lastKnown PE value can be wrong.
        if(toObj != -1) {
          objectComms[toObj][nborIdx] += commData.bytes; 
          internalBefore += commData.bytes;
        }
        else
          externalBefore += commData.bytes;
      }
      else { // External communication? - q
        externalBefore += commData.bytes;
        int nborIdx = findNborIdx(toNode);
        if(nborIdx == -1)
          nborIdx = EXT_IDX;//Store in last index if it is external bytes going to
//        non-immediate neighbors? -q
        if(fromNode == CkMyNode()/*peNodes[rank0PE]*/) {//ensure bytes are going from my node? -q
          int fromObj = nodeStats->getHash(from);
          CkPrintf("[%d] GRD Load Balancing from obj %d and pos %d\n", CkMyPe(), fromObj, nborIdx);
          objectComms[fromObj][nborIdx] += commData.bytes;
          obj++;
        }
      }
    }
    } // end for

    // calculate the gain value, initialize the heap.
    internalAfter = internalBefore;
    externalAfter = externalBefore;
    double threshold = THRESHOLD*avgLoadNeighbor/100.0;

    actualSend = 0;
    balanced.resize(toSendLoad.size());
    for(int i = 0; i < toSendLoad.size(); i++) {
      balanced[i] = false;
      if(toSendLoad[i] > 0) {
        balanced[i] = true;
        actualSend++;
      }
    }

    if(actualSend > 0) {

      if(obj_arr != NULL)
        delete[] obj_arr;

      obj_arr = new int[n_objs];

      for(int i = 0; i < n_objs; i++) {
        int sum_bytes = 0;
        //comm bytes with all neighbors
        vector<int> comm_w_nbors = objectComms[i];
        obj_arr[i] = i;
        //compute the sume of bytes of all comms for this obj
        for(int j = 0; j < comm_w_nbors.size(); j++)
            sum_bytes += comm_w_nbors[j];

        //This gives higher gain value to objects that have within node communication
        gain_val[i] = 2*objectComms[i][SELF_IDX] - sum_bytes;
      }

      // T1: create a heap based on gain values, and its position also.
      obj_heap.clear();
      heap_pos.clear();
//      objs.clear();

      obj_heap.resize(n_objs);
      heap_pos.resize(n_objs);
 //     objs.resize(n_objs);
      std::vector<CkVertex> objs_cpy = objs;

      //Creating a minheap of objects based on gain value
      InitializeObjHeap(nodeStats, obj_arr, n_objs, gain_val); 

      // T2: Actual load balancingDecide which node it should go, based on object comm data structure. Let node be n
      int v_id;
      double totalSent = 0;
      int counter = 0;
      CkPrintf("\n[PE-%d] my_loadAfterTransfer = %lf, actualSend=%d\n", CkMyPe(),my_loadAfterTransfer,actualSend);

      //return;
      while(my_loadAfterTransfer > 0 && actualSend > 0) {
        counter++;
        //pop the object id with the least gain (i.e least internal comm compared to ext comm)

        if(CkMyPe()==0)
          for(int ii=0;ii<n_objs;ii++)
            CkPrintf("\ngain_val[%d] = %d", ii, gain_val[ii]);

        v_id = heap_pop(obj_heap, ObjCompareOperator(&objs, gain_val), heap_pos);
        
        CkPrintf("\n On PE-%d, popped v_id = %d", CkMyPe(), v_id);
   
        /*If the heap becomes empty*/
        if(v_id==-1)          
            break;
        double currLoad = objs_cpy[v_id].getVertexLoad();
#if 0
        if(!objs[v_id].isMigratable()) {
          CkPrintf("not migratable \n");
          continue;
        }
#endif
        vector<int> comm = objectComms[v_id];
        int maxComm = 0;
        int maxi = -1;
#if 1
        // TODO: Get the object vs communication cost ratio and work accordingly.
        for(int i = 0 ; i < neighborCount; i++) {
            
          // TODO: if not underloaded continue
          if(toSendLoad[i] > 0 && currLoad <= toSendLoad[i]+threshold) {
            if(i!=SELF_IDX && (maxi == -1 || maxComm < comm[i])) {
                maxi = i;
               maxComm = comm[i];
            }
          }
        }
#endif

//        if(CkMyPe()==0)
          CkPrintf("\n[PE-%d] maxi = %d", CkMyPe(), maxi);
          
        if(maxi != -1) {
#if 1
          migrates++;
          int pos = neighborPos[CkNodeOf(rank0PE)];
          internalAfter -= comm[pos];
          internalAfter += comm[maxi];
          externalAfter += comm[pos];
          externalAfter -= comm[maxi];
          int node = nbors[maxi];
          toSendLoad[maxi] -= currLoad;
          if(toSendLoad[maxi] < threshold && balanced[maxi] == true) {
            balanced[maxi] = false;
            actualSend--;
          }
          totalSent += currLoad;
          objs[v_id].setCurrPe(-1); 
          // object Id changes to relative position in PE when passed to function getPENumber.
          int objId = objs_cpy[v_id].getVertexId();
          if(objId != v_id) {
              CkPrintf("\n%d!=%d", objId, v_id);fflush(stdout);
              CmiAbort("objectIds dont match \n");
          }
          int pe = GetPENumber(objId);
          migratedFrom[pe]++;
          int initPE = rank0PE + pe;
          pe_load[pe] -= currLoad;
          numObjects[pe]--;
          CkPrintf("[%d] GRD: Load Balancing object load %f to node %d and from pe %d and objID %d\n", CkMyPe(), currLoad, node, initPE, objId);
          // TODO: Change this to directly send the load to zeroth PE
          //thisProxy[nodes[node]].LoadTransfer(currLoad, initPE, objId);
          thisProxy[CkNodeFirst(CkMyNode())].LoadMetaInfo(nodeStats->objData[v_id].handle, currLoad);
          thisProxy[initPE].LoadReceived(objId, CkNodeFirst(node));
          my_loadAfterTransfer -= currLoad;
          int myPos = 0;//neighborPos[peNodes[rank0PE]];
          loadNeighbors[myPos] -= currLoad;
          loadNeighbors[maxi] += currLoad;   
#endif
        }
        else {
          CkPrintf("[%d] maxi is negative currLoad %f \n", CkMyPe(), currLoad);
        } 
      } //end of while
      CkPrintf("[%d] GRD: Load Balancing total load sent during LoadBalancing %f actualSend %d myloadB %f v_id %d counter %d nobjs %lu \n",
          CkMyPe(), totalSent, actualSend, my_loadAfterTransfer, v_id, counter, nodeStats->objData.size());
      for (int i = 0; i < neighborCount; i++) {
        CkPrintf("[%d] GRD: Load Balancing total load sent during LoadBalancing toSendLoad %f node %d\n", CkMyPe(), toSendLoad[i], nbors[i]);
        }
      }//end of if
      // TODO: Put QD in intra node
      /* Start quiescence detection at PE 0.
      if (CkMyPe() == 0) {
          CkCallback cb(CkIndex_Diffusion::DoneNodeLB(), thisProxy);
          CkStartQD(cb);
      }*/
}

// Load is sent from overloaded to underloaded nodes, now we should load balance the PE's within the node
void DiffusionLB::DoneNodeLB() {
  entered = false;
  return;
  if(CkMyPe() == rank0PE) {
    DEBUGR(("[%d] GRD: DoneNodeLB \n", CkMyPe()));
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
      DEBUGR(("[%d] GRD Intranode: Transfer obj %f from %d of load %f to %d of load %f avg %f and threshold %f \n", CkMyPe(), maxObj->load, pe, pe_load[pe], minPE->Id, minPE->load, avgPE, threshold));
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

void DiffusionLB::LoadReceived(int objId, int fromPE) {
  // load is received, hence create a migrate message for the object with id objId.
  if(objId > myStats->objData.size()) {
    DEBUGR(("[%d] GRD: objId %d total objects %d \n", objId, myStats->objData.size()));
    CmiAbort("this object does not exist \n");
  }
  MigrateInfo* migrateMe = new MigrateInfo;
  migrateMe->obj = myStats->objData[objId].handle;
  migrateMe->from_pe = CkMyPe();
  migrateMe->to_pe = fromPE;
  //migrateMe->async_arrival = myStats->objData[objId].asyncArrival;
  migrateInfo.push_back(migrateMe);
  total_migrates++;
  entered = false;
  CkPrintf("[%d] GRD Load Received objId %d  with load %f and toPE %d total_migrates %d total_migratesActual %d migrates_expected %d migrates_completed %d\n", CkMyPe(), objId, myStats->objData[objId].wallTime, fromPE, total_migrates, total_migratesActual, migrates_expected, migrates_completed);
}

void DiffusionLB::MigrationEnded() {
    // TODO: not deleted
    entered = true;
    DEBUGR(("[%d] GRD Migration Ended total_migrates %d total_migratesActual %d \n", CkMyPe(), total_migrates, total_migratesActual));
    msg = new(total_migrates,CkNumPes(),CkNumPes(),0) LBMigrateMsg;
    msg->n_moves = total_migrates;
    for(int i=0; i < total_migrates; i++) {
      MigrateInfo* item = (MigrateInfo*) migrateInfo[i];
      msg->moves[i] = *item;
      delete item;
      migrateInfo[i] = 0;
    }
    migrateInfo.clear();
    
    // Migrate messages from me to elsewhere
    for(int i=0; i < msg->n_moves; i++) {
        MigrateInfo& move = msg->moves[i];
        const int me = CkMyPe();
        if (move.from_pe == me && move.to_pe != me) {
	        lbmgr->Migrate(move.obj,move.to_pe);
        } else if (move.from_pe != me) {
	        CkPrintf("[%d] error, strategy wants to move from %d to  %d\n",
		    me,move.from_pe,move.to_pe);
        }
    }
    if (CkMyPe() == 0) {
        CkCallback cb(CkIndex_DiffusionLB::MigrationDone(), thisProxy);
        CkStartQD(cb);
    }
}

//What does Cascading migrations do?
void DiffusionLB::CascadingMigration(LDObjHandle h, double load) {
    double threshold = THRESHOLD*avgLoadNeighbor/100.0;
    int minNode = -1;
    int myPos = neighborPos[CkNodeOf(rank0PE)];
    if(actualSend > 0) {
        double minLoad;
        // Send to max underloaded node
        for(int i = 0; i < neighborCount; i++) {
            if(balanced[i] == true && load <= toSendLoad[i] && (minNode == -1 || minLoad < toSendLoad[i])) {
                minNode = i;
                minLoad = toSendLoad[i];
            }
        }
        DEBUGR(("[%d] GRD Cascading Migration actualSend %d to node %d\n", CkMyPe(), actualSend, nbors[minNode]));
        if(minNode != -1 && minNode != myPos) {
            // Send load info to receiving load
            toSendLoad[minNode] -= load;
            if(toSendLoad[minNode] < threshold && balanced[minNode] == true) {
                balanced[minNode] = false;
                actualSend--; 
            }
            thisProxy[CkNodeFirst(nbors[minNode])].LoadMetaInfo(h, load);
	        lbmgr->Migrate(h,CkNodeFirst(nbors[minNode]));
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

//What does this method do? - find out
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
    
    if(CkMyPe() == rank0PE) {
        double minLoadB = pe_loadBefore[0];
        double maxLoadB = pe_loadBefore[0];
        double sumBefore = 0.0;
        double minLoadA = pe_load[0];
        double maxLoadA = pe_load[0];
        double sumAfter = 0.0;
        double maxPEB = rank0PE;
        double maxPEA = rank0PE;
        double minPEB = rank0PE;
        double minPEA = rank0PE;
        if (_lb_args.debug()) {
            for(int i = 0; i < nodeSize; i++) {
                CkPrintf("[%d] GRD: load of PE before: %f after: %f\n",CkMyPe()+i, pe_loadBefore[i], pe_load[i] );
                if(minLoadB > pe_loadBefore[i]) {
                    minLoadB = pe_loadBefore[i];
                    minPEB = rank0PE + i;
                }
                if(maxLoadB < pe_loadBefore[i]) {
                    maxLoadB = pe_loadBefore[i];
                    maxPEB = rank0PE+i;
                }
                sumBefore += pe_loadBefore[i];
                if(minLoadA > pe_load[i]) {
                    minLoadA = pe_load[i];
                    minPEA = rank0PE + i;
                }
                if(maxLoadA < pe_load[i]) {
                    maxLoadA = pe_load[i];
                    maxPEA = rank0PE + i;
                }
                sumAfter += pe_load[i];
            }
            double loads[15];
            loads[0] = maxLoadB;
            loads[1] = minLoadB;
            loads[2] = sumBefore;
            loads[3] = maxLoadA;
            loads[4] = minLoadA;
            loads[5] = sumAfter;
            loads[6] = internalBefore;
            loads[7] = externalBefore;
            loads[8] = internalAfter;
            loads[9] = externalAfter;
            loads[10] = migrates;
            loads[11] = minPEB;
            loads[12] = maxPEB;
            loads[13] = minPEA;
            loads[14] = maxPEA;
            DEBUGL(("[%d] PE's %f %f %f %f \n", CkMyPe(), minPEB, maxPEB, minPEA, maxPEA));
            thisProxy[0].PrintDebugMessage(15, loads);
        }
    
        nodeStats->objData.clear();
        nodeStats->commData.clear();
        for(int i = 0; i < nodeSize; i++) {
            pe_load[i] = 0;
            pe_loadBefore[i] = 0;
            numObjects[i] = 0;
            migratedTo[i] = 0;
            migratedFrom[i] = 0;
        }
    }

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

