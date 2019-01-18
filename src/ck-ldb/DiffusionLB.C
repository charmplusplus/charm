#include "DiffusionLB.h"

#include "ck.h"
#include "ckgraph.h"
#include "envelope.h"
#include "LBDBManager.h"
#include "LBSimulation.h"
#include "elements.h"
#include "Heap_helper.C"
#define DEBUGR(x) /*CmiPrintf x*/;
#define DEBUGL(x) /*CmiPrintf x*/;
#define NUM_NEIGHBORS 2
#define ITERATIONS 8

// Percentage of error acceptable.
#define THRESHOLD 2

CreateLBFunc_Def(DiffusionLB, "The distributed graph refinement load balancer")

using std::vector;
// TODO'S: 
// Topology
// Non migratable objects

void DiffusionLB::staticMigrated(void* data, LDObjHandle h, int waitBarrier)
{
  DiffusionLB *me = (DiffusionLB*)(data);

  me->Migrated(h, waitBarrier);
}

void DiffusionLB::staticAtSync(void* data)
{
  DiffusionLB *me = (DiffusionLB*)(data);

  me->AtSync();
}

DiffusionLB::DiffusionLB(CkMigrateMessage *m) : CBase_DiffusionLB(m) {
}

DiffusionLB::DiffusionLB(const CkLBOptions &opt) : CBase_DiffusionLB(opt) {
#if CMK_LBDB_ON
    lbname = "DiffusionLB";
    if (CkMyPe() == 0)
        CkPrintf("[%d] Diffusion created\n",CkMyPe());
    if (_lb_args.statsOn()) theLbdb->CollectStatsOn();
    InitLB(opt);
#endif
}

DiffusionLB::~DiffusionLB()
{
#if CMK_LBDB_ON
    delete [] statsList;
    delete nodeStats;
    delete [] myStats->objData;
    delete [] myStats->commData;
    delete myStats;
    delete[] gain_val;
    delete[] obj_arr;
    theLbdb = CProxy_LBDatabase(_lbdb).ckLocalBranch();
    if (theLbdb) {
        theLbdb->getLBDB()->
          RemoveNotifyMigrated(notifier);
        theLbdb->
          RemoveStartLBFn((LDStartLBFn)(staticStartLB));
    }
#endif
}

void DiffusionLB::InitLB(const CkLBOptions &opt) {
    thisProxy = CProxy_DiffusionLB(thisgroup);
    receiver = theLbdb->
      AddLocalBarrierReceiver((LDBarrierFn)(staticAtSync),
                (void*)(this));
    notifier = theLbdb->getLBDB()->
      NotifyMigrated((LDMigratedFn)(staticMigrated), (void*)(this));
    startLbFnHdl = theLbdb->getLBDB()->
      AddStartLBFn((LDStartLBFn)(staticStartLB),(void*)(this));
 
    myspeed = theLbdb->ProcessorSpeed();
    // TODO: Initialize all class variables
    loadReceived = 0;
    statsReceived = 0;
    total_migrates = 0;
    total_migratesActual = -1;
    migrates_expected = -1;
    migrates_completed = 0;
    myStats = new DistBaseLB::LDStats;
    myStats->objData = NULL;
    myStats->commData = NULL;
//    ComputeNeighbors();


    nodeSize = CkNumPes()/CkNumNodes();
    nodeFirst = CkMyNode()*nodeSize;

    if(CkMyPe()==nodeFirst){
      int count = 0;

      int nbor =  CkMyNode()-1;

      if(CkMyNode() > 0) {
        neighbors.push_back(nbor);
        neighborPos[count++] = nbor;
      }

      for(int i=0;i<CkNumNodes();i++)
        nodes.push_back(i);

      if(CkMyNode() < CkNumNodes() - 1) {
        int nbor2 =  (CkMyNode()+1)%CkNumNodes();
        neighbors.push_back(nbor2);
      }

      neighborPos[count++] = nbor;
      neighborCount = 2;
      if(CkMyNode() == 0 || CkMyNode() == CkNumNodes()-1)
        neighborCount = 1;
    }
    for(int i=0;i<CkNumPes();i++)
    {
      peNodes[i] = i/nodeSize;
    }
    numNodes = CkNumNodes();

    if(CkMyPe() == nodeFirst) {
        gain_val = NULL;
        obj_arr = NULL;
        my_load = 0;
        my_loadB = 0;
        toSend = 0;
        statsList = new CLBStatsMsg*[nodeSize];
        nodeStats = new BaseLB::LDStats(nodeSize);
        loadPE.resize(nodeSize);
        loadPEBefore.resize(nodeSize);
        numObjects.resize(nodeSize);
        loadNeighbors.resize(neighborCount);
        prefixObjects.resize(nodeSize);
        migratedTo.resize(nodeSize);
        migratedFrom.resize(nodeSize);
        for(int i = 0; i < nodeSize; i++) {
            loadPE[i] = 0;
            loadPEBefore[i] = 0;
            numObjects[i] = 0;
            migratedTo[i] = 0;
            migratedFrom[i] = 0;
            prefixObjects[i] = 0;
        }
    }
}

void DiffusionLB::AtSync() {
#if CMK_LBDB_ON
    if (!QueryBalanceNow(step()) || CkNumPes() == 1) {
        finalBalancing = 0;
        MigrationDone();
        return;
    }
    finalBalancing = 1;
    if(CkMyPe() == 0 && _lb_args.debug()) {
        maxB = 0;
        minB = -1;
        avgB = 0;
        maxA = 0;
        minA = -1;
        avgA = 0;
        receivedStats = 0;
        internalBeforeFinal = 0;
        externalBeforeFinal = 0;
        internalAfterFinal = 0;
        externalAfterFinal = 0;
    }
    internalBefore = 0;
    externalBefore = 0;
    internalAfter = 0;
    externalAfter = 0;
    migrates = 0;
    migratesNode = 0;
    objectHandles.clear();
    objectLoads.clear();
    // TODO: Check is it is the first load balancing step and then only 
    // perform this sending and QD
    if(step() == 0) {
        //sendToNeighbors.clear();
        //toSend = 0;
        if(CkMyPe() == nodeFirst) {
            for(int i = 0; i < neighborCount; i++) {
                int nbor_node = neighbors[i];
                thisProxy[CkMyPe()].AddNeighbor(nbor_node);
            }
        }
        CkCallback cb(CkIndex_DiffusionLB::PEStarted(), thisProxy[0]);
        contribute(cb); 
    }
    else {
        if(CmiNodeAlive(CkMyPe())){
            thisProxy[CkMyPe()].ProcessAtSync();
        }
    }
#endif
}

void DiffusionLB::PEStarted() {
    if(CkMyPe() == 0) {
        CkCallback cb(CkIndex_DiffusionLB::ProcessAtSync(), thisProxy);
        CkStartQD(cb); 
    }
}

void DiffusionLB::AddNeighbor(int node) {
    toSend++;
    DEBUGR(("[%d] Send to neighbors node %d pe %d \n", CkMyPe(), node, nodes[node]));
    sendToNeighbors.push_back(node);
}

void DiffusionLB::ProcessAtSync()
{
#if CMK_LBDB_ON
  start_lb_time = 0;

    if(step() == 0) { 
        toSendLoad.resize(neighborCount);
        toReceiveLoad.resize(sendToNeighbors.size());

        for(int i = 0; i < sendToNeighbors.size(); i++) {
            neighborPosReceive[sendToNeighbors[i]] = i;
            toReceiveLoad[i] = 0;
        }

    }
    for(int i = 0; i < neighborCount; i++)
        toSendLoad[i] = 0;
    for(int i = 0; i < sendToNeighbors.size(); i++) {
        toReceiveLoad[i] = 0;
    }
  if (CkMyPe() == 0) {
    start_lb_time = CkWallTimer();
    if (_lb_args.debug())
      CkPrintf("[%s] Load balancing step %d starting at %f\n",
	       lbName(), step(), CkWallTimer());
  }
    
    // assemble LB database
    CLBStatsMsg* msg = AssembleStats();

    CkMarshalledCLBStatsMessage marshmsg(msg);

    // send to parent
    thisProxy[nodeFirst].ReceiveStats(marshmsg);
#endif
}

void DiffusionLB::ComputeNeighbors() {
    // TODO: Juan's topology aware mapping
//    preprocess(NUM_NEIGHBORS);

    int pdims[4];
    TopoManager_getPeCoordinates(CkMyPe(), pdims);
    std::vector<TCoord*> closest = closest_coords[TCoord(pdims).idx]; 
    TCoord &c = coord_table[TCoord(pdims).idx];
    nodeSize = c.ppn;
    nodeFirst = c.p;
    DEBUGR(("[%d] nodeSize %d and nodeFirst %d \n", CkMyPe(), nodeSize, nodeFirst)); 
    int dist = 0;
    if(CkMyPe() == nodeFirst)
    for(int i = 0; i < closest.size(); i++) {
        if(peNodes.find(closest[i]->p) == peNodes.end())
            CkAbort("all rank 0 pe's not included\n");
        int node = peNodes[closest[i]->p];
        if(neighborPos.find(node) == neighborPos.end()) {
            neighbors.push_back(node);
            neighborPos[neighbors[dist]] = dist;
            dist++;
            CkPrintf("[%d] GRD: neighbor is %d \n", CkMyPe(), node); 
        }
    }
    neighborCount = dist;    
}


// Assembling the stats for the PE
CLBStatsMsg* DiffusionLB::AssembleStats()
{
#if CMK_LBDB_ON
  // build and send stats
#if CMK_LB_CPUTIMER
    theLbdb->TotalTime(&myStats->total_walltime,&myStats->total_cputime);
    theLbdb->BackgroundLoad(&myStats->bg_walltime,&myStats->bg_cputime);
#else
    theLbdb->TotalTime(&myStats->total_walltime,&myStats->total_walltime);
    theLbdb->BackgroundLoad(&myStats->bg_walltime,&myStats->bg_walltime);
#endif
    theLbdb->IdleTime(&myStats->idletime);

    // TODO: myStats->move = QueryMigrateStep(step());

    myStats->n_objs = theLbdb->GetObjDataSz();
    if (myStats->objData != NULL) { CkPrintf("Freeing \n"); delete[] myStats->objData;
    myStats->objData = NULL;}
    myStats->objData = new LDObjData[myStats->n_objs];
    theLbdb->GetObjData(myStats->objData);

    myStats->n_comm = theLbdb->GetCommDataSz();
    if (myStats->commData != NULL) delete[] myStats->commData;
    myStats->commData = new LDCommData[myStats->n_comm];
    theLbdb->GetCommData(myStats->commData);

  const int osz = theLbdb->GetObjDataSz();
  const int csz = theLbdb->GetCommDataSz();

    // TODO: not deleted
  CLBStatsMsg* statsMsg = new CLBStatsMsg(osz, csz);
  statsMsg->from_pe = CkMyPe();

  // Get stats
#if CMK_LB_CPUTIMER
  theLbdb->GetTime(&statsMsg->total_walltime,&statsMsg->total_cputime,
                   &statsMsg->idletime, &statsMsg->bg_walltime,&statsMsg->bg_cputime);
#else
  theLbdb->GetTime(&statsMsg->total_walltime,&statsMsg->total_walltime,
                   &statsMsg->idletime, &statsMsg->bg_walltime,&statsMsg->bg_walltime);
#endif
//  msg->pe_speed = myspeed;
  // number of pes
  statsMsg->pe_speed = myStats->pe_speed;

  statsMsg->n_objs = osz;
  theLbdb->GetObjData(statsMsg->objData);
  statsMsg->n_comm = csz;
  theLbdb->GetCommData(statsMsg->commData);

  if(CkMyPe() == nodeFirst)
    numObjects[0] = osz;
  return statsMsg;
#else
  return NULL;
#endif
}

void DiffusionLB::ReceiveStats(CkMarshalledCLBStatsMessage &data)
{
#if CMK_LBDB_ON
    CLBStatsMsg *m = data.getMessage();
    DEBUGR(("[%d] GRD ReceiveStats from pe %d\n", CkMyPe(), m->from_pe)); 
    CmiAssert(CkMyPe() == nodeFirst);
    // store the message
    int fromRank = m->from_pe - nodeFirst;

    statsReceived++;
    AddToList(m, fromRank);

    if (statsReceived == nodeSize)  
    {
        // build LDStats
        BuildStats();
        statsReceived = 0;
        thisProxy[CkMyPe()].iterate();

        // Graph Refinement: Generate neighbors, Send load to neighbors
    }
#endif  
}

double DiffusionLB::average() {
    double sum = 0;
    DEBUGL(("\n[PE-%d load = %lf] n[0]=%lf, n[1]=%lf, ncount=%d\n", CkMyPe(), my_load, loadNeighbors[0], loadNeighbors[1], neighborCount));
    for(int i = 0; i < neighborCount; i++) {
        sum += loadNeighbors[i];
    }
    // TODO: check the value
    return (sum/neighborCount);
}
/*void Diffusion::ReceiveLoadInfo(double load, int node) {
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
}*/

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
    for(int i = 0; i < neighborCount; i++) {
        int node = neighbors[i];
        if(neighborPosReceive.find(node) != neighborPosReceive.end()) {

            // One of them will become negative
            int pos = neighborPosReceive[node];
            toSendLoad[i] -= toReceiveLoad[pos];
            if(toSendLoad[i] > 0)
                res= true;
            toReceiveLoad[pos] -= toSendLoad[i];
        }
        DEBUGL(("[%d] Diff: To Send load to node %d load %f res %d\n", CkMyPe(), node, toSendLoad[i], res));
        DEBUGR(("[%d] Diff: To Send load to node %d load %f res %d myLoadB %f\n", CkMyPe(), node, toSendLoad[i], res, my_loadB));
    }
    return res;
}

void DiffusionLB::InitializeObjHeap(BaseLB::LDStats *stats, int* obj_arr,int size,
    int* gain_val) {
    for(int obj = 0; obj <size; obj++) {
        obj_heap[obj]=obj_arr[obj];
        heap_pos[obj_arr[obj]]=obj;
    }
    heapify(obj_heap, ObjCompareOperator(&objs, gain_val), heap_pos);
}

void DiffusionLB::PseudoLoadBalancing() {
    DEBUGL(("[PE-%d] Pseudo Load Balancing , iteration %d my_load %f my_loadB %f avgLoadNeighbor %f\n", CkMyPe(), itr, my_load, my_loadB, avgLoadNeighbor));
    double threshold = THRESHOLD*avgLoadNeighbor/100.0;
    
    double totalOverload = my_load - avgLoadNeighbor;
    avgLoadNeighbor = (avgLoadNeighbor+my_load)/2;
    double totalUnderLoad = 0.0;
    double tempToSend[neighborCount];
    for(int i = 0 ;i < neighborCount; i++)
        tempToSend[i] = 0.;
    if(totalOverload > 0)
    for(int i = 0; i < neighborCount; i++) {
        tempToSend[i] = 0;
        if(loadNeighbors[i] < (avgLoadNeighbor - threshold) || (CkMyNode()==0 || CkMyNode() == CkNumNodes()-1)) {
            tempToSend[i] = avgLoadNeighbor - loadNeighbors[i];
            totalUnderLoad += avgLoadNeighbor - loadNeighbors[i];
            DEBUGL(("[PE-%d] iteration %d tempToSend %f avgLoadNeighbor %f loadNeighbors[i] %f to node %d\n",
                      CkMyPe(), itr, tempToSend[i], avgLoadNeighbor, loadNeighbors[i], neighbors[i]));
        }
    }
    if(totalUnderLoad > 0 && totalOverload > 0 && totalUnderLoad > totalOverload)
        totalOverload += threshold;
    else
        totalOverload = totalUnderLoad;
    DEBUGL(("[%d] GRD: Pseudo Load Balancing Sending, iteration %d totalUndeload %f totalOverLoad %f my_loadB %f\n", CkMyPe(), itr, totalUnderLoad, totalOverload, my_loadB));
    for(int i = 0; i < neighborCount; i++) {
        if(totalOverload > 0 && totalUnderLoad > 0 && tempToSend[i] > 0) {
            DEBUGL(("[%d] GRD: Pseudo Load Balancing Sending, iteration %d node %d toSend %lf totalToSend %lf\n", CkMyPe(), itr, neighbors[i], tempToSend[i], (tempToSend[i]*totalOverload)/totalUnderLoad));
            tempToSend[i] = (tempToSend[i]*totalOverload)/totalUnderLoad;
            toSendLoad[i] += tempToSend[i];
        }
        if(my_load - tempToSend[i] < 0)
            CkAbort("Get out");
        my_load -= tempToSend[i];
        int nbor_firstnode = (neighbors[i])*nodeSize;
        thisProxy[nbor_firstnode].PseudoLoad(itr, tempToSend[i], peNodes[nodeFirst]);
    }
}


void DiffusionLB::LoadBalancing() {
    DEBUGL(("[%d] GRD: Load Balancing \n", CkMyPe()));
    // iterate over the comm data and for each object, store its comm bytes to other neighbor nodes and own node.
    vector<vector<int>> objectComms;
    objectComms.resize(n_objs);
    
    if(gain_val != NULL)
        delete[] gain_val;
    gain_val = new int[nodeStats->n_objs];
    memset(gain_val, -1, nodeStats->n_objs);

    for(int i = 0; i < nodeStats->n_objs; i++) {
        objectComms[i].resize(neighborCount+1);
        for(int j = 0; j < neighborCount+1; j++)
            objectComms[i][j] = 0;
    }
    // TODO: Set objectComms to zero initially
    int obj = 0;
    for(int edge = 0; edge < nodeStats->n_comm; edge++) {
        LDCommData &commData = nodeStats->commData[edge];

        // ensure that the message is not from a processor but from an object
        // and that the type is an object to object message
        if( (!commData.from_proc()) && (commData.recv_type()==LD_OBJ_MSG) ) {
          LDObjKey from = commData.sender;
          LDObjKey to = commData.receiver.get_destObj();
          int fromNode = peNodes[nodeFirst];

          // Check the possible values of lastKnown.
          int toPE = commData.receiver.lastKnown();
          int toNode = toPE/nodeSize;
          if(fromNode == toNode) {
            int pos = neighborPos[toNode];
            pos = 0;
            int fromObj = nodeStats->getHash(from);
            int toObj = nodeStats->getHash(to);
            //DEBUGR(("[%d] GRD Load Balancing from obj %d and to obj %d and total objects %d\n", CkMyPe(), fromObj, toObj, nodeStats->n_objs));
            objectComms[fromObj][pos] += commData.bytes;
            // lastKnown PE value can be wrong.
            if(toObj != -1) {
                objectComms[toObj][pos] += commData.bytes; 
                internalBefore += commData.bytes;
            }
            else
                externalBefore += commData.bytes;
          }
          else {
            externalBefore += commData.bytes;
            int pos;
            if(neighborPos.find(toNode) != neighborPos.end()) {
                pos = neighborPos[toNode];
            }
            else {
                pos = neighborCount;
            }
            pos = 0;
            if(fromNode == peNodes[nodeFirst]) {
                int fromObj = nodeStats->getHash(from);
                //DEBUGR(("[%d] GRD Load Balancing from obj %d and pos %d\n", CkMyPe(), fromObj, pos));
                objectComms[fromObj][pos] += commData.bytes;
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
      obj_arr = new int[nodeStats->n_objs];
      for(int i = 0; i < nodeStats->n_objs; i++) {
        int sum = 0;
        vector<int> bytes = objectComms[i];
        int pos = neighborPos[peNodes[nodeFirst]];
        obj_arr[i] = i;
        for(int j = 0; j < bytes.size(); j++) {
            sum += objectComms[i][j];
        }
        gain_val[i] = 2*objectComms[i][pos] - sum;
      }

        // T1: create a heap based on gain values, and its position also.
      obj_heap.clear();
      heap_pos.clear();

        obj_heap.resize(nodeStats->n_objs);
        heap_pos.resize(nodeStats->n_objs);

        InitializeObjHeap(nodeStats, obj_arr, nodeStats->n_objs, gain_val); 

        // T2: Actual load balancingDecide which node it should go, based on object comm data structure. Let node be n
        int v_id;
        double totalSent = 0;
        int counter = 0;
        while(my_loadB > 0 && actualSend > 0) {

            counter++; 
            v_id = heap_pop(obj_heap, ObjCompareOperator(&objs, gain_val), heap_pos);
                
            /*If the heap becomes empty*/
            if(v_id==-1)          
                break;
            
            double currLoad = objs[v_id].getVertexLoad();
            if(!objs[v_id].isMigratable()) {
                DEBUGR(("not migratable \n"));
                continue;
            }
            
            vector<int> comm = objectComms[v_id];
            int maxComm = 0;
            int maxi = -1;
            
            // TODO: Get the object vs communication cost ratio and work accordingly.
            for(int i = 0 ; i < neighborCount; i++) {
                
                // TODO: if not underloaded continue
                if(toSendLoad[i] > 0 && currLoad <= toSendLoad[i]+threshold) {
                    if(neighborPos[peNodes[nodeFirst]] != i && (maxi == -1 || maxComm < comm[i])) {
                        maxi = i;
                       maxComm = comm[i];
                    }
                }
            }
            
            if(maxi != -1) {
                migrates++;
                int pos = neighborPos[peNodes[nodeFirst]];
                internalAfter -= comm[pos];
                internalAfter += comm[maxi];
                externalAfter += comm[pos];
                externalAfter -= comm[maxi];
                int node = neighbors[maxi];
                toSendLoad[maxi] -= currLoad;
                if(toSendLoad[maxi] < threshold && balanced[maxi] == true) {
                    balanced[maxi] = false;
                    actualSend--;
                }
                totalSent += currLoad;
                objs[v_id].setCurrPe(-1); 
                // object Id changes to relative position in PE when passed to function getPENumber.
                int objId = objs[v_id].getVertexId();
                if(objId != v_id)
                    CmiAbort("objectIds dont match \n");
                int pe = GetPENumber(objId);
                migratedFrom[pe]++;
                int initPE = nodeFirst + pe;
                loadPE[pe] -= currLoad;
                numObjects[pe]--;
                DEBUGR(("[%d] GRD: Load Balancing object load %f to node %d and from pe %d and objID %d\n", CkMyPe(), currLoad, node, initPE, objId));
                // TODO: Change this to directly send the load to zeroth PE
                //thisProxy[nodes[node]].LoadTransfer(currLoad, initPE, objId);
                thisProxy[nodes[node]].LoadMetaInfo(nodeStats->objData[v_id].handle, currLoad);
                thisProxy[initPE].LoadReceived(objId, nodes[node]);
                my_loadB -= currLoad;
                int myPos = neighborPos[peNodes[nodeFirst]];
                loadNeighbors[myPos] -= currLoad;
                loadNeighbors[maxi] += currLoad;   
            }
            else {
                DEBUGR(("[%d] maxi is negative currLoad %f \n", CkMyPe(), currLoad));
            } 
        }
            DEBUGR(("[%d] GRD: Load Balancing total load sent during LoadBalancing %f actualSend %d myloadB %f v_id %d counter %d nobjs %d \n", CkMyPe(), totalSent, actualSend, my_loadB, v_id, counter, nodeStats->n_objs));
            for (int i = 0; i < neighborCount; i++) {
                DEBUGR(("[%d] GRD: Load Balancing total load sent during LoadBalancing toSendLoad %f node %d\n", CkMyPe(), toSendLoad[i], nodes[neighbors[i]]));
            }
        }

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
    if(CkMyPe() == nodeFirst) {
        DEBUGR(("[%d] GRD: DoneNodeLB \n", CkMyPe()));
        double avgPE = averagePE();

        // Create a max heap and min heap for pe loads
        vector<double> objectSizes;
        vector<int> objectIds;
        minHeap minPes(nodeSize);
        double threshold = THRESHOLD*avgPE/100.0;
        
        for(int i = 0; i < nodeSize; i++) {
            if(loadPE[i] > avgPE + threshold) {
                DEBUGR(("[%d] GRD: DoneNodeLB rank %d is overloaded with load %f\n", CkMyPe(), i, loadPE[i]));
                double overLoad = loadPE[i] - avgPE;
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
            else if(loadPE[i] < avgPE - threshold) {
                DEBUGR(("[%d] GRD: DoneNodeLB rank %d is underloaded with load %f\n", CkMyPe(), i, loadPE[i]));
                InfoRecord* itemMin = new InfoRecord;
                itemMin->load = loadPE[i];
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
            if(maxObj->load > diff || loadPE[pe] < avgPE - threshold) {
                delete maxObj;
                continue;
            }
            migratedFrom[pe]++;
            DEBUGR(("[%d] GRD Intranode: Transfer obj %f from %d of load %f to %d of load %f avg %f and threshold %f \n", CkMyPe(), maxObj->load, pe, loadPE[pe], minPE->Id, minPE->load, avgPE, threshold));
            thisProxy[pe + nodeFirst].LoadReceived(objId, nodeFirst+minPE->Id);

            loadPE[minPE->Id] += maxObj->load;
            migratedTo[minPE->Id]++;
            loadPE[pe] -= maxObj->load;
            if(loadPE[minPE->Id] < avgPE) {
                minPE->load = loadPE[minPE->Id];
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
            thisProxy[nodeFirst + i].MigrationInfo(migratedTo[i], migratedFrom[i]);
        }*/
    }
}

double DiffusionLB::averagePE() {
    int size = nodeSize;
    double sum = 0;
    for(int i = 0; i < size; i++) {
        sum += loadPE[i];
    }
    return (sum/(size*1.0)); 
}

int DiffusionLB::FindObjectHandle(LDObjHandle h) {
    for(int i = 0; i < objectHandles.size(); i++)
        if(objectHandles[i].id == h.id)
            return i;
    return -1;  
}

void DiffusionLB::LoadReceived(int objId, int fromPE) {
    // load is received, hence create a migrate message for the object with id objId.
    if(objId > myStats->n_objs) {
        DEBUGR(("[%d] GRD: objId %d total objects %d \n", objId, myStats->n_objs));
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
    DEBUGR(("[%d] GRD Load Received objId %d  with load %f and toPE %d total_migrates %d total_migratesActual %d migrates_expected %d migrates_completed %d\n", CkMyPe(), objId, myStats->objData[objId].wallTime, fromPE, total_migrates, total_migratesActual, migrates_expected, migrates_completed));
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
	        theLbdb->Migrate(move.obj,move.to_pe);
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

void DiffusionLB::CascadingMigration(LDObjHandle h, double load) {
    double threshold = THRESHOLD*avgLoadNeighbor/100.0;
    int minNode = -1;
    int myPos = neighborPos[peNodes[nodeFirst]];
    if(actualSend > 0) {
        double minLoad;
        // Send to max underloaded node
        for(int i = 0; i < neighbors.size(); i++) {
            if(balanced[i] == true && load <= toSendLoad[i] && (minNode == -1 || minLoad < toSendLoad[i])) {
                minNode = i;
                minLoad = toSendLoad[i];
            }
        }
        DEBUGR(("[%d] GRD Cascading Migration actualSend %d to node %d\n", CkMyPe(), actualSend, neighbors[minNode]));
        if(minNode != -1 && minNode != myPos) {
            // Send load info to receiving load
            toSendLoad[minNode] -= load;
            if(toSendLoad[minNode] < threshold && balanced[minNode] == true) {
                balanced[minNode] = false;
                actualSend--; 
            }
            thisProxy[nodes[neighbors[minNode]]].LoadMetaInfo(h, load);
	        theLbdb->Migrate(h,nodes[neighbors[minNode]]);
        }
            
    }
    if(actualSend <= 0 || minNode == myPos || minNode == -1) {
        int minRank = -1;
        double minLoad = 0;
        for(int i = 0; i < nodeSize; i++) {
            if(minRank == -1 || loadPE[i] < minLoad) {
                minRank = i;
                minLoad = loadPE[i];
            }
        }
        DEBUGR(("[%d] GRD Cascading Migration actualSend %d sending to rank %d \n", CkMyPe(), actualSend, minRank));
        loadPE[minRank] += load;
        if(minRank > 0) {
	        theLbdb->Migrate(h, nodeFirst+minRank);
        }
    }
}

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
    if(CkMyPe() == nodeFirst) {
        thisProxy[CkMyPe()].MigratedHelper(h, waitBarrier);
    }
}

void DiffusionLB::MigratedHelper(LDObjHandle h, int waitBarrier) {
    DEBUGR(("[%d] GRD Migrated migrates_completed %d migrates_expected %d \n", CkMyPe(), migrates_completed, migrates_expected));
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
        for(int i = 0; i < numNodes; i++) 
            thisProxy[(i*nodeSize)].CallResumeClients();
    }
}

void DiffusionLB::MigrationDone() {
    DEBUGR(("[%d] GRD Migration Done \n", CkMyPe()));
#if CMK_LBDB_ON
  migrates_completed = 0;
  total_migrates = 0;
  migrates_expected = -1;
  total_migratesActual = -1;
  avgLoadNeighbor = 0;
  delete[] myStats->objData;
  myStats->objData = NULL;
  delete[] myStats->commData;
  myStats->commData = NULL; 
    if(CkMyPe() == 0) {
        end_lb_time = CkWallTimer();
        CkPrintf("Strategy Time %f \n", end_lb_time - start_lb_time);
    }
    
    if(CkMyPe() == nodeFirst) {
        double minLoadB = loadPEBefore[0];
        double maxLoadB = loadPEBefore[0];
        double sumBefore = 0.0;
        double minLoadA = loadPE[0];
        double maxLoadA = loadPE[0];
        double sumAfter = 0.0;
        double maxPEB = nodeFirst;
        double maxPEA = nodeFirst;
        double minPEB = nodeFirst;
        double minPEA = nodeFirst;
        if (_lb_args.debug()) {
            for(int i = 0; i < nodeSize; i++) {
                //CkPrintf("[%d] GRD: load of PE before: %f after: %f\n",CkMyPe()+i, loadPEBefore[i], loadPE[i] );
                if(minLoadB > loadPEBefore[i]) {
                    minLoadB = loadPEBefore[i];
                    minPEB = nodeFirst + i;
                }
                if(maxLoadB < loadPEBefore[i]) {
                    maxLoadB = loadPEBefore[i];
                    maxPEB = nodeFirst+i;
                }
                sumBefore += loadPEBefore[i];
                if(minLoadA > loadPE[i]) {
                    minLoadA = loadPE[i];
                    minPEA = nodeFirst + i;
                }
                if(maxLoadA < loadPE[i]) {
                    maxLoadA = loadPE[i];
                    maxPEA = nodeFirst + i;
                }
                sumAfter += loadPE[i];
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
    
        nodeStats->n_objs = 0;
        nodeStats->n_comm = 0;
        for(int i = 0; i < nodeSize; i++) {
            loadPE[i] = 0;
            loadPEBefore[i] = 0;
            numObjects[i] = 0;
            migratedTo[i] = 0;
            migratedFrom[i] = 0;
        }
    }

  // Increment to next step
  theLbdb->incStep();
  if(finalBalancing)
    theLbdb->ClearLoads();

  // if sync resume invoke a barrier
  if(!_lb_args.debug() || CkMyPe() != nodeFirst) {
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
    DEBUGR(("[%d] GRD: Call Resume clients \n", CkMyPe()));
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

  theLbdb->ResumeClients();
#endif
}

// Aggregates the stats messages of PE into LDStats, Computes total load of node
void DiffusionLB::BuildStats()
{
    DEBUGR(("[%d] GRD Build Stats  and objects %d\n", CkMyPe(), nodeStats->n_objs));

    n_objs = nodeStats->n_objs;
    nodeStats->nprocs() = statsReceived;
    // allocate space
    nodeStats->objData.clear();
    nodeStats->from_proc.clear();
    nodeStats->to_proc.clear();
    nodeStats->commData.clear();
    int prev = 0;
    for(int i = 0; i < nodeSize; i++) {
        prefixObjects[i] = prev + numObjects[i];
        prev = prefixObjects[i];
    }

    nodeStats->objData.resize(nodeStats->n_objs);
    nodeStats->from_proc.resize(nodeStats->n_objs);
    nodeStats->to_proc.resize(nodeStats->n_objs);
    nodeStats->commData.resize(nodeStats->n_comm);
    objs.clear();
    objs.resize(nodeStats->n_objs);
       
    /*if(nodeKeys != NULL)
        delete[] nodeKeys;
    nodeKeys = new LDObjKey[nodeStats->n_objs];*/

    int nobj = 0;
    int ncom = 0;
    int nmigobj = 0;
    int start = nodeFirst;
    my_load = 0;
    my_loadB = 0;
    
    // copy all data in individual message to this big structure
    for (int pe=0; pe<statsReceived; pe++) {
        int i;
        CLBStatsMsg *msg = statsList[pe];
        if(msg == NULL) continue;
        for (i = 0; i < msg->n_objs; i++) {
            nodeStats->from_proc[nobj] = nodeStats->to_proc[nobj] = start + pe;
            nodeStats->objData[nobj] = msg->objData[i];
            LDObjData &oData = nodeStats->objData[nobj];
            objs[nobj] = Vertex(nobj, oData.wallTime, nodeStats->objData[nobj].migratable, nodeStats->from_proc[nobj]);
            my_load += msg->objData[i].wallTime;
            loadPE[pe] += msg->objData[i].wallTime;
            loadPEBefore[pe] += msg->objData[i].wallTime;
            /*TODO Keys LDObjKey key;
            key.omID() = msg->objData[i].handle.omID;
            key.objID() =  msg->objData[i].handle.objID;
            nodeKeys[nobj] = key;*/
            if (msg->objData[i].migratable) 
                nmigobj++;
	        nobj++;
        }
        DEBUGR(("[%d] GRD BuildStats load of rank %d is %f \n", CkMyPe(), pe, loadPE[pe]));
        for (i = 0; i < msg->n_comm; i++) {
            nodeStats->commData[ncom] = msg->commData[i];
            //nodeStats->commData[ncom].receiver.dest.destObj.destObjProc = msg->commData[i].receiver.dest.destObj.destObjProc; 
            int dest_pe = nodeStats->commData[ncom].receiver.lastKnown();
            //CkPrintf("\n here dest_pe = %d\n", dest_pe);
            ncom++;
        }
        // free the memory TODO: Free the memory in Destructor
        delete msg;
        statsList[pe]=0;
    }
    my_loadB = my_load;
    nodeStats->n_migrateobjs = nmigobj;

    // Generate a hash with key object id, value index in objs vector
    nodeStats->deleteCommHash();
    nodeStats->makeCommHash();
}

void DiffusionLB::AddToList(CLBStatsMsg* m, int rank) {
    DEBUGR(("[%d] GRD Add To List num objects %d from rank %d load %f\n", CkMyPe(), m->n_objs, rank, m->total_walltime));
    nodeStats->n_objs += m->n_objs;
    nodeStats->n_comm += m->n_comm;
    numObjects[rank] = m->n_objs;
    statsList[rank] = m;
    
    struct ProcStats &procStat = nodeStats->procs[rank];
    procStat.pe = CkMyPe() + rank;	// real PE
    procStat.total_walltime = m->total_walltime;
    procStat.idletime = m->idletime;
    procStat.bg_walltime = m->bg_walltime;
    #if CMK_LB_CPUTIMER
    procStat.total_cputime = m->total_cputime;
    procStat.bg_cputime = m->bg_cputime;
    #endif
    procStat.pe_speed = m->pe_speed;		// important
    procStat.available = true;
    procStat.n_objs = m->n_objs;
}
#include "DiffusionLB.def.h"

