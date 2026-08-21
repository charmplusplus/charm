
/* At the highest level:
  - for each object compute the gain value (for comm, based on communication OUTWARD
    - this changes in new impl
  - while I have neighbors to send to, pick best object

  On completion, waits for QD then calls WITHINNODELB.
*/

#include "DiffusionJSON.h"
void DiffusionLB::AcrossNodeLB()
{
  if (thisIndex != rank0PE)
    return;

  if (thisIndex == 0)
  {
    if (_lb_args.debug() > 1) CkPrintf("--------STARTING ACROSS NODE LB--------\n");
    CkCallback cb(CkIndex_DiffusionLB::WithinNodeLB(), thisProxy);
    CkStartQD(cb);
  }

  if (numNodes == 1)
    return;  // nothing to do

  int n_objs = nodeStats->objData.size();

  gain_val = new int[n_objs];
  memset(gain_val, 100, n_objs);

  // build object comms
  // DiffusionMetric* metric =
  //     new MetricCommEI(nodeStats, myNodeId, nodeSize, neighborCount, toSendLoad);

  DiffusionMetric* metric;
  if (_lb_args.diffusionCommOn())
  {
    metric = new MetricComm(nodeStats, myNodeId, nodeSize, neighborCount, toSendLoad,
                            sendToNeighbors, myNodeInternalBytes, myNodeExternalBytes);
  }
  else
    metric = new MetricCentroid(nborCentroids, nborDistances, myCentroid, nodeStats,
                                myNodeId, toSendLoad, sendToNeighbors, nborObjCount);

  loadReceivers = std::count_if(toSendLoad.begin(), toSendLoad.end(),
                                [](double load) { return load > 0; });

  // TEMPORARY diagnostic: why does across-node diffusion move nothing?
  if (_lb_args.debug() > 1)
  {
    CkPrintf("[node %d] AcrossNodeLB: my_load=%f nbrs=%d loadReceivers=%d n_objs=%d\n",
             myNodeId, my_load, neighborCount, loadReceivers, n_objs);
    for (int i = 0; i < neighborCount; i++)
      CkPrintf("[node %d]   toSendLoad[%d] (-> node %d) = %f\n",
               myNodeId, i, sendToNeighbors[i], toSendLoad[i]);
  }

  // iterate through objects and set from_pe and to_pe correctly
  for (int i = 0; i < n_objs; i++)
  {
    int from = nodeStats->from_proc[i];
    CkAssert(from < numPes && from >= 0);
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
      int nborId = nid;//metric->getBestNeighbor();  // this is buggy (hangs)
      if (nborId == -1)
      {
        CkAbort("Error: no neighbor found to send to, but my_loadAfterTransfer = %f\n",
                my_loadAfterTransfer);
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
        else
          continue;
      }

      // getCompLoad(), not getVertexLoad(): currLoad is decremented from
      // my_loadAfterTransfer, which is in real measured seconds, and it is what the
      // loop's termination test reads. getVertexLoad()'s 0.1s floor would make the
      // loop retire its budget in one unit while testing against another, ending
      // transfers after far fewer objects than intended.
      double currLoad = objs[v_id].getCompLoad();
      objs[v_id].setCurrPe(-1);

      int rank = GetRank(v_id);
      int node = sendToNeighbors[nborId];
      int donorPE = rank0PE + rank;
      int destPE = node * nodeSize;  // send to rank0PE of dest node
      CkAssert(destPE != donorPE);   // if this is hit, our neighbor choice is not working

      if (nodeStats->from_proc[v_id] != donorPE) {
        CkAbort(
            "ERROR: Across Node LB - from_proc[%d] = %d does not match donorPE = %d\n",
            v_id, nodeStats->from_proc[v_id], donorPE);
      }

      my_loadAfterTransfer -= currLoad;
      num_migrations++;

      metric->updateState(v_id, nborId);  // update state to keep track of migrations

      LDObjHandle objHandle = nodeStats->objData[v_id].handle;

      int pe_local_id = v_id;
      if (donorPE != rank0PE) {
        pe_local_id = v_id - prefixObjects[donorPE - rank0PE - 1];
      }

      thisProxy[destPE].LoadMetaInfo(objHandle, pe_local_id, currLoad, donorPE, 0);     
      thisProxy[donorPE].LoadReceived(pe_local_id, destPE);
      nodeStats->to_proc[v_id] = destPE;
    }
  }


}

// When load balancing, remove object handle from your list, since it is about to be
// migrated
/* LoadMetaInfo is called on the receiver with the object that will be migrated to it
 * (via a MigrateMe in  LoadReceived). It is only called when migrating at the node
 * level. Not sure why the receiver would already have this handle though...*/
void DiffusionLB::LoadMetaInfo(LDObjHandle h, int local_id, double load, int senderPE, int only_mcount)
{

  // local_id should be PE local here
  migrates_expected++;
  if(only_mcount)
    return;
  if (thisIndex != rank0PE) {
    CkAbort("Error: LoadMetaInfo called during across node on non-rank0PE %d\n", thisIndex);
  }
  pe_load[0] += load;
  int idx = FindObjectHandle(h);  // if object is in my handles
  if (idx == -1)
  {
    objectHandles.push_back(h);
    objectSrcIds.push_back(local_id);
    objectLoads.push_back(load);
    objSenderPEs.push_back(senderPE);
  }
  else
  {
    CkAbort("Error: LoadMetaInfo called for object handle %d that already exists on PE %d\n",
            h.handle, thisIndex);
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



void DiffusionLB::ProcessFinalStats() {
  if (thisIndex == rank0PE)
  {
    int n_objs = nodeStats->objData.size();
    std::vector<bool> isMigratable(n_objs);
    for (int i = 0; i < n_objs; i++)
    {
      isMigratable[i] = nodeStats->objData[i].migratable;
    }

    std::vector<std::vector<LBRealType>> positions(n_objs);
    std::vector<double> load(n_objs);
    for (int i = 0; i < n_objs; i++)
    {
      // Simulator/dump path (LBSimulation::dumpStep only). Uses the same combined
      // figure as the balancer itself, so a dumped trace reflects the load the
      // decisions were actually made on -- note the receiving side stores it back
      // into wallTime, which is lossy on a CUDA run.
      load[i] = diffusionObjLoad(nodeStats->objData[i]);

      int size = nodeStats->objData[i].position.size();
      positions[i].resize(size);
      for (int j = 0; j < size; j++)
      {
        positions[i][j] = nodeStats->objData[i].position[j];
      }
    }
  thisProxy[0].ReceiveFinalStats(isMigratable, nodeStats->from_proc, nodeStats->to_proc,
                                    nodeStats->n_migrateobjs, positions, load,
                                    nodeStats->commData);

    // Clear nodeStats after sending to avoid accumulation in next round
    nodeStats->objData.clear();
    nodeStats->from_proc.clear();
    nodeStats->to_proc.clear();
    nodeStats->commData.clear();
    nodeStats->n_migrateobjs = 0;
    }

  if (thisIndex == 0)
  {
    CkCallback cb(CkIndex_DiffusionLB::ProcessMigrations(), thisProxy);
    CkStartQD(cb);
  }

}

void DiffusionLB::CollectStats() {

  double load_to_report = 0.0;
  double external_to_report = 0.0;
  double internal_to_report = 0.0;
  double avg_load = 0.0;
  double max_load = 0.0;

  int num_migrations = total_migrates;

  if (thisIndex == rank0PE) {
    for (int i = 0; i < nodeSize; i++) load_to_report += pe_load[i];
    avg_load = load_to_report;
    max_load = std::max_element(pe_load.begin(), pe_load.end())[0];
    external_to_report = myNodeExternalBytes;
    internal_to_report = myNodeInternalBytes;
  }

  
  CkCallback cb_max_load(CkReductionTarget(DiffusionLB, print_max_load), thisProxy[0]);
  contribute(sizeof(double), &max_load, CkReduction::max_double, cb_max_load);

  CkCallback cb_avg_load(CkReductionTarget(DiffusionLB, print_avg_load), thisProxy[0]);
  contribute(sizeof(double), &avg_load, CkReduction::sum_double, cb_avg_load);

  CkCallback cb_external_comm(CkReductionTarget(DiffusionLB, print_external_comm), thisProxy[0]);
  contribute(sizeof(double), &external_to_report, CkReduction::sum_double, cb_external_comm);

  CkCallback cb_internal_comm(CkReductionTarget(DiffusionLB, print_internal_comm), thisProxy[0]);
  contribute(sizeof(double), &internal_to_report, CkReduction::sum_double, cb_internal_comm);

  CkCallback cb_num_migrations(CkReductionTarget(DiffusionLB, print_num_migrations), thisProxy[0]);
  contribute(sizeof(int), &total_crossnode_migrates, CkReduction::sum_int, cb_num_migrations);

  if (thisIndex == 0){
    CkCallback cb(CkIndex_DiffusionLB::ProcessMigrations(), thisProxy);
    CkStartQD(cb);
  }
}

void DiffusionLB::print_max_load(double max){
  CkPrintf("Max load per PE AFTER LB: %f\n", max);
}
void DiffusionLB::print_avg_load(double sum){
    CkPrintf("Avg load per PE AFTER LB: %f\n", sum / numPes);

}
void DiffusionLB::print_num_migrations(int sum){
    CkPrintf("Number of cross node migrations AFTER LB: %d\n", sum);

}
void DiffusionLB::print_external_comm(double sum){
    CkPrintf("External comm BEFORE LB: %f MB\n", sum / (1024 * 1024 * 2));

}
void DiffusionLB::print_internal_comm(double sum){
      CkPrintf("Internal comm BEFORE LB: %f MB\n", sum / (1024 * 1024 * 2));
}


double DiffusionLB::averagePE()
{
  double avg = 0.0;
  for (int i = 0; i < nodeSize; i++) avg += pe_load[i];
  avg /= nodeSize;
  return avg;
}


int DiffusionLB::FindObjectHandle(LDObjHandle h)
{
  for (int i = 0; i < objectHandles.size(); i++)
    if (objectHandles[i].id == h.id)
      return i;
  return -1;
}


// all nodes call this to send final stats to 0. For printing to JSON
// TODO: this is broken rn, because of BaseLB::LDStats pup problems
void DiffusionLB::ReceiveFinalStats(std::vector<bool> isMigratable,
                                    std::vector<int> from_proc, std::vector<int> to_proc,
                                    int n_migrateobjs,
                                    std::vector<std::vector<LBRealType>> positions,
                                    std::vector<double> load,
                                    std::vector<LDCommData> commData)
{
  CkAssert(thisIndex == 0);

  // store the message
  statsReceived++;

  // Clear fullStats at the start of each new round
  if (statsReceived == 1) {
    fullStats->objData.clear();
    fullStats->from_proc.clear();
    fullStats->to_proc.clear();
    fullStats->commData.clear();
    fullStats->n_migrateobjs = 0;
  }

  int oldSize = fullStats->objData.size();

  fullStats->objData.resize(fullStats->objData.size() + isMigratable.size());

  fullStats->n_migrateobjs += n_migrateobjs;

  for (int i = 0; i < isMigratable.size(); i++)
  {
    fullStats->objData[i + oldSize].migratable = isMigratable[i];
    fullStats->objData[i + oldSize].wallTime = load[i];

    int poslen = positions[i].size();
    for (int j = 0; j < poslen; j++)
    {
      fullStats->objData[i + oldSize].position.push_back(positions[i][j]);
    }
  }

  fullStats->from_proc.insert(fullStats->from_proc.end(), from_proc.begin(),
                              from_proc.end());
  fullStats->to_proc.insert(fullStats->to_proc.end(), to_proc.begin(), to_proc.end());

  fullStats->commData.insert(fullStats->commData.end(), commData.begin(),
                              commData.end());

  if (statsReceived == numNodes)
  {
    statsReceived = 0;
    printf("Writing final stats with number of objects: %d\n", fullStats->objData.size());
    writeStatsMsgsJSON(fullStats);
  }
}

