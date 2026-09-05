
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
  }
  // WithinNodeLB now starts from the acrossDone barrier below, once every
  // handoff this phase issued has been acked as processed.

  if (numNodes == 1)
  {
    across_owed = true;
    migMaybeDone();
    return;  // nothing to do
  }

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
                            sendToNeighbors, myNodeInternalBytes, myNodeExternalBytes,
                            &diffusionCostCfg);
  }
  else
    metric = new MetricCentroid(nborCentroids, nborDistances, myCentroid, nodeStats,
                                myNodeId, toSendLoad, sendToNeighbors, nborObjCount);

  loadReceivers = std::count_if(toSendLoad.begin(), toSendLoad.end(),
                                [](double load) { return load > 0; });

  // Shed the EXCESS, not everything. my_loadAfterTransfer was initialised in
  // BuildStats to this node's TOTAL load, so the loop below ran until the
  // neighbours ran out of capacity rather than until this node reached its
  // fair share -- a node would hand over nearly everything it held, one object
  // at a time. The per-neighbour quotas in toSendLoad cap each RECIPIENT, but
  // nothing capped the donor, which is why object counts per PE still spread
  // 1..89 even once the geometry was constrained.
  {
    const double fair = avgNborLoad();
    const double excess = my_load - fair;
    my_loadAfterTransfer = (excess > 0.0) ? excess : 0.0;
    if (_lb_args.debug() > 1)
      CkPrintf("[node %d] AcrossNodeLB: my_load=%f fair=%f shedding=%f\n",
               myNodeId, my_load, fair, my_loadAfterTransfer);
  }

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

      // What this node still owes, so the metric can cap a candidate's benefit:
      // load shed beyond the fair share buys nothing and must not pay for a
      // move. Refreshed every iteration because each accepted move reduces it.
      metric->setRemainingShed(my_loadAfterTransfer);

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

      // Two different figures, because two different consumers.
      //
      // shedLoad is in the diffused dimension: it retires this node's obligation
      // (my_loadAfterTransfer) and the per-neighbour quota, both of which the
      // pseudo-LB rounds expressed in that dimension. Using anything else would
      // retire the budget in different units from the ones it was computed in.
      //
      // cpuLoad is host time, and is what travels in the message: the receiver adds
      // it to pe_load and hands it to the within-node heap, which balances host work
      // between PEs that share a device. Shipping GPU time would corrupt that.
      //
      // Both read getCompLoad()/objData rather than getVertexLoad(), whose
      // MAX(compLoad, 0.1) floor would retire the budget in yet another unit.
      const double shedLoad = diffusionObjLoad(nodeStats->objData[v_id]);
      const double cpuLoad  = objs[v_id].getCompLoad();
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

      my_loadAfterTransfer -= shedLoad;
      num_migrations++;

      metric->updateState(v_id, nborId);  // update state to keep track of migrations

      LDObjHandle objHandle = nodeStats->objData[v_id].handle;

      int pe_local_id = v_id;
      if (donorPE != rank0PE) {
        pe_local_id = v_id - prefixObjects[donorPE - rank0PE - 1];
      }

      mig_acksOut += 2;
      thisProxy[destPE].LoadMetaInfo(objHandle, pe_local_id, cpuLoad, donorPE, 0, CkMyPe());
      thisProxy[donorPE].LoadReceived(pe_local_id, destPE, CkMyPe());
      nodeStats->to_proc[v_id] = destPE;
    }
  }

  // Says which of the two reasons a quiet step had: nothing available to move,
  // or everything available priced out. Without this the two are
  // indistinguishable from the outside, and they call for opposite responses.
  if (_lb_args.debug() > 1 && metric != NULL)
    CkPrintf("[node %d] AcrossNodeLB: %d move(s) accepted, %d neighbour(s) with "
             "no move worth making, %.6f load left unshed\n",
             myNodeId, metric->acceptedCount(), metric->rejectedCount(),
             my_loadAfterTransfer > 0 ? my_loadAfterTransfer : 0.0);

  // Owned by this function since it was created here; the per-object vectors
  // inside it are sized by the node's object count, so leaking one per node per
  // balancer step is not free.
  delete metric;
  metric = NULL;

  across_owed = true;
  migMaybeDone();
}

// The phase-completion machinery shared by the across- and within-node
// handoffs. A rank0PE holds its barrier contribution until every
// LoadMetaInfo/LoadReceived it sent has been acked by its receiver's handler,
// so the barrier on PE 0 completes only when every migration record this
// phase created is in place everywhere. The phases are sequential, so one
// counter serves both.
void DiffusionLB::migMsgAck()
{
  // Every ack pairs with a += 2 at a handoff this step. One stray or
  // duplicated ack silently collapses the phase barrier: acrossDone or
  // withinDone then fires before every LoadReceived has landed, so
  // ProcessMigrations both loses migrations (the step never completes) and
  // reopens the dangling-map window it resets. Make that protocol violation
  // loud instead of letting it surface as heap corruption far away.
  if (mig_acksOut <= 0)
    CkAbort("DiffusionLB: migMsgAck on PE %d with no ack outstanding "
            "(mig_acksOut=%d)\n", CkMyPe(), mig_acksOut);
  mig_acksOut--;
  migMaybeDone();
}

void DiffusionLB::migMaybeDone()
{
  if (mig_acksOut > 0) return;
  if (across_owed)
  {
    across_owed = false;
    // Neighbour-local, not a rendezvous on PE 0. This node's across-node
    // handoffs are all acked, so it will send nothing further this phase --
    // tell the only nodes that could have been waiting on it. Diffusion moves
    // load between neighbours and nowhere else, so a node that has heard this
    // from every neighbour knows nothing more can arrive, which is the whole
    // property the job-wide barrier was buying.
    acrossSelfDone = true;
    for (size_t i = 0; i < sendToNeighbors.size(); i++)
      thisProxy[sendToNeighbors[i] * nodeSize].nbrAcrossDone();
    maybeStartWithin();
  }
  if (within_owed)
  {
    within_owed = false;
    // The debug and dump paths keep every node in lockstep -- group-wide
    // reductions in CollectStats, a PE 0 gather in ProcessFinalStats -- so
    // they still go through the PE 0 barrier. Otherwise the moves start
    // neighbour-locally, see maybeStartMigrations.
    if (diffusionGlobalPhases() || step() == LBSimulation::dumpStep ||
        _lb_args.debug() > 0)
      thisProxy[0].withinDone();
    else
    {
      withinSelfDone = true;
      for (size_t i = 0; i < sendToNeighbors.size(); i++)
        thisProxy[sendToNeighbors[i] * nodeSize].nbrWithinDone();
      maybeStartMigrations();
    }
  }
}

// A neighbour has finished its within-node phase, so it will retarget nothing
// further of ours this step.
void DiffusionLB::nbrWithinDone()
{
  withinNbrDoneCount++;
  maybeStartMigrations();
}

// Start this node's moves once its own within-node phase is acked AND every
// neighbour's is. The second half is what makes the move list final: a
// neighbour that received one of our tokens may hand it on to another of its
// PEs, and that retarget is a LoadReceived to OUR donor PE. It is acked to the
// neighbour before the neighbour reports its phase done, so once every
// neighbour has reported, every retarget of ours has been applied. Nodes that
// are not neighbours hold none of our tokens and are not waited for; with one
// node there are no neighbours and the moves start at once.
void DiffusionLB::maybeStartMigrations()
{
  if (!withinSelfDone) return;
  if (withinNbrDoneCount < (int)sendToNeighbors.size()) return;
  withinSelfDone = false;
  withinNbrDoneCount = 0;
  const int first = myNodeId * nodeSize;
  for (int r = 0; r < nodeSize; r++) thisProxy[first + r].ProcessMigrations();
}

// A neighbour has finished its across-node handoffs.
void DiffusionLB::nbrAcrossDone()
{
  acrossNbrDoneCount++;
  maybeStartWithin();
}

// Start within-node once this node is done sending and every neighbour has
// said the same. Both halves are needed and either can arrive first, so this
// is checked from both. Within-node is purely intra-node work, so it starts on
// this node's PEs alone -- no other node is involved and none is waited for.
void DiffusionLB::maybeStartWithin()
{
  if (!acrossSelfDone) return;
  if (acrossNbrDoneCount < (int)sendToNeighbors.size()) return;
  acrossSelfDone = false;
  acrossNbrDoneCount = 0;
  const int first = myNodeId * nodeSize;
  for (int r = 0; r < nodeSize; r++) thisProxy[first + r].WithinNodeLB();
}

// Retained for the job-wide path; no longer on the critical path.
void DiffusionLB::acrossDone()
{
  if (++acrossDoneCount < numNodes) return;
  acrossDoneCount = 0;
  thisProxy.WithinNodeLB();
}

// When load balancing, remove object handle from your list, since it is about to be
// migrated
/* LoadMetaInfo is called on the receiver with the object that will be migrated to it
 * (via a MigrateMe in  LoadReceived). It is only called when migrating at the node
 * level. Not sure why the receiver would already have this handle though...*/
void DiffusionLB::LoadMetaInfo(LDObjHandle h, int local_id, double load, int senderPE, int only_mcount, int ackPE)
{
  // The rank0PE that issued this handoff and is holding its phase barrier
  // open. Carried explicitly: a within-node token's donorPE is on another
  // node, so the issuer cannot be inferred from the other fields.
  thisProxy[ackPE].migMsgAck();


  // local_id should be PE local here
  // Diagnostic only: the step's barrier is the source-side move ledger (see
  // DistBaseLB::ProcessMigrationDecision), not this arrival count.
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

  // ProcessMigrations starts once PE 0 has every node's stats; see
  // ReceiveFinalStats.
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

  // Group reductions complete in contribution order, so the
  // print_num_migrations target -- the last contribute above -- is the end of
  // this phase, and it starts ProcessMigrations from there. No quiescence.
}

void DiffusionLB::print_max_load(double max){
  CkPrintf("Max load per PE AFTER LB: %f\n", max);
}
void DiffusionLB::print_avg_load(double sum){
    CkPrintf("Avg load per PE AFTER LB: %f\n", sum / numPes);

}
void DiffusionLB::print_num_migrations(int sum){
    CkPrintf("Number of cross node migrations AFTER LB: %d\n", sum);
    // Last of the CollectStats reductions: the stats phase is over.
    thisProxy.ProcessMigrations();
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
  if (statsReceived == numNodes)
  {
    statsReceived = 0;
    thisProxy.ProcessMigrations();
  }

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

