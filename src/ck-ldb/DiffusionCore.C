
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
    if (_lb_args.debug()) CkPrintf("--------STARTING ACROSS NODE LB--------\n");
    CkCallback cb(CkIndex_DiffusionLB::WithinNodeLB(), thisProxy);
    CkStartQD(cb);
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
                            sendToNeighbors);
  }
  else
    metric = new MetricCentroid(nborCentroids, nborDistances, myCentroid, nodeStats,
                                myNodeId, toSendLoad, sendToNeighbors, nborObjCount);

  loadReceivers = std::count_if(toSendLoad.begin(), toSendLoad.end(),
                                [](double load) { return load > 0; });

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
        else
          continue;
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
        CkAbort(
            "ERROR: not sure if this is supposed to work, but from_proc[%d] = %d, "
            "donorPE = %d\n",
            v_id, nodeStats->from_proc[v_id], donorPE);
      }

      my_loadAfterTransfer -= currLoad;

      metric->updateState(v_id, nborId);  // update state to keep track of migrations

      LDObjHandle objHandle = nodeStats->objData[v_id].handle;
      thisProxy[destPE].LoadMetaInfo(objHandle, objId, currLoad, donorPE, 0);
      thisProxy[donorPE].LoadReceived(objId, destPE);
      nodeStats->to_proc[v_id] = destPE;
    }
  }


  
}

// When load balancing, remove object handle from your list, since it is about to be
// migrated
/* LoadMetaInfo is called on the receiver with the object that will be migrated to it
 * (via a MigrateMe in  LoadReceived). It is only called when migrating at the node
 * level. Not sure why the receiver would already have this handle though...*/
void DiffusionLB::LoadMetaInfo(LDObjHandle h, int objId, double load, int senderPE, int only_mcount)
{
  migrates_expected++;
  if(only_mcount)
    return;
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
      load[i] = nodeStats->objData[i].wallTime;

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

                                  }

  if (thisIndex == 0)
  {
    CkCallback cb(CkIndex_DiffusionLB::ProcessMigrations(), thisProxy);
    CkStartQD(cb);
  }

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
    LBwriteStatsMsgs(fullStats);
  }
}

