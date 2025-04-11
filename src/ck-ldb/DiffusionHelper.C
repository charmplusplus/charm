#define SELF_IDX NUM_NEIGHBORS
#define EXT_IDX NUM_NEIGHBORS + 1
#define NUM_NEIGHBORS 2
// Assembling the stats for the PE
CLBStatsMsg* DiffusionLB::AssembleStats()
{
#if CMK_LB_CPUTIMER
  lbmgr->TotalTime(&myStats->total_walltime, &myStats->total_cputime);
  lbmgr->BackgroundLoad(&myStats->bg_walltime, &myStats->bg_cputime);
#else
  lbmgr->TotalTime(&myStats->total_walltime, &myStats->total_walltime);
  lbmgr->BackgroundLoad(&myStats->bg_walltime, &myStats->bg_walltime);
#endif
  lbmgr->IdleTime(&myStats->idletime);

  myStats->objData.resize(lbmgr->GetObjDataSz());  // = new LDObjData[myStats->n_objs];
  lbmgr->GetObjData(myStats->objData.data());

  myStats->commData.resize(lbmgr->GetCommDataSz());  // = new LDCommData[myStats->n_comm];
  lbmgr->GetCommData(myStats->commData.data());

  const int osz = lbmgr->GetObjDataSz();
  const int csz = lbmgr->GetCommDataSz();

  // TODO: not deleted
  CLBStatsMsg* statsMsg = new CLBStatsMsg(osz, csz);
  statsMsg->from_pe = CkMyPe();

  // Get stats
#if CMK_LB_CPUTIMER
  lbmgr->GetTime(&statsMsg->total_walltime, &statsMsg->total_cputime, &statsMsg->idletime,
                 &statsMsg->bg_walltime, &statsMsg->bg_cputime);
#else
  lbmgr->GetTime(&statsMsg->total_walltime, &statsMsg->total_walltime,
                 &statsMsg->idletime, &statsMsg->bg_walltime, &statsMsg->bg_walltime);
#endif
  //  msg->pe_speed = myspeed;
  // number of pes
  statsMsg->pe_speed = myStats->pe_speed;

  // statsMsg->n_objs = osz;
  lbmgr->GetObjData(statsMsg->objData.data());
  // statsMsg->n_comm = csz;
  lbmgr->GetCommData(statsMsg->commData.data());

  return statsMsg;
}

// Aggregates the stats messages of PE into LDStats, Computes total load of node
void DiffusionLB::BuildStats()
{
#if DEBUG_K
  CkPrintf("[%d] GRD Build Stats  and objects %lu\n", CkMyPe(),
           nodeStats->objData.size());
#endif
  int n_objs = nodeStats->objData.size();
  int n_comm = nodeStats->commData.size();
  //    nodeStats->nprocs() = statsReceived;
  // allocate space
  nodeStats->objData.clear();
  nodeStats->from_proc.clear();
  nodeStats->to_proc.clear();
  nodeStats->commData.clear();

  int prev = 0;
  for (int i = 0; i < nodeSize; i++)
  {
    prefixObjects[i] = prev + numObjects[i];
    prev = prefixObjects[i];
  }

  nodeStats->objData.resize(n_objs);
  nodeStats->from_proc.resize(n_objs);
  nodeStats->to_proc.resize(n_objs);
  nodeStats->commData.resize(n_comm);
  objs.clear();
  objs.resize(n_objs);

  /*if(nodeKeys != NULL)
      delete[] nodeKeys;
  nodeKeys = new LDObjKey[nodeStats->n_objs];*/
  int nobj = 0;
  int ncom = 0;
  int nmigobj = 0;
  int start = rank0PE;
  my_load = 0;
  my_loadAfterTransfer = 0;

  // copy all data in individual message to this big structure
  for (int pe = 0; pe < statsReceived; pe++)
  {
    int i;
    CLBStatsMsg* msg = statsList[pe];
    if (msg == NULL)
      continue;
    for (i = 0; i < msg->objData.size(); i++)
    {
      nodeStats->from_proc[nobj] = nodeStats->to_proc[nobj] = start + pe;

      nodeStats->objData[nobj] = msg->objData[i];
      LDObjData& oData = nodeStats->objData[nobj];
      //            CkPrintf("\n[PE-%d]Adding vertex id %d", CkMyPe(), nobj);
      objs[nobj] = CkVertex(nobj, oData.wallTime, nodeStats->objData[nobj].migratable,
                            nodeStats->from_proc[nobj]);
      my_load += msg->objData[i].wallTime;
      pe_load[pe] += msg->objData[i].wallTime;

      /*TODO Keys LDObjKey key;
      key.omID() = msg->objData[i].handle.omID;
      key.objID() =  msg->objData[i].handle.objID;
      nodeKeys[nobj] = key;*/
      if (msg->objData[i].migratable)
        nmigobj++;
      nobj++;
    }
    for (i = 0; i < msg->commData.size(); i++)
    {
      nodeStats->commData[ncom] = msg->commData[i];
      // nodeStats->commData[ncom].receiver.dest.destObj.destObjProc =
      // msg->commData[i].receiver.dest.destObj.destObjProc;
      int dest_pe = nodeStats->commData[ncom].receiver.lastKnown();
      // CkPrintf("\n here dest_pe = %d\n", dest_pe);
      ncom++;
    }
    // free the memory TODO: Free the memory in Destructor
    delete msg;
    statsList[pe] = 0;
  }
  my_loadAfterTransfer = my_load;
  nodeStats->n_migrateobjs = nmigobj;
  // Generate a hash with key object id, value index in objs vector
  nodeStats->deleteCommHash();
  nodeStats->makeCommHash();
}

void DiffusionLB::AddToList(CLBStatsMsg* m, int rank)
{
  nodeStats->objData.resize(nodeStats->objData.size() + m->objData.size());
  nodeStats->commData.resize(nodeStats->commData.size() + m->commData.size());
  numObjects[rank] = m->objData.size();
  statsList[rank] = m;

  struct ProcStats& procStat = nodeStats->procs[rank];
  procStat.pe = CkMyPe() + rank;  // real PE
  procStat.total_walltime = m->total_walltime;
  procStat.idletime = m->idletime;
  procStat.bg_walltime = m->bg_walltime;
#if CMK_LB_CPUTIMER
  procStat.total_cputime = m->total_cputime;
  procStat.bg_cputime = m->bg_cputime;
#endif
  procStat.pe_speed = m->pe_speed;  // important
  procStat.available = true;
  procStat.n_objs = m->objData.size();
}

int DiffusionLB::GetPENumber(int& obj_id)
{
  int i = 0;
  for (i = 0; i < nodeSize; i++)
  {
    if (obj_id < prefixObjects[i])
    {
      int prevAgg = 0;
      if (i != 0)
        prevAgg = prefixObjects[i - 1];
      obj_id = obj_id - prevAgg;
      break;
    }
  }
  return i;
}

int DiffusionLB::findNborIdx(int node)
{
  for (int i = 0; i < sendToNeighbors.size(); i++)
    if (sendToNeighbors[i] == node)
      return i;
  return -1;
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

double DiffusionLB::avgNborLoad()
{
  double sum = 0.0;

  for (int i = 0; i < neighborCount; i++) sum += loadNeighbors[i];
  return sum / neighborCount;
}

void DiffusionLB::buildObjComms(int n_objs)
{
  objectComms.resize(n_objs);
  for (int i = 0; i < n_objs; i++)
  {
    objectComms[i].resize(NUM_NEIGHBORS + 2);
    for (int j = 0; j < NUM_NEIGHBORS + 2; j++) objectComms[i][j] = 0;
  }

  // build object comms
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

      int toPE = commData.receiver.lastKnown();
      int toNode = toPE / nodeSize;

      // // remnants from simulator
      // int fromobj = get_obj_idx(from.objID());
      // int toobj = get_obj_idx(to.objID());

      // if (fromobj == -1 || toobj == -1)
      //   continue;
      // store internal bytes in the last index pos ? -q
      if (fromNode == toNode)
      {
        // internal communication
        int nborIdx = SELF_IDX;  // self ID at end of NUM_NEIGHBORS
        int fromObj = nodeStats->getHash(from);
        int toObj = nodeStats->getHash(to);

        CkAssert(fromObj != -1 && fromObj < n_objs);
        objectComms[fromObj][nborIdx] += commData.bytes;
        // lastKnown PE value can be wrong.
        if (toObj != -1 && toObj < n_objs)
          objectComms[toObj][nborIdx] += commData.bytes;
        else
          CkPrintf(
              "ERROR (MAYBE): toObj %d not found in objectComms, but we are "
              "destination\n",
              toObj);
      }
      else
      {  // External communication
        int nborIdx = findNborIdx(toNode);
        if (nborIdx == -1)
        {
          // object comm might be to a neighbor that we didn't decide on
          nborIdx = EXT_IDX;  // Store in last index if it is external bytes going to
                              // non-immediate neighbors
        }

        int fromObj = nodeStats->getHash(from);
        // CkPrintf("[%d] GRD Load Balancing from obj %d and pos %d\n", CkMyPe(), fromObj,
        // nborIdx);
        if (fromObj != -1 && fromObj < n_objs)
          objectComms[fromObj][nborIdx] += commData.bytes;
      }
    }
  }  // end for
}

// simple gain values, based only on internal comm
void DiffusionLB::buildGainValues(int n_objs)
{
  for (int i = 0; i < n_objs; i++)
  {
    int sum_bytes = 0;
    // comm bytes with all neighbors
    std::vector<int> comm_w_nbors = objectComms[i];
    // compute the sume of bytes of all comms for this obj
    for (int j = 0; j < comm_w_nbors.size(); j++) sum_bytes += comm_w_nbors[j];
    gain_val[i] = 2 * objectComms[i][SELF_IDX] - sum_bytes;  // gain val is only
  }
}

void DiffusionLB::buildGainValuesNbor(int n_objs, int nbor)
{
  for (int i = 0; i < n_objs; i++) gain_val[i] = -objectComms[i][nbor];
}

int DiffusionLB::getBestNeighbor()
{
  int bestNeighbor = -1;
  for (int i = 0; i < neighborCount; i++)
  {
    if (toSendLoad[i] > 0)
    {
      bestNeighbor = i;
      break;
    }
  }
  return bestNeighbor;
}

int DiffusionLB::getBestObject(int nbor)
{
  int v_id = heap_pop(obj_heap, ObjCompareOperator(&objs, gain_val), heap_pos);
  return v_id;
}

// all nodes call this to send final stats to 0. For printing to JSON
// TODO: this is broken rn, because of BaseLB::LDStats pup problems
void DiffusionLB::ReceiveFinalStats(std::vector<bool> isMigratable,
                                    std::vector<int> from_proc, std::vector<int> to_proc,
                                    int n_migrateobjs,
                                    std::vector<std::vector<LBRealType>> positions,
                                    std::vector<double> load)
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

  if (statsReceived == numNodes)
  {
    LBwriteStatsMsgs(fullStats);
  }
}

void DiffusionLB::pairedSort(int* A, std::vector<double> B)
{
  // sort array A based on corresponding values in B (both of size n)
  int n = B.size();
  std::vector<std::pair<long, int>> vp;
  for (int i = 0; i < n; ++i)
  {
    vp.push_back(std::make_pair(B[i], A[i]));
  }

  sort(vp.begin(), vp.end());

  // convert A back to array
  for (int i = 0; i < n; ++i)
  {
    A[i] = vp[i].second;
  }
}
