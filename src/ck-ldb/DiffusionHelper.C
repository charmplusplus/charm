#define SELF_IDX NUM_NEIGHBORS
#define EXT_IDX NUM_NEIGHBORS + 1
#define NUM_NEIGHBORS _lb_args.diffusionNumNbors()
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

  // TEMPORARY diagnostic: is the comm graph populated at all?
  if (_lb_args.debug() > 1)
    CkPrintf("[%d] DiffusionLB AssembleStats: objs=%d commRecords=%d traceComm=%d statsOn=%d\n",
             CkMyPe(), osz, csz, (int)_lb_args.traceComm(), (int)lbmgr->CollectingCommStats());

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
      // The origin of every load figure in this balancer, and the two levels take
      // different dimensions from it.
      //
      // my_load is what the pseudo-LB rounds diffuse across nodes: the resource that
      // is scarce at node granularity, GPU occupancy or host time per
      // +LBDiffusionGpuDim.
      //
      // pe_load and the CkVertex compLoad are always host time, because they feed the
      // within-node heap, and PEs inside a process share a device -- relocating a
      // chare between them moves host work only.
      objs[nobj] = CkVertex(nobj, diffusionObjCpuLoad(oData),
                            nodeStats->objData[nobj].migratable,
                            nodeStats->from_proc[nobj]);
      my_load += diffusionObjLoad(oData);
      pe_load[pe] += diffusionObjCpuLoad(oData);

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
  // Charge an unmeasured object the node's mean, in both dimensions.
  //
  // Every budget in this balancer is retired by subtraction --
  // my_loadAfterTransfer -= shedLoad across nodes, overLoad -= compLoad within
  // one -- and subtracting zero never retires anything, so a loop guarded by
  // "while budget > 0" hands over EVERY object on the PE. On a GPU-resident
  // application most objects report ~0 host time, which is how an even 43-44
  // objects/PE became 1 vs 95 in a single round while the spread being
  // minimised got worse.
  //
  // A fixed floor cannot work: getVertexLoad()'s MAX(compLoad, 0.1) is ~100x
  // larger than any real per-object load here, which makes every object look
  // identical and equally huge, while a floor small enough to be harmless
  // leaves the budget effectively un-retired. Scaling to the node's own mean
  // makes shedding k of n objects retire k/n of the budget -- proportional,
  // and self-cancelling when the loads are real. If a dimension is genuinely
  // all zero its mean is zero, the floor vanishes, and nothing moves: correct,
  // because there is no load to balance.
  // CHARM_LB_RAWLOAD: exactly what this node was handed, before any floor is
  // applied. my_load is a sum of these, so a node that reports 0.000000 to the
  // across-node phase is a node whose objects all measured zero.
  if (getenv("CHARM_LB_RAWLOAD"))
  {
    double gsum = 0.0, wsum = 0.0, gmax = 0.0, wmax = 0.0;
    int gz = 0, wz = 0;
    for (int i = 0; i < nobj; i++)
    {
      const LDObjData& od = nodeStats->objData[i];
#if CMK_CUDA
      const double g = od.gpuTime;
#else
      const double g = 0.0;
#endif
      const double w = od.wallTime;
      gsum += g; wsum += w;
      if (g > gmax) gmax = g;
      if (w > wmax) wmax = w;
      if (g <= 0.0) gz++;
      if (w <= 0.0) wz++;
    }
    // Per-PE object counts too: the CUPTI loads are built only once every PE of
    // the process has arrived at its LB barrier (hapiCuptiArrive, expected =
    // CkNodeSize). A PE holding no objects is the case to watch.
    std::string pes;
    int empty = 0;
    for (int pe = 0; pe < statsReceived; pe++)
    {
      char b[32];
      snprintf(b, sizeof(b), " %d", numObjects[pe]);
      pes += b;
      if (numObjects[pe] == 0) empty++;
    }
    CkPrintf("[RAWLOAD node %d] nobj %d migr %d | gpu sum %.6f max %.6f zero %d/%d"
             " | wall sum %.6f max %.6f zero %d/%d | statsReceived %d empty %d objs/pe%s\n",
             myNodeId, nobj, nmigobj, gsum, gmax, gz, nobj, wsum, wmax, wz, nobj,
             statsReceived, empty, pes.c_str());
  }

  objLoadFloor = 0.0;
  if (nobj > 0)
  {
    const double gpuFloor = my_load / (double)nobj;
    objLoadFloor = gpuFloor;
    double cpuTotal = 0.0;
    for (int r = 0; r < nodeSize; r++) cpuTotal += pe_load[r];
    const double cpuFloor = cpuTotal / (double)nobj;

    my_load = 0.0;
    for (int r = 0; r < nodeSize; r++) pe_load[r] = 0;
    int at = 0;
    for (int pe = 0; pe < statsReceived; pe++)
    {
      if (numObjects[pe] == 0) continue;
      for (int k = 0; k < numObjects[pe]; k++, at++)
      {
        const LDObjData& od = nodeStats->objData[at];
        const double g = std::max(diffusionObjLoad(od), gpuFloor);
        const double c = std::max(diffusionObjCpuLoad(od), cpuFloor);
        objs[at].setCompLoad(c);
        my_load += g;
        pe_load[pe] += c;
      }
    }
  }

  my_loadAfterTransfer = my_load;
  nodeStats->n_migrateobjs = nmigobj;

  // Is this node an interval of a 1-D ordering? Only when every object here
  // registered a width-1 position; one object without a key and the interval
  // rules stand down for the node (the metrics then choose freely).
  keyed1D = nobj > 0;
  myKeyLo = myKeyHi = 0.0;
  for (int i = 0; i < nobj && keyed1D; i++)
  {
    const double k = keyOf(nodeStats->objData[i]);
    if (k != k) { keyed1D = false; break; }
    if (i == 0 || k < myKeyLo) myKeyLo = k;
    if (i == 0 || k > myKeyHi) myKeyHi = k;
  }

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

// takes in node local id and returns rank
int DiffusionLB::GetRank(int obj_id)
{
  int i = 0;
  for (i = 0; i < nodeSize; i++)
  {
    if (obj_id < prefixObjects[i])
    {
      break;
    }
  }
  return i;
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
