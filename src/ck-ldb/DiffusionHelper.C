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
#if 0
            pe_loadBefore[pe] += msg->objData[i].wallTime;
#endif
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

bool DiffusionLB::AggregateToSend()
{
  bool res = false;
  for (int i = 0; i < neighborCount; i++) toSendLoad[i] -= toReceiveLoad[i];
  return res;
}