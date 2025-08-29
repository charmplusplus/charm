/**
 * \addtogroup CkLdb
*/
/*@{*/

#include "charm++.h"
#include "BaseLB.h"
#include "LBSimulation.h"

#if 1

void BaseLB::initLB(const CkLBOptions &opt) {
  seqno = opt.getSeqNo();
  lbmgr = CProxy_LBManager(_lbmgr).ckLocalBranch();
  lbname = "Unknown";
  // register this load balancer to LBManager at the sequence number
  lbmgr->addLoadbalancer(this, seqno);
}

void BaseLB::pup(PUP::er &p) { 
  p|seqno;
  if (p.isUnpacking())
  {
    if (CkMyPe()==0) {
      if (seqno!=-1) {
        int newseq = LBManagerObj()->getLoadbalancerTicket();
        CmiAssert(newseq == seqno);
      }
    }
    initLB(seqno);
  }
}

void BaseLB::flushStates() {
  Group::flushStates();
  lbmgr->ClearLoads();
}

#else
void BaseLB::initLB(const CkLBOptions &) {}
void BaseLB::pup(PUP::er &p) {}
void BaseLB::flushStates() {}
#endif

void BaseLB::turnOn()
{
#if 1
  lbmgr->TurnOnStartLBFn(startLbFnHdl);
#endif
}

void BaseLB::turnOff()
{
#if 1
  lbmgr->TurnOffStartLBFn(startLbFnHdl);
#endif
}

static inline int i_abs(int c) { return c>0?c:-c; }

// assume integer is 32 bits
inline static int ObjKey(const CmiUInt8 &oid, const int hashSize) {
  // make sure all positive
  return oid % hashSize;
}

BaseLB::LDStats::LDStats(int npes, int complete)
	: n_migrateobjs(0),
          complete_flag(complete)
{
  procs.resize(npes);
}

const static unsigned int doublingPrimes[] = {
3,
7,
17,
37,
73,
157,
307,
617,
1217,
2417,
4817,
9677,
20117,
40177,
80177,
160117,
320107,
640007,
1280107,
2560171,
5120117,
10000079,
20000077,
40000217,
80000111,
160000177,
320000171,
640000171,
1280000017,
2560000217u,
4200000071u
/* extra primes larger than an unsigned 32-bit integer:
51200000077,
100000000171,
200000000171,
400000000171,
800000000117,
1600000000021,
3200000000051,
6400000000081,
12800000000003,
25600000000021,
51200000000077,
100000000000067,
200000000000027,
400000000000063,
800000000000017,
1600000000000007,
3200000000000059,
6400000000000007,
12800000000000009,
25600000000000003,
51200000000000023,
100000000000000003,
200000000000000003,
400000000000000013,
800000000000000119,
1600000000000000031,
3200000000000000059 //This is a 62-bit number
*/
};

//This routine returns an arbitrary prime larger than x
static unsigned int primeLargerThan(unsigned int x)
{
	int i=0;
	while (doublingPrimes[i]<=x) i++;
	return doublingPrimes[i];
}

void BaseLB::LDStats::makeCommHash() {
  // hash table is already build
  if (!objHash.empty()) return;
   
  hashSize = primeLargerThan(objData.size() * 2);
  objHash.assign(hashSize, -1);
  int i = 0;
  for(const auto& obj : objData) {
        const CmiUInt8 &oid = obj.objID();
        int hash = ObjKey(oid, hashSize);
	CmiAssert(hash != -1);
        while(objHash[hash] != -1)
            hash = (hash+1)%hashSize;
        objHash[hash] = i++;
  }
}

void BaseLB::LDStats::deleteCommHash() {
  objHash.clear();
  for(auto& comm : commData) {
      comm.clearHash();
  }
}

int BaseLB::LDStats::getHash(const CmiUInt8 &oid, const LDOMid &mid)
{
#if 1
    CmiAssert(hashSize > 0);
    int hash = ObjKey(oid, hashSize);

    for(int id=0;id<hashSize;id++){
        int index = (id+hash)%hashSize;
	if (index == -1 || objHash[index] == -1) return -1;
        if (objData[objHash[index]].objID() == oid &&
            objData[objHash[index]].omID() == mid)
            return objHash[index];
    }
    //  CkPrintf("not found \n");
#endif
    return -1;
}

int BaseLB::LDStats::getHash(const LDObjKey &objKey)
{
  const CmiUInt8 &oid = objKey.objID();
  const LDOMid  &mid = objKey.omID();
  return getHash(oid, mid);
}

int BaseLB::LDStats::getSendHash(LDCommData &cData)
{
  if (cData.sendHash == -1) {
    cData.sendHash = getHash(cData.sender);
  }
  return cData.sendHash;
}

int BaseLB::LDStats::getRecvHash(LDCommData &cData)
{
  if (cData.recvHash == -1) {
    cData.recvHash =  getHash(cData.receiver.get_destObj());
  }
  return cData.recvHash;
}

void BaseLB::LDStats::clearCommHash() {
  for(auto& comm : commData) {
      comm.clearHash();
  }
}

void BaseLB::LDStats::computeNonlocalComm(int &nmsgs, int &nbytes)
{
#if 1
    	nmsgs = 0;
	nbytes = 0;

	makeCommHash();

	int mcast_count = 0;
        for (auto& cdata : commData) {
	    int senderPE, receiverPE;
	    if (cdata.from_proc())
	      senderPE = cdata.src_proc;
  	    else {
	      int idx = getHash(cdata.sender);
	      if (idx == -1) continue;    // sender has just migrated?
	      senderPE = to_proc[idx];
	      CmiAssert(senderPE != -1);
	    }
	    CmiAssert(senderPE < procs.size() && senderPE >= 0);

            // find receiver: point-to-point and multicast two cases
	    int receiver_type = cdata.receiver.get_type();
	    if (receiver_type == LD_PROC_MSG || receiver_type == LD_OBJ_MSG) {
              if (receiver_type == LD_PROC_MSG)
	        receiverPE = cdata.receiver.proc();
              else  {  // LD_OBJ_MSG
	        int idx = getHash(cdata.receiver.get_destObj());
		if (idx == -1) {		// receiver outside this domain
		  if (complete_flag) continue;
		  else receiverPE = -1;
		}
		else {
	          receiverPE = to_proc[idx];
                  CmiAssert(receiverPE < procs.size() && receiverPE >= 0);
		}
              }
	      if(senderPE != receiverPE)
	      {
	  	nmsgs += cdata.messages;
		nbytes += cdata.bytes;
	      }
	    }
            else if (receiver_type == LD_OBJLIST_MSG) {
              int nobjs;
              const LDObjKey *objs = cdata.receiver.get_destObjs(nobjs);
	      mcast_count ++;
	      std::vector<int> pes;
	      for (int i=0; i<nobjs; i++) {
	        int idx = getHash(objs[i]);
		CmiAssert(idx != -1);
	        if (idx == -1) continue;    // receiver has just been removed?
	        receiverPE = to_proc[idx];
		CmiAssert(receiverPE < procs.size() && receiverPE >= 0);
		int exist = 0;
	        for (auto pe : pes)
		  if (receiverPE == pe) { exist=1; break; }
		if (exist) continue;
		pes.push_back(receiverPE);
	        if(senderPE != receiverPE)
	        {
	  	  nmsgs += cdata.messages;
		  nbytes += cdata.bytes;
	        }
              }
	    }
	}   // end of for
#endif
}

void BaseLB::LDStats::normalize_speed() {
  double maxspeed = 0.0;

  for (const auto& proc : procs)
  {
    if (proc.pe_speed > maxspeed) maxspeed = proc.pe_speed;
  }
  for (auto& proc : procs) proc.pe_speed /= maxspeed;
}

void BaseLB::LDStats::print()
{
#if 1
  int i = 0;
  CkPrintf("------------- Processor Data: %zu -------------\n", procs.size());
  for (const auto& proc : procs)
  {
    CkPrintf("Proc %d (%d) Speed %f Total = %f Idle = %f Bg = %f nObjs = %d",
      i++, proc.pe, proc.pe_speed, proc.total_walltime, proc.idletime,
      proc.bg_walltime, proc.n_objs);
#if CMK_LB_CPUTIMER
    CkPrintf(" CPU Total %f Bg %f", proc.total_cputime, proc.bg_cputime);
#endif
    CkPrintf("\n");
  }

  i = 0;
  CkPrintf("------------- Object Data: %zu objects -------------\n", objData.size());
  for(auto &odata : objData) {
      CkPrintf("Object %d\n",i++);
      CkPrintf("     id = %" PRIu64 "\n",odata.objID());
      CkPrintf("  OM id = %d\t",odata.omID().id.idx);
      CkPrintf("   Mig. = %d\n",odata.migratable);
#if CMK_LB_CPUTIMER
      CkPrintf("    CPU = %f\t",odata.cpuTime);
#endif
      CkPrintf("   Wall = %f\n",odata.wallTime);
  }

  i = 0;
  CkPrintf("------------- Comm Data: %zu records -------------\n", commData.size());
  for(const auto& comm : commData) {
      CkPrintf("Link %d\n",i++);

      if (comm.from_proc())
	CkPrintf("    sender PE = %d\t",comm.src_proc);
      else
	CkPrintf("    sender id = %d:[%" PRIu64 "]\t",
		 comm.sender.omID().id.idx,comm.sender.objID());

      if (comm.recv_type() == LD_PROC_MSG)
	CkPrintf("  receiver PE = %d\n",comm.receiver.proc());
      else	
	CkPrintf("  receiver id = %d:[%" PRIu64 "]\n",
		 comm.receiver.get_destObj().omID().id.idx,comm.receiver.get_destObj().objID());
      
      CkPrintf("     messages = %d\t",comm.messages);
      CkPrintf("        bytes = %d\n",comm.bytes);
  }
  CkPrintf("------------- Object to PE mapping -------------\n");
  for (const auto& val : from_proc) CkPrintf(" %d", val);
  CkPrintf("\n");
#endif
}

double BaseLB::LDStats::computeAverageLoad()
{
  int numAvail = 0;
  double total = 0;
  for (const auto& obj : objData) total += obj.wallTime;

  for (const auto& proc : procs)
  {
    if (proc.available)
    {
      total += proc.bg_walltime;
      numAvail++;
    }
  }

  double averageLoad = total/numAvail;
  return averageLoad;
}

// remove the obj-th object from database
void BaseLB::LDStats::removeObject(int obj)
{
  CmiAssert(obj < objData.size());
  LDObjData odata = objData[obj];

  LDObjKey okey;		// build a key
  okey.omID() = odata.omID();
  okey.objID() = odata.objID();

  objData.erase(objData.begin() + obj);
  from_proc.erase(from_proc.begin() + obj);
  to_proc.erase(to_proc.begin() + obj);
  if (odata.migratable) n_migrateobjs --;

  // search for sender, can be multiple sender
  commData.erase(remove_if(commData.begin(),
                           commData.end(),
                           [&](LDCommData& cdata) {
                             return !cdata.from_proc() && cdata.sender == okey;
                           }),
                 commData.end());
}

void BaseLB::LDStats::pup(PUP::er &p)
{
  p(n_migrateobjs);
  p|procs;
  if (p.isUnpacking())
  {
    // user can specify simulated processors other than the real # of procs
    if (LBSimulation::simProcs > procs.size())
    {
      procs.resize(LBSimulation::simProcs);
    }
    // ignore the background load when unpacking if the user changed the # of procs
    if (LBSimulation::procsChanged)
    {
      const auto size = procs.size();
      procs.clear();
      procs.resize(size);
    }
  }
  p|objData;
  p|from_proc;
  // reset to_proc when unpacking
  if (p.isUnpacking())
    to_proc = from_proc;
  p|commData;
  if (p.isUnpacking()) {
    if (_lb_args.lbversion() <= 1)
      for (int i = 0; i < procs.size(); i++) procs[i].pe = i;
  }
}

int BaseLB::LDStats::useMem() { 
  // calculate the memory usage of this LB (superclass).
  return sizeof(LDStats) + sizeof(ProcStats) * procs.size() +
	 (sizeof(LDObjData) + 2 * sizeof(int)) * objData.size() +
	 sizeof(LDCommData) * commData.size();
}

#include "BaseLB.def.h"

/*@}*/
