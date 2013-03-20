/**
 * \addtogroup CkLdb
*/
/*@{*/

#include "charm++.h"
#include "BaseLB.h"
#include "LBSimulation.h"

#if CMK_LBDB_ON

void BaseLB::initLB(const CkLBOptions &opt) {
  seqno = opt.getSeqNo();
  CkpvAccess(numLoadBalancers) ++;
/*
  if (CkpvAccess(numLoadBalancers) - CkpvAccess(hasNullLB) > 1)
    CmiAbort("Error: try to create more than one load balancer strategies!");
*/
  theLbdb = CProxy_LBDatabase(_lbdb).ckLocalBranch();
  lbname = "Unknown";
  // register this load balancer to LBDatabase at the sequence number
  theLbdb->addLoadbalancer(this, seqno);
}

BaseLB::~BaseLB() {
  CkpvAccess(numLoadBalancers) --;
}

void BaseLB::unregister() {
  theLbdb->RemoveLocalBarrierReceiver(receiver);
  CkpvAccess(numLoadBalancers) --;
}

void BaseLB::pup(PUP::er &p) { 
  p|seqno;
  if (p.isUnpacking())
  {
    if (CkMyPe()==0) {
      if (seqno!=-1) {
        int newseq = LBDatabaseObj()->getLoadbalancerTicket();
        CmiAssert(newseq == seqno);
      }
    }
    initLB(seqno);
  }
}

void BaseLB::flushStates() {
  Group::flushStates();
  theLbdb->ClearLoads();
}

#else
BaseLB::~BaseLB() {} 
void BaseLB::initLB(const CkLBOptions &) {}
void BaseLB::unregister() {}
void BaseLB::pup(PUP::er &p) {}
void BaseLB::flushStates() {}
#endif

static inline int i_abs(int c) { return c>0?c:-c; }

// assume integer is 32 bits
inline static int ObjKey(const LDObjid &oid, const int hashSize) {
  // make sure all positive
  return (((i_abs(oid.id[2]) & 0x7F)<<24)
	 |((i_abs(oid.id[1]) & 0xFF)<<16)
	 |i_abs(oid.id[0])) % hashSize;
}

BaseLB::LDStats::LDStats(int c, int complete)
	: n_objs(0), n_migrateobjs(0), n_comm(0), 
          objHash(NULL), complete_flag(complete)
{
  count = c;
  if (count == 0) count = CkNumPes();
  procs = new ProcStats[count];
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
  if (objHash) return;
   
  int i;
  hashSize = n_objs*2;
  hashSize = primeLargerThan(hashSize);
  objHash = new int[hashSize];
  for(i=0;i<hashSize;i++)
        objHash[i] = -1;
   
  for(i=0;i<n_objs;i++){
        const LDObjid &oid = objData[i].objID();
        int hash = ObjKey(oid, hashSize);
	CmiAssert(hash != -1);
        while(objHash[hash] != -1)
            hash = (hash+1)%hashSize;
        objHash[hash] = i;
  }
}

void BaseLB::LDStats::deleteCommHash() {
  if (objHash) delete [] objHash;
  objHash = NULL;
  for(int i=0; i < n_comm; i++) {
      commData[i].clearHash();
  }
}

int BaseLB::LDStats::getHash(const LDObjid &oid, const LDOMid &mid)
{
#if CMK_LBDB_ON
    CmiAssert(hashSize > 0);
    int hash = ObjKey(oid, hashSize);

    for(int id=0;id<hashSize;id++){
        int index = (id+hash)%hashSize;
	if (index == -1 || objHash[index] == -1) return -1;
        if (LDObjIDEqual(objData[objHash[index]].objID(), oid) &&
            LDOMidEqual(objData[objHash[index]].omID(), mid))
            return objHash[index];
    }
    //  CkPrintf("not found \n");
#endif
    return -1;
}

int BaseLB::LDStats::getHash(const LDObjKey &objKey)
{
  const LDObjid &oid = objKey.objID();
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
  for(int i=0; i < n_comm; i++) {
      commData[i].clearHash();
  }
}

void BaseLB::LDStats::computeNonlocalComm(int &nmsgs, int &nbytes)
{
#if CMK_LBDB_ON
    	nmsgs = 0;
	nbytes = 0;

	makeCommHash();

	int mcast_count = 0;
        for (int cidx=0; cidx < n_comm; cidx++) {
	    LDCommData& cdata = commData[cidx];
	    int senderPE, receiverPE;
	    if (cdata.from_proc())
	      senderPE = cdata.src_proc;
  	    else {
	      int idx = getHash(cdata.sender);
	      if (idx == -1) continue;    // sender has just migrated?
	      senderPE = to_proc[idx];
	      CmiAssert(senderPE != -1);
	    }
	    CmiAssert(senderPE < nprocs() && senderPE >= 0);

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
                  CmiAssert(receiverPE < nprocs() && receiverPE >= 0);
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
              LDObjKey *objs = cdata.receiver.get_destObjs(nobjs);
	      mcast_count ++;
	      CkVec<int> pes;
	      for (int i=0; i<nobjs; i++) {
	        int idx = getHash(objs[i]);
		CmiAssert(idx != -1);
	        if (idx == -1) continue;    // receiver has just been removed?
	        receiverPE = to_proc[idx];
		CmiAssert(receiverPE < nprocs() && receiverPE >= 0);
		int exist = 0;
	        for (int p=0; p<pes.size(); p++) 
		  if (receiverPE == pes[p]) { exist=1; break; }
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
  int pe;
  double maxspeed = 0.0;

  for(int pe=0; pe < nprocs(); pe++) {
    if (procs[pe].pe_speed > maxspeed) maxspeed = procs[pe].pe_speed;
  }
  for(int pe=0; pe < nprocs(); pe++)
    procs[pe].pe_speed /= maxspeed;
}

void BaseLB::LDStats::print()
{
#if CMK_LBDB_ON
  int i;
  CkPrintf("------------- Processor Data: %d -------------\n", nprocs());
  for(int pe=0; pe < nprocs(); pe++) {
    struct ProcStats &proc = procs[pe];

    CkPrintf("Proc %d (%d) Speed %d Total = %f Idle = %f Bg = %f nObjs = %d",
      pe, proc.pe, proc.pe_speed, proc.total_walltime, proc.idletime,
      proc.bg_walltime, proc.n_objs);
#if CMK_LB_CPUTIMER
    CkPrintf(" CPU Total %f Bg %f", proc.total_cputime, proc.bg_cputime);
#endif
    CkPrintf("\n");
  }

  CkPrintf("------------- Object Data: %d objects -------------\n", n_objs);
  for(i=0; i < n_objs; i++) {
      LDObjData &odata = objData[i];
      CkPrintf("Object %d\n",i);
      CkPrintf("     id = %d %d %d %d\n",odata.objID().id[0],odata.objID().id[1
], odata.objID().id[2], odata.objID().id[3]);
      CkPrintf("  OM id = %d\t",odata.omID().id);
      CkPrintf("   Mig. = %d\n",odata.migratable);
#if CMK_LB_CPUTIMER
      CkPrintf("    CPU = %f\t",odata.cpuTime);
#endif
      CkPrintf("   Wall = %f\n",odata.wallTime);
  }

  CkPrintf("------------- Comm Data: %d records -------------\n", n_comm);
  CkVec<LDCommData> &cdata = commData;
  for(i=0; i < n_comm; i++) {
      CkPrintf("Link %d\n",i);

      LDObjid &sid = cdata[i].sender.objID();
      if (cdata[i].from_proc())
	CkPrintf("    sender PE = %d\t",cdata[i].src_proc);
      else
	CkPrintf("    sender id = %d:[%d %d %d %d]\t",
		 cdata[i].sender.omID().id,sid.id[0], sid.id[1], sid.id[2], sid.id[3]);

      LDObjid &rid = cdata[i].receiver.get_destObj().objID();
      if (cdata[i].recv_type() == LD_PROC_MSG)
	CkPrintf("  receiver PE = %d\n",cdata[i].receiver.proc());
      else	
	CkPrintf("  receiver id = %d:[%d %d %d %d]\n",
		 cdata[i].receiver.get_destObj().omID().id,rid.id[0],rid.id[1],rid.id[2],rid.id[3]);
      
      CkPrintf("     messages = %d\t",cdata[i].messages);
      CkPrintf("        bytes = %d\n",cdata[i].bytes);
  }
  CkPrintf("------------- Object to PE mapping -------------\n");
  for (i=0; i<n_objs; i++) CkPrintf(" %d", from_proc[i]);
  CkPrintf("\n");
#endif
}

double BaseLB::LDStats::computeAverageLoad()
{
  int i, numAvail=0;
  double total = 0;
  for (i=0; i<n_objs; i++) total += objData[i].wallTime;
                                                                                
  for (i=0; i<nprocs(); i++)
    if (procs[i].available == true) {
        total += procs[i].bg_walltime;
	numAvail++;
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

  objData.remove(obj);
  from_proc.remove(obj);
  to_proc.remove(obj);
  n_objs --;
  if (odata.migratable) n_migrateobjs --;

  // search for sender, can be multiple sender
  int removed = 0;
  for (int com=0; com<n_comm; com++) {
    LDCommData &cdata = commData[com-removed];
    if(!cdata.from_proc() && cdata.sender == okey) {
      commData.remove(com-removed);
      removed++;
    }
  }
  n_comm -= removed;
}

void BaseLB::LDStats::pup(PUP::er &p)
{
  int i;
  p(count);
  p(n_objs);
  p(n_migrateobjs);
  p(n_comm);
  if (p.isUnpacking()) {
    // user can specify simulated processors other than the real # of procs.
    int maxpe = nprocs() > LBSimulation::simProcs ? nprocs() : LBSimulation::simProcs;
    procs = new ProcStats[maxpe];
    objData.resize(n_objs);
    commData.resize(n_comm);
    from_proc.resize(n_objs);
    to_proc.resize(n_objs);
    objHash = NULL;
  }
  // ignore the background load when unpacking if the user change the # of procs
  // otherwise load everything
  if (p.isUnpacking() && LBSimulation::procsChanged) {
    ProcStats dummy;
    for (i=0; i<nprocs(); i++) p|dummy;
  }
  else
    for (i=0; i<nprocs(); i++) p|procs[i];
  for (i=0; i<n_objs; i++) p|objData[i]; 
  for (i=0; i<n_objs; i++) p|from_proc[i]; 
  for (i=0; i<n_objs; i++) p|to_proc[i]; 
  // reset to_proc when unpacking
  if (p.isUnpacking())
    for (i=0; i<n_objs; i++) to_proc[i] = from_proc[i];
  for (i=0; i<n_comm; i++) p|commData[i];
  if (p.isUnpacking())
    count = LBSimulation::simProcs;
  if (p.isUnpacking()) {
    objHash = NULL;
    if (_lb_args.lbversion() <= 1) 
      for (i=0; i<nprocs(); i++) procs[i].pe = i;
  }
}

int BaseLB::LDStats::useMem() { 
  // calculate the memory usage of this LB (superclass).
  return sizeof(LDStats) + sizeof(ProcStats) * nprocs() +
	 (sizeof(LDObjData) + 2 * sizeof(int)) * n_objs +
 	 sizeof(LDCommData) * n_comm;
}

#include "BaseLB.def.h"

/*@}*/
