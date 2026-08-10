/*
Charm++ File: Checkpoint Library
added 01/03/2003 by Chao Huang, chuang10@uiuc.edu

More documentation goes here...
--- Updated 12/14/2003 by Gengbin, gzheng@uiuc.edu
    see ckcheckpoint.h for change log
*/

#include <stdio.h>
#include <stdlib.h>
#ifndef _WIN32
#include <unistd.h>
#include <sys/time.h>
#endif
#include <string.h>
#include <sstream>
using std::ostringstream;
#include <errno.h>
#include <fstream>
#include <cstring>
#include "charm++.h"
#include "ck.h"
#include "ckrescale.h"
#include "ckcheckpoint.h"
#include "CkCheckpoint.decl.h"

void noopit(const char*, ...)
{}

//#define DEBCHK   CkPrintf
#define DEBCHK noopit

#define SUBDIR_SIZE 256

CkGroupID _sysChkptWriteMgr;
CkGroupID _sysChkptMgr;

bool _restarted = false;
int _oldNumPes = 0;
bool _chareRestored = false;
double chkptStartTimer = 0;
// PE-0 wall-clock (gettimeofday seconds) timestamps spanning a rescale, used
// to break down where the post-checkpoint restart-overhead time goes. All
// fields are 0 outside an active rescale.
//
// Must be a wall clock (not CmiWallTimer), since the rescale crosses a
// ConverseInit re-init that resets CmiWallTimer's epoch.
double rescale_overhead_start_timer = 0; // entry to ResumeFromReallocCheckpoint
double rescale_t_cleanup_enter      = 0; // ConverseCleanup entry
double rescale_t_commit_done        = 0; // after coordinator COMMIT
double rescale_t_ep_reinit_done     = 0; // after UcxReInitEpsFromView (or MPI equiv)
double rescale_t_barrier_done       = 0; // after coord::barrier post-reinit
double rescale_t_longjmp            = 0; // just before longjmp
double rescale_t_after_longjmp      = 0; // just after setjmp != 0 returns
double rescale_t_converseinit_call  = 0; // just before re-entering ConverseInit
double rescale_t_lrtsinit_done      = 0; // after LrtsInit (UCX/PMIx init)
double rescale_t_converserunpe_enter= 0; // ConverseRunPE entry
double rescale_t_commoninit_done    = 0; // after ConverseCommonInit
// ConverseCommonInit sub-phase stamps (PE 0):
double rescale_t_cci_basics         = 0; // after CmiIOInit
double rescale_t_cci_tmp            = 0; // after CmiTmpInit
double rescale_t_cci_timer_only     = 0; // after CmiTimerInit
double rescale_t_cci_stats          = 0; // after CstatsInit
double rescale_t_cci_timers         = 0; // after CmiInitCPUAffinityUtil
double rescale_t_cci_handlers       = 0; // after CIdleTimeoutInit
double rescale_t_cci_iso_predeps    = 0; // after CldModuleInit (before CmiIsomallocInit)
double rescale_t_cci_trace          = 0; // after traceInit
double rescale_t_cci_persistent     = 0; // after CmiOnesidedDirectInit
double rescale_t_cci_ccs            = 0; // after CcsInit
double rescale_t_cci_threads        = 0; // after CmiInitMultipleSend
double rescale_t_initcharm_enter    = 0; // _initCharm entry
double rescale_t_register_done      = 0; // after _register pass (or skip)
// Post-register sub-phase stamps (PE 0):
double rescale_t_pr_initcalls_done  = 0; // after _initCallTable.enumerateInitCalls
double rescale_t_pr_aff_done        = 0; // after CmiInitCPUAffinity + CmiInitMemAffinity
double rescale_t_pr_cputopo_done    = 0; // after CmiInitCPUTopology
double rescale_t_pr_topo_done       = 0; // after TopoManager_reset + tree
double rescale_t_pr_to_faultfunc    = 0; // just before faultFunc(CkRestartMain) call
double rescale_t_restart_main_enter = 0; // CkRestartMain entry
double rescale_wall_now()
{
#ifdef _WIN32
  return 0;
#else
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec + tv.tv_usec * 1e-6;
#endif
}
static void rescale_clear_timers()
{
  rescale_overhead_start_timer  = 0;
  rescale_t_cleanup_enter       = 0;
  rescale_t_commit_done         = 0;
  rescale_t_ep_reinit_done      = 0;
  rescale_t_barrier_done        = 0;
  rescale_t_longjmp             = 0;
  rescale_t_after_longjmp       = 0;
  rescale_t_converseinit_call   = 0;
  rescale_t_lrtsinit_done       = 0;
  rescale_t_converserunpe_enter = 0;
  rescale_t_commoninit_done     = 0;
  rescale_t_cci_basics          = 0;
  rescale_t_cci_tmp             = 0;
  rescale_t_cci_timer_only      = 0;
  rescale_t_cci_stats           = 0;
  rescale_t_cci_timers          = 0;
  rescale_t_cci_handlers        = 0;
  rescale_t_cci_iso_predeps     = 0;
  rescale_t_cci_trace           = 0;
  rescale_t_cci_persistent      = 0;
  rescale_t_cci_ccs             = 0;
  rescale_t_cci_threads         = 0;
  rescale_t_initcharm_enter     = 0;
  rescale_t_register_done       = 0;
  rescale_t_pr_initcalls_done   = 0;
  rescale_t_pr_aff_done         = 0;
  rescale_t_pr_cputopo_done     = 0;
  rescale_t_pr_topo_done        = 0;
  rescale_t_pr_to_faultfunc     = 0;
  rescale_t_restart_main_enter  = 0;
}
#if CMK_SHRINK_EXPAND
int originalnumGroups = -1;
extern int Cmi_isOldProcess;
extern bool _shrinkexpand_isNewcomer;
extern char *_shrinkexpand_basedir;
// PE-0-only: callback that resumes the LB after the rescale restart.
// On the no-disk path this is stashed here at checkpoint time and consumed by
// CkRestartMain when it builds the in-memory broadcast for newcomers.
static CkCallback _rescaleResumeCb;
static bool _rescaleResumeCbValid = false;
#endif


// Required for broadcasting RO Data after recovering from failure
#if CMK_SMP
extern std::atomic<UInt> numZerocopyROops;
#else
extern UInt  numZerocopyROops; 
#endif

#ifndef CMK_CHARE_USE_PTR

CkpvExtern(std::vector<void *>, chare_objs);
CkpvExtern(std::vector<int>, chare_types);
CkpvExtern(std::vector<VidBlock *>, vidblocks);

#endif

void CkCreateLocalChare(int epIdx, envelope *env);

// helper class to get number of array elements
class ElementCounter : public CkLocIterator {
private:
	int count;
public:
        ElementCounter():count(0){};
        void addLocation(CkLocation &loc)  { count++; }
	int getCount() { return count; }
};

// helper class to pup all elements that belong to same ckLocMgr
class ElementCheckpointer : public CkLocIterator {
private:
        CkLocMgr *locMgr;
        PUP::er &p;
public:
        ElementCheckpointer(CkLocMgr* mgr_, PUP::er &p_):locMgr(mgr_),p(p_){};
        void addLocation(CkLocation &loc) {
          CkArrayIndex idx=loc.getIndex();
          //CkPrintf("[%d] Packing index dim = %i, %s\n", CkMyPe(), idx.dimension, idx2str(idx));
          CkGroupID gID = locMgr->ckGetGroupID();
          CmiUInt8 id = loc.getID();
          p|gID;	    // store loc mgr's GID as well for easier restore
          p|idx;
          p|id;
          p|loc;
		      //CkPrintf("[%d] addLocation: ", CkMyPe()), idx.print();
        }
};


extern void _initDone();

static void bdcastRO(void){
	int i;
	// Determine the size of the RODataMessage
	PUP::sizer ps(PUP::er::IS_CHECKPOINT);
	UInt numZerocopyROopsSize; // only used for sizing.
	ps|numZerocopyROopsSize;
	for(i=0;i<_readonlyTable.size();i++) _readonlyTable[i]->pupData(ps);

	// Allocate and fill out the RODataMessage
	envelope *env = _allocEnv(RODataMsg, ps.size());
	PUP::toMem pp((char *)EnvToUsr(env), PUP::er::IS_CHECKPOINT);
	// Messages of type 'RODataMsg' need to have numZerocopyROops pupped in order
	// to be processed inside _processRODataMsg
#if CMK_SMP
	UInt numZerocopyROopsTemp = numZerocopyROops.load(std::memory_order_relaxed);
	pp|numZerocopyROopsTemp;
#else
	pp|numZerocopyROops;
#endif
	for(i=0;i<_readonlyTable.size();i++) _readonlyTable[i]->pupData(pp);
	
	env->setCount(++_numInitMsgs);
	env->setSrcPe(CkMyPe());
	CmiSetHandler(env, _roRestartHandlerIdx);
	CmiSyncBroadcastAndFree(env->getTotalsize(), (char *)env);
}

#if CMK_SHRINK_EXPAND
static void bdcastROGroupData(void){
	int i;
	//Determine the size of the RODataMessage
	PUP::sizer ps(PUP::er::IS_CHECKPOINT), ps1(PUP::er::IS_CHECKPOINT);
	CkPupROData(ps);
	int ROSize = ps.size();

	//CkPupGroupData(ps1);
	int GroupSize = ps1.size();

	char *msg = (char *)CmiAlloc(CmiMsgHeaderSizeBytes + 2*sizeof(int) + ps.size() + ps1.size());
	char *payloadOffset = msg + CmiMsgHeaderSizeBytes;

	// how much data to send
	*(int*)payloadOffset = ps.size();
	payloadOffset += sizeof(int);
	*(int*)payloadOffset = ps1.size();
	payloadOffset += sizeof(int);

	//Allocate and fill out the RODataMessage
	PUP::toMem pp((char *)payloadOffset, PUP::er::IS_CHECKPOINT);
	CkPupROData(pp);

	//CkPupGroupData(pp);

	CmiSetHandler(msg, _ROGroupRestartHandlerIdx);
	CmiSyncBroadcastAllAndFree(CmiMsgHeaderSizeBytes + 2*sizeof(int) + pp.size(), msg);
}
#endif

// Print out an array index to this string as decimal fields
// separated by underscores.
void printIndex(const CkArrayIndex &idx,char *dest) {
	const int *idxData=idx.data();
	for (int i=0;i<idx.nInts;i++) {
		snprintf(dest,12,"%s%d",i==0?"":"_", idxData[i]);
		dest+=strlen(dest);
	}
}

static bool checkpointOne(const char* dirname, CkCallback& cb, bool requestStatus);

static void addPartitionDirectory(ostringstream &path) {
  if (CmiNumPartitions() > 1) {
    path << "/part-" << CmiMyPartition();
  }
}

static std::string getCheckpointFileName(const char* dirname, const char* basename,
                                         const int id = -1)
{
  ostringstream out;
  out << dirname;
  addPartitionDirectory(out);
  if (id != -1)
  {
    const int subdir_id = id / SUBDIR_SIZE;
    out << "/sub" << subdir_id;
  }
  out << "/" << basename;
  if (id != -1)
  {
    out << "_" << id;
  }
  out << ".dat";
  return out.str();
}

static FILE* openCheckpointFile(const char *dirname, const char *basename,
    const char *mode, const int id = -1) {
  std::string filename = getCheckpointFileName(dirname, basename, id);
  FILE *fp = CmiFopen(filename.c_str(), mode);
  if (!fp) {
    CkAbort("PE %d failed to open checkpoint file: %s, mode: %s, status: %s",
        CkMyPe(), filename.c_str(), mode, strerror(errno));
  }
  return fp;
}

class CkCheckpointWriteMgr : public CBase_CkCheckpointWriteMgr
{
private:
  const int firstPE = CkNodeFirst(CkMyNode());
  const int nodeSize = CkMyNodeSize();
  int numWriters = CkMyNodeSize();
  int numComplete = 0;
  int index = 0;
  bool inProgress = false;

  const char* dirname;
  CkCallback cb;
  bool requestStatus;

public:
  CkCheckpointWriteMgr() {}

  CkCheckpointWriteMgr(CkMigrateMessage* m) : CBase_CkCheckpointWriteMgr(m) {}

  void Checkpoint(const char* dirname, CkCallback cb, bool requestStatus = false,
                  int writersPerNode = 0)
  {
    // If currently checkpointing, drop new requests
    if (inProgress) return;
    inProgress = true;
    numComplete = 0;

    if (writersPerNode > 0) numWriters = std::min(writersPerNode, nodeSize);

    // Save params for future invocations and kick off the first numWriters PEs to start
    // checkpointing
    this->dirname = dirname;
    this->cb = cb;
    this->requestStatus = requestStatus;
    for (index = firstPE; index < firstPE + numWriters; index++)
      CProxy_CkCheckpointMgr(_sysChkptMgr)[index].Checkpoint(dirname, cb, requestStatus);
  }

  void RescaleCheckpoint(const char* dirname, CkCallback cb, std::vector<char> avail,
    bool requestStatus = false, int writersPerNode = 0)
  {
    // If currently checkpointing, drop new requests
    if (inProgress) return;
    inProgress = true;
    numComplete = 0;

    set_shrinkexpand_exit(true); // Set this flag to indicate that we are in the process of shrinking/expanding

    if (writersPerNode > 0) numWriters = std::min(writersPerNode, nodeSize);

    // Save params for future invocations and kick off the first numWriters PEs to start
    // checkpointing
    this->dirname = dirname;
    this->cb = cb;
    this->requestStatus = requestStatus;

#if CMK_SHRINK_EXPAND
    // All PEs, including PE 0, refresh se_avail_vector from the carried
    // copy here: PE 0's original handler-time malloc has been observed
    // clobbered by the time the exit path reads it.
    se_avail_vector = (char*) malloc(CkNumPes() * sizeof(char));
    memcpy(se_avail_vector, avail.data(), CkNumPes() * sizeof(char));
#endif

    for (index = firstPE; index < firstPE + numWriters; index++)
      CProxy_CkCheckpointMgr(_sysChkptMgr)[index].Checkpoint(dirname, cb, requestStatus);
  }

  void FinishedCheckpoint()
  {
    numComplete++;

    // If there's another PE to kick off, do so
    if (index < firstPE + nodeSize)
    {
      CProxy_CkCheckpointMgr(_sysChkptMgr)[index].Checkpoint(dirname, cb, requestStatus);
      index++;
    }
    // If there isn't, then check if all the PEs are finished
    else if (numComplete == nodeSize)
    {
      inProgress = false;
    }
  }
};

/**
 * There is only one Checkpoint Manager in the whole system
**/
class CkCheckpointMgr : public CBase_CkCheckpointMgr {
private:
	CkCallback restartCB;
	double chkptStartTimer;
	bool requestStatus;
	int chkpStatus;
public:
	CkCheckpointMgr() { }
	CkCheckpointMgr(CkMigrateMessage *m):CBase_CkCheckpointMgr(m) { }
	void Checkpoint(const char *dirname, CkCallback cb, bool requestStatus = false);
	void SendRestartCB(void);
	void pup(PUP::er& p){ p|restartCB; }
};

// broadcast
void CkCheckpointMgr::Checkpoint(const char *dirname, CkCallback cb, bool _requestStatus){
#if CMK_SHRINK_EXPAND
  std::vector<char> avail(se_avail_vector, se_avail_vector + CkNumPes());
  int chckPtId = CmiPhysicalRank(CmiMyPe());
#else
  int chckPtId = CmiPhysicalRank(CmiMyPe());
#endif
	chkptStartTimer = CmiWallTimer();
  
#if CMK_SHRINK_EXPAND
  // pending_realloc_state only carries the SHRINK_IN_PROGRESS / EXPAND_IN_PROGRESS
  // distinction on PE 0 (set in CentralLB::CheckForRealloc); other PEs reach
  // here via the RescaleCheckpoint broadcast which sets shrinkexpand_exit on
  // every PE. Trust shrinkexpand_exit as the rescale indicator; fall back to
  // pending_realloc_state on PE 0 for the SHRINK vs EXPAND callback selection
  // below.
  const bool isRescale = get_shrinkexpand_exit()
                         || pending_realloc_state == SHRINK_IN_PROGRESS
                         || pending_realloc_state == EXPAND_IN_PROGRESS;
#else
  const bool isRescale = false;
#endif

#if CMK_SHRINK_EXPAND
  if (avail[CkMyPe()])
#endif
  {
    requestStatus = _requestStatus;
    bool success = true;

  #if CMK_SHRINK_EXPAND
    if (isRescale) {
      // No-disk rescale path. The survivor's groups/nodegroups/arrays/chares/RO
      // are still live in memory across the longjmp (init.C gates the table
      // overwrites on _reuseRegistrationStateOnRestart), and newcomers will
      // receive groups + RO via an in-memory broadcast that PE 0 builds in
      // CkRestartMain. So we skip every disk write here.
      if (CkMyPe() == 0) {
        if (pending_realloc_state == SHRINK_IN_PROGRESS) {
          CkPrintf("Shrink in progress on PE%i\n", CkMyPe());
          _rescaleResumeCb = CkCallback(CkIndex_LBManager::ResumeClients(), _lbmgr);
        } else {
          CkPrintf("Expand in progress on PE%i\n", CkMyPe());
          _rescaleResumeCb = CkCallback(CkIndex_LBManager::StartLB(),
                                        CProxy_LBManager(_lbmgr)[0]);
        }
        _rescaleResumeCbValid = true;
      }
      pending_realloc_state = NO_REALLOC;
    } else
  #endif
    {
      // make dir on all PEs in case it is a local directory
      CmiMkdir(dirname);

      // Create partition directories (if applicable)
      ostringstream dirPath;
      dirPath << dirname;
      if (CmiNumPartitions() > 1) {
        addPartitionDirectory(dirPath);
        CmiMkdir(dirPath.str().c_str());
      }

      // Due to file system issues we have observed, divide checkpoints
      // into subdirectories to avoid having too many files in a single directory.
      // Nodegroups should be checked separately since they could go into
      // different subdirectory.

      // Save current path for later use with nodegroups
      ostringstream dirPathNode;
      dirPathNode << dirPath.str();

      // Create subdirectories
      int mySubDir = chckPtId / SUBDIR_SIZE;
      dirPath << "/sub" << mySubDir;
      CmiMkdir(dirPath.str().c_str());

      // Create Nodegroup subdirectory if needed
      if (CkMyRank() == 0) {
        int mySubDirNode = CkMyNode() / SUBDIR_SIZE;
        if (mySubDirNode != mySubDir) {
          dirPathNode << "/sub" << mySubDirNode;
          CmiMkdir(dirPathNode.str().c_str());
        }
      }

      if (CkMyPe() == 0) {
        success &= checkpointOne(dirname, cb, requestStatus);
      }

  #ifndef CMK_CHARE_USE_PTR
      // only create chare checkpoint file if this PE actually has data
      if (CkpvAccess(chare_objs).size() > 0 || CkpvAccess(vidblocks).size() > 0)
      {
        // save plain singleton chares into Chares.dat
        FILE* fChares = openCheckpointFile(dirname, "Chares", "wb", chckPtId);
        PUP::toDisk pChares(fChares, PUP::er::IS_CHECKPOINT);
        CkPupChareData(pChares);
        if (pChares.checkError()) success = false;
        if (CmiFclose(fChares) != 0) success = false;
      }
  #endif

      // save groups into Groups.dat
      // content of the file: numGroups, GroupInfo[numGroups], _groupTable(PUP'ed),
      // groups(PUP'ed)
      FILE* fGroups = openCheckpointFile(dirname, "Groups", "wb", chckPtId);
      PUP::toDisk pGroups(fGroups, PUP::er::IS_CHECKPOINT);
      CkPupGroupData(pGroups);
      if (pGroups.checkError()) success = false;
      if (CmiFclose(fGroups) != 0) success = false;

      // save nodegroups into NodeGroups.dat
      // content of the file: numNodeGroups, GroupInfo[numNodeGroups],
      // _nodeGroupTable(PUP'ed), nodegroups(PUP'ed)
      if (CkMyRank() == 0)
      {
        FILE* fNodeGroups = openCheckpointFile(dirname, "NodeGroups", "wb", 0);
        PUP::toDisk pNodeGroups(fNodeGroups, PUP::er::IS_CHECKPOINT);
        CkPupNodeGroupData(pNodeGroups);
        if (pNodeGroups.checkError()) success = false;
        if (CmiFclose(fNodeGroups) != 0) success = false;
      }

      FILE* datFile = openCheckpointFile(dirname, "arr", "wb", chckPtId);
      PUP::toDisk p(datFile, PUP::er::IS_CHECKPOINT);
      CkPupArrayElementsData(p);
      if (p.checkError()) success = false;
      if (CmiFclose(datFile) != 0) success = false;

  #if ! CMK_DISABLE_SYNC
  #if CMK_HAS_SYNC_FUNC
            sync();
  #elif CMK_HAS_SYNC
      system("sync");
  #endif
  #endif
    }

    chkpStatus = success?CK_CHECKPOINT_SUCCESS:CK_CHECKPOINT_FAILURE;
    restartCB = cb;
    DEBCHK("[%d]restartCB installed\n",CkMyPe());
  }
	// Use barrier instead of contribute here:
	// barrier is stateless and multiple calls to it do not overlap.
	barrier(CkCallback(CkReductionTarget(CkCheckpointMgr, SendRestartCB), 0, thisgroup));
	CProxy_CkCheckpointWriteMgr(_sysChkptWriteMgr)[CkMyNode()].FinishedCheckpoint();
}

void CkCheckpointMgr::SendRestartCB(void){
	DEBCHK("[%d]Sending out the cb\n",CkMyPe());
#if CMK_SHRINK_EXPAND
	const bool isRescale = get_shrinkexpand_exit();
#else
	const bool isRescale = false;
#endif
	CkPrintf("%s finished in %fs, sending out the cb...\n",
		isRescale ? "Rescale snapshot (no-op)" : "Checkpoint to disk",
		CmiWallTimer() - chkptStartTimer);
	if(requestStatus)
	{
	  CkCheckpointStatusMsg * m = new CkCheckpointStatusMsg(chkpStatus);
	  restartCB.send(m);
	}
	else
	  restartCB.send();
}

void CkPupROData(PUP::er &p)
{
	int _numReadonlies = 0;
	int _numReadonlyMsgs = 0;
	if (!p.isUnpacking()) _numReadonlies=_readonlyTable.size();

	p|_numReadonlies;

	if (p.isUnpacking()) {
	  if (_numReadonlies != _readonlyTable.size())
	    CkAbort("You cannot add readonlies and restore from checkpoint...");
	}
	for(int i=0;i<_numReadonlies;i++) _readonlyTable[i]->pupData(p);
	if (!p.isUnpacking()) _numReadonlyMsgs=_readonlyMsgs.size();
        p|_numReadonlyMsgs;
	for(int i=0;i<_numReadonlyMsgs; i++){
		ReadonlyMsgInfo *c = _readonlyMsgs[i];
		CkPupMessage(p,c->pMsg);
	}
}

// handle main chare
void CkPupMainChareData(PUP::er &p, CkArgMsg *args)
{
	int nMains=_mainTable.size();
	//CkPrintf("[%d] CkPupMainChareData %s: nMains = %d\n", CkMyPe(),p.typeString(),nMains);
	for(int i=0;i<nMains;i++){  /* Create all mainchares */
		const auto& chareIdx = _mainTable[i]->chareIdx;
		ChareInfo *entry = _chareTable[chareIdx];
		int entryMigCtor = entry->getMigCtor();
		if(entryMigCtor!=-1) {
			Chare* obj;
			if (p.isUnpacking()) {
				//CkPrintf("MainChare PUP'ed: name = %s, idx = %d, size = %d\n", entry->name, i, entry->size);
				obj = CkAllocateChare(chareIdx);
        //CkPrintf("Allocated mainchare %s\n", entry->name);
				_mainTable[i]->setObj(obj);
        //CkPrintf("Set mainchare %s\n", entry->name);
				//void *m = CkAllocSysMsg();
				CkInvokeEP(obj, entryMigCtor, args);
        //CkPrintf("Invoked migration constructor for mainchare %s\n", entry->name);
			}
			else 
			 	obj = (Chare *)_mainTable[i]->getObj();
			obj->virtual_pup(p);
		}
	}
	// to update mainchare proxy
	// only readonly variables of Chare Proxy are taken care of here;
	// in general, if chare proxy is contained in some data structure,
	// such as CkCallback, it is user's responsibility to
	// update them after restarting
#if !CMK_SHRINK_EXPAND
	if (p.isUnpacking() && CkMyPe()==0)
		bdcastRO();
#endif

}

#ifndef CMK_CHARE_USE_PTR

// handle plain non-migratable chare
void CkPupChareData(PUP::er &p)
{
  int i, n = 0;
  if (!p.isUnpacking()) n = CkpvAccess(chare_objs).size();
  p|n;
  for (i=0; i<n; i++) {
        int chare_type = 0;
	if (!p.isUnpacking()) {
		chare_type = CkpvAccess(chare_types)[i];
	}
	p | chare_type;
	bool pup_flag = true;
	if (!p.isUnpacking()) {
	  if(CkpvAccess(chare_objs)[i] == NULL){
	    pup_flag = false;
	  }
	}
	p|pup_flag;
	if(pup_flag)
	{
	  if (p.isUnpacking()) {
		  int migCtor = _chareTable[chare_type]->migCtor;
		  if(migCtor==-1) {
			  CkAbort("Chare %s needs a migration constructor and PUP'er routine for restart.\n", _chareTable[chare_type]->name);
		  }
		  void *m = CkAllocSysMsg();
		  envelope* env = UsrToEnv((CkMessage *)m);
		  CkCreateLocalChare(migCtor, env);
		  CkFreeSysMsg(m);
	  }
	  Chare *obj = (Chare*)CkpvAccess(chare_objs)[i];
	  obj->virtual_pup(p);
	}
	else
	{
	  CkpvAccess(chare_objs)[i] = NULL;
	}
  }

  if (!p.isUnpacking()) n = CkpvAccess(vidblocks).size();
  p|n;
  for (i=0; i<n; i++) {
	VidBlock *v;
	bool pup_flag = true;
	if (!p.isUnpacking()) {
	  if(CkpvAccess(vidblocks)[i]==NULL)
	  {
	    pup_flag = false;
	  }
	}
	p|pup_flag;
	if(pup_flag)
	{
	  if (p.isUnpacking()) {
		  v = new VidBlock();
		  CkpvAccess(vidblocks).push_back(v);
	  }
	  else{
		  v = CkpvAccess(vidblocks)[i];
	  }
	  v->pup(p);
	}
  }
}
#else
void CkPupChareData(PUP::er &p)
{
   // not implemented
}
#endif

typedef void GroupCreationFn(CkGroupID groupID, int constructorIdx, envelope *env);



static void CkPupPerPlaceData(PUP::er &p, GroupIDTable *idTable, GroupTable *objectTable,
                              unsigned int &numObjects, int constructionMsgType,
                              GroupCreationFn creationFn
                             )
{
  int numGroups = 0, i;

  if (!p.isUnpacking()) {
    numGroups = idTable->size();
  }
  p|numGroups;
  CkPrintf("[%d] CkPupPerPlaceData %s: numGroups = %d\n", CkMyPe(),p.typeString(),numGroups);

  std::vector<GroupInfo> tmpInfo(numGroups);
  if (!p.isUnpacking()) {
    for (i = 0; i < numGroups; i++) {
      tmpInfo[i].gID = (*idTable)[i];
      TableEntry ent = objectTable->find(tmpInfo[i].gID);
      tmpInfo[i].present = ent.getObj() != NULL;
      tmpInfo[i].MigCtor = _chareTable[ent.getcIdx()]->migCtor;
      tmpInfo[i].name = _chareTable[ent.getcIdx()]->name;
      //CkPrintf("[%d] CkPupPerPlaceData: %s group %s \n", CkMyPe(), p.typeString(), tmpInfo[i].name);

      if(tmpInfo[i].MigCtor==-1) {
        CkAbort("(Node)Group %s needs a migration constructor and PUP'er routine for restart.\n", tmpInfo[i].name.c_str());
      }
    }
  }
  p|tmpInfo;

  int maxGroup = 0;
  for (i = 0; i < numGroups; i++) 
  {
    if (!tmpInfo[i].present)
      continue;

    CkGroupID gID = tmpInfo[i].gID;
    if (p.isUnpacking()) {
      int eIdx = tmpInfo[i].MigCtor;
      if (eIdx == -1) {
        CkPrintf("[%d] ERROR> (Node)Group %s's migration constructor is not defined!\n", CkMyPe(), tmpInfo[i].name.c_str());
        CkAbort("Abort");
      }
      void *m = CkAllocSysMsg();
      envelope* env = UsrToEnv((CkMessage *)m);
      env->setMsgtype(constructionMsgType);

      {
        creationFn(gID, eIdx, env);
      }
      if(gID.idx > maxGroup)
          maxGroup = gID.idx;

      CkFreeSysMsg(m);
    }   // end of unPacking
    IrrGroup *gobj = objectTable->find(gID).getObj();


    // if using migration constructor, you'd better have a pup
    gobj->virtual_pup(p);
  }

  if (p.isUnpacking()) {
    if(CkMyPe()==0)
      numObjects = maxGroup+1;
    else
      numObjects = 1;
  }
}

void CkPupGroupData(PUP::er &p)
{
  CkPupPerPlaceData(p, CkpvAccess(_groupIDTable), CkpvAccess(_groupTable),
    CkpvAccess(_numGroups), BocInitMsg, &CkCreateLocalGroup
  );
}

void CkPupNodeGroupData(PUP::er &p
  )
{
          CkPupPerPlaceData(p, &CksvAccess(_nodeGroupIDTable),
                           CksvAccess(_nodeGroupTable), CksvAccess(_numNodeGroups),
                           NodeBocInitMsg, &CkCreateLocalNodeGroup
                          );
}

// handle chare array elements for this processor
void CkPupArrayElementsData(PUP::er &p, int notifyListeners)
{
 	int i;
	// safe in both packing/unpacking at this stage
  int numGroups = CkpvAccess(_groupIDTable)->size();

	// number of array elements on this processor
	int numElements = 0;
	if (!p.isUnpacking()) {
	  ElementCounter  counter;
	  CKLOCMGR_LOOP(mgr->iterate(counter););
          numElements = counter.getCount();
	}
	p|numElements;

	DEBCHK("[%d] CkPupArrayElementsData %s numGroups:%d numElements:%d \n",CkMyPe(),p.typeString(), numGroups, numElements);

	if (!p.isUnpacking())
	{
	  // let CkLocMgr iterate over and store every array element
    CKLOCMGR_LOOP(ElementCheckpointer chk(mgr, p); mgr->iterate(chk););
  }
	else {
	  // loop and create all array elements ourselves
	  //CkPrintf("total chare array cnts: %d\n", numElements);
	  for (int i=0; i<numElements; i++) {
      CkGroupID gID;
      CkArrayIndex idx;
      CmiUInt8 id;
      p|gID;
      p|idx;
      p|id;
      //CkPrintf("[%d] Unpacked dim = %i: %s\n", CkMyPe(), idx.dimension, idx2str(idx));
      CkLocMgr *mgr = (CkLocMgr*)CkpvAccess(_groupTable)->find(gID).getObj();
      if (notifyListeners){
        mgr->resume(idx, id, p, true);
      } else{
        mgr->restore(idx, id, p);
      }
	  }
	}
	// finish up
        if (notifyListeners)
        for(i=0;i<numGroups;i++) {
                IrrGroup *obj = CkpvAccess(_groupTable)->find((*CkpvAccess(_groupIDTable))[i]).getObj();
		if (obj)
                  obj->ckJustMigrated();
	}
}

#if __FAULT__
int  CkCountArrayElements(){
    int numGroups = CkpvAccess(_groupIDTable)->size();
    int i;
    ElementCounter  counter;
    CKLOCMGR_LOOP(mgr->iterate(counter););
  int numElements = counter.getCount();
    return numElements;
}
#endif

void CkPupProcessorData(PUP::er &p)
{
    // save readonlys, and callback BTW
    if(CkMyRank()==0) {
        CkPupROData(p);
    }

    // save mainchares into MainChares.dat
    if(CkMyPe()==0) {
      CkPupMainChareData(p, NULL);
    }
	
    // save non-migratable chare
    CkPupChareData(p);

    // save groups 
    //CkPupGroupData(p);

    // save nodegroups
    if(CkMyRank()==0) {
        CkPupNodeGroupData(p);
    }

    // pup array elements
    CkPupArrayElementsData(p);
}

// called only on pe 0
static bool checkpointOne(const char* dirname, CkCallback& cb, bool requestStatus){
	CmiAssert(CkMyPe()==0);
	
	// save readonlys, and callback BTW
	FILE* fRO = openCheckpointFile(dirname, "RO", "wb", -1);
	PUP::toDisk pRO(fRO, PUP::er::IS_CHECKPOINT);
	int _numPes = CkNumPes();
	pRO|_numPes;
	int _numNodes = CkNumNodes();

	pRO|_numNodes;
	pRO|cb;
	CkPupROData(pRO);
	pRO|requestStatus;

	if(pRO.checkError())
	{
	  return false;
	}

	if(CmiFclose(fRO)!=0)
	{
	  return false;
	}

	// save mainchares into MainChares.dat
	{
		FILE* fMain = openCheckpointFile(dirname, "MainChares", "wb", -1);
		PUP::toDisk pMain(fMain, PUP::er::IS_CHECKPOINT);
		CkPupMainChareData(pMain, NULL);
		if(pMain.checkError())
		{
		  return false;
		}
		if(CmiFclose(fMain) != 0)
		{
		  return false;
		}
	}
	return true;
}

void CkRemoveArrayElements()
{
  int i;
  int numGroups = CkpvAccess(_groupIDTable)->size();
  CKLOCMGR_LOOP(mgr->flushAllRecs(););
/*  GroupTable *gTbl = CkpvAccess(_groupTable);
  for(i=0; i<numGroups; i++){
    IrrGroup *obj = CkpvAccess(_groupTable)->find((*CkpvAccess(_groupIDTable))[i]).getObj();
    if(obj->isLocMgr()) {
	CkLocMgr *mgr = (CkLocMgr *)obj;
	mgr->flushAllRecs();
    }
  }*/
}

/*
void CkTestArrayElements()
{
  int i;
  int numGroups = CkpvAccess(_groupIDTable)->size();
  //CKLOCMGR_LOOP(mgr->flushAllRecs(););
  GroupTable *gTbl = CkpvAccess(_groupTable);
  for(i=0; i<numGroups; i++){
    IrrGroup *obj = CkpvAccess(_groupTable)->find((*CkpvAccess(_groupIDTable))[i]).getObj();
    CkPrintf("An object at [%d]: %p | isLocMgr: %d\n", i, obj, obj->isLocMgr());
  }
}
*/

void CkStartCheckpoint(const char* dirname, const CkCallback& cb, bool requestStatus,
                       int writersPerNode)
{
  if (cb.isInvalid())
    CkAbort("callback after checkpoint is not set properly");

  if (cb.containsPointer())
    CkAbort("Cannot restart from a callback based on a pointer");

  CkPrintf("[%d] Checkpoint starting in %s\n", CkMyPe(), dirname);

  // hand over to checkpoint managers for per-processor checkpointing
  CProxy_CkCheckpointWriteMgr(_sysChkptWriteMgr)
      .Checkpoint(dirname, cb, requestStatus, writersPerNode);
}

void CkStartRescaleCheckpoint(const char* dirname, const CkCallback& cb, 
  std::vector<char> avail, bool requestStatus, int writersPerNode)
{
#if CMK_SHRINK_EXPAND
  // Refresh PE 0's se_avail_vector as well (see RescaleCheckpoint).
  se_avail_vector = (char*) malloc(CkNumPes() * sizeof(char));
  memcpy(se_avail_vector, avail.data(), CkNumPes() * sizeof(char));

  if (cb.isInvalid())
  CkAbort("callback after checkpoint is not set properly");

  if (cb.containsPointer())
  CkAbort("Cannot restart from a callback based on a pointer");

  // hand over to checkpoint managers for per-processor checkpointing
  CProxy_CkCheckpointWriteMgr(_sysChkptWriteMgr)
      .RescaleCheckpoint(dirname, cb, avail, requestStatus, writersPerNode);
#endif
}

/**
  * Restart: There's no such object as restart manager is created
  *          because a group cannot restore itself anyway.
  *          The mechanism exists as converse code and get invoked by
  *          broadcast message.
  **/
CkCallback globalCb;
void CkRecvGroupROData(char* msg)
{
  char* origMsg = msg;
  msg = msg + CmiMsgHeaderSizeBytes;
  int dirSize = *reinterpret_cast<int*>(msg);
  msg += sizeof(int);
  std::string dirname(msg, dirSize);
  msg += dirSize;
  int ROsize = *reinterpret_cast<int*>(msg);
  msg += sizeof(int);

  //CkPrintf("dirname = %s, groupsize = %i\n", dirname.c_str(), groupSize);
  PUP::fromMem bRO(msg, PUP::er::IS_CHECKPOINT);

  int _numPes = -1;
  bRO|_numPes;
	int _numNodes = -1;
	bRO|_numNodes;
	bRO|globalCb;
	/*if (CmiMyRank() == 0)*/ CkPupROData(bRO);
	bool requestStatus = false;
	bRO|requestStatus;

  CkPrintf("[%d]Number of PE: %d -> %d\n",CkMyPe(),_numPes,CkNumPes());

  msg += ROsize;

  if (_shrinkexpand_isNewcomer) {
    PUP::fromMem bGroups(msg, PUP::er::IS_CHECKPOINT);
    CkPupGroupData(bGroups);
  }
  // Reset reduction state on EVERY rank (survivor and newcomer). The broadcast
  // payload was packed on PE 0 *before* the survivor's resetForRescale ran, so
  // newcomers unpack stale reductionInfo.redNo values and emit reduction
  // messages stamped with the pre-rescale redNo, which then sit in the
  // survivor parent's futureRemoteMsgs queue forever (msg.redNo > myRedNo=0).
  // Drop pending state on every group so survivor and newcomer are aligned.
  {
    int numGroups = CkpvAccess(_groupIDTable)->size();
    for (int i = 0; i < numGroups; i++) {
      CkGroupID gID = (*CkpvAccess(_groupIDTable))[i];
      IrrGroup *obj = CkpvAccess(_groupTable)->find(gID).getObj();
      if (obj && obj->isReductionMgr()) {
        ((CkReductionMgr *)obj)->resetForRescale();
      }
      // Survivor sends were crashing UCX with destPE = killed-PE. Cause: the
      // location cache and home-PE encoded in chare IDs were both stale after
      // the longjmp. Recompute home for every local element, rekey hash entries
      // under the new ID, clear the ID->PE cache, and re-publish to the new
      // home so remote PEs can resolve.
      if (obj && obj->isLocMgr()) {
        ((CkLocMgr *)obj)->resetForRescale();
      }
    }
    if (CkMyRank() == 0) {
      int numNodeGroups = CksvAccess(_nodeGroupIDTable).size();
      for (int i = 0; i < numNodeGroups; i++) {
        CkGroupID gID = CksvAccess(_nodeGroupIDTable)[i];
        IrrGroup *obj = CksvAccess(_nodeGroupTable)->find(gID).getObj();
        if (obj && obj->isNodeGroup()) {
          ((CkNodeReductionMgr *)obj)->resetForRescale();
        }
      }
    }
    // The pre-rescale LB step set lb_in_progress=true in CentralLB::InvokeLB
    // but never reached ResumeClients (the rescale path forks at
    // CheckForRealloc → StartCleanup → longjmp). On the survivor the flag is
    // stuck true, so the realloc CCS handler on PE 0 rejects every subsequent
    // rescale request with "Rescaling called while load balancing is in
    // progress". Clear the flag on every survivor PE; on PE 0 also replay any
    // rescale requests that arrived during the LB step (bufferRealloc) so
    // their pending_realloc_state takes effect on the next AtSync.
    LBManager *_lbm = LBManager::Object();
    if (_lbm) _lbm->resetForRescale();
  }

#ifndef CMK_CHARE_USE_PTR
  // restore chares only when number of pes is the same
  if (CkNumPes() == _numPes)
  {
    // A chare checkpoint file only exists when the PE actually contained singleton
    // chares at checkpoint time, so check to see if the file exists before trying
    // to restore
    std::string filename = getCheckpointFileName(dirname.c_str(), "Chares", CkMyPe());
    FILE* fChares = CmiFopen(filename.c_str(), "rb");
    if (fChares)
    {
      PUP::fromDisk pChares(fChares, PUP::er::IS_CHECKPOINT);
      CkPupChareData(pChares);
      CmiFclose(fChares);
      _chareRestored = true;
    }
  }
#endif
  CmiFree(origMsg);

	// for each location, restore arrays
	//DEBCHK("[%d]Trying to find location manager\n",CkMyPe());
	
  // Survivor branch: groups, nodegroups, and array elements are all live in
  // memory across the longjmp (preserved by the gated allocations in
  // _initCharm). Nothing to restore — no disk reads, no PUP cycle. Newcomers
  // already populated their groups from the in-memory broadcast above; their
  // array elements arrive later via LB-driven migration.

  set_in_restart(false);

  // Once the integrating restart completes, this rank is a survivor for any
  // future rescale events. Clear the flag so subsequent CkRecvGroupROData
  // calls take the survivor branches.
  _shrinkexpand_isNewcomer = false;

  if (CmiMyRank()==0) _initDone();  // this rank will trigger other ranks

	if(CkMyPe()==0) {
		CmiPrintf("[%d]CkRestartMain done. sending out callback.\n",CkMyPe());
		if(requestStatus)
		{
		  CkCheckpointStatusMsg * m = new CkCheckpointStatusMsg(CK_CHECKPOINT_SUCCESS);
		  globalCb.send(m);
		}
		else
		{
		  globalCb.send();
		}
	}
  
  if (CmiMyRank() == 0) CkMemCheckPT::inRestarting = false;

  if (CmiMyPe() == 0) {
    double restore_s = CmiWallTimer() - chkptStartTimer;
    CkPrintf("Rescale restore (in-memory) finished in %fs, sending out the cb...\n", restore_s);
    if (rescale_overhead_start_timer > 0) {
      double now = rescale_wall_now();
      double total_s    = now - rescale_overhead_start_timer;
      double overhead_s = total_s - restore_s;
      // Break the overhead into the segments that span the longjmp. Any
      // segment whose endpoint wasn't stamped (e.g. on machines other than
      // UCX) shows up as 0.
      auto seg = [](double a, double b) { return (a > 0 && b > 0) ? (b - a) : 0.0; };
      double s_orch        = seg(rescale_overhead_start_timer, rescale_t_cleanup_enter);
      double s_commit      = seg(rescale_t_cleanup_enter,      rescale_t_commit_done);
      double s_ep_reinit   = seg(rescale_t_commit_done,        rescale_t_ep_reinit_done);
      double s_post_barrier= seg(rescale_t_ep_reinit_done,     rescale_t_barrier_done);
      double s_to_longjmp  = seg(rescale_t_barrier_done,       rescale_t_longjmp);
      double s_jmp_to_init = seg(rescale_t_longjmp,            rescale_t_converseinit_call);
      double s_lrts_init    = seg(rescale_t_converseinit_call,  rescale_t_lrtsinit_done);
      double s_lrts_to_runpe= seg(rescale_t_lrtsinit_done,      rescale_t_converserunpe_enter);
      double s_common_init  = seg(rescale_t_converserunpe_enter,rescale_t_commoninit_done);
      double s_cci_basics   = seg(rescale_t_converserunpe_enter,rescale_t_cci_basics);
      double s_cci_timers   = seg(rescale_t_cci_basics,         rescale_t_cci_timers);
      double s_cci_tmp      = seg(rescale_t_cci_basics,         rescale_t_cci_tmp);
      double s_cci_timer    = seg(rescale_t_cci_tmp,            rescale_t_cci_timer_only);
      double s_cci_stats    = seg(rescale_t_cci_timer_only,     rescale_t_cci_stats);
      double s_cci_aff_util = seg(rescale_t_cci_stats,          rescale_t_cci_timers);
      double s_cci_handlers = seg(rescale_t_cci_timers,         rescale_t_cci_handlers);
      double s_cci_trace    = seg(rescale_t_cci_handlers,       rescale_t_cci_trace);
      double s_cci_persist  = seg(rescale_t_cci_trace,          rescale_t_cci_persistent);
      double s_cci_ccs      = seg(rescale_t_cci_persistent,     rescale_t_cci_ccs);
      double s_cci_threads  = seg(rescale_t_cci_ccs,            rescale_t_cci_threads);
      double s_cci_isopre   = seg(rescale_t_cci_threads,        rescale_t_cci_iso_predeps);
      double s_cci_isoonly  = seg(rescale_t_cci_iso_predeps,    rescale_t_commoninit_done);
      double s_cci_isomalloc= seg(rescale_t_cci_threads,        rescale_t_commoninit_done);
      double s_to_initcharm = seg(rescale_t_commoninit_done,    rescale_t_initcharm_enter);
      double s_register     = seg(rescale_t_initcharm_enter,    rescale_t_register_done);
      double s_post_register= seg(rescale_t_register_done,      rescale_t_restart_main_enter);
      double s_pr_initcalls = seg(rescale_t_register_done,      rescale_t_pr_initcalls_done);
      double s_pr_topo      = seg(rescale_t_pr_initcalls_done,  rescale_t_pr_topo_done);
      double s_pr_aff       = seg(rescale_t_pr_initcalls_done,  rescale_t_pr_aff_done);
      double s_pr_cputopo   = seg(rescale_t_pr_aff_done,        rescale_t_pr_cputopo_done);
      double s_pr_topomgr   = seg(rescale_t_pr_cputopo_done,    rescale_t_pr_topo_done);
      double s_pr_to_fault  = seg(rescale_t_pr_topo_done,       rescale_t_pr_to_faultfunc);
      double s_pr_dispatch  = seg(rescale_t_pr_to_faultfunc,    rescale_t_restart_main_enter);
      double s_post_restore = seg(rescale_t_restart_main_enter, now) - restore_s;
      CkPrintf("Charm> Rescale timing (PE 0): total=%.6fs restore=%.6fs overhead=%.6fs\n"
               "  orchestration  (cb -> ConverseCleanup)             : %.6fs\n"
               "  coord COMMIT   (cleanup -> commit returned)        : %.6fs\n"
               "  ep reinit      (commit -> UcxReInitEpsFromView)    : %.6fs\n"
               "  post barrier   (ep reinit -> coord::barrier)       : %.6fs\n"
               "  to longjmp     (barrier -> longjmp)                : %.6fs\n"
               "  longjmp->init  (after setjmp -> ConverseInit call) : %.6fs\n"
               "  ConverseInit:\n"
               "    LrtsInit       (Converse entry -> LrtsInit done) : %.6fs\n"
               "    -> ConverseRunPE (LrtsInit -> RunPE entry)       : %.6fs\n"
               "    ConverseCommonInit (RunPE -> CommonInit done)    : %.6fs\n"
               "      cci basics    (RunPE -> CmiIOInit)             : %.6fs\n"
               "      cci timers    (-> CmiInitCPUAffinityUtil)      : %.6fs\n"
               "        cci tmp     (-> CmiTmpInit)                  : %.6fs\n"
               "        cci timer   (-> CmiTimerInit)                : %.6fs\n"
               "        cci stats   (-> CstatsInit)                  : %.6fs\n"
               "        cci affutil (-> CmiInitCPUAffinityUtil)      : %.6fs\n"
               "      cci handlers  (-> CIdleTimeoutInit)            : %.6fs\n"
               "      cci trace     (-> traceInit)                   : %.6fs\n"
               "      cci persist   (-> CmiOnesidedDirectInit)       : %.6fs\n"
               "      cci ccs       (-> CcsInit)                     : %.6fs\n"
               "      cci threads   (-> CmiInitMultipleSend)         : %.6fs\n"
               "      cci isomalloc (-> CmiIsomallocInit/end)        : %.6fs\n"
               "        cci isopre  (CrnInit + CldModuleInit)        : %.6fs\n"
               "        cci isoonly (CmiIsomallocInit only)          : %.6fs\n"
               "    -> _initCharm (CommonInit -> _initCharm entry)   : %.6fs\n"
               "    _register pass  (_initCharm -> register done)    : %.6fs\n"
               "    post-register   (register -> CkRestartMain)      : %.6fs\n"
               "      pr initcalls  (-> _initCallTable.enumerate)    : %.6fs\n"
               "      pr topo       (-> CmiInitCPUTopology+tree)     : %.6fs\n"
               "        pr aff      (-> CmiInit{CPU,Mem}Affinity)    : %.6fs\n"
               "        pr cputopo  (-> CmiInitCPUTopology)          : %.6fs\n"
               "        pr topomgr  (-> TopoManager+tree)            : %.6fs\n"
               "      pr to fault   (-> just before faultFunc call)  : %.6fs\n"
               "      pr dispatch   (faultFunc -> CkRestartMain ent) : %.6fs\n"
               "  post-restore   (after restore I/O -> now)          : %.6fs\n",
               total_s, restore_s, overhead_s,
               s_orch, s_commit, s_ep_reinit, s_post_barrier, s_to_longjmp,
               s_jmp_to_init,
               s_lrts_init, s_lrts_to_runpe, s_common_init,
               s_cci_basics, s_cci_timers,
               s_cci_tmp, s_cci_timer, s_cci_stats, s_cci_aff_util,
               s_cci_handlers, s_cci_trace,
               s_cci_persist, s_cci_ccs, s_cci_threads, s_cci_isomalloc,
               s_cci_isopre, s_cci_isoonly,
               s_to_initcharm,
               s_register, s_post_register,
               s_pr_initcalls, s_pr_topo,
               s_pr_aff, s_pr_cputopo, s_pr_topomgr,
               s_pr_to_fault, s_pr_dispatch,
               s_post_restore);
      rescale_clear_timers();
    }
  }
}

void CkRestartMain(const char* dirname, CkArgMsg *args){
#if CMK_SHRINK_EXPAND
  chkptStartTimer = CmiWallTimer();
  if (CmiMyPe() == 0) rescale_t_restart_main_enter = rescale_wall_now();
	int i;
	
  if (CmiMyRank() == 0) {
    set_in_restart(true);
    _restarted = true;
    CkMemCheckPT::inRestarting = true;
  }

  // Mainchares are live in memory across the longjmp (they were preserved by
  // gating the table allocations in _initCharm), so no restore is needed.

  if (CkMyPe() == 0)
  {
    // Build the rescale broadcast in-memory: PE 0's live readonly data + group
    // table become the source of truth that newcomers will deserialize. The
    // dirname slot in the message format is preserved for receiver-side parser
    // compatibility but is left empty — no file paths are involved any more.
    const int strLen = 0;

    if (!_rescaleResumeCbValid) {
      CmiAbort("[CkRestartMain] _rescaleResumeCb was not stashed before the "
               "rescale longjmp — checkpoint path bypassed?");
    }

    int _numPes = CkNumPes();
    int _numNodes = CkNumNodes();

    PUP::sizer pROsz(PUP::er::IS_CHECKPOINT);
    pROsz | _numPes;
    pROsz | _numNodes;
    pROsz | _rescaleResumeCb;
    CkPupROData(pROsz);
    bool requestStatusLocal = false;
    pROsz | requestStatusLocal;
    const int ROSizeInt = (int)pROsz.size();

    PUP::sizer pGrpsz(PUP::er::IS_CHECKPOINT);
    CkPupGroupData(pGrpsz);
    const int GroupSizeInt = (int)pGrpsz.size();

    const size_t totalSize = CmiMsgHeaderSizeBytes
                           + 2 * sizeof(int) + strLen
                           + ROSizeInt + GroupSizeInt;
    char* msg = (char*) CmiAlloc(totalSize);
    char* buffer = msg + CmiMsgHeaderSizeBytes;
    std::memcpy(buffer, &strLen, sizeof(int));
    buffer += sizeof(int);
    // (no dirname bytes — strLen == 0)
    std::memcpy(buffer, &ROSizeInt, sizeof(int));
    buffer += sizeof(int);

    {
      PUP::toMem pRO(buffer, PUP::er::IS_CHECKPOINT);
      pRO | _numPes;
      pRO | _numNodes;
      pRO | _rescaleResumeCb;
      CkPupROData(pRO);
      pRO | requestStatusLocal;
    }
    buffer += ROSizeInt;

    {
      PUP::toMem pGrp(buffer, PUP::er::IS_CHECKPOINT);
      CkPupGroupData(pGrp);
    }
    buffer += GroupSizeInt;

    _rescaleResumeCbValid = false;

    CmiSetHandler(msg, _shrinkExpandRestartHandlerIdx);
    CmiSyncBroadcastAllAndFree(totalSize, msg);
  }

   	//_initDone();
#endif
}

#if CMK_SHRINK_EXPAND
// NOTE - This function doesn't appear to be called anywhere
// after resume and getting message
void CkResumeRestartMain(char * msg) {
}

int GetNewPeNumber(std::vector<char> avail){
  int mype = CkMyPe();
  int count =0;
  for (int i =0; i <mype; i++){
    if(avail[i] ==0) count++;
  }
  return (mype - count);
}
#endif

// Main chare: initialize system checkpoint manager
class CkCheckpointInit : public Chare {
public:
  CkCheckpointInit(CkArgMsg *msg) {
    _sysChkptWriteMgr = CProxy_CkCheckpointWriteMgr::ckNew();
    _sysChkptMgr = CProxy_CkCheckpointMgr::ckNew();
    delete msg;
  }
  CkCheckpointInit(CkMigrateMessage *m) {delete m;}
};

#include "CkCheckpoint.def.h"
#include "CkCheckpointStatus.def.h"
