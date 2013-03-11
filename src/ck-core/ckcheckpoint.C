/*
Charm++ File: Checkpoint Library
added 01/03/2003 by Chao Huang, chuang10@uiuc.edu

More documentation goes here...
--- Updated 12/14/2003 by Gengbin, gzheng@uiuc.edu
    see ckcheckpoint.h for change log
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sstream>
using std::ostringstream;
#include <errno.h>
#include "charm++.h"
#include "ck.h"
#include "ckcheckpoint.h"

void noopit(const char*, ...)
{}

//#define DEBCHK   CkPrintf
#define DEBCHK noopit

#define DEBUGC(x) x
//#define DEBUGC(x) 

CkGroupID _sysChkptMgr;

typedef struct _GroupInfo{
        CkGroupID gID;
        int MigCtor, DefCtor;
        char name[256];
} GroupInfo;
PUPbytes(GroupInfo)
PUPmarshall(GroupInfo)

int _inrestart = 0;
int _restarted = 0;
int _oldNumPes = 0;
int _chareRestored = 0;

void CkCreateLocalChare(int epIdx, envelope *env);

// help class to find how many array elements
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
		CkGroupID gID = locMgr->ckGetGroupID();
		p|gID;	    // store loc mgr's GID as well for easier restore
                p|idx;
	        p|loc;
		//CkPrintf("[%d] addLocation: ", CkMyPe()), idx.print();
        }
};


extern void _initDone();

static void bdcastRO(void){
	int i;
	//Determine the size of the RODataMessage
	PUP::sizer ps;
	for(i=0;i<_readonlyTable.size();i++) _readonlyTable[i]->pupData(ps);

	//Allocate and fill out the RODataMessage
	envelope *env = _allocEnv(RODataMsg, ps.size());
	PUP::toMem pp((char *)EnvToUsr(env));
	for(i=0;i<_readonlyTable.size();i++) _readonlyTable[i]->pupData(pp);
	
	env->setCount(++_numInitMsgs);
	env->setSrcPe(CkMyPe());
	CmiSetHandler(env, _roRestartHandlerIdx);
	CmiSyncBroadcastAndFree(env->getTotalsize(), (char *)env);
}

// Print out an array index to this string as decimal fields
// separated by underscores.
void printIndex(const CkArrayIndex &idx,char *dest) {
	const int *idxData=idx.data();
	for (int i=0;i<idx.nInts;i++) {
		sprintf(dest,"%s%d",i==0?"":"_", idxData[i]);
		dest+=strlen(dest);
	}
}

static void checkpointOne(const char* dirname, CkCallback& cb);

static void addPartitionDirectory(ostringstream &path) {
        if (CmiNumPartitions() > 1) {
          path << "/part-" << CmiMyPartition() << '/';
        }
}

static FILE* openCheckpointFile(const char *dirname, const char *basename,
                                const char *mode, int id = -1) {
        ostringstream out;
        out << dirname << '/';
        addPartitionDirectory(out);
        out << basename;
        if (id != -1)
                out << '_' << id;
        out << ".dat";

        FILE *fp = CmiFopen(out.str().c_str(), mode);
        if (!fp) {
                ostringstream error;
                error << "PE " << CkMyPe() << " failed to open checkpoint file: " << out.str()
                      << ", mode: " << mode << " status: " << strerror(errno);
                CkAbort(error.str().c_str());
        }
        return fp;
}

// broadcast
void CkCheckpointMgr::Checkpoint(const char *dirname, CkCallback& cb){
	chkptStartTimer = CmiWallTimer();
	// every body make dir in case it is local directory
	CmiMkdir(dirname);

        if (CmiNumPartitions() > 1) {
          ostringstream partDir;
          partDir << dirname;
          addPartitionDirectory(partDir);
          CmiMkdir(partDir.str().c_str());
        }

	if (CkMyPe() == 0) {
          checkpointOne(dirname, cb);
 	}

#ifndef CMK_CHARE_USE_PTR
	// save plain singleton chares into Chares.dat
	FILE* fChares = openCheckpointFile(dirname, "Chares", "wb", CkMyPe());
	PUP::toDisk pChares(fChares);
	CkPupChareData(pChares);
	CmiFclose(fChares);
#endif

	// save groups into Groups.dat
	// content of the file: numGroups, GroupInfo[numGroups], _groupTable(PUP'ed), groups(PUP'ed)
	FILE* fGroups = openCheckpointFile(dirname, "Groups", "wb", CkMyPe());
	PUP::toDisk pGroups(fGroups);
#if (defined(_FAULT_MLOG_) || defined(_FAULT_CAUSAL_))
    CkPupGroupData(pGroups,CmiTrue);
#else
    CkPupGroupData(pGroups);
#endif
	CmiFclose(fGroups);

	// save nodegroups into NodeGroups.dat
	// content of the file: numNodeGroups, GroupInfo[numNodeGroups], _nodeGroupTable(PUP'ed), nodegroups(PUP'ed)
	if (CkMyRank() == 0) {
	  FILE* fNodeGroups = openCheckpointFile(dirname, "NodeGroups", "wb", CkMyNode());
	  PUP::toDisk pNodeGroups(fNodeGroups);
#if (defined(_FAULT_MLOG_) || defined(_FAULT_CAUSAL_))
      CkPupNodeGroupData(pNodeGroups,CmiTrue);
#else
      CkPupNodeGroupData(pNodeGroups);
#endif
	  CmiFclose(fNodeGroups);
  	}

	//DEBCHK("[%d]CkCheckpointMgr::Checkpoint called dirname={%s}\n",CkMyPe(),dirname);
	FILE *datFile = openCheckpointFile(dirname, "arr", "wb", CkMyPe());
	PUP::toDisk  p(datFile);
	CkPupArrayElementsData(p);
	CmiFclose(datFile);

#if CMK_HAS_SYNC && ! CMK_DISABLE_SYNC
	system("sync");
#endif

	restartCB = cb;
	DEBCHK("[%d]restartCB installed\n",CkMyPe());
	CkCallback localcb(CkIndex_CkCheckpointMgr::SendRestartCB(NULL),0,thisgroup);
	//contribute(0,NULL,CkReduction::sum_int,localcb);
	barrier(localcb);
}

void CkCheckpointMgr::SendRestartCB(CkReductionMsg *m){ 
	delete m; 
	DEBCHK("[%d]Sending out the cb\n",CkMyPe());
	CkPrintf("Checkpoint to disk finished in %fs, sending out the cb...\n", CmiWallTimer() - chkptStartTimer);
	restartCB.send(); 
}

void CkPupROData(PUP::er &p)
{
	int _numReadonlies = 0;
	if (!p.isUnpacking()) _numReadonlies=_readonlyTable.size();
        p|_numReadonlies;
	if (p.isUnpacking()) {
	  if (_numReadonlies != _readonlyTable.size())
	    CkAbort("You cannot add readonlies and restore from checkpoint...");
	}
	for(int i=0;i<_numReadonlies;i++) _readonlyTable[i]->pupData(p);
}

// handle main chare
void CkPupMainChareData(PUP::er &p, CkArgMsg *args)
{
	int nMains=_mainTable.size();
	DEBCHK("[%d] CkPupMainChareData %s: nMains = %d\n", CkMyPe(),p.typeString(),nMains);
	for(int i=0;i<nMains;i++){  /* Create all mainchares */
		ChareInfo *entry = _chareTable[_mainTable[i]->chareIdx];
		int entryMigCtor = entry->getMigCtor();
		if(entryMigCtor!=-1) {
			Chare* obj;
			if (p.isUnpacking()) {
				int size = entry->size;
				DEBCHK("MainChare PUP'ed: name = %s, idx = %d, size = %d\n", entry->name, i, size);
				obj = (Chare*)malloc(size);
				_MEMCHECK(obj);
				_mainTable[i]->setObj(obj);
				//void *m = CkAllocSysMsg();
				_entryTable[entryMigCtor]->call(args, obj);
			}
			else 
			 	obj = (Chare *)_mainTable[i]->getObj();
			obj->pup(p);
		}
	}
	// to update mainchare proxy
	// only readonly variables of Chare Proxy is taken care of here;
	// in general, if chare proxy is contained in some data structure
	// for example CkCallback, it is user's responsibility to
	// update them after restarting
	if (p.isUnpacking() && CkMyPe()==0)
		bdcastRO();
}

#ifndef CMK_CHARE_USE_PTR

CkpvExtern(CkVec<void *>, chare_objs);
CkpvExtern(CkVec<int>, chare_types);
CkpvExtern(CkVec<VidBlock *>, vidblocks);

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
	if (p.isUnpacking()) {
		int migCtor = _chareTable[chare_type]->migCtor;
		if(migCtor==-1) {
			char buf[512];
			sprintf(buf,"Chare %s needs a migration constructor and PUP'er routine for restart.\n", _chareTable[chare_type]->name);
			CkAbort(buf);
		}
	        void *m = CkAllocSysMsg();
	        envelope* env = UsrToEnv((CkMessage *)m);
		CkCreateLocalChare(migCtor, env);
		CkFreeSysMsg(m);
	}
	Chare *obj = (Chare*)CkpvAccess(chare_objs)[i];
	obj->pup(p);
  }

  if (!p.isUnpacking()) n = CkpvAccess(vidblocks).size();
  p|n;
  for (i=0; i<n; i++) {
	VidBlock *v;
	if (p.isUnpacking()) {
		v = new VidBlock();
		CkpvAccess(vidblocks).push_back(v);
	}
	else
		v = CkpvAccess(vidblocks)[i];
	v->pup(p);
  }
}
#else
void CkPupChareData(PUP::er &p)
{
   // not implemented
}
#endif

#if (defined(_FAULT_MLOG_) || defined(_FAULT_CAUSAL_))
// handle GroupTable and data
void CkPupGroupData(PUP::er &p, CmiBool create)
{
	int numGroups, i;

	if (!p.isUnpacking()) {
	  numGroups = CkpvAccess(_groupIDTable)->size();
	}
	p|numGroups;
	if (p.isUnpacking()) {
	  if(CkMyPe()==0)  
            CkpvAccess(_numGroups) = numGroups+1; 
          else 
	    CkpvAccess(_numGroups) = 1;
	}
	DEBCHK("[%d] CkPupGroupData %s: numGroups = %d\n", CkMyPe(),p.typeString(),numGroups);

	GroupInfo *tmpInfo = new GroupInfo [numGroups];
	if (!p.isUnpacking()) {
	  for(i=0;i<numGroups;i++) {
		tmpInfo[i].gID = (*CkpvAccess(_groupIDTable))[i];
		TableEntry ent = CkpvAccess(_groupTable)->find(tmpInfo[i].gID);
		tmpInfo[i].MigCtor = _chareTable[ent.getcIdx()]->migCtor;
		tmpInfo[i].DefCtor = _chareTable[ent.getcIdx()]->defCtor;
		strncpy(tmpInfo[i].name,_chareTable[ent.getcIdx()]->name,255);
		//CkPrintf("[%d] CkPupGroupData: %s group %s \n", CkMyPe(), p.typeString(), tmpInfo[i].name);

		if(tmpInfo[i].MigCtor==-1) {
			char buf[512];
			sprintf(buf,"Group %s needs a migration constructor and PUP'er routine for restart.\n", tmpInfo[i].name);
			CkAbort(buf);
		}
	  }
  	}
	for (i=0; i<numGroups; i++) p|tmpInfo[i];

	for(i=0;i<numGroups;i++) 
	{
	  CkGroupID gID = tmpInfo[i].gID;
	  if (p.isUnpacking()) {
	    //CkpvAccess(_groupIDTable)->push_back(gID);
	    int eIdx = tmpInfo[i].MigCtor;
	    // error checking
	    if (eIdx == -1) {
	      CkPrintf("[%d] ERROR> Group %s's migration constructor is not defined!\n", CkMyPe(), tmpInfo[i].name); CkAbort("Abort");
	    }
	    void *m = CkAllocSysMsg();
	    envelope* env = UsrToEnv((CkMessage *)m);
		if(create)
		    CkCreateLocalGroup(gID, eIdx, env);
	  }   // end of unPacking
	  IrrGroup *gobj = CkpvAccess(_groupTable)->find(gID).getObj();
	  // if using migration constructor, you'd better have a pup
	  	if(!create)
			gobj->mlogData->teamRecoveryFlag = 1;
          gobj->pup(p);
         // CkPrintf("Group PUP'ed: gid = %d, name = %s\n",gobj->ckGetGroupID().idx, tmpInfo[i].name);
	}
	delete [] tmpInfo;
}

// handle NodeGroupTable and data
void CkPupNodeGroupData(PUP::er &p, CmiBool create)
{
	int numNodeGroups, i;
	if (!p.isUnpacking()) {
	  numNodeGroups = CksvAccess(_nodeGroupIDTable).size();
	}
	p|numNodeGroups;
	if (p.isUnpacking()) {
	  if(CkMyPe()==0){ CksvAccess(_numNodeGroups) = numNodeGroups+1; }
	  else { CksvAccess(_numNodeGroups) = 1; }
	}

	GroupInfo *tmpInfo = new GroupInfo [numNodeGroups];
	if (!p.isUnpacking()) {
	  for(i=0;i<numNodeGroups;i++) {
		tmpInfo[i].gID = CksvAccess(_nodeGroupIDTable)[i];
		TableEntry ent2 = CksvAccess(_nodeGroupTable)->find(tmpInfo[i].gID);
		tmpInfo[i].MigCtor = _chareTable[ent2.getcIdx()]->migCtor;
		if(tmpInfo[i].MigCtor==-1) {
			char buf[512];
			sprintf(buf,"NodeGroup %s either need a migration constructor and\n\
				     declared as [migratable] in .ci to be able to checkpoint.",\
				     _chareTable[ent2.getcIdx()]->name);
			CkAbort(buf);
		}
	  }
	}
	for (i=0; i<numNodeGroups; i++) p|tmpInfo[i];
	for (i=0;i<numNodeGroups;i++) {
		CkGroupID gID = tmpInfo[i].gID;
		if (p.isUnpacking()) {
			//CksvAccess(_nodeGroupIDTable).push_back(gID);
			int eIdx = tmpInfo[i].MigCtor;
			void *m = CkAllocSysMsg();
			envelope* env = UsrToEnv((CkMessage *)m);
			if(create){
				CkCreateLocalNodeGroup(gID, eIdx, env);
			}
		}
		TableEntry ent2 = CksvAccess(_nodeGroupTable)->find(gID);
		IrrGroup *obj = ent2.getObj();
		obj->pup(p);
	}
	delete [] tmpInfo;
}
#else
// handle GroupTable and data
void CkPupGroupData(PUP::er &p)
{
	int numGroups = 0, i;

	if (!p.isUnpacking()) {
	  numGroups = CkpvAccess(_groupIDTable)->size();
	}
	p|numGroups;
	if (p.isUnpacking()) {
	  if(CkMyPe()==0)  
            CkpvAccess(_numGroups) = numGroups+1; 
          else 
	    CkpvAccess(_numGroups) = 1;
	}
	DEBCHK("[%d] CkPupGroupData %s: numGroups = %d\n", CkMyPe(),p.typeString(),numGroups);

	GroupInfo *tmpInfo = new GroupInfo [numGroups];
	if (!p.isUnpacking()) {
	  for(i=0;i<numGroups;i++) {
		tmpInfo[i].gID = (*CkpvAccess(_groupIDTable))[i];
		TableEntry ent = CkpvAccess(_groupTable)->find(tmpInfo[i].gID);
		tmpInfo[i].MigCtor = _chareTable[ent.getcIdx()]->migCtor;
		tmpInfo[i].DefCtor = _chareTable[ent.getcIdx()]->defCtor;
		strncpy(tmpInfo[i].name,_chareTable[ent.getcIdx()]->name,255);
		DEBCHK("[%d] CkPupGroupData: %s group %s \n",
			CkMyPe(), p.typeString(), tmpInfo[i].name);

		if(tmpInfo[i].MigCtor==-1) {
			char buf[512];
			sprintf(buf,"Group %s needs a migration constructor and PUP'er routine for restart.\n", tmpInfo[i].name);
			CkAbort(buf);
		}
	  }
  	}
	for (i=0; i<numGroups; i++) p|tmpInfo[i];

	for(i=0;i<numGroups;i++) 
	{
	  CkGroupID gID = tmpInfo[i].gID;
	  if (p.isUnpacking()) {
	    //CkpvAccess(_groupIDTable)->push_back(gID);
	    int eIdx = tmpInfo[i].MigCtor;
	    // error checking
	    if (eIdx == -1) {
	      CkPrintf("[%d] ERROR> Group %s's migration constructor is not defined!\n", CkMyPe(), tmpInfo[i].name); CkAbort("Abort");
	    }
	    void *m = CkAllocSysMsg();
	    envelope* env = UsrToEnv((CkMessage *)m);
	    CkCreateLocalGroup(gID, eIdx, env);
	  }   // end of unPacking
	  IrrGroup *gobj = CkpvAccess(_groupTable)->find(gID).getObj();
	  // if using migration constructor, you'd better have a pup
          gobj->pup(p);
          DEBCHK("Group PUP'ed: gid = %d, name = %s\n",
			gobj->ckGetGroupID().idx, tmpInfo[i].name);
	}
	delete [] tmpInfo;
}

// handle NodeGroupTable and data
void CkPupNodeGroupData(PUP::er &p)
{
	int numNodeGroups = 0, i;
	if (!p.isUnpacking()) {
	  numNodeGroups = CksvAccess(_nodeGroupIDTable).size();
	}
	p|numNodeGroups;
	if (p.isUnpacking()) {
	  if(CkMyPe()==0){ CksvAccess(_numNodeGroups) = numNodeGroups+1; }
	  else { CksvAccess(_numNodeGroups) = 1; }
	}
	DEBCHK("[%d] CkPupNodeGroupData %s: numNodeGroups = %d\n",CkMyPe(),p.typeString(),numNodeGroups);

	GroupInfo *tmpInfo = new GroupInfo [numNodeGroups];
	if (!p.isUnpacking()) {
	  for(i=0;i<numNodeGroups;i++) {
		tmpInfo[i].gID = CksvAccess(_nodeGroupIDTable)[i];
		TableEntry ent2 = CksvAccess(_nodeGroupTable)->find(tmpInfo[i].gID);
		tmpInfo[i].MigCtor = _chareTable[ent2.getcIdx()]->migCtor;
		if(tmpInfo[i].MigCtor==-1) {
			char buf[512];
			sprintf(buf,"NodeGroup %s either need a migration constructor and\n\
				     declared as [migratable] in .ci to be able to checkpoint.",\
				     _chareTable[ent2.getcIdx()]->name);
			CkAbort(buf);
		}
	  }
	}
	for (i=0; i<numNodeGroups; i++) p|tmpInfo[i];
	for (i=0;i<numNodeGroups;i++) {
		CkGroupID gID = tmpInfo[i].gID;
		if (p.isUnpacking()) {
			//CksvAccess(_nodeGroupIDTable).push_back(gID);
			int eIdx = tmpInfo[i].MigCtor;
			void *m = CkAllocSysMsg();
			envelope* env = UsrToEnv((CkMessage *)m);
			CkCreateLocalNodeGroup(gID, eIdx, env);
		}
		TableEntry ent2 = CksvAccess(_nodeGroupTable)->find(gID);
		IrrGroup *obj = ent2.getObj();
		obj->pup(p);
		DEBCHK("Nodegroup PUP'ed: gid = %d, name = %s\n",
			obj->ckGetGroupID().idx,
			_chareTable[ent2.getcIdx()]->name);
	}
	delete [] tmpInfo;
}
#endif

// handle chare array elements for this processor
void CkPupArrayElementsData(PUP::er &p, int notifyListeners)
{
 	int i;
	// safe in both packing/unpakcing at this stage
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
	  // let CkLocMgr to iterate and store every array elements
          CKLOCMGR_LOOP(ElementCheckpointer chk(mgr, p); mgr->iterate(chk););
        }
	else {
	  // loop and create all array elements ourselves
	  //CkPrintf("total chare array cnts: %d\n", numElements);
	  for (int i=0; i<numElements; i++) {
		CkGroupID gID;
		CkArrayIndex idx;
		p|gID;
                p|idx;
		CkLocMgr *mgr = (CkLocMgr*)CkpvAccess(_groupTable)->find(gID).getObj();
		if (notifyListeners){
  		  mgr->resume(idx,p,CmiTrue);
		}
                else{
  		  mgr->restore(idx,p);
		}
	  }
	}
	// finish up
        if (notifyListeners)
        for(i=0;i<numGroups;i++) {
                IrrGroup *obj = CkpvAccess(_groupTable)->find((*CkpvAccess(_groupIDTable))[i]).getObj();
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
#if (defined(_FAULT_MLOG_) || defined(_FAULT_CAUSAL_))
    CkPupGroupData(p,CmiTrue);
#else
    CkPupGroupData(p);
#endif

    // save nodegroups
    if(CkMyRank()==0) {
#if (defined(_FAULT_MLOG_) || defined(_FAULT_CAUSAL_))
        CkPupNodeGroupData(p,CmiTrue);	
#else
        CkPupNodeGroupData(p);
#endif
    }

    // pup array elements
    CkPupArrayElementsData(p);
}

// called only on pe 0
static void checkpointOne(const char* dirname, CkCallback& cb){
	CmiAssert(CkMyPe()==0);
	char filename[1024];
	
	// save readonlys, and callback BTW
	FILE* fRO = openCheckpointFile(dirname, "RO", "wb");
	PUP::toDisk pRO(fRO);
	int _numPes = CkNumPes();
	pRO|_numPes;
	CkPupROData(pRO);
	pRO|cb;
	CmiFclose(fRO);

	// save mainchares into MainChares.dat
	{
		FILE* fMain = openCheckpointFile(dirname, "MainChares", "wb");
		PUP::toDisk pMain(fMain);
		CkPupMainChareData(pMain, NULL);
		CmiFclose(fMain);
	}
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

void CkStartCheckpoint(const char* dirname,const CkCallback& cb)
{

	CkPrintf("[%d] Checkpoint starting in %s\n", CkMyPe(), dirname);
	
	// hand over to checkpoint managers for per-processor checkpointing
	CProxy_CkCheckpointMgr(_sysChkptMgr).Checkpoint(dirname, cb);
}

/**
  * Restart: There's no such object as restart manager is created
  *          because a group cannot restore itself anyway.
  *          The mechanism exists as converse code and get invoked by
  *          broadcast message.
  **/

void CkRestartMain(const char* dirname, CkArgMsg *args){
	int i;
	char filename[1024];
	CkCallback cb;
	
        _inrestart = 1;
	_restarted = 1;
	CkMemCheckPT::inRestarting = 1;

	// restore readonlys
	FILE* fRO = openCheckpointFile(dirname, "RO", "rb");
	int _numPes = -1;
	PUP::fromDisk pRO(fRO);
	pRO|_numPes;
	CkPupROData(pRO);
	pRO|cb;
	CmiFclose(fRO);
	DEBCHK("[%d]CkRestartMain: readonlys restored\n",CkMyPe());
        _oldNumPes = _numPes;

	CmiNodeBarrier();

	// restore mainchares
	FILE* fMain = openCheckpointFile(dirname, "MainChares", "rb");
	if(fMain && CkMyPe()==0){ // only main chares have been checkpointed, we restart on PE0
		PUP::fromDisk pMain(fMain);
		CkPupMainChareData(pMain, args);
		CmiFclose(fMain);
		DEBCHK("[%d]CkRestartMain: mainchares restored\n",CkMyPe());
		//bdcastRO(); // moved to CkPupMainChareData()
	}
	
#ifndef CMK_CHARE_USE_PTR
	// restore chares only when number of pes is the same 
	if(CkNumPes() == _numPes) {
		FILE* fChares = openCheckpointFile(dirname, "Chares", "rb", CkMyPe());
		PUP::fromDisk pChares(fChares);
		CkPupChareData(pChares);
		CmiFclose(fChares);
		_chareRestored = 1;
	}
#endif

	// restore groups
	// content of the file: numGroups, GroupInfo[numGroups], _groupTable(PUP'ed), groups(PUP'ed)
	// restore from PE0's copy if shrink/expand
	FILE* fGroups = openCheckpointFile(dirname, "Groups", "rb",
                                           (CkNumPes() == _numPes) ? CkMyPe() : 0);
	PUP::fromDisk pGroups(fGroups);
#if (defined(_FAULT_MLOG_) || defined(_FAULT_CAUSAL_))
    CkPupGroupData(pGroups,CmiTrue);
#else
    CkPupGroupData(pGroups);
#endif
	CmiFclose(fGroups);

	// restore nodegroups
	// content of the file: numNodeGroups, GroupInfo[numNodeGroups], _nodeGroupTable(PUP'ed), nodegroups(PUP'ed)
	if(CkMyRank()==0){
                FILE* fNodeGroups = openCheckpointFile(dirname, "NodeGroups", "rb",
                                                       (CkNumPes() == _numPes) ? CkMyNode() : 0);
                PUP::fromDisk pNodeGroups(fNodeGroups);
#if (defined(_FAULT_MLOG_) || defined(_FAULT_CAUSAL_))
        CkPupNodeGroupData(pNodeGroups,CmiTrue);
#else
        CkPupNodeGroupData(pNodeGroups);
#endif
		CmiFclose(fNodeGroups);
	}

	// for each location, restore arrays
	//DEBCHK("[%d]Trying to find location manager\n",CkMyPe());
	DEBCHK("[%d]Number of PE: %d -> %d\n",CkMyPe(),_numPes,CkNumPes());
	if(CkMyPe() < _numPes) 	// in normal range: restore, otherwise, do nothing
          for (i=0; i<_numPes;i++) {
            if (i%CkNumPes() == CkMyPe()) {
              FILE *datFile = openCheckpointFile(dirname, "arr", "rb", i);
	      PUP::fromDisk  p(datFile);
	      CkPupArrayElementsData(p);
	      CmiFclose(datFile);
            }
	  }

        _inrestart = 0;

   	_initDone();
	CkMemCheckPT::inRestarting = 0;
	if(CkMyPe()==0) {
		CmiPrintf("[%d]CkRestartMain done. sending out callback.\n",CkMyPe());
		
		cb.send();
	}
}

// Main chare: initialize system checkpoint manager
class CkCheckpointInit : public Chare {
public:
  CkCheckpointInit(CkArgMsg *msg) {
    _sysChkptMgr = CProxy_CkCheckpointMgr::ckNew();
    delete msg;
  }
  CkCheckpointInit(CkMigrateMessage *m) {delete m;}
};

#include "CkCheckpoint.def.h"

