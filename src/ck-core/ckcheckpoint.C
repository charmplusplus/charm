/*
Charm++ File: Checkpoint Library
added 01/03/2003 by Chao Huang, chuang10@uiuc.edu

More documentation goes here...
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "charm++.h"
#include "ck.h"
//#include "ckcheckpoint.h"

#if 1
#define DEBCHK CkPrintf
#else
#define DEBCHK //CkPrintf
#endif

CkGroupID _sysChkptMgr;

// helper class to pup all elements that belong to same ckLocMgr
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
                CkArrayIndexMax idx=loc.getIndex();
		//CkPrintf("[%d] addLocation: ", CkMyPe()), idx.print();
		CkGroupID gID = locMgr->ckGetGroupID();
		p|gID;
                p|idx;
	        p|loc;
        }
};


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
	CmiSetHandler(env, _roHandlerIdx);
	CmiSyncBroadcastAndFree(env->getTotalsize(), (char *)env);
	CpvAccess(_qd)->create(CkNumPes()-1);
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

void CkCheckpointMgr::Checkpoint(const char *dirname, CkCallback& cb){
	//DEBCHK("[%d]CkCheckpointMgr::Checkpoint called dirname={%s}\n",CkMyPe(),dirname);
	IrrGroup* obj;
	int numGroups = CkpvAccess(_groupIDTable)->size();

	char fileName[1024];
	sprintf(fileName,"%s/arr_%d.dat",dirname, CkMyPe());
	FILE *datFile=fopen(fileName,"wb");
	if (datFile==NULL) CkAbort("Could not create data file");
	PUP::toDisk  p(datFile);
	CkPupArrayElementsData(p);
	fclose(datFile);

	restartCB = cb;
	DEBCHK("[%d]restartCB installed\n",CkMyPe());
	CkCallback localcb(CkIndex_CkCheckpointMgr::SendRestartCB(NULL),thisgroup);
	contribute(sizeof(int),&numGroups,CkReduction::sum_int,localcb);
}

void CkCheckpointMgr::SendRestartCB(CkReductionMsg *m){ 
	delete m; 
	DEBCHK("[%d]Sending out the cb\n",CkMyPe());
	restartCB.send(); 
}

void CkPupROData(PUP::er &p)
{
	int _numReadonlies;
	if (!p.isUnpacking()) _numReadonlies=_readonlyTable.size();
        p|_numReadonlies;
	if (p.isUnpacking()) {
	  if (_numReadonlies != _readonlyTable.size())
	    CkAbort("You cannot add readonlies and restore from checkpoint...");
	}
	for(int i=0;i<_numReadonlies;i++) _readonlyTable[i]->pupData(p);
}

// handle main chare
void CkPupMainChareData(PUP::er &p)
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
				void *m = CkAllocSysMsg();
				_entryTable[entryMigCtor]->call(m, obj);
			}
			else 
			 	obj = (Chare *)_mainTable[i]->getObj();
			obj->pup(p);
		}
	}
	// to update mainchare proxy
	// only readonly variables of Chare Proxy is taken care of here
	// in general, if chare proxy is contained in some data structure
	// for example CkCallback, it is user's responsibility to
	// update them after restarting
	if (p.isUnpacking() && CkMyPe()==0)
		bdcastRO();
}

// handle GroupTable and data
void CkPupGroupData(PUP::er &p)
{
	int numGroups, i;

	if (!p.isUnpacking()) {
	  numGroups = CkpvAccess(_groupIDTable)->size();
	}
	p|numGroups;
	if (p.isUnpacking()) {
	  if(CkMyPe()==0) { CkpvAccess(_numGroups) = numGroups+1; }else{ CkpvAccess(_numGroups) = 1; }
	}
	DEBCHK("[%d] CkPupGroupData %s: numGroups = %d\n", CkMyPe(),p.typeString(),numGroups);

	GroupInfo *tmpInfo = new GroupInfo [numGroups];
	if (!p.isUnpacking()) {
	  for(i=0;i<numGroups;i++) {
		tmpInfo[i].gID = (*CkpvAccess(_groupIDTable))[i];
		TableEntry ent = CkpvAccess(_groupTable)->find(tmpInfo[i].gID);
		tmpInfo[i].useDefCtor = ent.getObj()->useDefCtor();
		tmpInfo[i].MigCtor = _chareTable[ent.getcIdx()]->migCtor;
		tmpInfo[i].DefCtor = _chareTable[ent.getcIdx()]->defCtor;
		strncpy(tmpInfo[i].name,_chareTable[ent.getcIdx()]->name,255);
		DEBCHK("[%d] CkPupGroupData: %s group %s has useDefCtor=%d\n",
			CkMyPe(), p.typeString(), tmpInfo[i].name,
			tmpInfo[i].useDefCtor);

		if(tmpInfo[i].useDefCtor==0 && tmpInfo[i].MigCtor==-1) {
			char buf[512];
			sprintf(buf,"Group %s needs a migration constructor and PUP'er routine for restart.\n",
				     tmpInfo[i].name);
			CkAbort(buf);
		}
	  }
  	}
	for (i=0; i<numGroups; i++) p|tmpInfo[i];

	for(i=0;i<numGroups;i++) {
		CkGroupID gID = tmpInfo[i].gID;
		if (p.isUnpacking()) {
		  //CkpvAccess(_groupIDTable)->push_back(gID);
		  int eIdx = (tmpInfo[i].useDefCtor)?(tmpInfo[i].DefCtor):(tmpInfo[i].MigCtor);
		  void *m = CkAllocSysMsg();
		  envelope* env = UsrToEnv((CkMessage *)m);
		  CkCreateLocalGroup(gID, eIdx, env);
		}
		IrrGroup *gobj = CkpvAccess(_groupTable)->find(gID).getObj();
		if(!tmpInfo[i].useDefCtor) {
                        gobj->pup(p);
                        DEBCHK("Group PUP'ed: gid = %d, name = %s\n",
				gobj->ckGetGroupID().idx,
				tmpInfo[i].name);
		}else{
                        DEBCHK("Group NOT PUP'ed : gid = %d, name = %s\n",
				gobj->ckGetGroupID().idx,
				tmpInfo[i].name);
		}
	}
	delete [] tmpInfo;
}

// handle NodeGroupTable and data
void CkPupNodeGroupData(PUP::er &p)
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
	DEBCHK("[%d] CkPupNodeGroupData %s: numNodeGroups = %d\n",CkMyPe(),p.typeString(),numNodeGroups);

	GroupInfo *tmpInfo2 = new GroupInfo [numNodeGroups];
	if (!p.isUnpacking()) {
	  for(i=0;i<numNodeGroups;i++) {
		tmpInfo2[i].gID = CksvAccess(_nodeGroupIDTable)[i];
		TableEntry ent2 = CksvAccess(_nodeGroupTable)->find(tmpInfo2[i].gID);
		tmpInfo2[i].MigCtor = _chareTable[ent2.getcIdx()]->migCtor;
		if(tmpInfo2[i].MigCtor==-1) {
			char buf[512];
			sprintf(buf,"NodeGroup %s either need a migration constructor and\n\
				     declared as [migratable] in .ci to be able to checkpoint.",\
				     _chareTable[ent2.getcIdx()]->name);
			CkAbort(buf);
		}
	  }
	}
	for (i=0; i<numNodeGroups; i++) p|tmpInfo2[i];
	for(i=0;i<numNodeGroups;i++) {
		CkGroupID gID = tmpInfo2[i].gID;
		if (p.isUnpacking()) {
			CksvAccess(_nodeGroupIDTable).push_back(gID);
			int eIdx = tmpInfo2[i].MigCtor;
			void *m = CkAllocSysMsg();
			envelope* env = UsrToEnv((CkMessage *)m);
			CkCreateLocalNodeGroup(gID, eIdx, env);
		}
		TableEntry ent2 = CksvAccess(_nodeGroupTable)->find(gID);
		ent2.getObj()->pup(p);
		DEBCHK("Nodegroup PUP'ed: gid = %d, name = %s\n",
			ent2.getObj()->ckGetGroupID().idx,
			_chareTable[ent2.getcIdx()]->name);
	}
	delete [] tmpInfo2;
}

// handle chare array elements for this processor
void CkPupArrayElementsData(PUP::er &p)
{
	// safe in both packing/unpakcing
        int numGroups = CkpvAccess(_groupIDTable)->size();

	int numElements;
	if (!p.isUnpacking()) {
	  ElementCounter  counter;
          for(int i=0;i<numGroups;i++) {
                IrrGroup *obj = CkpvAccess(_groupTable)->find((*CkpvAccess(_groupIDTable))[i]).getObj();
                if(obj->isLocMgr())  ((CkLocMgr*)obj)->iterate(counter);
          }
          numElements = counter.getCount();
	}
	p|numElements;

	DEBCHK("[%d] CkPupArrayElementsData %s numGroups:%d numElements:%d \n",CkMyPe(),p.typeString(), numGroups, numElements);

	if (!p.isUnpacking())
	{
          for(int i=0;i<numGroups;i++) {
                IrrGroup *obj = CkpvAccess(_groupTable)->find((*CkpvAccess(_groupIDTable))[i]).getObj();
                if(obj->isLocMgr()){
			CmiPrintf("Found a location manager\n");
			CkLocMgr *mgr = (CkLocMgr*)obj;
                        ElementCheckpointer chk(mgr, p);
                        mgr->iterate(chk);
                }
	  }
        }
	else {
	  for (int i=0; i<numElements; i++) {
		CkGroupID gID;
		CkArrayIndexMax idx;
		p|gID;
                p|idx;
		CkLocMgr *mgr = (CkLocMgr*)CkpvAccess(_groupTable)->find(gID).getObj();
		mgr->resume(idx,p);
	  }
	}
	// finish up
        for(int i=0;i<numGroups;i++) {
                IrrGroup *obj = CkpvAccess(_groupTable)->find((*CkpvAccess(_groupIDTable))[i]).getObj();
	  	obj->ckJustMigrated();
	}
}

void CkStartCheckpoint(char* dirname,const CkCallback& cb){
	int i;
	char filename[1024];
	CkPrintf("[%d] Checkpoint starting in %s\n", CkMyPe(), dirname);
	CmiMkdir(dirname);
	
	// save readonlys, and callback BTW
	sprintf(filename,"%s/RO.dat",dirname);
	FILE* fRO = fopen(filename,"wb");
	if(!fRO) CkAbort("Failed to create checkpoint file for readonly data!");
	int _numPes = CkNumPes();
	fwrite(&_numPes,sizeof(int),1,fRO);
	PUP::toDisk pRO(fRO);
	CkPupROData(pRO);
	fwrite(&cb,sizeof(CkCallback),1,fRO);
	fclose(fRO);

	// save mainchares into MainChares.dat
	if(CkMyPe()==0){
		sprintf(filename,"%s/MainChares.dat",dirname);
		FILE* fMain = fopen(filename,"wb");
		if(!fMain) CkAbort("Failed to open checkpoint file for mainchare data!");
		PUP::toDisk pMain(fMain);
		CkPupMainChareData(pMain);
		fclose(fMain);
	}
	
	// save groups into Groups.dat
	// content of the file: numGroups, GroupInfo[numGroups], _groupTable(PUP'ed), groups(PUP'ed)
	sprintf(filename,"%s/Groups.dat",dirname);
	FILE* fGroups = fopen(filename,"wb");
	if(!fGroups) CkAbort("Failed to create checkpoint file for group table!");
	PUP::toDisk pGroups(fGroups);
	CkPupGroupData(pGroups);
	fclose(fGroups);

	// save nodegroups into NodeGroups.dat
	// content of the file: numNodeGroups, GroupInfo[numNodeGroups], _nodeGroupTable(PUP'ed), nodegroups(PUP'ed)
	sprintf(filename,"%s/NodeGroups.dat",dirname);
	FILE* fNodeGroups = fopen(filename,"wb");
	if(!fNodeGroups) CkAbort("Failed to create checkpoint file for nodegroup table!");
	PUP::toDisk pNodeGroups(fNodeGroups);
	CkPupNodeGroupData(pNodeGroups);
	fclose(fNodeGroups);

	// hand over to checkpoint managers for per-processor checkpointing
	CProxy_CkCheckpointMgr(_sysChkptMgr).Checkpoint((char *)dirname, cb);
}

/**
  * Restart: There's no such object as restart manager is created
  *          because a group cannot restore itself anyway.
  *          The mechanism exists as converse code and get invoked by
  *          broadcast message.
  **/

void CkRestartMain(const char* dirname){
	int i;
	char filename[1024];
	CkCallback cb;
	
	// restore readonlys
	sprintf(filename,"%s/RO.dat",dirname);
	FILE* fRO = fopen(filename,"rb");
	if(!fRO) CkAbort("Failed to open checkpoint file for readonly data!");
	int _numPes = -1;
	fread(&_numPes,sizeof(int),1,fRO);
	PUP::fromDisk pRO(fRO);
	CkPupROData(pRO);
	fread(&cb,sizeof(CkCallback),1,fRO);
	fclose(fRO);
	DEBCHK("[%d]CkRestartMain: readonlys restored\n",CkMyPe());

	// restore mainchares
	sprintf(filename,"%s/MainChares.dat",dirname);
	FILE* fMain = fopen(filename,"rb");
	if(fMain && CkMyPe()==0){ // only main chares have been checkpointed, we restart on PE0
		PUP::fromDisk pMain(fMain);
		CkPupMainChareData(pMain);
		fclose(fMain);
		DEBCHK("[%d]CkRestartMain: mainchares restored\n",CkMyPe());
		//bdcastRO(); // moved to CkPupMainChareData()
	}
	
	// restore groups
	// content of the file: numGroups, GroupInfo[numGroups], _groupTable(PUP'ed), groups(PUP'ed)
	sprintf(filename,"%s/Groups.dat",dirname);
	FILE* fGroups = fopen(filename,"rb");
	if(!fGroups) CkAbort("Failed to open checkpoint file for group table!");
	PUP::fromDisk pGroups(fGroups);
	CkPupGroupData(pGroups);
	fclose(fGroups);

	// restore nodegroups
	// content of the file: numNodeGroups, GroupInfo[numNodeGroups], _nodeGroupTable(PUP'ed), nodegroups(PUP'ed)
	if(CkMyRank()==0){
		sprintf(filename,"%s/NodeGroups.dat",dirname);
		FILE* fNodeGroups = fopen(filename,"rb");
		if(!fNodeGroups) CkAbort("Failed to open checkpoint file for nodegroup table!");
		PUP::fromDisk pNodeGroups(fNodeGroups);
		CkPupNodeGroupData(pNodeGroups);
		fclose(fNodeGroups);
	}

	// for each location, restore arrays
	//DEBCHK("[%d]Trying to find location manager\n",CkMyPe());
	DEBCHK("[%d]Number of PE: %d -> %d\n",CkMyPe(),_numPes,CkNumPes());
	if(CkMyPe() < _numPes) 	// in normal range: restore, o/w, do nothing
          for (i=0; i<_numPes;i++) {
            if (i%CkNumPes() == CkMyPe()) {
	      sprintf(filename,"%s/arr_%d.dat",dirname, i);
	      FILE *datFile=fopen(filename,"rb");
	      if (datFile==NULL) CkAbort("Could not read data file");
	      PUP::fromDisk  p(datFile);
	      CkPupArrayElementsData(p);
	      fclose(datFile);
            }
	  }

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


