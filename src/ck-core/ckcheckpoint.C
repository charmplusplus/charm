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

#if 0
#define DEBCHK CkPrintf
#else
#define DEBCHK //CkPrintf
#endif

CkGroupID _sysChkptMgr;

typedef struct _GroupInfo{
	CkGroupID gID;
	int MigCtor, DefCtor;
	int useDefCtor;
	char name[256];
} GroupInfo;

// Print out an array index to this string as decimal fields
// separated by underscores.
void printIndex(const CkArrayIndex &idx,char *dest) {
	const int *idxData=idx.data();
	for (int i=0;i<idx.nInts;i++) {
		sprintf(dest,"%s%d",i==0?"":"_", idxData[i]);
		dest+=strlen(dest);
	}
}
ElementSaver::ElementSaver(const char *dirName_,const int locMgrIdx_) :dirName(dirName_),locMgrIdx(locMgrIdx_){
	char fileName[1024];
	sprintf(fileName,"%s/loc_%d_%d.idx",dirName,locMgrIdx,CkMyPe());
	indexFile=fopen(fileName,"w");
	if (indexFile==NULL) CkAbort("Could not create index file");
	fprintf(indexFile,"CHARM++_Checkpoint_File 1.0 %d %d\n",CkMyPe(),CkNumPes());

	sprintf(fileName,"%s/arr_%d_%d.dat",dirName,locMgrIdx,CkMyPe());
	datFile=fopen(fileName,"wb");
	if (indexFile==NULL) CkAbort("Could not create data file");
}
ElementSaver::~ElementSaver() {
	fclose(datFile);
	fclose(indexFile);
}
void ElementSaver::addLocation(CkLocation &loc) {
	const CkArrayIndex &idx=loc.getIndex();
	const int *idxData=idx.data();
	char idxName[128]; printIndex(idx,idxName);
	char fileName[1024]; sprintf(fileName,"arr_%d_%d.dat",locMgrIdx,CkMyPe());

	//Write a file index entry
	fprintf(indexFile,"%s %d ",fileName,idx.nInts);
	for (int i=0;i<idx.nInts;i++) fprintf(indexFile,"%d ",idxData[i]);
	fprintf(indexFile,"\n");

	//Save the actual array element data to the file:
	if (!datFile) CkAbort("Could not write checkpoint file");
	PUP::toDisk p(datFile);
	loc.pup(p);
	//DEBCHK("Saved array index %s to datFile\n",idxName);
}

void CkCheckpointMgr::Checkpoint(const char *dirname,CkCallback& cb){
	//DEBCHK("[%d]CkCheckpointMgr::Checkpoint called dirname={%s}\n",CkMyPe(),dirname);
	IrrGroup* obj;
	int numGroups = CkpvAccess(_groupIDTable)->size();
	for(int i=0;i<numGroups;i++) {
		obj = CkpvAccess(_groupTable)->find((*CkpvAccess(_groupIDTable))[i]).getObj();
		if(obj->isLocMgr()){
			DEBCHK("\tThis is a location manager!\n");
			ElementSaver saver(dirname,obj->ckGetGroupID().idx);
			((CkLocMgr*)(obj))->iterate(saver);
		}
	}
	if(CkMyPe()!=0)
		DEBCHK("[%d]CkCheckpointMgr::Checkpoint DONE.\n",CkMyPe());
	else{
		CkPrintf("[%d]CkCheckpointMgr::Checkpoint DONE. Invoking callback.\n",CkMyPe());
		cb.send();
	}
}

void CkStartCheckpoint(char* dirname,const CkCallback& cb){
	int i;
	char filename[1024];
	CkPrintf("Checkpoint starting in %s\n",dirname);
	CmiMkdir(dirname);
	
	// save readonlys, and callback BTW
	sprintf(filename,"%s/RO.dat",dirname);
	FILE* fRO = fopen(filename,"wb");
	if(!fRO) CkAbort("Failed to create checkpoint file for readonly data!");
	int _numPes = CkNumPes();
	fwrite(&_numPes,sizeof(int),1,fRO);
	int _numReadonlies=_readonlyTable.size();
	fwrite(&_numReadonlies,sizeof(int),1,fRO);
	PUP::toDisk pRO(fRO);
	for(i=0;i<_numReadonlies;i++) _readonlyTable[i]->pupData(pRO);
	fwrite(&cb,sizeof(CkCallback),1,fRO);
	fclose(fRO);

	// save mainchares into MainChares.dat
	sprintf(filename,"%s/MainChares.dat",dirname);
	FILE* fMain = fopen(filename,"wb");
	if(!fMain) CkAbort("Failed to open checkpoint file for mainchare data!");
	PUP::toDisk pMain(fMain);
	int nMains=_mainTable.size();
	for(i=0;i<nMains;i++){  /* Create all mainchares */
		int entryMigCtor = _chareTable[_mainTable[i]->chareIdx]->getMigCtor();
		if(entryMigCtor!=-1){
			Chare* obj = (Chare *)_mainTable[i]->getObj();
			obj->pup(pMain);
		}
	}

	fclose(fMain);
	
	// save groups into Groups.dat
	// content of the file: numGroups, GroupInfo[numGroups], _groupTable(PUP'ed), groups(PUP'ed)
	int numGroups = CkpvAccess(_groupIDTable)->size();
	sprintf(filename,"%s/Groups.dat",dirname);
	FILE* fGroups = fopen(filename,"wb");
	if(!fGroups) CkAbort("Failed to create checkpoint file for group table!");
	fwrite(&numGroups,sizeof(UInt),1,fGroups);
	DEBCHK("[%d]CkStartCheckpoint: numGroups = %d\n",CkMyPe(),numGroups);

	GroupInfo *tmpInfo = new GroupInfo [numGroups];
	TableEntry ent;
	for(i=0;i<numGroups;i++) {
		tmpInfo[i].gID = (*CkpvAccess(_groupIDTable))[i];
		ent = CkpvAccess(_groupTable)->find(tmpInfo[i].gID);
		tmpInfo[i].useDefCtor = ent.getObj()->useDefCtor();
		tmpInfo[i].MigCtor = _chareTable[ent.getcIdx()]->migCtor;
		tmpInfo[i].DefCtor = _chareTable[ent.getcIdx()]->defCtor;
		strncpy(tmpInfo[i].name,_chareTable[ent.getcIdx()]->name,255);
		DEBCHK("[%d]CkStartCheckpoint: group %s has useDefCtor=%d\n",CkMyPe(),
			tmpInfo[i].name,tmpInfo[i].useDefCtor);

		if(tmpInfo[i].useDefCtor==0 && tmpInfo[i].MigCtor==-1) {
			char buf[512];
			sprintf(buf,"Group %s needs a migration constructor and PUP'er routine for restart.\n",
				     tmpInfo[i].name);
			CkAbort(buf);
		}
	}
	if(numGroups != fwrite(tmpInfo,sizeof(GroupInfo),numGroups,fGroups)) CkAbort("error writing groupinfo");

	PUP::toDisk pGroups(fGroups);
	for(i=0;i<numGroups;i++) {
		if(!tmpInfo[i].useDefCtor){
                        CkpvAccess(_groupTable)->find(tmpInfo[i].gID).getObj()->pup(pGroups);
                        DEBCHK("Group PUP'ed in: gid = %d, name = %s\n",
				CkpvAccess(_groupTable)->find(tmpInfo[i].gID).getObj()->ckGetGroupID().idx,
				tmpInfo[i].name);
		}else{
                        DEBCHK("Group NOT PUP'ed in: gid = %d, name = %s\n",
				CkpvAccess(_groupTable)->find(tmpInfo[i].gID).getObj()->ckGetGroupID().idx,
				tmpInfo[i].name);
		}
	}
	delete [] tmpInfo;
	fclose(fGroups);

	// save nodegroups into NodeGroups.dat
	// content of the file: numNodeGroups, GroupInfo[numNodeGroups], _nodeGroupTable(PUP'ed), nodegroups(PUP'ed)
	int numNodeGroups = CksvAccess(_nodeGroupIDTable).size();
	sprintf(filename,"%s/NodeGroups.dat",dirname);
	FILE* fNodeGroups = fopen(filename,"wb");
	if(!fNodeGroups) CkAbort("Failed to create checkpoint file for nodegroup table!");
	fwrite(&numNodeGroups,sizeof(UInt),1,fNodeGroups);
	DEBCHK("[%d]CkStartCheckpoint: numNodeGroups = %d\n",CkMyPe(),numNodeGroups);

	GroupInfo *tmpInfo2 = new GroupInfo [numNodeGroups];
	TableEntry ent2;
	for(i=0;i<numNodeGroups;i++) {
		tmpInfo2[i].gID = CksvAccess(_nodeGroupIDTable)[i];
		ent2 = CksvAccess(_nodeGroupTable)->find(tmpInfo2[i].gID);
		tmpInfo2[i].MigCtor = _chareTable[ent2.getcIdx()]->migCtor;
		if(tmpInfo2[i].MigCtor==-1) {
			char buf[512];
			sprintf(buf,"NodeGroup %s either need a migration constructor and\n\
				     declared as [migratable] in .ci to be able to checkpoint.",\
				     _chareTable[ent2.getcIdx()]->name);
			CkAbort(buf);
		}
	}
	if(numNodeGroups != fwrite(tmpInfo2,sizeof(GroupInfo),numNodeGroups,fNodeGroups)) CkAbort("error writing nodegroupinfo");
	PUP::toDisk pNodeGroups(fNodeGroups);
	for(i=0;i<numNodeGroups;i++) {
		ent2 = CksvAccess(_nodeGroupTable)->find(tmpInfo2[i].gID);
		ent2.getObj()->pup(pNodeGroups);
		DEBCHK("Nodegroup PUP'ed in: gid = %d, name = %s\n",
			ent2.getObj()->ckGetGroupID().idx,
			_chareTable[ent2.getcIdx()]->name);
	}
	delete [] tmpInfo2;
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
ElementRestorer::ElementRestorer(const char *dirName_,CkLocMgr *dest_,int destPe)
	:dirName(dirName_), dest(dest_)
{
	char indexName[1024];
	sprintf(indexName,"%s/loc_%d_%d.idx",dirName,dest->ckGetGroupID().idx,destPe);
	indexFile=fopen(indexName,"r");
	if (indexFile==NULL)  CkAbort("Could not read index file");
	char ignored[128]; double version; int srcPE; int srcSize;
	if (4!=fscanf(indexFile,"%s%lf%d%d",ignored,&version,&srcPE,&srcSize))
		CkAbort("Checkpoint index file format error");
	if (version>=2.0) CkAbort("Checkpoint index file format is too new");

	sprintf(indexName,"%s/arr_%d_%d.dat",dirName,dest->ckGetGroupID().idx,destPe);
	datFile=fopen(indexName,"rb");
	if (datFile==NULL)  CkAbort("Could not read data file");
}
ElementRestorer::~ElementRestorer() {
	fclose(datFile);
	fclose(indexFile);
}
// Try to restore one array element.  If it worked, return true.
bool ElementRestorer::restore(void) {
	// Find the index and filename from the file index:
	CkArrayIndexMax idx;
	char fileName[1024];

	if (fscanf(indexFile,"%s%d",fileName,&idx.nInts)!=2) return false;
	int *idxData=1+((int *)&idx);
	for (int i=0;i<idx.nInts;i++) fscanf(indexFile,"%d",&idxData[i]);

	//Restore the actual array element data from the file:
	if (!datFile) CkAbort("Could not read checkpoint file");
	PUP::fromDisk p(datFile);
	dest->resume(idx,p);
	//DEBCHK("Restored an index from datFile\n");
	return true;
}

void CkRestartMain(const char* dirname){
	int i;
	char filename[1024];
	CkCallback cb;
	int numGroups,numNodeGroups;
	// restore readonlys
	sprintf(filename,"%s/RO.dat",dirname);
	FILE* fRO = fopen(filename,"rb");
	if(!fRO) CkAbort("Failed to open checkpoint file for readonly data!");
	int _numPes = -1;
	fread(&_numPes,sizeof(int),1,fRO);
	int _numReadonlies=-1;
	fread(&_numReadonlies,sizeof(int),1,fRO);
	if (_numReadonlies != _readonlyTable.size())
		CkAbort("You cannot add readonlies and restore from checkpoint...");
	PUP::fromDisk pRO(fRO);
	for(i=0;i<_numReadonlies;i++) _readonlyTable[i]->pupData(pRO);
	fread(&cb,sizeof(CkCallback),1,fRO);
	fclose(fRO);
	DEBCHK("[%d]CkRestartMain: readonlys restored\n",CkMyPe());

	// restore mainchares
	sprintf(filename,"%s/MainChares.dat",dirname);
	FILE* fMain = fopen(filename,"rb");
	if(!fMain) CkAbort("Failed to open checkpoint file for readonly data!");
	PUP::fromDisk pMain(fMain);
	int nMains=_mainTable.size();
	for(i=0;i<nMains;i++){  /* Create all mainchares */
		int entryMigCtor = _chareTable[_mainTable[i]->chareIdx]->getMigCtor();
		if(entryMigCtor!=-1){
			int size = _chareTable[_mainTable[i]->chareIdx]->size;
			void *obj = malloc(size);
			_MEMCHECK(obj);
			_mainTable[i]->setObj(obj);
			void *m = CkAllocSysMsg();
			_entryTable[entryMigCtor]->call(m, obj);
			((Chare *)obj)->pup(pMain);
		}
	}
	fclose(fMain);
	DEBCHK("[%d]CkRestartMain: mainchares restored\n",CkMyPe());
	
	// restore groups
	// content of the file: numGroups, GroupInfo[numGroups], _groupTable(PUP'ed), groups(PUP'ed)
	sprintf(filename,"%s/Groups.dat",dirname);
	FILE* fGroups = fopen(filename,"rb");
	if(!fGroups) CkAbort("Failed to open checkpoint file for group table!");
	fread(&numGroups,sizeof(UInt),1,fGroups);
	if(CkMyPe()==0) { CkpvAccess(_numGroups) = numGroups+1; }else{ CkpvAccess(_numGroups) = 1; }
	//CkpvAccess(_numGroups) = 1;

	GroupInfo *tmpInfo = new GroupInfo [numGroups];
	if(numGroups != fread(tmpInfo,sizeof(GroupInfo),numGroups,fGroups)) CkAbort("error reading groupinfo");

	PUP::fromDisk pGroups(fGroups);
	for(i=0;i<numGroups;i++) {
		CkGroupID gID = tmpInfo[i].gID;
		//CkpvAccess(_groupIDTable)->push_back(gID);
		int eIdx = (tmpInfo[i].useDefCtor)?(tmpInfo[i].DefCtor):(tmpInfo[i].MigCtor);
		void *m = CkAllocSysMsg();
		envelope* env = UsrToEnv((CkMessage *)m);
		CkCreateLocalGroup(gID, eIdx, env);
		if (!tmpInfo[i].useDefCtor){
			CkpvAccess(_groupTable)->find(gID).getObj()->pup(pGroups);
			DEBCHK("Group PUP'ed out: gid = %d, name = %s\n",
				CkpvAccess(_groupTable)->find(gID).getObj()->ckGetGroupID().idx,tmpInfo[i].name);
		}else{
			DEBCHK("Group NOT PUP'ed out: gid = %d, name = %s\n",
				CkpvAccess(_groupTable)->find(gID).getObj()->ckGetGroupID().idx,tmpInfo[i].name);
	    }
	}
	fclose(fGroups);

	// restore nodegroups
	// content of the file: numNodeGroups, GroupInfo[numNodeGroups], _nodeGroupTable(PUP'ed), nodegroups(PUP'ed)
	if(CkMyRank()==0){
		sprintf(filename,"%s/NodeGroups.dat",dirname);
		FILE* fNodeGroups = fopen(filename,"rb");
		if(!fNodeGroups) CkAbort("Failed to open checkpoint file for nodegroup table!");
		fread(&numNodeGroups,sizeof(UInt),1,fNodeGroups);
		if(CkMyPe()==0){ CksvAccess(_numNodeGroups) = numNodeGroups+1; }
		else { CksvAccess(_numNodeGroups) = 1; }

		GroupInfo* tmpInfo2 = new GroupInfo [numNodeGroups];
		if(numNodeGroups != fread(tmpInfo2,sizeof(GroupInfo),numNodeGroups,fNodeGroups)) CkAbort("error reading nodegroupinfo");

		PUP::fromDisk pNodeGroups(fNodeGroups);
		for(i=0;i<numNodeGroups;i++) {
			CkGroupID gID = tmpInfo2[i].gID;
			CksvAccess(_nodeGroupIDTable).push_back(gID);
			int eIdx = tmpInfo2[i].MigCtor;
			void *m = CkAllocSysMsg();
			envelope* env = UsrToEnv((CkMessage *)m);
			CkCreateLocalNodeGroup(gID, eIdx, env);
			CksvAccess(_nodeGroupTable)->find(gID).getObj()->pup(pNodeGroups);
			DEBCHK("Nodegroup PUP'ed out: gid = %d, name = %s\n",CksvAccess(_nodeGroupTable)->find(gID).getObj()->ckGetGroupID().idx,_chareTable[CksvAccess(_nodeGroupTable)->find(gID).getcIdx()]->name);
		}
		fclose(fNodeGroups);
		delete [] tmpInfo2;
	}

	// for each location, restore arrays
	//DEBCHK("[%d]Trying to find location manager\n",CkMyPe());
	DEBCHK("[%d]Number of PE: %d -> %d\n",CkMyPe(),_numPes,CkNumPes());
	IrrGroup* obj;
	CkGroupID gID;
	if(CkMyPe() < _numPes) 	// in normal range: restore, o/w, do nothing
		for(i=0;i<numGroups;i++) {
			gID = tmpInfo[i].gID;
			obj = CkpvAccess(_groupTable)->find(gID).getObj();
			if(obj->isLocMgr()){
				CkLocMgr *mgr = (CkLocMgr *)obj;
				for(int j=0;j<_numPes;j++)
					if(j%CkNumPes()==CkMyPe()){
						ElementRestorer restorer(dirname,mgr,j);
						while (restorer.restore()) {}
					}
			}
			obj->ckJustMigrated();
		}
	delete [] tmpInfo;

	if(CkMyPe()==0) {
		DEBCHK("[%d]CkRestartMain done. sending out callback.\n",CkMyPe());
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
  CkCheckpointInit(CkMigrateMessage *m) {}
};

#include "CkCheckpoint.def.h"


