/*
Charm++ File: Checkpoint Library
added 01/03/2003 by Chao Huang, chuang10@uiuc.edu

More documentation goes here...
*/

#include <string.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include "charm++.h"
#include "ck.h"
//#include "ckcheckpoint.h"

#if 1
#define DEBCHK CkPrintf
#else
#define DEBCHK //CkPrintf
#endif

CkGroupID _sysChkptMgr;

typedef struct _GroupInfo{
	CkGroupID gID;
	int eIdx;
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
	char indexName[1024];
	//memset(indexName,0,1024);
	//chdir(dirName);
	sprintf(indexName,"%s/loc_%d_%d.idx",dirName,locMgrIdx,CkMyPe());
	indexName[strlen(indexName)]='\0';
	indexFile=fopen(indexName,"w");
	if (indexFile==NULL)  CkAbort("Could not create index file");
	fprintf(indexFile,"CHARM++_Checkpoint_File 1.0 %d %d\n",CkMyPe(),CkNumPes());
}
ElementSaver::~ElementSaver() {
	fclose(indexFile);
}
void ElementSaver::addLocation(CkLocation &loc) {
	const CkArrayIndex &idx=loc.getIndex();
	const int *idxData=idx.data();
	char idxName[128]; printIndex(idx,idxName);
	char fileName[1024]; sprintf(fileName,"arr_%d_%s.dat",locMgrIdx,idxName);

	//Write a file index entry
	fprintf(indexFile,"%s %d ",fileName,idx.nInts);
	for (int i=0;i<idx.nInts;i++) fprintf(indexFile,"%d ",idxData[i]);
	fprintf(indexFile,"\n");

	//Save the actual array element data to the file:
	char pathName[1024];
	sprintf(pathName,"%s",fileName);
	FILE* f=fopen(pathName,"wb");
	if(!f) CkAbort("Could not create checkpoint file");
	PUP::toDisk p(f);
	loc.pup(p);
	fclose(f);
	DEBCHK("Saved array index %s to file %s\n",idxName,pathName);
}

void CkCheckpointMgr::Checkpoint(int len, char dirname[],CkCallback& cb){
	dirname[len]='\0';
	//DEBCHK("[%d]CkCheckpointMgr::Checkpoint called, len=%d, dirname={%s}\n",CkMyPe(),len,dirname);
	IrrGroup* obj;
	int numGroups = CkpvAccess(_groupIDTable).size();
	for(int i=0;i<numGroups;i++) {
		obj = CkpvAccess(_groupTable)->find(CkpvAccess(_groupIDTable)[i]).getObj();
		//DEBCHK("tmpInfo[%d]:gID = %d, eIdx = %d, obj->ckGetGroupID() = %d\n",i,CkpvAccess(_groupIDTable)[i].idx,CkpvAccess(_groupTable)->find(CkpvAccess(_groupIDTable)[i]).getMigCtor(),obj->ckGetGroupID().idx);
		if(obj->isLocMgr()){
			DEBCHK("\tThis is a location manager!\n");
			ElementSaver saver(dirname,obj->ckGetGroupID().idx);
			dynamic_cast<CkLocMgr*>(obj)->iterate(saver);
		}
	}
	DEBCHK("[%d]CkCheckpointMgr::Checkpoint DONE. Invoking callback.\n",CkMyPe());
	if(CkMyPe()==0) cb.send();
}

void CkStartCheckpoint(char* dirname,const CkCallback& cb){
	char filename[1024];
	int len = strlen(dirname);
	//dirname[len]='\0';
	CkPrintf("CkStartCheckpoint() making dir: len=%d,dir=%s\n",len,dirname);
	mkdir(dirname,0777);
	// save readonlys, and callback BTW
	sprintf(filename,"%s/RO.dat",dirname);
	FILE* fRO = fopen(filename,"wb");
	if(!fRO) CkAbort("Failed to create checkpoint file for readonly data!");
	fwrite(&_numReadonlies,sizeof(int),1,fRO);
	PUP::toDisk pRO(fRO);
	for(int i=0;i<_numReadonlies;i++) _readonlyTable[i]->pupData(pRO);
	fwrite(&cb,sizeof(CkCallback),1,fRO);
	fclose(fRO);

	// save groups into Groups.dat
	// content of the file: numGroups, GroupInfo[numGroups], _groupTable(PUP'ed), groups(PUP'ed)
	int numGroups = CkpvAccess(_groupIDTable).size();
	sprintf(filename,"%s/Groups.dat",dirname);
	FILE* fGroups = fopen(filename,"wb");
	if(!fGroups) CkAbort("Failed to create checkpoint file for group table!");
	fwrite(&numGroups,sizeof(UInt),1,fGroups);
	DEBCHK("[%d]CkStartCheckpoint: numGroups = %d\n",CkMyPe(),numGroups);

	GroupInfo *tmpInfo = new GroupInfo [numGroups];
	for(int i=0;i<numGroups;i++) {
		int tmpCtor;
		tmpInfo[i].gID = CkpvAccess(_groupIDTable)[i];
		tmpCtor = (tmpInfo[i].eIdx = CkpvAccess(_groupTable)->find(tmpInfo[i].gID).getMigCtor());
		//DEBCHK("[%d]tmpInfo[%d].gID:%d,migCtor:%d\n",CkMyPe(),i,tmpInfo[i].gID.idx,tmpCtor);
		if(tmpCtor==-1) {
			char buf[512];
			sprintf(buf,"Group %s either need a migration constructor and\n\
				     declared as [migratable] in .ci to be able to checkpoint.",\
				     CkpvAccess(_groupTable)->find(tmpInfo[i].gID).getName());
			CkAbort(buf);
		}
	}
	if(numGroups != fwrite(tmpInfo,sizeof(GroupInfo),numGroups,fGroups)) CkAbort("error writing groupinfo");
	PUP::toDisk pGroups(fGroups);
	for(int i=0;i<numGroups;i++) {
		if(1||0!=strcmp("LBDatabase",CkpvAccess(_groupTable)->find(tmpInfo[i].gID).getName())){
			CkpvAccess(_groupTable)->find(tmpInfo[i].gID).getObj()->pup(pGroups);
			DEBCHK(" group just PUP'ed out: gid = %d, name = %s\n",CkpvAccess(_groupTable)->find(tmpInfo[i].gID).getObj()->ckGetGroupID().idx,CkpvAccess(_groupTable)->find(tmpInfo[i].gID).getName());
		}else{
			DEBCHK(" group NOT PUP'ed out: gid = %d, name = %s\n",CkpvAccess(_groupTable)->find(tmpInfo[i].gID).getObj()->ckGetGroupID().idx,CkpvAccess(_groupTable)->find(tmpInfo[i].gID).getName());
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

	tmpInfo = new GroupInfo [numNodeGroups];
	for(int i=0;i<numNodeGroups;i++) {
		int tmpCtor;
		tmpInfo[i].gID = CksvAccess(_nodeGroupIDTable)[i];
		tmpCtor = (tmpInfo[i].eIdx = CksvAccess(_nodeGroupTable)->find(tmpInfo[i].gID).getMigCtor());
		//DEBCHK("[%d]tmpInfo[%d].gID:%d,migCtor:%d\n",CkMyPe(),i,tmpInfo[i].gID.idx,tmpCtor);
		if(tmpCtor==-1) {
			char buf[512];
			sprintf(buf,"NodeGroup %s either need a migration constructor and\n\
				     declared as [migratable] in .ci to be able to checkpoint.",\
				     CksvAccess(_nodeGroupTable)->find(tmpInfo[i].gID).getName());
			CkAbort(buf);
		}
	}
	if(numNodeGroups != fwrite(tmpInfo,sizeof(GroupInfo),numNodeGroups,fNodeGroups)) CkAbort("error writing nodegroupinfo");
	PUP::toDisk pNodeGroups(fNodeGroups);
	for(int i=0;i<numNodeGroups;i++) {
		CksvAccess(_nodeGroupTable)->find(tmpInfo[i].gID).getObj()->pup(pNodeGroups);
		DEBCHK(" nodegroup just PUP'ed out: gid = %d, name = %s\n",CksvAccess(_nodeGroupTable)->find(tmpInfo[i].gID).getObj()->ckGetGroupID().idx,CksvAccess(_nodeGroupTable)->find(tmpInfo[i].gID).getName());
	}
	delete [] tmpInfo;
	fclose(fNodeGroups);

	// hand over to checkpoint managers for per-processor checkpointing
	CProxy_CkCheckpointMgr(_sysChkptMgr).Checkpoint(len,(char *)dirname, cb);
}

/**
  * Restart: There's no such object as restart manager is created
  *          because a group cannot restore itself anyway.
  *          The mechanism exists as converse code and get invoked by
  *          broadcast message.
  **/
ElementRestorer::ElementRestorer(const char *dirName_,CkLocMgr *dest_)
	:dirName(dirName_), dest(dest_)
{
	char indexName[1024];
	//chdir(dirName);
	sprintf(indexName,"%s/loc_%d_%d.idx",dirName,dest->ckGetGroupID().idx,CkMyPe());
	indexFile=fopen(indexName,"r");
	if (indexFile==NULL)  CkAbort("Could not read index file");
	char ignored[128]; double version; int srcPE; int srcSize;
	if (4!=fscanf(indexFile,"%s%lf%d%d",ignored,&version,&srcPE,&srcSize))
		CkAbort("Checkpoint index file format error");
	if (version>=2.0) CkAbort("Checkpoint index file format is too new");
}
ElementRestorer::~ElementRestorer() {
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
	char pathName[1024];
	sprintf(pathName,"%s",fileName);
	FILE *f=fopen(pathName,"rb");
	if (!f) CkAbort("Could not read checkpoint file");
	PUP::fromDisk p(f);
	dest->resume(idx,p);
	fclose(f);
	DEBCHK("Restored an index from file %s\n",pathName);
	return true;
}

void CkRestartMain(const char* dirname){
	char filename[1024];
	CkCallback cb;
	int numGroups,numNodeGroups;

	// restore readonlys
	sprintf(filename,"%s/RO.dat",dirname);
	FILE* fRO = fopen(filename,"rb");
	if(!fRO) CkAbort("Failed to open checkpoint file for readonly data!");
	fread(&_numReadonlies,sizeof(int),1,fRO);
	PUP::fromDisk pRO(fRO);
	for(int i=0;i<_numReadonlies;i++) _readonlyTable[i]->pupData(pRO);
	fread(&cb,sizeof(CkCallback),1,fRO);
	fclose(fRO);
	DEBCHK("[%d]CkRestartMain: readonlys restored\n",CkMyPe());

	// restore groups
	// content of the file: numGroups, GroupInfo[numGroups], _groupTable(PUP'ed), groups(PUP'ed)
	sprintf(filename,"%s/Groups.dat",dirname);
	FILE* fGroups = fopen(filename,"rb");
	if(!fGroups) CkAbort("Failed to open checkpoint file for group table!");
	fread(&numGroups,sizeof(UInt),1,fGroups);
	if(CkMyPe()==0){ CkpvAccess(_numGroups) = numGroups+1; }
	else { CkpvAccess(_numGroups) = 1; }

	GroupInfo *tmpInfo = new GroupInfo [numGroups];
	if(numGroups != fread(tmpInfo,sizeof(GroupInfo),numGroups,fGroups)) CkAbort("error reading groupinfo");

	PUP::fromDisk pGroups(fGroups);
	for(int i=0;i<numGroups;i++) {
		CkGroupID gID = tmpInfo[i].gID;
		//DEBCHK("tmpInfo[%d]:gID = %d, eIdx = %d\n",i,gID.idx,tmpInfo[i].eIdx);
		CkpvAccess(_groupIDTable).push_back(gID);
		int eIdx = tmpInfo[i].eIdx;
		CkMigrateMessage m;
		envelope* env = UsrToEnv(&m);
		CkCreateLocalGroup(gID, eIdx, env);
		if(1){//0!=strcmp("LBDatabase",CkpvAccess(_groupTable)->find(gID).getName())){
			CkpvAccess(_groupTable)->find(gID).getObj()->pup(pGroups);
			DEBCHK(" group just PUP'ed out: gid = %d, name = %s\n",CkpvAccess(_groupTable)->find(gID).getObj()->ckGetGroupID().idx,CkpvAccess(_groupTable)->find(gID).getName());
		}else{
			DEBCHK(" group NOT PUP'ed out: gid = %d, name = %s\n",CkpvAccess(_groupTable)->find(gID).getObj()->ckGetGroupID().idx,CkpvAccess(_groupTable)->find(gID).getName());
		}
	}
	fclose(fGroups);

	// for each location, restore arrays
	DEBCHK("[%d]Trying to find location manager\n",CkMyPe());
	IrrGroup* obj;
	for(int i=0;i<numGroups;i++) {
		CkGroupID gID = tmpInfo[i].gID;
		obj = CkpvAccess(_groupTable)->find(gID).getObj();
		//DEBCHK("tmpInfo[%d]:gID = %d, eIdx = %d, obj->ckGetGroupID() = %d\n",i,gID.idx,tmpInfo[i].eIdx, obj->ckGetGroupID().idx);
		if(obj->isLocMgr()){
			CkLocMgr *mgr = (CkLocMgr *)obj;
			DEBCHK("\tThis is a location manager! lbdb=%d\n",lbdb.idx);
			ElementRestorer restorer(dirname,mgr);
			while (restorer.restore()) {}
		}
		obj->ckJustMigrated();
	}
	delete [] tmpInfo;

	// restore nodegroups
	// content of the file: numNodeGroups, GroupInfo[numNodeGroups], _nodeGroupTable(PUP'ed), nodegroups(PUP'ed)
	if(CkMyRank()==0){
	sprintf(filename,"%s/NodeGroups.dat",dirname);
	FILE* fNodeGroups = fopen(filename,"rb");
	if(!fNodeGroups) CkAbort("Failed to open checkpoint file for nodegroup table!");
	fread(&numNodeGroups,sizeof(UInt),1,fNodeGroups);
	if(CkMyPe()==0){ CksvAccess(_numNodeGroups) = numNodeGroups+1; }
	else { CksvAccess(_numNodeGroups) = 1; }

	tmpInfo = new GroupInfo [numNodeGroups];
	if(numNodeGroups != fread(tmpInfo,sizeof(GroupInfo),numNodeGroups,fNodeGroups)) CkAbort("error reading nodegroupinfo");

	PUP::fromDisk pNodeGroups(fNodeGroups);
	for(int i=0;i<numNodeGroups;i++) {
		CkGroupID gID = tmpInfo[i].gID;
		//DEBCHK("tmpInfo[%d]:gID = %d, eIdx = %d\n",i,gID.idx,tmpInfo[i].eIdx);
		CksvAccess(_nodeGroupIDTable).push_back(gID);
		int eIdx = tmpInfo[i].eIdx;
		CkMigrateMessage m;
		envelope* env = UsrToEnv(&m);
		CkCreateLocalNodeGroup(gID, eIdx, env);
		CksvAccess(_nodeGroupTable)->find(gID).getObj()->pup(pNodeGroups);
		DEBCHK(" nodegroup just PUP'ed out: gid = %d, name = %s\n",CksvAccess(_nodeGroupTable)->find(gID).getObj()->ckGetGroupID().idx,CksvAccess(_nodeGroupTable)->find(gID).getName());
	}
	fclose(fNodeGroups);
	delete [] tmpInfo;
	}

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


