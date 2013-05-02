#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "charm++.h"
#include "ck.h"
#include "ckevacuation.h"

//#define DEBUGC(x) x
#define DEBUGC(x) 

/***********************************************************************************************/
/*
	FAULT_EVAC
*/



int _ckEvacBcastIdx;
int _ckAckEvacIdx;
int numValidProcessors;

double evacTime;

int remainingElements;
int allowMessagesOnly; /*this processor has started evacuating but not yet complete 
												So allow messages to it but not immigration
												*/
double firstRecv;
/*
	Called on other processors that msg->pe processor has been evacuated
*/
void _ckEvacBcast(struct evacMsg *msg){
	if(msg->remainingElements == -1){
			firstRecv = CmiWallTimer();
			return;
	}
	printf("[%d]<%.6f> Processor %d is being evacuated \n",CkMyPe(),CmiWallTimer(),msg->pe);
	fprintf(stderr,"[%d] <%.6f> Processor %d is being evacuated \n",CkMyPe(),CmiWallTimer(),msg->pe);
	CpvAccess(_validProcessors)[msg->pe] = 0;
	set_avail_vector(CpvAccess(_validProcessors));
	if(msg->pe == CpvAccess(serializer)){
		CpvAccess(serializer) = getNextSerializer();
	}
	/*
		Inform all processors about the current position of 
		the elements that have a home on them.
		Useful for the case where an element on the crashing
		processor has migrated away previously
	*/
	int numGroups = CkpvAccess(_groupIDTable)->size();
	int i;
	CkElementInformHome inform;
	CKLOCMGR_LOOP(((CkLocMgr*)(obj))->iterate(inform););
	
	if(msg->remainingElements == 0){
		struct evacMsg reply;
		reply.pe = CkMyPe();
	//	printf("[%d] Last broadcast received at %.6lf in %.6lf \n",CkMyPe(),CmiWallTimer(),CmiWallTimer()-firstRecv);
	CmiSetHandler(&reply,_ckAckEvacIdx);
	CmiSyncSend(msg->pe,sizeof(struct evacMsg),(char *)&reply);
		allowMessagesOnly = -1;
	}else{
		allowMessagesOnly = msg->pe;
	}
}




/*
	Acks that all the valid processors have received the 
	evacuate broadcast
*/
void _ckAckEvac(struct evacMsg *msg){
	numValidProcessors--;
	if(numValidProcessors == 0){
		set_avail_vector(CpvAccess(_validProcessors));
		printf("[%d] <%.6f> Reply from all processors took %.6lf s \n",CkMyPe(),CmiWallTimer(),CmiWallTimer()-evacTime);
//		CcdCallOnCondition(CcdPERIODIC_1s,(CcdVoidFn)CkStopScheduler,0);
//		CkStopScheduler();
	}
}


void CkAnnounceEvac(int remain){
	//	Tell all the processors
	struct evacMsg msg;
	msg.pe = CkMyPe();
	msg.remainingElements = remain;
	CmiSetHandler(&msg,_ckEvacBcastIdx);
	CmiSyncBroadcast(sizeof(struct evacMsg),(char *)&msg);
}


void CkStopScheduler(){
	if(remainingElements > 0){
		return;
	}
		/*
		Tell the reduction managers that this processor is going down now
		and that they should tell their parents to cut them off
	*/	
	int numNodeGroups = CksvAccess(_nodeGroupIDTable).size();
	for(int i=0;i<numNodeGroups;i++){
    IrrGroup *obj = CksvAccess(_nodeGroupTable)->find((CksvAccess(_nodeGroupIDTable))[i]).getObj();	
		obj->doneEvacuate();
	}
	int thisPE = CkMyPe();
	printf("[%d] Stopping Scheduler \n", thisPE);
	/*stops putting messages into the scheduler queue*/
	CpvAccess(_validProcessors)[thisPE]=0;
}

void CkEmmigrateElement(void *arg){
	CkLocRec *rec = (CkLocRec *)arg;
	const CkArrayIndex &idx = rec->getIndex();
	int targetPE=getNextPE(idx);
	//set this flag so that load balancer is not informed when
	//this element migrates
	rec->AsyncMigrate(true);
	rec->migrateMe(targetPE);
	CkEvacuatedElement();
	
}

void CkEvacuatedElement(){
	if(!CpvAccess(_validProcessors)[CkMyPe()]){
		return;
	}
	if(!CkpvAccess(startedEvac)){
		return;
	}
	remainingElements=0;
	//	Go through all the groups and find the location managers.
	//	For each location manager migrate away all the elements
	// Recalculate the number of remaining elements
	int numGroups = CkpvAccess(_groupIDTable)->size();
	int i;
  CkElementEvacuate evac;
	CKLOCMGR_LOOP(((CkLocMgr*)(obj))->iterate(evac););
	
	CmiAssert(remainingElements >= 0);
	DEBUGC(printf("[%d] remaining elements %d \n",CkMyPe(),remainingElements));
	if(remainingElements == 0){
		printf("[%d] Processor empty in %.6lfs \n",CkMyPe(),CmiWallTimer()-evacTime);
		CpvAccess(_validProcessors)[CkMyPe()] = 0;
		CkAnnounceEvac(0);
		int numNodeGroups = CksvAccess(_nodeGroupIDTable).size();
		for(int i=0;i<numNodeGroups;i++){
  	  IrrGroup *obj = CksvAccess(_nodeGroupTable)->find((CksvAccess(_nodeGroupIDTable))[i]).getObj();	
			obj->doneEvacuate();
		}
	}	
}

int evacuate;
extern "C" void CkClearAllArrayElements();

void CkDecideEvacPe(){
	if(evacuate > 0){
		return;
	}
	evacuate = 1;
	evacTime = CmiWallTimer();
	CkClearAllArrayElements();
}



int numEvacuated;

/*
	Code for moving off all the array elements on a processor
*/
extern "C"
void CkClearAllArrayElements(){
	if(evacuate != 1){
			return;
	}
	evacuate=2;
	remainingElements=0;
	numEvacuated=0;
//	evacTime = CmiWallTimer();
	printf("[%d] <%.6lf> Start Evacuation \n",CkMyPe(),evacTime);
	CkpvAccess(startedEvac)=1;
	//	Make sure the broadcase serializer changes
	if(CkMyPe() == CpvAccess(serializer)){
		CpvAccess(serializer) = getNextSerializer();
	}
//	CkAnnounceEvac(-1);
	
	//	Go through all the groups and find the location managers.
	//	For each location manager migrate away all the elements
	int numGroups = CkpvAccess(_groupIDTable)->size();
	int i;
  CkElementEvacuate evac;
	CKLOCMGR_LOOP(((CkLocMgr*)(obj))->iterate(evac););

	/*
		Tell the nodegroup reduction managers that they need to 
		start changing their reduction trees
	*/
	int numNodeGroups = CksvAccess(_nodeGroupIDTable).size();
	for(i=0;i<numNodeGroups;i++){
    IrrGroup *obj = CksvAccess(_nodeGroupTable)->find((CksvAccess(_nodeGroupIDTable))[i]).getObj();	
		obj->evacuate();
	}
	
	DEBUGC(printf("[%d] remaining elements %d number Evacuated %d \n",CkMyPe(),remainingElements,numEvacuated));
	numValidProcessors = CkNumValidPes()-1;
	CkAnnounceEvac(remainingElements);
	if(remainingElements == 0){
		/*
			Tell the nodegroup reduction managers when all the elements have been
			removed
		*/
		printf("[%d] Processor empty in %.6lfs \n",CkMyPe(),CmiWallTimer()-evacTime);
		CpvAccess(_validProcessors)[CkMyPe()] = 0;
		int numNodeGroups = CksvAccess(_nodeGroupIDTable).size();
		for(int i=0;i<numNodeGroups;i++){
  	  IrrGroup *obj = CksvAccess(_nodeGroupTable)->find((CksvAccess(_nodeGroupIDTable))[i]).getObj();	
			obj->doneEvacuate();
		}
	}	
}

void CkClearAllArrayElementsCPP(){
	CkClearAllArrayElements();
}

void CkElementEvacuate::addLocation(CkLocation &loc){
	CkLocMgr *locMgr = loc.getManager();
	CkLocRec *rec = loc.getLocalRecord();
	const CkArrayIndex &i = loc.getIndex();
	int targetPE=getNextPE(i);
	if(rec->isAsyncEvacuate()){
		numEvacuated++;
		printf("[%d]<%.6lf> START to emigrate array element \n",CkMyPe(),CmiWallTimer());
		rec->AsyncMigrate(true);
		locMgr->emigrate(rec,targetPE);
		printf("[%d]<%.6lf> emigrated array element \n",CkMyPe(),CmiWallTimer());
	}else{
		/*
			This is in all probability a location containing an ampi, ampiParent and their
			associated TCharm thread.
		*/
		CkVec<CkMigratable *>list;
		locMgr->migratableList(rec,list);
		DEBUGC(printf("[%d] ArrayElement not ready to Evacuate number of migratable %d \n",CkMyPe(),list.size()));
		for(int i=0;i<list.size();i++){
			if(list[i]->isAsyncEvacuate()){
				DEBUGC(printf("[%d] possible TCharm element decides to migrate \n",CkMyPe()));
//				list[i]->ckMigrate(targetPE);
				rec->AsyncMigrate(true);
				locMgr->emigrate(rec,targetPE);
				numEvacuated++;
			}
		}
	//	remainingElements++;
		//inform new home that this element is here
	//	locMgr->informHome(i,CkMyPe());
	}
}

void CkElementInformHome::addLocation(CkLocation &loc){
	const CkArrayIndex &i = loc.getIndex();
	CkLocMgr *locMgr = loc.getManager();
	locMgr->informHome(i,CkMyPe());	
}


/*
	Find the homePE of an array element,  given an index. Used only on a
	processor that is being evacuated
*/

int getNextPE(const CkArrayIndex &i){
	if (i.nInts==1) {
      //Map 1D integer indices in simple round-robin fashion
      int ans= (i.data()[0])%CkNumPes();
			while(!CpvAccess(_validProcessors)[ans] || ans == CkMyPe()){
				ans = (ans +1 )%CkNumPes();
			}
			return ans;
  }else{
		//Map other indices based on their hash code, mod a big prime.
			unsigned int hash=(i.hash()+739)%1280107;
			int ans = (hash % CkNumPes());
			while(!CpvAccess(_validProcessors)[ans] || ans == CkMyPe()){
				ans = (ans +1 )%CkNumPes();
			}
			return ans;

	}

}

/*
	If it is found that the serializer processor has crashed, decide on a 
	new serializer, should return the same answer on all processors
*/
int getNextSerializer(){
	int currentSerializer = CpvAccess(serializer);
	int nextSerializer = (currentSerializer+1)%CkNumPes();

	while(!(CpvAccess(_validProcessors)[nextSerializer])){
		nextSerializer = (nextSerializer + 1)%CkNumPes();
		if(nextSerializer == currentSerializer){
			CkAbort("All processors are invalid ");
		}
	}
	return nextSerializer;
}

int CkNumValidPes(){
#if CMK_BIGSIM_CHARM
        return CkNumPes();
#else
	int count=0;
	for(int i=0;i<CkNumPes();i++){
		if(CpvAccess(_validProcessors)[i]){
			count++;
		}
	}
	return count;
#endif
}


void processRaiseEvacFile(char *raiseEvacFile){
	FILE *fp = fopen(raiseEvacFile,"r");
	if(fp == NULL){
		printf("Could not open raiseevac file %s. Ignoring raiseevac \n",raiseEvacFile);
		return;
	}
	char line[100];
	while(fgets(line,99,fp)!=0){
		int pe,faultTime;
		sscanf(line,"%d %d",&pe,&faultTime);
		if(pe == CkMyPe()){
			printf("[%d] Processor to be evacuated after %ds\n",CkMyPe(),faultTime);
			CcdCallFnAfter((CcdVoidFn)CkDecideEvacPe, 0, faultTime*1000);
		}
	}
	fclose(fp);	
}
