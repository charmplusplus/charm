
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>
#include <assert.h>

#include "converse.h"
#include "traceCore.h"
#include "traceCoreCommon.h"

#include "converseEvents.h"	//TODO: remove this hack for REGISTER_CONVESE
#include "charmEvents.h"	//TODO: remove this hack for REGISTER_CHARM
#include "machineEvents.h"	// for machine events
//#include "ampiEvents.h" 	/* for ampi events */

CpvExtern(double, _traceCoreInitTime);
CpvExtern(char*, _traceCoreRoot);
CpvExtern(int, _traceCoreBufferSize);
CpvExtern(TraceCore*, _traceCore);
//CpvStaticDeclare(int ,staticNumEntries);

/* Trace Timer */
#define  TRACE_CORE_TIMER   CmiWallTimer
inline double TraceCoreTimer() { return TRACE_CORE_TIMER() - CpvAccess(_traceCoreInitTime); }
inline double TraceCoreTimer(double t) { return t - CpvAccess(_traceCoreInitTime); }

/***************** Class TraceCore Definition *****************/
TraceCore::TraceCore(char** argv)
{
	int binary = CmiGetArgFlag(argv,"+binary-trace");
	

	if(CpvAccess(_traceCoreOn) == 0){
		traceCoreOn=0;
		return;
	}
	traceCoreOn=1;
	traceLogger = new TraceLogger(CpvAccess(_traceCoreRoot), binary);
	//CmiPrintf("[%d]In TraceCore Constructor\n",CmiMyPe());
	startPtc();
	REGISTER_CONVERSE
	REGISTER_CHARM
	REGISTER_MACHINE
	//REGISTER_AMPI
	//closePtc();
}

TraceCore::~TraceCore()
{
	closePtc();
	if(traceLogger) delete traceLogger; 
}

void TraceCore::RegisterLanguage(int lID, const char* ln)
{
	//CmiPrintf("Register Language called for %s at %d \n",ln,lID);
  if(traceCoreOn == 0){
		return;
  }
  traceLogger->RegisterLanguage(lID, ln);

  // code for ptc file generation
  if(maxlID < lID){
	maxlID = lID;
  }
  lIDList[numLangs] = lID;
  lNames[numLangs] = new char[strlen(ln)+1];
  sprintf(lNames[numLangs],"%s",ln);
  numLangs++;

 }

struct TraceCoreEvent *insert_TraceCoreEvent(struct TraceCoreEvent *root,int eID){
	struct TraceCoreEvent *p;
	
	if(root == NULL){
		p = (struct TraceCoreEvent *)malloc(sizeof(struct TraceCoreEvent));
		p->next = NULL;
		p->eID = eID;
		return p;
	}
	p = root;
	while(p->next != NULL)
		p = p->next;
	p->next = (struct TraceCoreEvent *)malloc(sizeof(struct TraceCoreEvent));
	p->next->next = NULL;
	p->next->eID = eID;
	//cppcheck-suppress memleak
	return root;
}


void print_TraceCoreEvent(FILE *fpPtc,struct TraceCoreEvent *root,char *lang){
	struct TraceCoreEvent *p;
	
	p = root;
	while(p!=NULL){
		fprintf(fpPtc,"%d %s%d ",p->eID,lang,p->eID);
		p = p->next;

	}
}

//TODO: currently these are dummy definitions
void TraceCore::RegisterEvent(int lID, int eID)
{
	//CmiPrintf("registering event (%d, %d)\n", lID, eID);
	if(traceCoreOn == 0){
		return;
	}
	// code for ptc file generation
	for(int i=0;i<numLangs;i++){
		if(lIDList[i] == lID){
			if(maxeID[i] < eID){
				maxeID[i] = eID;
			}
			numEvents[i]++;
			eventLists[i] = insert_TraceCoreEvent(eventLists[i],eID);
			break;
		}
	}
}

void TraceCore::startPtc(){
	if(traceCoreOn ==0){
		return;
	}
	char *str = new char[strlen(CpvAccess(_traceCoreRoot))+strlen(".ptc")+1];
	sprintf(str,"%s.ptc",CpvAccess(_traceCoreRoot));
	fpPtc = fopen(str,"w");
	if(fpPtc == NULL){
		CmiAbort("Can't generate Ptc file");
	}
	fprintf(fpPtc,"%d\n",CmiNumPes());
	for(int i=0;i<MAX_NUM_LANGUAGES;i++){
		eventLists[i] = NULL;
		maxeID[i] = 0;
		numEvents[i] = 0;
	}
	maxlID = 0;
	numLangs = 0;
    delete [] str;
}


void TraceCore::closePtc(){
	int i;
	if(traceCoreOn ==0){
		return;
	}
	fprintf(fpPtc,"%d %d ",maxlID,numLangs);
	for(i=0;i<numLangs;i++){
		fprintf(fpPtc,"%d %s ",lIDList[i],lNames[i]);
	}
	fprintf(fpPtc,"\n");
	for(i=0;i<numLangs;i++){
		fprintf(fpPtc,"%d %d %d ",lIDList[i],maxeID[i],numEvents[i]);
		print_TraceCoreEvent(fpPtc,eventLists[i],lNames[i]);
		fprintf(fpPtc,"\n");
	}
	fclose(fpPtc);
}




//TODO: only for compatibility with incomplete converse instrumentation
void TraceCore::LogEvent(int lID, int eID)
{ 
	if(traceCoreOn == 0){
		return;
	}
	LogEvent(lID, eID, 0, NULL, 0, NULL); 
}

void TraceCore::LogEvent(int lID, int eID, int iLen, const int* iData)
{ 
	if(traceCoreOn == 0){
		return;
	}
	LogEvent(lID, eID, iLen, iData, 0, NULL); 
}

void TraceCore::LogEvent(int lID, int eID, int iLen, const int* iData,double t){
	if(traceCoreOn == 0){
		return;
	}
	CmiPrintf("TraceCore LogEvent called \n");
#if CMK_TRACE_ENABLED	
	int *iDataalloc;
	if(iLen != 0){
		iDataalloc = (int *)malloc(iLen*sizeof(int));
		for(int i=0;i<iLen;i++){
			iDataalloc[i] = iData[i];
		}
	}else{
		iDataalloc = NULL;
	}
	traceLogger->add(lID,eID,TraceCoreTimer(t),iLen,iDataalloc,0,NULL);
#endif
}


void TraceCore::LogEvent(int lID, int eID, int sLen, const char* sData)
{ 
	if(traceCoreOn == 0){
		return;
	}
	LogEvent(lID, eID, 0, NULL, sLen, sData); 
}

void TraceCore::LogEvent(int lID, int eID, int iLen, const int* iData, int sLen,const char* sData)
{
	//CmiPrintf("lID: %d, eID: %d", lID, eID);
	if(traceCoreOn == 0){
		return;
	}
		

#if CMK_TRACE_ENABLED
	int *iDataalloc;
	char *sDataalloc;
	if(iLen != 0){
		iDataalloc = (int *)malloc(iLen*sizeof(int));
		for(int i=0;i<iLen;i++){
			iDataalloc[i] = iData[i];
		}
	}else{
		iDataalloc = NULL;
	}
	if(sLen != 0){
		sDataalloc = (char *)malloc(sLen*sizeof(char));
		for(int i=0;i<sLen;i++){
			sDataalloc[i] = sData[i];
		}
	}else{
		sDataalloc = NULL;
	}

	traceLogger->add(lID, eID, TraceCoreTimer(), iLen, iDataalloc, sLen, sDataalloc);
#endif
}

/***************** Class TraceEntry Definition *****************/
TraceEntry::TraceEntry(TraceEntry& te)
{
	languageID = te.languageID;
	eventID    = te.eventID;
	timestamp  = te.timestamp;
	eLen	   = te.eLen;
	entity	   = te.entity;
	iLen	   = te.iLen;
	iData	   = te.iData;
	sLen	   = te.sLen;
	sData	   = te.sData;
}

TraceEntry::~TraceEntry()
{
	if(entity) free(entity);
	if(iData)  free(iData);
	if(sData)  free(sData);
}

void TraceEntry::write(FILE* fp, int prevLID, int prevSeek, int nextLID, int nextSeek)
{
	//NOTE: no need to write languageID to file
	if(prevLID == 0 && nextLID ==0)
		fprintf(fp, "%d %f %d %d ", eventID, timestamp, 0, 0);
	else if(prevLID == 0 && nextLID !=0)
		fprintf(fp, "%d %f %d %d %d", eventID, timestamp, 0, nextLID, nextSeek);
	else if(prevLID != 0 && nextLID ==0)
		fprintf(fp, "%d %f %d %d %d", eventID, timestamp, prevLID, prevSeek, 0);
	else // if(prevLID != 0 && nextLID !=0)
		fprintf(fp, "%d %f %d %d %d %d", eventID, timestamp, prevLID, prevSeek, nextLID, nextSeek);

	fprintf(fp, " %d", eLen);
	if(eLen != 0) {
	  for(int i=0; i<eLen; i++) fprintf(fp, " %d", entity[i]);
	}

	fprintf(fp, " %d", iLen);
	if(iLen != 0) {
	  for(int i=0; i<iLen; i++) fprintf(fp, " %d", iData[i]);
	}

	if(sLen !=0) fprintf(fp, " %s\n", sData);
	else fprintf(fp, "\n");

	// free memory
	if(entity){
		free(entity);
	}
	entity = NULL;
	if(iData){
		free(iData);
	}
	iData = NULL;
	if(sData){
		free(sData);
	}
	sData=NULL;
}

/***************** Class TraceLogger Definition *****************/
TraceLogger::TraceLogger(char* program, int b):
	numLangs(1), numEntries(0), lastWriteFlag(0), prevLID(0), prevSeek(0)
{
  binary = b;

  

  poolSize = 10000; // CkpvAccess(CtrLogBufSize);
  pool = new TraceEntry[poolSize+5];
//  CmiPrintf("CtrLogBufSize %d \n",CkpvAccess(CtrLogBufSize));
 // CmiPrintf("PoolSize = %d \n",poolSize);
  for (int lID=0;lID<MAX_NUM_LANGUAGES;lID++) {
    lName[lID]=NULL;
    fName[lID]=NULL;
  }

  pgm = new char[strlen(program)+1];
  sprintf(pgm, "%s", program);
  numEntries = 0;
  isWriting = 0;
  buffer = NULL;

  //CmiPrintf("In TraceLogger Constructor %s %d",pgm,strlen(program)+1);
  //initlogfiles();

}

 void TraceLogger::initlogfiles(){
	openLogFiles();
  	closeLogFiles();
}


TraceLogger::~TraceLogger()
{
  if(binary)
  { lastWriteFlag = 1; writeBinary(); }
  else
  { lastWriteFlag = 1; write(); }
  for (int lID=0;lID<MAX_NUM_LANGUAGES;lID++) {
    delete[] lName[lID];
    delete[] fName[lID];
  }
}

void TraceLogger::RegisterLanguage(int lID, const char* ln)
{
	numLangs++;

	lName[lID] = new char[strlen(ln)+1];
	sprintf(lName[lID], "%s", ln);

	char pestr[10]; sprintf(pestr, "%d\0", CmiMyPe());
	fName[lID] = new char[strlen(pgm)+1+strlen(pestr)+1+strlen(ln)+strlen(".log")+10];
	sprintf(fName[lID], "%s.%s.%s.log", pgm, pestr, ln);

	// my debug code - schak
	//CmiPrintf("%s at %d in %d \n",fName[lID],lID,fName[lID]);
	if(CpvAccess(_traceCoreOn) == 0){
		CmiPrintf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1_traceCoreOn = 0 in RegisterLanguage \n");
                return;
	}
	FILE* fp = NULL;
	do
  	{
    	fp = fopen(fName[lID], "w");
  	} while (!fp && (errno == EINTR || errno == EMFILE));
  	if(!fp) {
    	CmiAbort("Cannot open Projector Trace File for writing ... \n");
  	}
  	if(!binary) {
    	fprintf(fp, "PROJECTOR-RECORD: %s.%s\n", pestr, lName[lID]);
  	}
  	//fclose(fp);
	fptrs[lID] = fp;
}

void TraceLogger::verifyFptrs(){
	
  for(int i=1; i<numLangs; i++){
		if(!fptrs[i]){
			CmiPrintf("Null File Pointer Found after Open\n");
		}
  }
}

void TraceLogger::write(void)
{
	if(CpvAccess(_traceCoreOn) == 0){
		return;
	}
 	//openLogFiles();
	verifyFptrs();
	int currLID=0, nextLID=0;
	int pLID=0, nLID=0;
	int currSeek=0, nextSeek=0;
	int i;
  	for(i=0; i<numEntries-1; i++) {
		currLID  = pool[i].languageID;
		FILE* fp = fptrs[currLID];
		if(fp ==  NULL)
			return;
		currSeek = ftell(fp);
		nextLID  = pool[i+1].languageID;
		nextSeek = ftell(fptrs[nextLID]);

		pLID = ((prevLID==currLID)?0:prevLID);
		nLID = ((nextLID==currLID)?0:nextLID);
		pool[i].write(fp, pLID, prevSeek, nLID, nextSeek);

		prevSeek = currSeek; prevLID = currLID;
		flushLogFiles();
  	}
	if(lastWriteFlag==1) {
		currLID  = pool[i].languageID;
		FILE* fp = fptrs[currLID];
		if(fp == NULL)
			return;
		currSeek = ftell(fp);
		nextLID  = nextSeek = 0;

		pLID = ((prevLID==currLID)?0:prevLID);
		nLID = ((nextLID==currLID)?0:nextLID);
		pool[i].write(fp, pLID, prevSeek, nLID, nextSeek);
		closeLogFiles();
	}

  	
}

//TODO
void TraceLogger::writeBinary(void) {}
//TODO
void TraceLogger::writeSts(void) {}

void TraceLogger::add(int lID, int eID, double timestamp, int iLen, int* iData, int sLen, char* sData)
{
	
  if(isWriting){
	//  CmiPrintf("Printing in buffer \n");
	  buffer = new TraceEntry(lID, eID, timestamp, iLen, iData, sLen, sData);
  }else{
  new (&pool[numEntries]) TraceEntry(lID, eID, timestamp, iLen, iData, sLen, sData);
  numEntries = numEntries+1;
if(numEntries>= poolSize) {
    double writeTime = TraceCoreTimer();
    isWriting = 1;
    if(binary) writeBinary();
	else 	   write();


    new (&pool[0]) TraceEntry(pool[numEntries-1]);
    //numEntries = 1;
    numEntries=1;
    if(buffer != NULL){
	    new (&pool[1]) TraceEntry(*buffer);
	    numEntries=2;
	    delete buffer;
	    buffer = NULL;
    }
        isWriting = 0;
	//TODO
    //new (&pool[numEntries++]) TraceEntry(0, BEGIN_INTERRUPT, writeTime);
    //new (&pool[numEntries++]) TraceEntry(0, END_INTERRUPT, TraceCoreTimer());
  }
 }
}

void TraceLogger::openLogFiles()
{
  CmiPrintf("[%d]Entering openLogFile \n",CmiMyPe());
  for(int i=1; i<numLangs; i++) {

	FILE* fp = NULL;
	do
  	{

    			fp = fopen(fName[i], "a");

  	} while (!fp && (errno == EINTR || errno == EMFILE));
  	if(!fp) {
	//	CmiPrintf("FILE NAME %s at %d \n",fName[i],i);
	    	CmiAbort("Cannot open Projector Trace File for writing ... \n");
  	}
	CmiPrintf("[%d]Iteration %d : fp %d \n",CmiMyPe(),i,fp);
	fptrs[i] = fp;

	if(i == 1)
		assert(fptrs[1]);
	else if(i == 2)
	{
		assert(fptrs[1]);
	        assert(fptrs[2]);
	}
	else if(i>= 3)
	{
		assert(fptrs[1]);
	        assert(fptrs[2]);
	        assert(fptrs[3]);
	}
  }
  CmiAssert(fptrs[1]);
    CmiAssert(fptrs[2]);
      CmiAssert(fptrs[3]);
  CmiPrintf("[%d]In Open log files ........\n",CmiMyPe());
  verifyFptrs();
  CmiPrintf("[%d].....................\n",CmiMyPe());
}

void TraceLogger::closeLogFiles()
{

  for(int i=1; i<numLangs; i++){
		if(fptrs[i])
			fclose(fptrs[i]);
		fptrs[i]=NULL;
		
  }
}

void TraceLogger::flushLogFiles(){
	for(int i=1;i<numLangs;i++){
		fflush(fptrs[i]);
	}
}

