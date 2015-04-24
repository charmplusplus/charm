#include "blue.h"
#include "blue_impl.h"
#include "blue_types.h"
#include "bigsim_logs.h"
#include "assert.h"

BgTimeLineRec* currTline = NULL;
int currTlineIdx=0;
char* traceFileName="bgTrace";

//Used for parallel file I/O
int BgReadProc(int procNum, int numWth ,int numEmulatingPes, int totalWorkerProcs, int* allNodeOffsets, BgTimeLineRec& tlinerec){

  currTline = &tlinerec;

  /*Right now works only for cyclicMapInfo - needs a more general scheme*/
  int arrayIdx, fileNum;
  int arrayID = CyclicMapInfo::FileOffset(procNum, numWth, numEmulatingPes, totalWorkerProcs, fileNum, arrayIdx);
  int fileOffset = allNodeOffsets[arrayIdx];

   //   CmiPrintf("nodeNum: %d arrayId:%d numNodes:%d numPes:%d\n",nodeNum,arrayID,numNodes,numPes);
 
  char fName[20];
  sprintf(fName,"%s%d",traceFileName,fileNum);
  FILE*  f = fopen(fName,"rb");
  if (f == NULL) {
    printf("Error> Open failed with %s. \n", fName);
    return -1;
  }
//  PUP::fromDisk p(f);
  PUP::fromDisk pd(f);
  PUP::machineInfo machInfo;
  //pd((char *)&machInfo, sizeof(machInfo));
  pd | machInfo;
  if (!machInfo.valid()) CmiAbort("Invalid machineInfo on disk file!\n");
  PUP::xlater p(machInfo, pd);

  fseek(f,fileOffset,SEEK_SET);
  tlinerec.pup(p);
  fclose(f);

  currTline = NULL;
  return fileNum;
}

// This version only reads in a part (window) of the time line
int BgReadProcWindow(int procNum, int numWth ,int numEmulatingPes, int totalWorkerProcs, int* allNodeOffsets, BgTimeLineRec& tlinerec,
		     int& fileLoc, int& totalTlineLength, int firstLog, int numLogs) {

  int firstLogToRead, numLogsToRead, tLineLength;

  currTline = &tlinerec;
  
  /*Right now works only for cyclicMapInfo - needs a more general scheme*/
  int arrayIdx, fileNum;
  int arrayID = CyclicMapInfo::FileOffset(procNum, numWth, numEmulatingPes, totalWorkerProcs, fileNum, arrayIdx);
  int fileOffset = allNodeOffsets[arrayIdx];

  //  CmiPrintf("   BgReadProc: fileOffset=%d arrayID=%d nodeNum=%d numNodes=%d fileNum=%d\n", fileOffset, arrayID, nodeNum, numNodes, fileNum);

  char fName[20];
  sprintf(fName,"bgTrace%d",fileNum);
  FILE* f = fopen(fName,"r");
  if (f == NULL) {
    printf("Error> Open failed with %s. \n", fName);
    return -1;
  }
  PUP::fromDisk pd(f);
  PUP::machineInfo machInfo;
  //pd((char *)&machInfo, sizeof(machInfo));
  pd | machInfo;
  if (!machInfo.valid()) CmiAbort("Invalid machineInfo on disk file!\n");
  PUP::xlater p(machInfo, pd);

  //  CmiPrintf("BgReadProc: procNum=%d, ftell=%d\n", procNum, ftell(f));

  if (firstLog == 0) {
    fseek(f, fileOffset, SEEK_SET);
  } else {
    fseek(f, fileLoc, 0);
  }
  firstLogToRead = firstLog;
  if (numLogs < 1) {
    numLogsToRead = 0x7FFFFFFF;
  } else {
    numLogsToRead = numLogs;
  }
  if (firstLog != 0) {
    tLineLength = totalTlineLength;
  } else {
    tLineLength = -1;
  }
  //  CmiPrintf("   BgReadProc: procNum=%d, ftell=%d\n", procNum, ftell(f));
  tlinerec.winPup(p, firstLogToRead, numLogsToRead, tLineLength);
  if (firstLog == 0) {
    totalTlineLength = tLineLength;
  }
  //  CmiPrintf("      BgReadProc: procNum=%d, ftell=%d\n", procNum, ftell(f));
  fileLoc = ftell(f);
  fclose(f);
  currTline = NULL;

  return fileNum;
}

// user has to remember to free allProcOffsets
int* BgLoadOffsets(int totalProcs, int numPes){

  int* allProcOffsets = new int[totalProcs];
  int arrayOffset=0, procsInPe;
  char d[128];

  PUP::machineInfo machInfo;
  for (int i=0; i<numPes; i++){
    sprintf(d,"%s%d",traceFileName,i);
    FILE *f = fopen(d,"rb");
    if (f == NULL) {
      CmiPrintf("BgLoadOffsets: can not open file %s!\n", d);
      CmiAbort("BgLoadOffsets failed!\n");
    }  
    PUP::fromDisk pd(f);
    //pd((char *)&machInfo, sizeof(machInfo));
    pd | machInfo;
    PUP::xlater p(machInfo, pd);
    if (!machInfo.valid()) CmiAbort("Invalid machineInfo on disk file!\n");
    p|procsInPe;

    p(allProcOffsets+arrayOffset,procsInPe);
    arrayOffset += procsInPe;
    fclose(f);
  }
  return  allProcOffsets;
}

static int thread_ep = -1;

int BgLoadTraceSummary(const char *fname, int &totalWorkerProcs, int &numX, int &numY, int &numZ, int &numCth, int &numWth, int &numEmulatingPes)
{
  BGMach  bgMach ;
  PUP::machineInfo machInfo;

  FILE* f = fopen(fname,"rb");
  if (f == NULL) {
    printf("Error> Open failed with %s. \n", fname);
    return -1;
  }

  PUP::fromDisk pd(f);
//  pd((char *)&machInfo, sizeof(machInfo));	// load machine info
  pd | machInfo;
  if (!machInfo.valid()) CmiAbort("Invalid machineInfo on disk file!\n");
  PUP::xlater p(machInfo, pd);
  p|totalWorkerProcs;
  p|bgMach;
  numX = bgMach.x;
  numY = bgMach.y;
  numZ = bgMach.z;
  numCth = bgMach.numCth;
  numWth = bgMach.numWth;
  p|numEmulatingPes;

  bglog_version = 0;
  if (!feof(f)) p|bglog_version;

  if (!feof(f)) p|thread_ep;    // EP (CthResumeBigSimThreadIdx) to identify thread resume events

  fclose(f);
  return 0;
}


// return CthResumeBigSimThreadIdx, the thread resume event EP number
int BgLogGetThreadEP()
{
  return thread_ep;
}

