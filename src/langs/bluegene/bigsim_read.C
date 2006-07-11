#include "blue.h"
#include "blue_impl.h"
#include "blue_types.h"
#include "bigsim_logs.h"
#include "assert.h"

BgTimeLineRec* currTline = NULL;
int currTlineIdx=0;

//Used for parallel file I/O
int BgReadProc(int procNum, int numWth ,int numPes, int totalProcs, int* allNodeOffsets, BgTimeLineRec& tlinerec){

  /*Right now works only for cyclicMapInfo - needs a more general scheme*/
  int nodeNum = procNum/numWth;
  int numNodes = totalProcs/numWth;
  int fileNum = nodeNum%numPes;
  int arrayID=0, fileOffset;
  char* fName = new char[10];
  //BgTimeLineRec* tlinerec = new BgTimeLineRec;
  
  for(int i=0;i<fileNum;i++)
    arrayID += (numNodes/numPes + ((i < numNodes%numPes)?1:0))*numWth;
  
  arrayID += (nodeNum/numPes)*numWth + procNum%numWth;
  fileOffset = allNodeOffsets[arrayID];

   //   CmiPrintf("nodeNum: %d arrayId:%d numNodes:%d numPes:%d\n",nodeNum,arrayID,numNodes,numPes);
 
  sprintf(fName,"bgTrace%d",fileNum);
  FILE*  f = fopen(fName,"r");
  if (f == NULL) {
    printf("Error> Open failed with %s. \n", fName);
    return -1;
  }
//  PUP::fromDisk p(f);
  PUP::fromDisk pd(f);
  PUP::machineInfo machInfo;
  pd((char *)&machInfo, sizeof(machInfo));
  if (!machInfo.valid()) CmiAbort("Invalid machineInfo on disk file!\n");
  PUP::xlater p(machInfo, pd);

  fseek(f,fileOffset,SEEK_SET);
  tlinerec.pup(p);
  fclose(f);

  return fileNum;
}


// user has to remember to free allProcOffsets
int* BgLoadOffsets(int totalProcs, int numPes){

  int* allProcOffsets = new int[totalProcs];
  int arrayOffset=0, procsInPe;
  char d[128];

  PUP::machineInfo machInfo;
  for (int i=0; i<numPes; i++){
    sprintf(d,"bgTrace%d",i);
    FILE *f = fopen(d,"r");
    PUP::fromDisk pd(f);
    pd((char *)&machInfo, sizeof(machInfo));
    PUP::xlater p(machInfo, pd);
    if (!machInfo.valid()) CmiAbort("Invalid machineInfo on disk file!\n");
    p|procsInPe;

    p(allProcOffsets+arrayOffset,procsInPe);
    arrayOffset += procsInPe;
    fclose(f);
  }
  return  allProcOffsets;
}


int BgLoadTraceSummary(char *fname, int &totalProcs, int &numX, int &numY, int &numZ, int &numCth, int &numWth, int &numPes)
{
  BGMach  bgMach ;
  PUP::machineInfo machInfo;

  FILE* f = fopen(fname,"r");
  if (f == NULL) {
    printf("Error> Open failed with %s. \n", fname);
    return -1;
  }

  PUP::fromDisk pd(f);
  pd((char *)&machInfo, sizeof(machInfo));	// load machine info
  if (!machInfo.valid()) CmiAbort("Invalid machineInfo on disk file!\n");
  PUP::xlater p(machInfo, pd);
  p|totalProcs;
  p|bgMach;
  numX = bgMach.x;
  numY = bgMach.y;
  numZ = bgMach.z;
  numCth = bgMach.numCth;
  numWth = bgMach.numWth;
  p|numPes;

  bglog_version = 0;
  if (!feof(f)) p|bglog_version;

  fclose(f);
  return 0;
}

