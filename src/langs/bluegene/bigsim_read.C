#include "blue.h"
#include "blue_impl.h"
#include "blue_types.h"
#include "blue_logs.h"
#include "assert.h"

BgTimeLineRec* currTline = NULL;
int currTlineIdx=0;

//Used for parallel file I/O
void BgReadProc(int procNum, int numWth ,int numPes, int totalProcs, int* allNodeOffsets, BgTimeLineRec& tlinerec){

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
//  PUP::fromDisk p(f);
  PUP::fromDisk p(f);
  PUP::machineInfo machInfo;
  p((char *)&machInfo, sizeof(machInfo));
  PUP::xlater p_xlater(machInfo, p);

  fseek(f,fileOffset,SEEK_SET);
  tlinerec.pup(p_xlater);
  fclose(f);

  return;
}


int* BgLoadOffsets(int totalProcs, int numPes){

  int* allProcOffsets = new int[totalProcs];
  int arrayOffset=0, procsInPe;
  char* d = new char[10];

  PUP::machineInfo machInfo;
  for (int i=0; i<numPes; i++){
    sprintf(d,"bgTrace%d",i);
    FILE *f = fopen(d,"r");
    PUP::fromDisk p(f);
    p((char *)&machInfo, sizeof(machInfo));
    PUP::xlater p_xlater(machInfo, p);
    p_xlater|procsInPe;

    p_xlater(allProcOffsets+arrayOffset,procsInPe);
    arrayOffset += procsInPe;
    fclose(f);
  }
  return  allProcOffsets;
}


int BgLoadTraceSummary(char *fname, int &totalProcs, int &numX, int &numY, int &numZ, int &numCth, int &numWth, int &numPes)
{
  PUP::machineInfo machInfo;

  FILE* f = fopen(fname,"r");
  PUP::fromDisk p(f);

  p((char *)&machInfo, sizeof(machInfo));	// load machine info
  PUP::xlater p_xlater(machInfo, p);
  p_xlater|totalProcs;
  p_xlater|numX; p_xlater|numY; p_xlater|numZ;
  p_xlater|numCth;p_xlater|numWth;
  p_xlater|numPes;
  fclose(f);
  return 0;
}

