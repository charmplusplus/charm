#include "blue.h"
#include "blue_impl.h"
#include "blue_types.h"
#include "blue_logs.h"
#include "assert.h"

BgTimeLineRec* currTline = NULL;
int currTlineIdx=0;

//Used for parallel file I/O
void readProc(int procNum, int numWth ,int numPes, int totalProcs, int* allNodeOffsets, BgTimeLineRec& tlinerec){

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
  PUP::fromDisk p(f);

  fseek(f,fileOffset,SEEK_SET);
  tlinerec.pup(p);
  fclose(f);

  return;
}


int* loadOffsets(int totalProcs, int numPes){

  int* allProcOffsets = new int[totalProcs];
  int arrayOffset=0, procsInPe;
  char* d = new char[10];
  FILE* f;

 //TODO: right now works only for BG/L, later have to pup number of worker and comm threads per node also
  for (int i=0; i<numPes; i++){
    sprintf(d,"bgTrace%d",i);
    f = fopen(d,"r");
    PUP::fromDisk p(f);
    p|procsInPe;

    p(allProcOffsets+arrayOffset,procsInPe);
    arrayOffset += procsInPe;
    fclose(f);
  }
  return  allProcOffsets;
}
