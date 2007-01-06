/*

   Author: Isaac Dooley
   email : idooley2@uiuc.edu
   date  : Jan 06, 2007

   This program will take in a set of bgTrace files from a run
   of the Big Emulator and it will rewrite the files so that
   named events can be replaced with better approximate timings
   supplied interactively by the user of this program. For example
   a cycle-accurate simulator might be able to give the exact times
   for some Charm++ entry methods.

   The modified bgTrace files are put into a directory called
   "newtraces"

   The traces are currently all dumped into a single bgTrace0 file,
   as if the emulation was run on one processor, but all the virtual
   processors are contained in that file.

   This program was written by someone with little experience with
   the POSE based Big Simulator, so these traces might not quite
   act as the author expected. Please report any problems to
   ppl@cs.uiuc.edu

*/



#include "blue.h"
#include "blue_impl.h"
#include "blue.h"
#include "blue_impl.h"
#include "blue_types.h"
#include "bigsim_logs.h"
#include "assert.h"


extern BgTimeLineRec* currTline;
extern int currTlineIdx;


#define OUTPUTDIR "newtraces/"

void MyBgWriteTraceSummary(int nlocalProcs, int numPes, int x, int y=1, int z=1, int numWth=1, int numCth=1, char *traceroot=NULL);
void MyBgWriteTimelines(int seqno, BgTimeLineRec *tlinerecs, int nlocalProcs, int numWth=1, char *traceroot=NULL);

int MyBgLoadTraceSummary(char *fname, int &totalProcs, int &numX, int &numY, int &numZ, int &numCth, int &numWth, int &numPes);
int* MyBgLoadOffsets(int totalProcs, int numPes);
int MyBgReadProc(int procNum, int numWth, int numPes, int totalProcs, int* allNodeOffsets, BgTimeLineRec& tlinerec);


int main()
{
  int totalProcs, numX, numY, numZ, numCth, numWth, numPes;
  BgTimeLineRec *tlinerecs;

  // load bg trace summary file
  printf("Loading bgTrace ... \n");
  int status = MyBgLoadTraceSummary("bgTrace", totalProcs, numX, numY, numZ, numCth, numWth, numPes);
  if (status == -1) exit(1);
  printf("========= BgLog Version: %d ========= \n", bglog_version);
  printf("Found %d (%dx%dx%d:%dw-%dc) simulated procs on %d real procs.\n", totalProcs, numX, numY, numZ, numWth, numCth, numPes);

  int* allNodeOffsets = MyBgLoadOffsets(totalProcs,numPes);

  tlinerecs = new BgTimeLineRec[totalProcs];

  printf("========= Loading All Logs ========= \n");

  // load each individual trace file for each bg proc
  for (int i=0; i<totalProcs; i++)
  {
    int procNum = i;
    currTline = &tlinerecs[i];
    currTlineIdx = procNum;
    int fileNum = MyBgReadProc(procNum,numWth,numPes,totalProcs,allNodeOffsets,tlinerecs[i]);
    CmiAssert(fileNum != -1);
    printf("Load log of BG proc %d from bgTrace%d... \n", i, fileNum);

    // dump bg timeline log to disk in ascii format
    BgWriteThreadTimeLine("detail", 0, 0, 0, procNum, tlinerecs[i].timeline);
  }




// We should write out the timelines to the same number of files as we started with.
// The mapping from VP to file was probably round robin. Here we cheat and make just one file
// TODO : fix this to write out in same initial pattern
    MyBgWriteTraceSummary(totalProcs, 1, numX, numY, numZ, numCth, numWth, OUTPUTDIR);
//  for(int i=0; i<numPes; i++)
      MyBgWriteTimelines(0, &tlinerecs[0], totalProcs, numWth, OUTPUTDIR);



  delete [] allNodeOffsets;
  printf("End of program\n");
}







void MyBgWriteTraceSummary(int nlocalProcs, int numPes, int x, int y, int z, int numWth, int numCth, char *traceroot)
{
  char* d = new char[512];
  BGMach bgMach;

  const PUP::machineInfo &machInfo = PUP::machineInfo::current();

  if (!traceroot) traceroot="";
  sprintf(d, "%sbgTrace", traceroot);
  FILE *f = fopen(d,"w");
  if(f==NULL) {
      CmiPrintf("Creating trace file %s  failed\n", d);
      CmiAbort("BG> Abort");
  }
  PUP::toDisk p(f);
  p((char *)&machInfo, sizeof(machInfo));
  p|nlocalProcs;
  bgMach.x = x;
  bgMach.y = y;
  bgMach.z = z;
  bgMach.numWth = numWth;
  bgMach.numCth = numCth;
  p|(BGMach &)bgMach;
  p|numPes;
  p|bglog_version;

  printf("BgWriteTraceSummary> Number is numX:%d numY:%d numZ:%d numCth:%d numWth:%d numPes:%d totalProcs:%d bglog_ver:%d\n",bgMach.x,bgMach.y,bgMach.z,bgMach.numCth,bgMach.numWth,numPes,nlocalProcs,bglog_version);

  fclose(f);
}

void MyBgWriteTimelines(int seqno, BgTimeLineRec *tlinerecs, int nlocalProcs, int numWth, char *traceroot)
{
  int *procOffsets = new int[nlocalProcs];

  char *d = new char[512];
  sprintf(d, "%sbgTrace%d", traceroot?traceroot:"", seqno);
  FILE *f = fopen(d,"w");
  PUP::toDisk p(f);
  const PUP::machineInfo &machInfo = PUP::machineInfo::current();

  p((char *)&machInfo, sizeof(machInfo));   // machine info
  p|nlocalProcs;

  // CmiPrintf("Timelines are: \n");
  int procTablePos = ftell(f);
  int procTableSize = (nlocalProcs)*sizeof(int);
  fseek(f,procTableSize,SEEK_CUR);

  int numNodes = nlocalProcs / numWth;
  for(int i=0;i<nlocalProcs;i++) {
    BgTimeLineRec &t = tlinerecs[i];
    procOffsets[i] = ftell(f);
    t.pup(p);
  }

  fseek(f,procTablePos,SEEK_SET);
  p(procOffsets,nlocalProcs);
  fclose(f);

  CmiPrintf("BgWriteTimelines> Wrote to disk for %d simulated nodes. \n", nlocalProcs);
  delete [] d;
}


int MyBgLoadTraceSummary(char *fname, int &totalProcs, int &numX, int &numY, int &numZ, int &numCth, int &numWth, int &numPes)
{
  BGMach  bgMach ;
  PUP::machineInfo machInfo;

  FILE* f = fopen(fname,"r");
  if (f == NULL) {
    printf("Error> Open failed with %s. \n", fname);
    return -1;
  }

  PUP::fromDisk pd(f);
  pd((char *)&machInfo, sizeof(machInfo));  // load machine info
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



// user has to remember to free allProcOffsets
int* MyBgLoadOffsets(int totalProcs, int numPes){

  int* allProcOffsets = new int[totalProcs];
  int arrayOffset=0, procsInPe;
  char d[128];

  PUP::machineInfo machInfo;
  for (int i=0; i<numPes; i++){
    sprintf(d,"bgTrace%d",i);
    FILE *f = fopen(d,"r");
    if (f == NULL) {
      CmiPrintf("BgLoadOffsets: can not open file %s!\n", d);
      CmiAbort("BgLoadOffsets failed!\n");
    }
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

//Used for parallel file I/O
int MyBgReadProc(int procNum, int numWth ,int numPes, int totalProcs, int* allNodeOffsets, BgTimeLineRec& tlinerec){

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


