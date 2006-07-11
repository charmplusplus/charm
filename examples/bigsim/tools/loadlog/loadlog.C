
#include "blue.h"
#include "blue_impl.h"

extern BgTimeLineRec* currTline;
extern int currTlineIdx;

int main()
{
  int totalProcs, numX, numY, numZ, numCth, numWth, numPes;

  // load bg trace summary file
  printf("Loading bgTrace ... \n");
  int status = BgLoadTraceSummary("bgTrace", totalProcs, numX, numY, numZ, numCth, numWth, numPes);
  if (status == -1) exit(1);
  printf("========= BgLog Version: %d ========= \n", bglog_version);
  printf("Found %d (%dx%dx%d:%dw-%dc) simulated procs on %d real procs.\n", totalProcs, numX, numY, numZ, numWth, numCth, numPes);
                                                                                
  int* allNodeOffsets = BgLoadOffsets(totalProcs,numPes);

  // load each individual trace file for each bg proc
  for (int i=0; i<totalProcs; i++) 
  {
    BgTimeLineRec tline;
    int procNum = i;
    currTline = &tline;
    currTlineIdx = procNum;
    int fileNum = BgReadProc(procNum,numWth,numPes,totalProcs,allNodeOffsets,tline);
    CmiAssert(fileNum != -1);
    printf("Load log of BG proc %d from bgTrace%d... \n", i, fileNum);
                                                                                
    // dump bg timeline log to disk in asci format
    BgWriteThreadTimeLine("detail", 0, 0, 0, procNum, tline.timeline);
  }

  delete [] allNodeOffsets;
  printf("End of program\n");
}

