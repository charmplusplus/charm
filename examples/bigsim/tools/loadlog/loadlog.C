
#include "blue.h"
#include "blue_impl.h"
#include "blue_logs.h"

extern BgTimeLineRec* currTline;
extern int currTlineIdx;

int main()
{
  int totalProcs, numX, numY, numZ, numCth, numWth, numPes;

  // load bg trace summary file
  BgLoadTraceSummary("bgTrace", totalProcs, numX, numY, numZ, numCth, numWth, numPes);
                                                                                
  int* allNodeOffsets = BgLoadOffsets(totalProcs,numPes);

  // load each individual trace file for each bg proc
  for (int i=0; i<numPes; i++) 
  {
    BgTimeLineRec tline;
    int procNum = i;
    currTline = &tline;
    currTlineIdx = procNum;
    BgReadProc(procNum,numWth,numPes,totalProcs,allNodeOffsets,tline);
                                                                                
    // dump bg timeline log to disk in asci format
    BgWriteThreadTimeLine("detail", 0, 0, 0, procNum, tline.timeline);
  }

  delete [] allNodeOffsets;
}

