
#include "blue.h"
#include "blue_impl.h"
#include "blue_logs.h"
#include "loadlog.decl.h"

extern BgTimeLineRec* currTline;
extern int currTlineIdx;

class Main: public Chare {
public:
  Main()
  {
  int totalProcs, numX, numY, numZ, numCth, numWth, numPes;

  BgLoadTraceSummary("bgTrace", totalProcs, numX, numY, numZ, numCth, numWth, numPes);
                                                                                
  int* allNodeOffsets = BgLoadOffsets(totalProcs,numPes);

  for (int i=0; i<numPes; i++) 
  {
    BgTimeLineRec tline;
    int procNum = i;
    currTline = &tline;
    currTlineIdx = procNum;
    BgReadProc(procNum,numWth,numPes,totalProcs,allNodeOffsets,tline);
                                                                                
    // dump bg timeline log to disk
    BgWriteThreadTimeLine("detail", 0, 0, 0, procNum, tline.timeline);
  }

  delete [] allNodeOffsets;
  CkExit();
  }
};

#include "loadlog.def.h"
