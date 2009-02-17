
#include <math.h>
#include "log2proj.decl.h"
#include "blue.h"
#include "blue_impl.h"
#include "bigsim_logs.h"
#include "charm++.h"
#include "trace-projections.h"

extern BgTimeLineRec* currTline;
extern int currTlineIdx;

class Main : public Chare
{
public:
Main(CkArgMsg* m)
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
    printf("Loading bglog of proc %d from bgTrace%d succeed. \n", i, fileNum);
                                                                                
    // some senity checking and generate projection log
    printf("Proc %d has %d logs. \n", i, tline.length());
    char fname[128];
    sprintf(fname, "%s.%d.log", "app", procNum);
    FILE *lf = fopen(fname, "w");
    fprintf(lf, "PROJECTIONS-RECORD\n");
    printf("generate projections log %s\n", fname);
    toProjectionsFile p(lf);
    LogEntry *foo = new LogEntry(0.0, 6);
    foo->pup(p);

    double lastT = 0;
    for (int idx = 0; idx < tline.length(); idx ++)
    {
      BgTimeLog *bglog = tline[idx];
#if 1
      if (fabs(bglog->execTime - ( bglog->endTime - bglog->startTime)) > 1e-6)
        printf("Invalid log [%d,%d]: startT: %f endT: %f execT: %f\n", i, idx,
	       bglog->startTime, bglog->endTime, bglog->execTime);
#endif

      LogEntry *beginLog = new LogEntry(bglog->startTime, 2);
      beginLog->event = bglog->msgId.msgID();
      beginLog->pe = bglog->msgId.pe();
      beginLog->eIdx = bglog->charm_ep==-1?0:bglog->charm_ep; 
      beginLog->pup(p);
      delete beginLog;

      for(int midx=0; midx < bglog->msgs.length(); midx++){
        BgMsgEntry *msg = bglog->msgs[midx];
        if (msg->sendTime < bglog->startTime || msg->sendTime > bglog->endTime)
          printf("[%d] Invalid MsgEntry [%d]: sendTime: %f in log startT: %f endT: %f execT: %f\n", i, idx, msg->sendTime, bglog->startTime, bglog->endTime, bglog->execTime);
        LogEntry *sendLog = new LogEntry(msg->sendTime, 1);
        sendLog->event = msg->msgID;
        sendLog->pup(p);
        delete sendLog;
      }

      LogEntry *endLog = new LogEntry(bglog->startTime+bglog->execTime, 3);
      endLog->pup(p);
      delete endLog;
      lastT = bglog->endTime;
    }

    foo = new LogEntry(lastT, 7);
    foo->pup(p);
    fclose(lf);
  }

  delete [] allNodeOffsets;
  printf("End of program\n");
  CkExit();
}

};

#include "log2proj.def.h"
