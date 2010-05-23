#include <stdio.h>
#include <math.h>
#include "blue.h"
#include "bigsim_timing.h"
#include "bigsim_logs.h"

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
    printf("Loading bglog of proc %d from bgTrace%d succeed. \n", i, fileNum);

    // some sanity checking
    printf("Proc %d has %d events. \n", i, tline.length());
    for (int idx = 0; idx < tline.length(); idx ++)
    {
      BgTimeLog *bglog = tline[idx];
#if 1
      if (fabs(bglog->execTime - ( bglog->endTime - bglog->startTime)) > 1e-6)
        printf("Error: Invalid log [%d,#%d]: startT: %f endT: %f execT: %f\n", 
               i, idx, bglog->startTime, bglog->endTime, bglog->execTime);
#endif
      int bDepLen = bglog->backwardDeps.length();
#if 0
      if (bDepLen>0 && bglog->msgId.pe()!=-1) {
        if (bglog->msgId.pe() != i) {
          printf("Error: [%d] Invalid log entry --- bDepLen:%d from PE %d\n", i, bDepLen, bglog->msgId.pe());
        }
      }
#else
      if (bDepLen>0 && bglog->msgId.pe()!=-1 && bglog->msgId.msgID()!=-1) {
          printf("Error: [%d] Invalid log entry (event with both incoming message and backward deps )--- bDepLen:%d from msg (PE:%d, id:%d)\n", i, bDepLen, bglog->msgId.pe(), bglog->msgId.msgID());
      }
#endif
      for(int midx=0; midx < bglog->msgs.length(); midx++){
        BgMsgEntry *msg = bglog->msgs[midx];
        if (msg->sendTime < bglog->startTime || msg->sendTime > bglog->endTime)
          printf("[%d] Invalid MsgEntry [%d]: sendTime: %f in log startT: %f endT: %f execT: %f\n", i, idx, msg->sendTime, bglog->startTime, bglog->endTime, bglog->execTime);
        if (msg->sendTime > msg->recvTime)
          printf("[%d] Invalid recvTime in MsgEntry [%d]: sendTime: %f recvTime: %f\n", i, idx, msg->sendTime, msg->recvTime);
      }
    }

    // dump bg timeline log to disk in ASCII format
    BgWriteThreadTimeLine("detail", 0, 0, 0, procNum, tline.timeline);
  }

  delete [] allNodeOffsets;
  printf("End of program\n");
}

