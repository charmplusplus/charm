
#include <math.h>
#include "blue.h"
#include "blue_impl.h"

#define OUTPUTDIR "newtraces/"

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

  int numNodes = totalProcs / numWth;

  // load timelines from each emulating processor
  for (int i=0; i<numPes; i++) 
  {
    BgTimeLineRec *tlinerecs = new BgTimeLineRec[totalProcs/numPes+1];
    int rec_count = 0;

      // procNum is the target PE on this emulating processor
    for (int nodeNum=i;nodeNum<numNodes;nodeNum+=numPes) {
     for (int procNum=nodeNum*numWth; procNum<(nodeNum+1)*numWth; procNum++) {
       //for(int procNum=i;procNum<totalProcs;procNum+=numPes){

      BgTimeLineRec &tlinerec = tlinerecs[rec_count];
      rec_count++;

      currTline = &tlinerec;
      currTlineIdx = procNum;
      int fileNum = BgReadProc(procNum,numWth,numPes,totalProcs,allNodeOffsets,tlinerec);
      CmiAssert(fileNum != -1 && fileNum==i);
      printf("Loading bglog of proc %d from bgTrace%d succeed. \n", procNum, fileNum);
                                                                                
      BgTimeLine &timeLine = tlinerec.timeline;

      // some senity checking
      printf("Proc %d has %d events. \n", procNum, timeLine.length());
      for (int idx = 0; idx < timeLine.length(); idx ++)
      {
        BgTimeLog *bglog = timeLine[idx];
        for(int midx=0; midx < bglog->msgs.length(); midx++){
          BgMsgEntry *msg = bglog->msgs[midx];
          if (msg->sendTime < bglog->startTime || msg->sendTime > bglog->endTime)
            printf("[%d] Invalid MsgEntry [%d]: sendTime: %f in log startT: %f endT: %f execT: %f\n", i, idx, msg->sendTime, bglog->startTime, bglog->endTime, bglog->execTime);
        }  
      }

    }
    }
    BgWriteTimelines(i,tlinerecs,rec_count,OUTPUTDIR);
    delete[] tlinerecs;
  }

  delete [] allNodeOffsets;
  printf("End of program\n");
}

