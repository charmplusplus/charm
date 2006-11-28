
#include "bigsim_logs.h"

#define totalPEs  2

int main()
{
  BgTimeLineRec tlinerecs[totalPEs];

  // turn on log generation
  BgGenerateLogs();

  // PE 0
  double curTime = 1.0;
  BgTimeLog *newlog = new BgTimeLog();
  newlog->setEP(1);
  newlog->setName("Method0");
  newlog->setTime(curTime, curTime+0.5);
  curTime = 1.4;
  int seqno = 0;
    // send first message to 1
  BgMsgID msgID = BgMsgID(0, seqno++);      // (mype, seqno)
  BgMsgEntry *msgEntry = new BgMsgEntry(0, 1024, curTime, curTime+0.001, 1, 0);
  newlog->addMsg(msgEntry);
  curTime += 0.003;
    // send second message to 1
  msgID = BgMsgID(0, seqno++);      // (mype, seqno)
  msgEntry = new BgMsgEntry(0, 1024, curTime, curTime+0.001, 1, 0);
  newlog->addMsg(msgEntry);

  tlinerecs[0].logEntryInsert(newlog);


  // PE 1
  curTime = 1.0;
  newlog = new BgTimeLog(BgMsgID(0,0));   // receive the first message with ID of (0,0)
  newlog->setTime(curTime, curTime+1.0);  // set start/end time of execution
  newlog->setEP(1);                       // function handle
  newlog->setName("Method1");             // human-readable name
  tlinerecs[1].logEntryInsert(newlog);
  curTime += 2.0;

  BgTimeLog *prevlog = newlog;
  newlog = new BgTimeLog();   // no msg for this event, depend on the previous timelog
  newlog->setTime(curTime, curTime+1.0);
  newlog->setEP(2);
  newlog->setName("Method2");
  newlog->addBackwardDep(prevlog);
  tlinerecs[1].logEntryInsert(newlog);
  curTime += 1.5;

  newlog = new BgTimeLog(BgMsgID(0,1));   // receive the second message with ID of (0,1)
  newlog->setTime(curTime, curTime+2.0);
  newlog->setEP(3);
  newlog->setName("Method3");
  tlinerecs[1].logEntryInsert(newlog);

  // start output
    // write summary file for totalPEs processor traces, running on 
    // 1 REAL processor
  BgWriteTraceSummary(totalPEs, 1, totalPEs, 1, 1);

    // write all timelines into one bgTrace file
  BgWriteTimelines(0, tlinerecs, totalPEs);

#if 0
    // print ascii format of the files
  char fname[128];
  strcpy(fname, "bgTrace");
  for (int i=0; i<totalPEs; i++)
  BgWriteThreadTimeLine(fname, i, 0, 0, 0, tlinerecs[i].timeline);
#endif
}

