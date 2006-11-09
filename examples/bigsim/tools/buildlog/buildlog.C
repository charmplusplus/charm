
#include "blue.h"
#include "blue_impl.h"

extern BgTimeLineRec* currTline;
extern int currTlineIdx;

#define totalPEs  2

int main()
{
  genTimeLog = 1;

  BgTimeLineRec tlinerecs[totalPEs];
  BgMsgID msgID;

  // PE 0
  double curTime = 1.0;
  BgTimeLog *newlog = new BgTimeLog(1, "logname", curTime, curTime+0.5);
  curTime = 1.4;
    // send a message to 1
  msgID = BgMsgID(0, 0);      // (mype, seqno)
  BgMsgEntry *msgEntry = new BgMsgEntry(0, 1024, curTime, curTime+0.001, 1, 0);
  newlog->addMsg(msgEntry);
  tlinerecs[0].logEntryInsert(newlog);

  // PE 1
  curTime = 1.0;
  newlog = new BgTimeLog(1, 0, 0, curTime, curTime+1.0);
  tlinerecs[1].logEntryInsert(newlog);

  // start output

  BgWriteTraceSummary(totalPEs, totalPEs, totalPEs, 1, 1);

  BgWriteTimelines(0, tlinerecs, totalPEs);

  char fname[128];
  strcpy(fname, "bgTrace");
  for (int i=0; i<totalPEs; i++)
  BgWriteThreadTimeLine(fname, i, 0, 0, 0, tlinerecs[i].timeline);

}

