
#include "blue.h"
#include "blue_timing.h"

CpvStaticDeclare(int, msgCounter);

void BgInitTiming()
{
  CpvInitialize(int, msgCounter);
  CpvAccess(msgCounter) = 0;
}

void BgMsgSetTiming(char *msg)
{
  CmiBgMsgID(msg) = CpvAccess(msgCounter)++;
  CmiBgMsgSrcPe(msg) = BgMyNode();
}

bgMsgEntry::bgMsgEntry(char *msg)
{
  msgID = CmiBgMsgID(msg);
  sendtime = BgGetTime();
}

void bgMsgEntry::print()
{
  CmiPrintf("msgID: %d sendtime: %f\n", msgID, sendtime);
}

bgTimingLog::bgTimingLog(int epc, char *msg)
{
  if (msg == NULL) CmiAbort("bgTimingLog: msg is NULL!");
  ep = epc;
  time = BgGetTime();
  srcpe = CmiBgMsgSrcPe(msg);
  msgID = CmiBgMsgID(msg);
}

bgTimingLog::~bgTimingLog()
{
  for (int i=0; i<msgs.length(); i++)
    delete msgs[i];
}

void bgTimingLog::addMsg(char *msg)
{
  msgs.push_back(new bgMsgEntry(msg));
}

void bgTimingLog::print()
{
  CmiPrintf("<< == [%d] ep:%d time:%f srcpe:%d msgID:%d\n", BgMyNode(), ep, time, srcpe, msgID);
  for (int i=0; i<msgs.length(); i++)
    msgs[i]->print();
  CmiPrintf("== >>\n");
}
