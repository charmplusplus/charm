
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
  dstPe = CmiBgMsgNodeID(msg);
}

void bgMsgEntry::print()
{
  CmiPrintf("msgID:%d sendtime:%f dstPe:%d\n", msgID, sendtime, dstPe);
}

bgTimingLog::bgTimingLog(int epc, char *msg)
{
  if (msg == NULL) CmiAbort("bgTimingLog: msg is NULL!");
  ep = epc;
  startTime = endTime = BgGetTime();
  srcpe = CmiBgMsgSrcPe(msg);
  msgID = CmiBgMsgID(msg);
}

bgTimingLog::~bgTimingLog()
{
  for (int i=0; i<msgs.length(); i++)
    delete msgs[i];
}

void bgTimingLog::closeLog()
{
  endTime = BgGetTime();
}

void bgTimingLog::addMsg(char *msg)
{
  msgs.push_back(new bgMsgEntry(msg));
}

void bgTimingLog::print(int node, int th)
{
  CmiPrintf("<<== [%d th:%d] ep:%d startTime:%f endTime:%f srcpe:%d msgID:%d\n", node, th, ep, startTime, endTime, srcpe, msgID);
  for (int i=0; i<msgs.length(); i++)
    msgs[i]->print();
  CmiPrintf("==>>\n");
}

void BgPrintThreadTimeLine(int pe, int th, BgTimeLine &tline)
{
  for (int i=0; i<tline.length(); i++)
    tline[i]->print(pe, th);
}

