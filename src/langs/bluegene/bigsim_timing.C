
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

void bgTimingLog::adjustTimingLog(double tAdjust)
{
	startTime += tAdjust;
	endTime   += tAdjust;

	for(int i=0; i<msgs.length(); i++) {
		msgs[i]->sendtime += tAdjust;
		//TODO send correction
	}
}

void BgAdjustTimeLineInsert(bgTimingLog* tlog, BgTimeLine &tline)
{
	//FIXME
	/* ASSUMPTION: BgAdjustTimeLineInit is called only if necessary */

	/* search appropriate index, 'idx' of 'msg' in timeline */
	int idx = 0;
	while((idx < tline.length()) && (tline[idx]->startTime > tlog->startTime))
		idx++;
	
	/* store entry corresponding to 'msg' in timeline at 'idx' */
	tline->insert(idx, tlog);

	/* adjust all entries following 'idx' in timeline */
	while(idx < tline.length()-1) {
		double tAdjust = tline[idx]->endTime - tline[idx+1]->startTime;
		if(tAdjust <= 0)	// log fits in the idle time
			break; 
		else {
			idx++;
			tline[idx]->adjustTimeLog(tAdjust);
		}
	}
}

void BgAdjustTimeLineForward(int msgID, double tAdjust, BgTimeLine &tline)
{
	/* search index, 'idx' of 'msgID' in timeline */
	int idx = 0;
	while((idx < tline.length()) && (tline[idx]->msgID != msgID))
		idx++;

	//FIXME is remove implemented ?
	/* remove entry at 'idx' from timeline */
	bgTimingLog* tlog = (bgTimingLog*)(tline->remove(idx));

	/* adjust timing of 'tlog' */
	tlog->adjustTimingLog(tAdjust);

	/* insert entry at proper place in timeline */
    BgAdjustTimeLineInit(bgTimingLog* tlog, BgTimeLine &tline);
}

void BgPrintThreadTimeLine(int pe, int th, BgTimeLine &tline)
{
  for (int i=0; i<tline.length(); i++)
    tline[i]->print(pe, th);
}

