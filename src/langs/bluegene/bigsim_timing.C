
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

bgTimeLog::bgTimeLog(int epc, char *msg)
{
  ep = epc;
  startTime = BgGetTime();
  endTime = 0.0;
  srcpe = msg?CmiBgMsgSrcPe(msg):-1;
  msgID = msg?CmiBgMsgID(msg):-1;
}

bgTimeLog::~bgTimeLog()
{
  for (int i=0; i<msgs.length(); i++)
    delete msgs[i];
}

void bgTimeLog::closeLog()
{
  endTime = BgGetTime();
}

void bgTimeLog::addMsg(char *msg)
{
  msgs.push_back(new bgMsgEntry(msg));
}

void bgTimeLog::print(int node, int th)
{
  CmiPrintf("<<== [%d th:%d] ep:%d startTime:%f endTime:%f srcnode:%d msgID:%d\n", node, th, ep, startTime, endTime, srcpe, msgID);
  for (int i=0; i<msgs.length(); i++)
    msgs[i]->print();
  CmiPrintf("==>>\n");
}

void bgTimeLog::adjustTimeLog(double tAdjust)
{
	startTime += tAdjust;
	endTime   += tAdjust;

	for(int i=0; i<msgs.length(); i++) {
		msgs[i]->sendtime += tAdjust;
		//TODO send correction
	}
}

void BgAdjustTimeLineInsert(bgTimeLog* tlog, BgTimeLine &tline)
{
	//FIXME
	/* ASSUMPTION: BgAdjustTimeLineInit is called only if necessary */

	/* search appropriate index, 'idx' of 'msg' in timeline */
	int idx = 0;
	while((idx < tline.length()) && (tline[idx]->startTime > tlog->startTime))
		idx++;
	
	/* store entry corresponding to 'msg' in timeline at 'idx' */
	tline.insert(idx, tlog);

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
//	bgTimeLog* tlog = (bgTimeLog*)(tline->remove(idx));

	/* adjust timing of 'tlog' */
//	tlog->adjustTimeLog(tAdjust);

	/* insert entry at proper place in timeline */
//    BgAdjustTimeLineInit(bgTimeLog* tlog, BgTimeLine &tline);
}

void BgPrintThreadTimeLine(int pe, int th, BgTimeLine &tline)
{
  for (int i=0; i<tline.length(); i++)
    tline[i]->print(pe, th);
}

