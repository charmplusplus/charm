
#include "stdio.h"

#include "blue.h"
#include "blue_impl.h"
#include "blue_timing.h"

#define max(a,b) ((a)>=(b)?(a):(b))

CpvStaticDeclare(int, msgCounter);

/**
  init Cpvs of timing module
*/
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
  sendtime = BgGetCurTime();
  dstPe = CmiBgMsgNodeID(msg);
  tID = CmiBgMsgThreadID(msg);
}

void bgMsgEntry::print()
{
  CmiPrintf("msgID:%d sendtime:%f dstPe:%d\n", msgID, sendtime, dstPe);
}

void bgMsgEntry::write(FILE *fp)
{
  fprintf(fp, "msgID:%d sendtime:%f dstPe:%d\n", msgID, sendtime, dstPe);
}

bgTimeLog::bgTimeLog(int epc, char *msg)
{
  ep = epc;
  startTime = BgGetCurTime();
  recvTime = CmiBgMsgRecvTime(msg);
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
  endTime = BgGetCurTime(); 
}

void bgTimeLog::print(int node, int th)
{
  CmiPrintf("<<== [%d th:%d] ep:%d startTime:%f endTime:%f srcnode:%d msgID:%d\n", node, th, ep, startTime, endTime, srcpe, msgID);
  for (int i=0; i<msgs.length(); i++)
    msgs[i]->print();
  CmiPrintf("==>>\n");
}

void bgTimeLog::write(FILE *fp)
{
  fprintf(fp, "<<== ep:%d startTime:%f endTime:%f srcnode:%d msgID:%d\n", ep, startTime, endTime, srcpe, msgID);
  for (int i=0; i<msgs.length(); i++)
    msgs[i]->write(fp);
  fprintf(fp, "==>>\n");
}

void bgTimeLog::adjustTimeLog(double tAdjust)
{
	if(tAdjust == 0) return;

	startTime += tAdjust;
	endTime   += tAdjust;

	for(int i=0; i<msgs.length(); i++) {
		msgs[i]->sendtime += tAdjust;
		bgCorrectionMsg *msg = (bgCorrectionMsg *)CmiAlloc(sizeof(bgCorrectionMsg));
		msg->msgID = msgs[i]->msgID;
		msg->tID = msgs[i]->tID;
		msg->tAdjust = tAdjust;
		msg->destNode = msgs[i]->dstPe;
		
		CmiSetHandler(msg, CpvAccess(bgCorrectionHandler));
		CmiSyncSendAndFree(BgNodeToPE(msgs[i]->dstPe), sizeof(bgCorrectionMsg), (char*)msg);
	}
}

void BgGetMsgStartTime(double recvTime, BgTimeLine &tline, double* startTime, int* index)
{
	/* ASSUMPTION: BgGetMsgStartTime is called only if necessary */

	//FIXME
	//replace linear search by binary search algorithm
	int idx = 0;
	while((idx < tline.length()) && (recvTime >= tline[idx]->recvTime))
		idx++;

	if(idx==0 || tline[idx-1]->endTime <= recvTime) {
		*startTime = recvTime;
		/* ASSUMPTION: the overhead of transferring msg from inbuffer 
		 * to executing thread is negligble, since the thread Q is empty.  
	     */ 
	}
	else {
		*startTime = tline[idx-1]->endTime;
	}

	*index = idx;
}

void BgAdjustTimeLineInsert(BgTimeLine &tline)
{
	/* ASSUMPTION: no error testing needed */

	/* check is 'bgTimingLog' is for an in-order message */
	if(tline.length() == 1)
		return;
//	CmiPrintf("BgAdjustTimeLineInsert:: last %f secondlast %f\n", tline[tline.length()-1]->recvTime, tline[tline.length()-2]->recvTime); 
	if(tline[tline.length()-1]->recvTime >= tline[tline.length()-2]->recvTime) {
		return;
	}

	bgTimeLog* tlog = tline.remove(tline.length()-1);

	int idx = 0;
	double startTime = 0;
	BgGetMsgStartTime(tlog->recvTime, tline, &startTime, &idx);
	
	/* store entry corresponding to 'msg' in timeline at 'idx' */
	tline.insert(idx, tlog);

	double tAdjust = startTime - tlog->startTime;
	tline[idx]->adjustTimeLog(tAdjust);	// tAdjust would be '0' or -ve

	/* adjust all entries following 'idx' in timeline */
	while(idx < tline.length()-1) {
		tAdjust = max(tline[idx]->endTime,tline[idx+1]->recvTime) - tline[idx+1]->startTime;
		if(tAdjust <= 0)	// log fits in the idle time; would never be -ve
			return;
		else {
			idx++;
			tline[idx]->adjustTimeLog(tAdjust); // tAdjust would be +ve
		}
	}
}

void BgAdjustTimeLineForward(int msgID, double tAdjust, BgTimeLine &tline)
{
	/* ASSUMPTION: BgAdjustTimeLineForward is called only if necessary */
	/* ASSUMPTION: no error testing needed */

//	CmiPrintf("BgAdjustTimeLineForward\n"); 
	if(tAdjust == 0) return;

	//FIXME can this search be made faster than linear search ?
	int idxOld = 0;
	while((idxOld < tline.length()) && (tline[idxOld]->msgID != msgID))
		idxOld++;

	int idx=0;
	double startTime=0;
	bgTimeLog* tlog = tline.remove(idxOld);
	tlog->recvTime += tAdjust;
	BgGetMsgStartTime(tlog->recvTime, tline, &startTime, &idx);
	tline.insert(idx, tlog);
	tAdjust = startTime - tlog->startTime;
	tline[idx]->adjustTimeLog(tAdjust);

	if(tAdjust==0) return;
	if(tAdjust<0) {
		// move log forward if required.
		while(idx < idxOld) {
			tAdjust = max(tline[idx]->endTime,tline[idx+1]->recvTime) - tline[idx+1]->startTime;
			if(tAdjust <= 0)	// log fits in the idle time, would never be -ve
				break;
			else {
				idx++;
				tline[idx]->adjustTimeLog(tAdjust); // tAdjust would be +ve
			}
		}
		// move log backward if required
		idx = idxOld+1;
		while(idx < tline.length()) {
			tAdjust = max(tline[idx-1]->endTime,tline[idx]->recvTime) - tline[idx]->startTime;
			if(tAdjust >= 0)	// would never be positive
				break;
			else {
				tline[idx]->adjustTimeLog(tAdjust);
				idx++;
			}
		}
	}
	else {
		// move log forward if required
		while(idx < tline.length()-1) {
			tAdjust = max(tline[idx]->endTime,tline[idx+1]->recvTime) - tline[idx+1]->startTime;
			if(tAdjust <= 0)	// log fits in the idle time
				break;
			else {
				idx++;
				tline[idx]->adjustTimeLog(tAdjust); // tAdjust would be +ve
			}

		}
		// move log backward if required
		while(idxOld < idx) {
			if(idxOld==0)
				tAdjust = tline[idxOld]->recvTime - tline[idxOld]->startTime;
			else
				tAdjust = max(tline[idxOld-1]->endTime,tline[idxOld]->recvTime) - tline[idxOld]->startTime;
			if(tAdjust >= 0)	// would never be positive
				break;
			else {
				tline[idxOld]->adjustTimeLog(tAdjust);
				idxOld++;
			}
		}
	}
}

void BgPrintThreadTimeLine(int pe, int th, BgTimeLine &tline)
{
  for (int i=0; i<tline.length(); i++)
    tline[i]->print(pe, th);
}

void BgWriteThreadTimeLine(char **argv, int x, int y, int z, int th, BgTimeLine &tline)
{
  char *fname = (char *)malloc(strlen(argv[0])+100);
  sprintf(fname, "%s-%d-%d-%d.%d.log", argv[0], x,y,z,th);
  FILE *fp = fopen(fname, "w+");
  for (int i=0; i<tline.length(); i++)
    tline[i]->write(fp);
  fclose(fp);
  free(fname);
}



