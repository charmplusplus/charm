
#include "stdio.h"

#include "blue.h"
#include "blue_impl.h"
#include "blue_timing.h"

#define max(a,b) ((a)>=(b)?(a):(b))

//temporary
static int testCount = 0;

CpvStaticDeclare(int, msgCounter);


void *BgCreateEvent(int eidx)
{
  bgTimeLog *entry = new bgTimeLog();
  entry->ep = eidx;
  entry->srcnode = BgMyNode();
  entry->startTime = BgGetCurTime();
  return (void *)entry;
}

void *BgInsertEvent(int eidx)
{
  bgTimeLog *entry = (bgTimeLog *)BgCreateEvent(eidx);
  tTIMELINE.enq(entry);
  return (void *)entry;
}

// must be called inside a timelog 
double BgGetRecvTime() 
{
  if (genTimeLog)  {
    int len = tTIMELINE.length();
    if (len) return tTIMELINE[len-1]->recvTime;
  }
  return 0.0;
}

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
  recvTime = CmiBgMsgRecvTime(msg);
  dstPe = CmiBgMsgNodeID(msg);
  tID = CmiBgMsgThreadID(msg);
}

bgTimeLog::bgTimeLog(int epc, char *msg)
{
  ep = epc;
  startTime = BgGetCurTime();
  recvTime = msg?CmiBgMsgRecvTime(msg):startTime;
  endTime = 0.0;
  srcnode = msg?CmiBgMsgSrcPe(msg):-1;
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
  CmiPrintf("<<== [%d th:%d] ep:%d startTime:%f endTime:%f srcnode:%d msgID:%d\n", node, th, ep, startTime, endTime, srcnode, msgID);
  for (int i=0; i<msgs.length(); i++)
    msgs[i]->print();
  CmiPrintf("==>>\n");
}

void bgTimeLog::write(FILE *fp)
{
  fprintf(fp, "<<== ep:%d startTime:%f endTime:%f srcnode:%d msgID:%d\n", ep, startTime, endTime, srcnode, msgID);
  for (int i=0; i<msgs.length(); i++)
    msgs[i]->write(fp);
  fprintf(fp, "==>>\n");
}

void bgTimeLog::adjustTimeLog(double tAdjust)
{
	//arg tAdjust is a relative time
	if(tAdjust == .0) return;

	startTime += tAdjust;
	endTime   += tAdjust;

	for(int i=0; i<msgs.length(); i++) {
          if (msgs[i]->dstPe < 0) {
            // FIXME broadcast here
            continue;
          }
          else {
	    msgs[i]->sendtime += tAdjust;
	    bgCorrectionMsg *msg = (bgCorrectionMsg *)CmiAlloc(sizeof(bgCorrectionMsg));
	    msg->msgID = msgs[i]->msgID;
	    msg->tID = msgs[i]->tID;
	    //msg->tAdjust is absolute recvTime at destination node
	    msg->tAdjust = msgs[i]->recvTime + tAdjust;
	    msg->destNode = msgs[i]->dstPe;
		
	    CmiSetHandler(msg, CpvAccess(bgCorrectionHandler));
	    CmiSyncSendAndFree(BgNodeToPE(msgs[i]->dstPe), sizeof(bgCorrectionMsg), (char*)msg);
          }
	}
}


static void BgGetMsgStartTime(double recvTime, BgTimeLine &tline, double* startTime, int* index)
{
	/* ASSUMPTION: BgGetMsgStartTime is called only if necessary */

	// binary search: index of FIRST entry in tline whose recvTime is 
	// greater then arg 'recvTime'
	int low = 0, high = tline.length();
	int idx = 0;
	while(1) {
		idx = (low + high)/2;
		if(recvTime < tline[idx]->recvTime) {
			if(idx == low || recvTime >= tline[idx-1]->recvTime) { break; }
			else { high = idx; }
		}
		else {
			low = idx+1;
			if(low == tline.length()) {
				idx = low;
				break;
			}
		}
	}

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

// move the last entry in BgTimeLine to it's proper position
int BgAdjustTimeLineInsert(BgTimeLine &tline)
{
	/* ASSUMPTION: no error testing needed */

	/* check is 'bgTimingLog' is for an in-order message */
	if(tline.length() == 1) return 0;

//	CmiPrintf("BgAdjustTimeLineInsert:: last %f secondlast %f\n", tline[tline.length()-1]->recvTime, tline[tline.length()-2]->recvTime); 
	if(tline[tline.length()-1]->recvTime >= tline[tline.length()-2]->recvTime) {
		return 0;
	}

	bgTimeLog* tlog = tline.remove(tline.length()-1);

	int idx = 0;
	double startTime = 0;
	BgGetMsgStartTime(tlog->recvTime, tline, &startTime, &idx);
	
	/* store entry corresponding to 'msg' in timeline at 'idx' */
	tline.insert(idx, tlog);

	// tAdjust is relative time
	double tAdjust = startTime - tlog->startTime;
	tline[idx]->adjustTimeLog(tAdjust);	// tAdjust would be '0' or -ve

	/* adjust all entries following 'idx' in timeline */
	while(idx < tline.length()-1) {
		tAdjust = max(tline[idx]->endTime,tline[idx+1]->recvTime) - tline[idx+1]->startTime;
		if(tAdjust <= 0.0)	// log fits in the idle time; would never be -ve
			return 1;
		else {
			idx++;
			tline[idx]->adjustTimeLog(tAdjust); // tAdjust would be +ve
		}
	}
	return 1;
}

int BgAdjustTimeLineForward(int msgID, double tAdjustAbs, BgTimeLine &tline)
{
  /* ASSUMPTION: BgAdjustTimeLineForward is called only if necessary */
  /* ASSUMPTION: no error testing needed */


  // FIXME can this search be made faster than linear search ?
  // It cannot be made binary search, since there is no ordering for msgIDs
  int idxOld = tline.length()-1;
  while((idxOld >= 0) && (tline[idxOld]->msgID != msgID))
  	idxOld--;
  // msg come earlier
  // CmiPrintf("msg come earlier\n");
  if (idxOld == -1) return 0;

  testCount++;
  if((testCount%1000)==0)
  	CmiPrintf("BgAdjustTimeLineForward\n");

  // remove the log temporarily
  int idx=0;
  double startTime=0;
  bgTimeLog* tlog = tline.remove(idxOld);

  // get the new startTime and proper idx for this tlog
  tlog->recvTime = tAdjustAbs;
  BgGetMsgStartTime(tlog->recvTime, tline, &startTime, &idx);
  // update to the new start time and insert back
  tline.insert(idx, tlog);

  double tAdjust = startTime - tlog->startTime;
  if(tAdjust==0.) return 1;
  tline[idx]->adjustTimeLog(tAdjust);

  if(tAdjust<0) {
  	// move log forward if required.
  	while(idx < idxOld) {
  	  tAdjust = max(tline[idx]->endTime,tline[idx+1]->recvTime) - 
                        tline[idx+1]->startTime;
  	  if(tAdjust <= 0.0)	// log fits in the idle time, would never be -ve
  		break;
  	  else {
  		idx++;
  		tline[idx]->adjustTimeLog(tAdjust); // tAdjust would be +ve
  	  }
  	}
  	// move log backward if required
  	idx = idxOld+1;
  	while(idx < tline.length()) {
  	  tAdjust = max(tline[idx-1]->endTime,tline[idx]->recvTime) - 
                        tline[idx]->startTime;
  	  if(tAdjust >= 0.0)	// would never be positive
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
  	  tAdjust = max(tline[idx]->endTime,tline[idx+1]->recvTime) - 
                        tline[idx+1]->startTime;
  	  if(tAdjust <= 0.0)	// log fits in the idle time
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
  	  if(tAdjust >= 0.0)	// would never be positive
  		break;
  	  else {
  		tline[idxOld]->adjustTimeLog(tAdjust);
  		idxOld++;
  	  }
  	}
  }
  return 1;
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




