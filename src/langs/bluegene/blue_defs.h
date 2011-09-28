#ifndef BLUE_DEFS_H
#define BLUE_DEFS_H

#define BIGSIM_TIMING     	1

#if (CMK_HAS_VALUES_H && (!CMK_BLUEGENEQ))
#   include <values.h>
#   define INVALIDTIME  MAXDOUBLE
#   define CMK_MAXINT   MAXINT
#else
#   define INVALIDTIME  (9999999999.99)
#   define CMK_MAXINT   (1000000000)
#endif

/* optimization parameters */
//#define SCHEDULE_WORK       1     now is schedule_flag
#define USE_MULTISEND	    0		/* must be BATCH_PROCESSING */
#define DELAY_CHECK         1
#define LIMITED_SEND        0		/* BATCH_PROCESSING must be 1 or 2 */
#define THROTTLE_WORK       1
#define DELAY_SEND          1

#define BG_LEASH               0.00005         /* 50us leash window */

#define CHECK_INTERVAL   10
#define CHECK_THRESHOLD  80000

#define CORRECTSENDLEN  5

#define PRIO_FACTOR      (1e7)

//Reads the logs from disk to do seq correction
#define SEQ_CORRECT 0
//Dumps the logs to disk
#define WRITE_TO_DISK 1


#define BG_EPSILON      (1e-9)

#define BG_MAX(a,b) ((a)>=(b)?(a):(b))
#define BG_MIN(a,b) ((a)<=(b)?(a):(b))
#define BG_ABS(x)   ((x)>=0?(x):(-(x)))


inline int isZero(double input){
  return (input < BG_EPSILON && input > -BG_EPSILON);
}

inline int isLess(double v1, double v2){
  return (v1 < v2-BG_EPSILON);
}

inline int isEqual(double v1, double v2){
  return isZero(v1-v2);
}


#if BIGSIM_TIMING

#define BG_ENTRYSTART(m)  \
	if (genTimeLog) tTIMELINEREC.logEntryStart(m);

#define BG_ENTRYEND()  \
	if (genTimeLog) BgLogEntryCommit(tTIMELINEREC);

#define BG_ADDMSG(m, node, tid, sendT, local, group)  	\
	BgMsgSetTiming(m); 	\
        if (genTimeLog)	{ \
	  if (tTHREADTYPE == WORK_THREAD) {	\
            BgTimeLineRec &tlinerec = tTIMELINEREC;	\
            int n = tlinerec.length();			\
            if (n>0) {					\
              BgTimeLog *tlog = tlinerec[n-1];		\
	      if (tlog->endTime < 0.0)			\
                tlog->addMsg(m, node, tid, sendT, local, group);	\
	      else {	 /* standalone msg */		\
		  /*CmiAssert(0);*/ 			\
		  /*double curT = CmiBgMsgRecvTime(m);*/		\
		  double curT = BgGetTime();		\
		  BgTimeLog *newLog = new BgTimeLog(-1, (char*)"addMsg", curT, curT); \
		  newLog->recvTime = newLog->effRecvTime = curT;	\
                  newLog->addMsg(m, node, tid, curT, local, group);	\
		  tlinerec.logEntryInsert(newLog);		\
		  tlinerec.clearSendingLogs();		\
		}					\
            }						\
	    /* log[log.length()-1]->print(); */		\
          }	\
	}

#else
#define BG_ENTRYSTART(m)
#define BG_ENTRYEND()
#define BG_ADDMSG(m, node, tid, local)
#endif

#endif
