/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

/**
 * \addtogroup CkPerf
*/
/*@{*/

#ifndef __TRACE_COMMON_H__
#define __TRACE_COMMON_H__

#define  CREATION           1
#define  BEGIN_PROCESSING   2
#define  END_PROCESSING     3
#define  ENQUEUE            4
#define  DEQUEUE            5
#define  BEGIN_COMPUTATION  6
#define  END_COMPUTATION    7
#define  BEGIN_INTERRUPT    8
#define  END_INTERRUPT      9
#define  USER_EVENT         13
#define  BEGIN_IDLE         14
#define  END_IDLE           15
#define  BEGIN_PACK         16
#define  END_PACK           17
#define  BEGIN_UNPACK       18
#define  END_UNPACK         19

CkpvExtern(int, CtrLogBufSize);
CkpvExtern(char*, traceRoot);
CkpvExtern(double, traceInitTime);

#if CMK_BLUEGENE_CHARM
#define  TRACE_TIMER   BgGetTime
inline double TraceTimer() { return TRACE_TIMER(); }
#else
#define  TRACE_TIMER   CmiWallTimer
inline double TraceTimer() { return TRACE_TIMER() - CkpvAccess(traceInitTime); }
#endif

#endif

/*@}*/
