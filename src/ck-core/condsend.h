
/***************************************************************************
 * RCS INFORMATION:
 *
 *	$RCSfile$
 *	$Author$	$Locker$		$State$
 *	$Revision$	$Date$
 *
 ***************************************************************************
 * DESCRIPTION:
 *
 ***************************************************************************
 * REVISION HISTORY:
 *
 * $Log$
 * Revision 2.4  1995-10-13 18:15:53  jyelon
 * K&R changes.
 *
 * Revision 2.3  1995/06/29  22:34:30  narain
 * Changed prototype of  SendMsgIfConditionArises to static
 *
 * Revision 2.2  1995/06/18  21:30:15  sanjeev
 * separated charm and converse condsends
 *
 ***************************************************************************/


#ifndef CH_CONDS_H
#define CH_CONDS_H


typedef struct {
  int entry;
  void *msg;
  ChareIDType *cid;
} SendMsgStuff;

typedef struct {
  FUNCTION_PTR fn_ptr;
  int bocNum;
} CallBocStuff;

/* Function implemented but not to be used .. */
static void SendMsgIfConditionArises  CMK_PROTO((int condnum, int entry, void *msg, int size, ChareIDType *cid));
void CallBocIfConditionArises CMK_PROTO((int condnum, FUNCTION_PTR fnp, int bocNum));
void SendMsgAfter CMK_PROTO((unsigned int deltaT, int entry, void *msg, int size, ChareIDType *cid));
void CallBocAfter CMK_PROTO((FUNCTION_PTR fnp, int bocNum, unsigned int deltaT));
void CallBocOnCondition CMK_PROTO((FUNCTION_PTR fnp, int bocNum));

int NoDelayedMsgs CMK_PROTO((void));

#endif
