
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
 * Revision 2.3  1995-06-29 22:34:30  narain
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
static void SendMsgIfConditionArises(int condnum, int entry, void *msg, int size,
			      ChareIDType *cid);
void CallBocIfConditionArises(int condnum, FUNCTION_PTR fnp, int bocNum);      
void SendMsgAfter(unsigned int deltaT, int entry, void *msg, int size, 
		  ChareIDType *cid);
void CallBocAfter(FUNCTION_PTR fnp, int bocNum, unsigned int deltaT);      
void CallBocOnCondition(FUNCTION_PTR fnp, int bocNum);

int NoDelayedMsgs();

#endif
