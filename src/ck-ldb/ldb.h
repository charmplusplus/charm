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
 * Revision 2.1  1995-07-06 22:40:05  narain
 * LdbBocNum, interface to newseed fns.
 *
 * Revision 2.0  1995/06/29  21:19:36  narain
 * *** empty log message ***
 *
 ***************************************************************************/

readonly int LdbBocNum;

extern int CmiNumNeighbours();
extern CmiGetNodeNeighbours();
extern trace_creation();
extern int netSend();
extern CmiNeighboursIndex();
extern abs();
extern CallBocAfter();
extern CqsEmpty();
extern CqsLength();
extern CqsDequeue();
extern CqsEnqueue();
extern log();
extern CkMakeFreeCharesMessage();
extern CkQueueFreeCharesMessage();
extern CldMyLoad();
extern void CldPickFreeChare();


#define TRACE(x) 



