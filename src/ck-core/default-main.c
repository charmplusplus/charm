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
 * Revision 2.0  1995-06-02 17:27:40  brunner
 * Reorganized directory structure
 *
 * Revision 1.4  1995/04/13  20:54:18  sanjeev
 * Changed Mc to Cmi
 *
 * Revision 1.3  1995/04/02  00:48:53  sanjeev
 * changes for separating Converse
 *
 * Revision 1.2  1995/03/17  23:38:21  sanjeev
 * changes for better message format
 *
 * Revision 1.1  1994/11/18  20:38:11  narain
 * Initial revision
 *
 ***************************************************************************/
static char ident[] = "@(#)$Header$";

#include "converse.h"

int          disable_sys_msgs=0;
extern int   numHeapEntries;
extern int   numCondChkArryElts;
extern int   CsdStopFlag;

main(argc, argv)
int argc;
char *argv[];
{
  ConverseInit(argv);
  StartCharm(argv);
  CsdStopFlag=0;
  while (1) {
    void *msg;
    msg = CsdGetMsg();
    if (msg) (CmiGetHandlerFunction(msg))(msg);
    if (CsdStopFlag) break;
    if (!disable_sys_msgs)
        { if (numHeapEntries > 0) TimerChecks();
          if (numCondChkArryElts > 0) PeriodicChecks(); }
  }
  EndCharm();
  ConverseExit() ;
}

