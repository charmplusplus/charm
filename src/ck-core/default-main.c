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
 * Revision 2.10  1995-09-19 17:56:25  sanjeev
 * moved Charm's module inits from user_main to InitializeCharm
 *
 * Revision 2.9  1995/07/19  22:15:24  jyelon
 * *** empty log message ***
 *
 * Revision 2.8  1995/07/12  20:59:58  brunner
 * Added argv[0] to perfModuleInit() call, so performance data files
 * can use the prgram name in the log file name.
 *
 * Revision 2.7  1995/07/10  22:30:49  brunner
 * Added call to perfModuleInit() for CPV macros
 *
 * Revision 2.6  1995/07/03  17:55:55  gursoy
 * changed charm_main to user_main
 *
 * Revision 2.5  1995/06/18  21:55:06  sanjeev
 * removed loop from charm_main, put in CsdScheduler()
 *
 * Revision 2.4  1995/06/13  17:00:16  jyelon
 * *** empty log message ***
 *
 * Revision 2.3  1995/06/13  14:33:55  gursoy
 * *** empty log message ***
 *
 * Revision 2.2  1995/06/09  16:37:40  gursoy
 * Csv accesses modified
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


CpvExtern(int, numHeapEntries);
CpvExtern(int, numCondChkArryElts);
CpvExtern(int, CsdStopFlag);



user_main(argc, argv)
int argc;
char *argv[];
{
  ConverseInit(argv);

  InitializeCharm(argv) ;
  StartCharm(argv);

  CpvAccess(CsdStopFlag)=0;

  CsdScheduler(-1) ;

  EndCharm();
  ConverseExit() ;
}

