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
 * Revision 2.2  1995-06-09 16:37:40  gursoy
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

CpvDeclare(int, disable_sys_msgs);

CpvExtern(int, numHeapEntries);
CpvExtern(int, numCondChkArryElts);
CpvExtern(int, CsdStopFlag);


void defaultmainModuleInit()
{
   CpvInitialize(int, disable_sys_msgs);
   CpvAccess(disable_sys_msgs) = 0; 
}

main(argc, argv)
int argc;
char *argv[];
{
  if (CmiMyRank() != 0) CmiNodeBarrier();

  defaultmainModuleInit();
  bocModuleInit();
  ckModuleInit();
  condsendModuleInit();
  globalsModuleInit();
  initModuleInit();
  mainModuleInit();
  quiesModuleInit();
  registerModuleInit();
  statModuleInit();
  tblModuleInit(); 
  ldbModuleInit();


  if (CmiMyRank() == 0) CmiNodeBarrier();

  ConverseInit(argv);
  StartCharm(argv);
  CpvAccess(CsdStopFlag)=0;
  while (1) {
    void *msg;
    msg = CsdGetMsg();
    if (msg) (CmiGetHandlerFunction(msg))(msg);
    if (CpvAccess(CsdStopFlag)) break;
    if (!CpvAccess(disable_sys_msgs))
        { if (CpvAccess(numHeapEntries) > 0) TimerChecks();
          if (CpvAccess(numCondChkArryElts) > 0) PeriodicChecks(); }
  }
  EndCharm();
  ConverseExit() ;
}

