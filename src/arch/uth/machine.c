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
 * Revision 1.35  1998-07-08 22:44:38  jyelon
 * Was storing argv in global var, not good, since stack-allocated.
 *
 * Revision 1.34  1998/06/29 16:51:31  milind
 * Fixed the argv passing to all the processors as only processor 0 was
 * processing and removing the command-line arguments.
 *
 * Revision 1.33  1998/02/13 23:55:30  pramacha
 * Removed CmiAlloc, CmiFree and CmiSize
 * Added CmiAbort
 *
 * Revision 1.32  1997/12/10 21:59:23  jyelon
 * Modified CmiDeliverSpecificMsg so that it works with uth version.
 *
 * Revision 1.31  1997/12/10 21:01:28  jyelon
 * *** empty log message ***
 *
 * Revision 1.30  1997/10/10 18:26:23  jyelon
 * I have no idea what i changed.
 *
 * Revision 1.29  1997/10/03 19:51:52  milind
 * Made charmc to work again, after inserting trace calls in converse part,
 * i.e. threads and user events.
 *
 * Revision 1.28  1997/08/04 09:50:42  jyelon
 * *** empty log message ***
 *
 * Revision 1.27  1997/07/30 19:58:09  jyelon
 * *** empty log message ***
 *
 * Revision 1.26  1997/07/30 17:31:13  jyelon
 * *** empty log message ***
 *
 * Revision 1.25  1997/07/29 16:09:47  milind
 * Added CmiNodeLock macros and functions to the machine layer for all except
 * solaris SMP.
 *
 * Revision 1.24  1997/04/25 20:48:12  jyelon
 * Corrected CmiNotifyIdle
 *
 * Revision 1.23  1997/04/24 22:37:02  jyelon
 * Added CmiNotifyIdle
 *
 * Revision 1.22  1997/03/21 19:23:57  milind
 * removed the alignment bug in Common.uth/machine.c
 *
 * Revision 1.21  1997/03/19 04:31:36  jyelon
 * Redesigned ConverseInit
 *
 * Revision 1.20  1997/02/13 09:31:39  jyelon
 * Updated for new main/ConverseInit structure.
 *
 * Revision 1.19  1997/01/17 15:49:57  jyelon
 * Minor adjustments to deal with recent changes to Common code.
 *
 * Revision 1.18  1996/11/20 06:46:54  jyelon
 * Repaired rob's HP/C++ mods.
 *
 * Revision 1.17  1996/11/08 22:22:53  brunner
 * Put _main in for HP-UX CC compilation.  It is ignored according to the
 * CMK_USE_HP_MAIN_FIX flag.
 *
 * Revision 1.16  1996/07/19 17:07:37  jyelon
 * *** empty log message ***
 *
 * Revision 1.15  1996/07/15 20:59:22  jyelon
 * Moved much timer, signal, etc code into common.
 *
 * Revision 1.14  1996/07/02 21:25:22  jyelon
 * *** empty log message ***
 *
 * Revision 1.13  1996/02/10 18:57:29  sanjeev
 * fixed bugs in CmiGetNodeNeighbours, CmiNeighboursIndex
 *
 * Revision 1.12  1996/02/10 18:54:24  sanjeev
 * fixed bug in CmiNumNeighbours
 *
 * Revision 1.11  1995/11/07 23:20:32  jyelon
 * removed neighbour_init residue.
 *
 * Revision 1.10  1995/11/07  18:24:53  jyelon
 * Corrected a bug in GetNodeNeighbours
 *
 * Revision 1.9  1995/11/07  18:16:45  jyelon
 * Corrected 'neighbour' functions (they now make a hypercube).
 *
 * Revision 1.8  1995/10/27  21:45:35  jyelon
 * Changed CmiNumPe --> CmiNumPes
 *
 * Revision 1.7  1995/10/18  22:22:53  jyelon
 * I forget.
 *
 * Revision 1.6  1995/10/13  22:36:29  jyelon
 * changed exit() --> exit(1)
 *
 * Revision 1.5  1995/10/13  22:34:42  jyelon
 * added CmiNext to CmiCallMain.
 *
 * Revision 1.4  1995/10/13  20:05:13  jyelon
 * *** empty log message ***
 *
 * Revision 1.3  1995/10/10  06:10:58  jyelon
 * removed program_name
 *
 * Revision 1.2  1995/09/30  15:44:59  jyelon
 * fixed a bug.
 *
 * Revision 1.1  1995/09/30  15:00:00  jyelon
 * Initial revision
 *
 * Revision 2.5  1995/09/29  09:50:07  jyelon
 * CmiGet-->CmiDeliver, added protos, etc.
 *
 * Revision 2.4  1995/09/20  15:56:29  gursoy
 * made the arg of CmiFree and CmiSize void*
 *
 * Revision 2.3  1995/09/07  22:33:07  gursoy
 * now the processor specific variables in machine files also accessed thru macros (because macros modifies the var names)
 *
 * Revision 2.2  1995/07/26  19:04:11  gursoy
 * fixed some timer-system-include-file related problems
 *
 * Revision 2.1  1995/07/11  16:53:57  gursoy
 * added CsdStopCount
 *
 * Revision 2.0  1995/07/05  23:37:59  gursoy
 * *** empty log message ***
 *
 *
 *
 ***************************************************************************/
static char ident[] = "@(#)$Header$";

#include <stdio.h>
#include <math.h>
#include "converse.h"

static char *DeleteArg(argv)
  char **argv;
{
  char *res = argv[0];
  if (res==0) { CmiError("Bad arglist."); exit(1); }
  while (*argv) { argv[0]=argv[1]; argv++; }
  return res;
}

int CountArgs(argv)
  char **argv;
{
  int n = 0;
  while (*argv) { n++; argv++; }
  return n;
}

/***********************************************************************
 *
 * Abort function:
 *
 ************************************************************************/

void CmiAbort(char *message)
{
  CmiError(message);
  exit(1);
}

/*****************************************************************************
 *
 * Module variables
 * 
 ****************************************************************************/

typedef void *Fifo;

int        Cmi_mype;
int        Cmi_myrank;
int        Cmi_numpes;
int        Cmi_nodesize;
int        Cmi_stacksize = 64000;
char     **CmiArgv;
CmiStartFn CmiStart;
int        CmiUsched;
CthThread *CmiThreads;
Fifo      *CmiQueues;
int       *CmiBarred;
int        CmiNumBarred=0;

Fifo FIFO_Create();
CpvDeclare(Fifo, CmiLocalQueue);

/******************************************************************************
 *
 * Load-Balancer needs
 *
 * These neighbour functions impose a (possibly incomplete)
 * hypercube on the machine.
 *
 *****************************************************************************/


long CmiNumNeighbours(node)
int node;
{
  int bit, count=0;
  bit = 1;
  while (1) {
    int neighbour = node ^ bit;
    if (neighbour < CmiNumPes()) count++;
    bit = bit<<1; 
    if (bit > CmiNumPes()) break;
  }
  return count;
}

int CmiGetNodeNeighbours(node, neighbours)
int node, *neighbours;
{
  int bit, count=0;
  bit = 1;
  while (1) {
    int neighbour = node ^ bit;
    if (neighbour < CmiNumPes()) neighbours[count++] = neighbour;
    bit = bit<<1; 
    if (bit > CmiNumPes()) break;
  }
  return count;
}
 
int CmiNeighboursIndex(node, nbr)
int node, nbr;
{
  int bit, count=0;
  bit = 1;
  while (1) {
    int neighbour = node ^ bit;
    if (neighbour < CmiNumPes()) { if (nbr==neighbour) return count; count++; }
    bit = bit<<=1; 
    if (bit > CmiNumPes()) break;
  }
  return(-1);
}


/*****************************************************************************
 *
 * Comm handles are nonexistent in uth version
 *
 *****************************************************************************/

int CmiAsyncMsgSent(c)
CmiCommHandle c ;
{
  return 1;
}

void CmiReleaseCommHandle(c)
CmiCommHandle c ;
{
}

/********************* CONTEXT-SWITCHING FUNCTIONS ******************/

static void CmiNext()
{
  CthThread t; int index; int orig;
  index = (CmiMyPe()+1) % CmiNumPes();
  orig = index;
  while (1) {
    t = CmiThreads[index];
    if ((t)&&(!CmiBarred[index])) break;
    index = (index+1) % CmiNumPes();
    if (index == orig) exit(0);
  }
  Cmi_mype = index;
  CthResume(t);
}

void CmiExit()
{
  CmiThreads[CmiMyPe()] = 0;
  CmiFree(CthSelf());
  CmiNext();
}

void *CmiGetNonLocal()
{
  CmiThreads[CmiMyPe()] = CthSelf();
  CmiNext();
  return 0;
}

void CmiNotifyIdle()
{
  CmiThreads[CmiMyPe()] = CthSelf();
  CmiNext();
}

void CmiNodeBarrier()
{
  int i;
  CmiNumBarred++;
  CmiBarred[CmiMyPe()] = 1;
  if (CmiNumBarred == CmiNumPes()) {
    for (i=0; i<CmiNumPes(); i++) CmiBarred[i]=0;
    CmiNumBarred=0;
  }
  CmiGetNonLocal();
}

CmiNodeLock CmiCreateLock()
{
  return (CmiNodeLock)malloc(sizeof(int));
}

void CmiLock(CmiNodeLock lk)
{
  while (*lk) CmiGetNonLocal();
  *lk = 1;
}

void CmiUnlock(CmiNodeLock lk)
{
  if (*lk==0) {
    CmiError("CmiNodeLock not locked, can't unlock.");
    exit(1);
  }
  *lk = 0;
}

int CmiTryLock(CmiNodeLock lk)
{
  if (*lk==0) { *lk=1; return 0; }
  return -1;
}

void CmiDestroyLock(CmiNodeLock lk)
{
  free(lk);
}



/********************* MESSAGE SEND FUNCTIONS ******************/

void CmiSyncSendFn(destPE, size, msg)
int destPE;
int size;
char * msg;
{
  char *buf = (char *)CmiAlloc(size);
  memcpy(buf,msg,size);
  FIFO_EnQueue(CmiQueues[destPE],buf);
}

CmiCommHandle CmiAsyncSendFn(destPE, size, msg) 
int destPE;
int size;
char * msg;
{
  char *buf = (char *)CmiAlloc(size);
  memcpy(buf,msg,size);
  FIFO_EnQueue(CmiQueues[destPE],buf);
}

void CmiFreeSendFn(destPE, size, msg)
int destPE;
int size;
char * msg;
{
  FIFO_EnQueue(CmiQueues[destPE], msg);
}

void CmiSyncBroadcastFn(size, msg)
int size;
char * msg;
{
  int i;
  for(i=0; i<CmiNumPes(); i++)
    if (i != CmiMyPe()) CmiSyncSendFn(i,size,msg);
}

CmiCommHandle CmiAsyncBroadcastFn(size, msg)
int size;
char * msg;
{
  CmiSyncBroadcastFn(size, msg);
  return 0;
}

void CmiFreeBroadcastFn(size, msg)
int size;
char * msg;
{
  CmiSyncBroadcastFn(size, msg);
  CmiFree(msg);
}

void CmiSyncBroadcastAllFn(size, msg)
int size;
char * msg;
{
  int i;
  for(i=0; i<CmiNumPes(); i++)
    CmiSyncSendFn(i,size,msg);
}

CmiCommHandle CmiAsyncBroadcastAllFn(size, msg)
int size;
char * msg;
{
  CmiSyncBroadcastAllFn(size,msg);
  return 0 ;
}

void CmiFreeBroadcastAllFn(size, msg)
int size;
char * msg;
{
  int i;
  for(i=0; i<CmiNumPes(); i++)
    if (i!=CmiMyPe()) CmiSyncSendFn(i,size,msg);
  FIFO_EnQueue(CpvAccess(CmiLocalQueue),msg);
}



/************************** SETUP ***********************************/

static void CmiParseArgs(argv)
char **argv;
{
  char **argp;
  
  for (argp=argv; *argp; ) {
    if ((strcmp(*argp,"++stacksize")==0)&&(argp[1])) {
      DeleteArg(argp);
      Cmi_stacksize = atoi(*argp);
      DeleteArg(argp);
    } else if ((strcmp(*argp,"+p")==0)&&(argp[1])) {
      Cmi_numpes = atoi(argp[1]);
      argp+=2;
    } else if (sscanf(*argp, "+p%d", &CmiNumPes()) == 1) {
      argp+=1;
    } else argp++;
  }
  
  if (CmiNumPes()<1) {
    printf("Error: must specify number of processors to simulate with +pXXX\n",CmiNumPes());
    exit(1);
  }
}

static char **CopyArgvec(char **src)
{
  int argc; char **argv;
  for (argc=0; src[argc]; argc++);
  argv = (char **)malloc((argc+1)*sizeof(char *));
  memcpy(argv, src, (argc+1)*sizeof(char *));
  return argv;
}

char **CmiInitPE()
{
  int argc; char **argv;
  argv = CopyArgvec(CmiArgv);
  CpvAccess(CmiLocalQueue) = CmiQueues[CmiMyPe()];
  CmiSpanTreeInit();
  CmiTimerInit();
  ConverseCommonInit(argv);
  return argv;
}

void CmiCallMain()
{
  char **argv;
  int argc;
  argv = CmiInitPE();
  for (argc=0; argv[argc]; argc++);
  CmiStart(argc, argv);
  if (CmiUsched==0) CsdScheduler(-1);
  ConverseExit();
}

void ConverseExit()
{
  ConverseCommonExit();
  CmiThreads[CmiMyPe()] = 0;
  CmiNext();
}

void ConverseInit(argc,argv,fn,usched,initret)
int argc;
char **argv;
CmiStartFn fn;
int usched, initret;
{
  CthThread t; int stacksize, i;
  
#if CMK_USE_HP_MAIN_FIX
#if FOR_CPLUS
  _main(argc,argv);
#endif
#endif
  
  CmiArgv = CopyArgvec(argv);
  CmiStart = fn;
  CmiUsched = usched;
  CmiParseArgs(argv);
  CthInit(argv);
  CpvInitialize(void*, CmiLocalQueue);
  CmiThreads = (CthThread *)CmiAlloc(CmiNumPes()*sizeof(CthThread));
  CmiBarred  = (int       *)CmiAlloc(CmiNumPes()*sizeof(int));
  CmiQueues  = (Fifo      *)CmiAlloc(CmiNumPes()*sizeof(Fifo));
  
  /* Create threads for all PE except PE 0 */
  for(i=0; i<CmiNumPes(); i++) {
    t = (i==0) ? CthSelf() : CthCreate(CmiCallMain, 0, Cmi_stacksize);
    CmiThreads[i] = t;
    CmiBarred[i] = 0;
    CmiQueues[i] = FIFO_Create();
  }
  Cmi_mype = 0;
  argv = CmiInitPE();
  if (initret==0) {
    fn(CountArgs(argv), argv);
    if (usched==0) CsdScheduler(-1);
    ConverseExit();
  }
}

