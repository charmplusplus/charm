/*
Converse-level debugger support

Collected from convcore.c, conv-ccs.c, register.c by
Orion Sky Lawlor, olawlor@acm.org, 4/10/2001
 */
#include <stdio.h> /*for sscanf*/
#include <string.h> /*for strcmp*/
#include "converse.h"
#include "conv-trace.h"
#include "queueing.h"
#include "conv-ccs.h"
#include <errno.h>

CpvStaticDeclare(int, freezeModeFlag);
CpvStaticDeclare(int, continueFlag);
CpvStaticDeclare(int, stepFlag);
CpvStaticDeclare(void *, debugQueue);

/***************************************************
  The CCS interface to the debugger
*/

#include <string.h>


extern void CpdContinueFromBreakPoint(void);

static void CpdDebugHandler(char *msg)
{
    char name[128];
    sscanf(msg+CmiMsgHeaderSizeBytes, "%s", name);

    if (strcmp(name, "freeze") == 0) {
      CpdFreeze();
      //CmiPrintf("freeze received\n");
    }
    else if (strcmp(name, "unfreeze") == 0) {
      CpdUnFreeze();
      //CmiPrintf("unfreeze received\n");
    }
    else if (strncmp(name, "step", strlen("step")) == 0){
      CmiPrintf("step received\n");
      CpvAccess(stepFlag) = 1;
      CpdUnFreeze();
    }
    else if (strncmp(name, "continue", strlen("continue")) == 0){
      CmiPrintf("continue received\n");
      CpvAccess(continueFlag) = 1;
      CpdUnFreeze();
    }
#if 0
    else if (strncmp(name, "setBreakPoint", strlen("setBreakPoint")) == 0){
      CmiPrintf("setBreakPoint received\n");
      temp = strstr(name, "#");
      temp++;
      setBreakPoints(temp);
    }
#endif
    else if (strcmp(name, "quit") == 0){
      //CpdUnFreeze();
      CpdContinueFromBreakPoint(); 
      CsdExitScheduler();
    }
    else if (strcmp(name, "startgdb") == 0){
       //start gdb on processor and attach process to it
       CpdContinueFromBreakPoint(); 
       CpdStartGdb();
    }
    else{
      CmiPrintf("bad debugger command:%s received,len=%ld\n",name,strlen(name));
    }
}


/*
 Start the freeze-- call will not return until unfrozen
 via a CCS request.
 */
void CpdFreeze(void)
{
  if (CpvAccess(freezeModeFlag)) return; /*Already frozen*/
  CpvAccess(freezeModeFlag) = 1;
  CpdFreezeModeScheduler();
}

void CpdUnFreeze(void)
{
  CpvAccess(freezeModeFlag) = 0;
}

/* Special scheduler-type loop only executed while in
freeze mode-- only executes CCS requests.
*/
void CcsServerCheck(void);
extern int _isCcsHandlerIdx(int idx);

void CpdFreezeModeScheduler(void)
{
#if CMK_CCS_AVAILABLE
    void *msg;
    void *debugQ=CpvAccess(debugQueue);
    CsdSchedulerState_t state;
    CsdSchedulerState_new(&state);

    /* While frozen, queue up messages */
    while (CpvAccess(freezeModeFlag)) {
#if NODE_0_IS_CONVHOST
      CcsServerCheck(); /*Make sure we can get CCS messages*/
#endif
      msg = CsdNextMessage(&state);

      if (msg!=NULL) {
	  int hIdx=CmiGetHandler(msg);
	  if(_isCcsHandlerIdx(hIdx))
	  /*A CCS request-- handle it immediately*/
          {
            //CmiPrintf("Before handling ccs call in freeze mode\n"); 
	    CmiHandleMessage(msg);
            //CmiPrintf("After handling ccs call in freeze mode\n"); 
          }
	  else
	  /*An ordinary message-- queue it up*/
	    CdsFifo_Enqueue(debugQ, msg);
      } else CmiNotifyIdle();
    }
    /* Before leaving freeze mode, execute the messages 
       in the order they would have executed before.*/
    while (!CdsFifo_Empty(debugQ))
    {
	char *queuedMsg = (char *)CdsFifo_Dequeue(debugQ);
        CmiHandleMessage(queuedMsg);
    }
#endif
}

CpvExtern(char *, displayArgument);

void CpdInit(void)
{
  CpvInitialize(int, freezeModeFlag);
  CpvAccess(freezeModeFlag) = 0;

  CpvInitialize(void *, debugQueue);
  CpvAccess(debugQueue) = CdsFifo_Create();

  CcsRegisterHandler("ccs_debug", (CmiHandler)CpdDebugHandler);
  
#if 0
  CpdInitializeObjectTable();
  CpdInitializeHandlerArray();
  CpdInitializeBreakPoints();

  /* To allow start in freeze state: */
  msgListCleanup();
  msgListCache();
#endif
  
/*  if (CmiGetArgStringDesc(argv, "+DebugDisplay",&displayArgument, "X display for gdb used only in cpd mode"))
  {
    if (!displayArgument)
       CmiPrintf("WARNING> NULL parameter for +DebugDisplay\n***");
  }
  else
  {
    CmiPrintf("WARNING> x term for gdb needs to be specified as +DebugDisplay by debugger\n***");
  }*/
}


/* The following code is for only the net-linux version */
/* Has to be taken care for other versions */

void CpdStartGdb(void)
{
  FILE *f;
  char gdbScript[200];
  int pid;
  if (CpvAccess(displayArgument) != NULL)
  {
     CmiPrintf("MY NODE IS %d  and process id is %d\n", CmiMyPe(), getpid());
     sprintf(gdbScript, "/tmp/cpdstartgdb.%d.%d", getpid(), CmiMyPe());
     f = fopen(gdbScript, "w");
     fprintf(f,"#!/bin/sh\n");
     fprintf(f,"cat > /tmp/start_gdb.$$ << END_OF_SCRIPT\n");
     fprintf(f,"shell /bin/rm -f /tmp/start_gdb.$$\n");
     //fprintf(f,"handle SIGPIPE nostop noprint\n");
     fprintf(f,"handle SIGWINCH nostop noprint\n");
     fprintf(f,"handle SIGWAITING nostop noprint\n");
     fprintf(f, "attach %d\n", getpid());
     fprintf(f,"END_OF_SCRIPT\n");
     fprintf(f, "DISPLAY='%s';export DISPLAY\n",CpvAccess(displayArgument));
     fprintf(f,"/usr/X11R6/bin/xterm ");
     fprintf(f," -title 'Node %d ' ",CmiMyPe());
     fprintf(f," -e /usr/bin/gdb -x /tmp/start_gdb.$$ \n");
     fprintf(f, "exit 0\n");
     fclose(f);
     if( -1 == chmod(gdbScript, 0755))
     {
        CmiPrintf("ERROR> chmod on script failed!\n");
        return;
     }
     pid = fork();
     if (pid < 0)
        { perror("ERROR> forking to run debugger script\n"); exit(1); }
     if (pid == 0)
     {
         //CmiPrintf("In child process to start script %s\n", gdbScript);
         if (-1 == execvp(gdbScript, NULL))
            CmiPrintf ("Error> Could not Execute Debugger Script: %s\n",strerror
(errno));

      }
    }
}















