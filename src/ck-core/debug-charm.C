/*****************************************************************************
 * A few useful built-in CPD handlers.
 *****************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>

#include "converse.h"
#include "ckhashtable.h"
#include "conv-ccs.h"
#include "sockRoutines.h"
#include "queueing.h"

#if CMK_CCS_AVAILABLE


#include "charm.h"
#include "middle.h"
#include "cklists.h"
#include "register.h"

typedef CkHashtableTslow<int,EntryInfo *> CpdBpFuncTable_t;


extern void CpdFreeze(void);
extern void CpdUnFreeze(void);
extern int CkMessageToEpIdx(void *msg);


CpvStaticDeclare(int, _debugMsg);
CpvStaticDeclare(int, _debugChare);

CpvStaticDeclare(CpdBpFuncTable_t *, breakPointEntryTable);

CpvStaticDeclare(void *, lastBreakPointMsg);
CpvStaticDeclare(void *, lastBreakPointObject);
CpvStaticDeclare(int, lastBreakPointIndex);

void CpdBreakPointInit()
{
  CpvInitialize(void *, lastBreakPointMsg);
  CpvInitialize(void *, lastBreakPointObject);
  CpvInitialize(int, lastBreakPointIndex);
  CpvInitialize(int, _debugMsg);
  CpvInitialize(int, _debugChare);
  CpvInitialize(CpdBpFuncTable_t *, breakPointEntryTable);
  CpvAccess(lastBreakPointMsg) = NULL;
  CpvAccess(lastBreakPointObject) = NULL;
  CpvAccess(lastBreakPointIndex) = 0;
  CpvAccess(_debugMsg) = CkRegisterMsg("debug_msg",0,0,0);
  CpvAccess(_debugChare) = CkRegisterChare("debug_Chare",0);
  CpvAccess(breakPointEntryTable) = new CpdBpFuncTable_t(10,0.5,CkHashFunction_int,CkHashCompare_int );
}



static void _call_freeze_on_break_point(void * msg, void * object)
{
      //Save breakpoint entry point index. This is retrieved from msg.
      //So that the appropriate EntryInfo can be later retrieved from the hash table 
      //of break point function entries, on continue.
      CpvAccess(lastBreakPointMsg) = msg;
      CpvAccess(lastBreakPointObject) = object;
      CpvAccess(lastBreakPointIndex) = CkMessageToEpIdx(msg);
      EntryInfo * breakPointEntryInfo = CpvAccess(breakPointEntryTable)->get(CpvAccess(lastBreakPointIndex));
      CmiPrintf("Break point reached for Function = %s\n", breakPointEntryInfo->name);
      CpdFreeze();
}



//ccs handler when continue from a break point
extern "C"
void CpdContinueFromBreakPoint ()
{
    CpdUnFreeze();
    if ( (CpvAccess(lastBreakPointMsg) != NULL) && (CpvAccess(lastBreakPointObject) != NULL) )
    {
        EntryInfo * breakPointEntryInfo = CpvAccess(breakPointEntryTable)->get(CpvAccess(lastBreakPointIndex));
        if (breakPointEntryInfo != NULL)
           breakPointEntryInfo->call(CpvAccess(lastBreakPointMsg), CpvAccess(lastBreakPointObject));
    }
    CpvAccess(lastBreakPointMsg) = NULL;
    CpvAccess(lastBreakPointObject) = NULL;
}

//ccs handler to set a breakpoint with entry function name msg
void CpdSetBreakPoint (char *msg)
{
  char functionName[128];
  int tableSize, tableIdx = 0;
  sscanf(msg+CmiMsgHeaderSizeBytes, "%s", functionName);
  if (strlen(functionName) > 0)
  {
    tableSize = _entryTable.size();
    // Replace entry in entry table with _call_freeze_on_break_point
    // retrieve epIdx for entry method
    for (tableIdx=0; tableIdx < tableSize; tableIdx++)
    {
       if (strstr(_entryTable[tableIdx]->name, functionName) != NULL)
       {
            EntryInfo * breakPointEntryInfo = new EntryInfo(_entryTable[tableIdx]->name, _entryTable[tableIdx]->call, _entryTable[tableIdx]->msgIdx, _entryTable[tableIdx]->chareIdx );
           CmiPrintf("Breakpoint is set for function %s with an epIdx = %ld\n", _entryTable[tableIdx]->name, tableIdx);
           CpvAccess(breakPointEntryTable)->put(tableIdx) = breakPointEntryInfo;  
           _entryTable[tableIdx]->name = "debug_breakpoint_ep";  
           _entryTable[tableIdx]->call = (CkCallFnPtr)_call_freeze_on_break_point;
           _entryTable[tableIdx]->msgIdx = CpvAccess(_debugMsg); 
           _entryTable[tableIdx]->chareIdx = CpvAccess(_debugChare);
           break;
       }
    }
    if (tableIdx == tableSize)
    {
      CmiPrintf("[ERROR]Entrypoint was not found for function %s\n", functionName); 
      return;
    }

  }

}

void CpdQuitDebug()
{
  CpdContinueFromBreakPoint();
  CkExit();
}

void CpdRemoveBreakPoint (char *msg)
{
  char functionName[128];
  sscanf(msg+CmiMsgHeaderSizeBytes, "%s", functionName);
  void *objPointer;
  void *keyPointer; 
  CkHashtableIterator *it = CpvAccess(breakPointEntryTable)->iterator();
  while(NULL!=(objPointer = it->next(&keyPointer)))
  {
    EntryInfo * breakPointEntryInfo = *(EntryInfo **)objPointer;
    int idx = *(int *)keyPointer;
    if (strstr(breakPointEntryInfo->name, functionName) != NULL){
        _entryTable[idx]->name =  breakPointEntryInfo->name;
        _entryTable[idx]->call = (CkCallFnPtr)breakPointEntryInfo->call;
        _entryTable[idx]->msgIdx = breakPointEntryInfo->msgIdx;
        _entryTable[idx]->chareIdx = breakPointEntryInfo->chareIdx;
        CmiPrintf("Breakpoint is removed for function %s with epIdx %ld\n", _entryTable[idx]->name, idx);
    }
  }
}

void CpdRemoveAllBreakPoints ()
{
  //all breakpoints removed
  void *objPointer;
  void *keyPointer; 
  CkHashtableIterator *it = CpvAccess(breakPointEntryTable)->iterator();
  while(NULL!=(objPointer = it->next(&keyPointer)))
  {
    EntryInfo * breakPointEntryInfo = *(EntryInfo **)objPointer;
    int idx = *(int *)keyPointer;
    _entryTable[idx]->name =  breakPointEntryInfo->name;
    _entryTable[idx]->call = (CkCallFnPtr)breakPointEntryInfo->call;
    _entryTable[idx]->msgIdx = breakPointEntryInfo->msgIdx;
    _entryTable[idx]->chareIdx = breakPointEntryInfo->chareIdx;
  }
}

CpvExtern(char *, displayArgument);

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



void CpdCharmInit()
{
  CpdBreakPointInit();
  CcsRegisterHandler("ccs_set_break_point",(CmiHandler)CpdSetBreakPoint);
  CcsRegisterHandler("ccs_remove_break_point",(CmiHandler)CpdRemoveBreakPoint);
  CcsRegisterHandler("ccs_remove_all_break_points",(CmiHandler)CpdRemoveAllBreakPoints);
  CcsRegisterHandler("ccs_continue_break_point",(CmiHandler)CpdContinueFromBreakPoint);
  CcsRegisterHandler("ccs_debug_quit",(CmiHandler)CpdQuitDebug);
  CcsRegisterHandler("ccs_debug_startgdb",(CmiHandler)CpdStartGdb);

}


#endif /*CMK_CCS_AVAILABLE*/












