
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
 * Revision 2.7  1997-10-29 23:52:45  milind
 * Fixed CthInitialize bug on uth machines.
 *
 * Revision 2.6  1995/10/19 18:21:18  jyelon
 * added typecasts to remove warnings.
 *
 * Revision 2.5  1995/10/13  18:15:53  jyelon
 * K&R changes.
 *
 * Revision 2.4  1995/06/19  17:45:27  sanjeev
 * bug in CallBocFn
 *
 * Revision 2.3  1995/06/18  22:10:45  sanjeev
 * Added Ccd
 *
 * Revision 2.2  1995/06/18  21:30:38  sanjeev
 * separated charm and converse condsends
 *
 ***************************************************************************/
static char ident[] = "@(#)$Header$";


#include "converse.h"
#include "const.h"
#include "chare.h"
#include "globals.h"
#include "condsend.h"

CpvStaticDeclare(int, outstanding_sends);



void condsendModuleInit()
{
    CpvInitialize(int, outstanding_sends);
    CpvAccess(outstanding_sends) = 0;
}




/*****************************************************************************
  This function sends out a message using fields that it extracts from its one
  argument
 *****************************************************************************/
static void SendMsgFn(arg)
    void *arg;
{
  SendMsgStuff *sendstruct;

  CpvAccess(outstanding_sends)--;

  sendstruct = (SendMsgStuff *)arg;
  SendMsg(sendstruct->entry, sendstruct->msg, sendstruct->cid);
}

/*****************************************************************************
  This function makes a BOC call using fields that it extracts from its one
  argument
 *****************************************************************************/
static void CallBocFn(arg)
    void *arg;
{
  CallBocStuff *cbocstruct;
  
  cbocstruct = (CallBocStuff *)arg;
  if ( !((*(cbocstruct->fn_ptr))(cbocstruct->bocNum)) )
      CcdPeriodicallyCall((CcdVoidFn)CallBocFn, (void *)arg);

}

/*****************************************************************************
  This function adds a call that will send a message if a particular condition
  is raised
 *****************************************************************************/
/* Function not to be used ..  */
static void SendMsgIfConditionArises(condnum, entry, msgToSend, size, pChareID)
int condnum; int entry; void *msgToSend; 
int size; ChareIDType *pChareID;
{
  SendMsgStuff *newEntry;
  if((newEntry = (SendMsgStuff *) CmiAlloc(sizeof(SendMsgStuff))) == NULL) {
    CkMemError(newEntry);
    return;
  }
  else {
    newEntry->entry     = entry;
    newEntry->msg       = msgToSend;
    newEntry->cid   = pChareID;

    CpvAccess(outstanding_sends)++;
    
    CcdCallOnCondition(condnum, (CcdVoidFn)SendMsgFn, (void *)newEntry);
    }
} 


/*****************************************************************************
  This function adds a call that will make a BOC call if a particular condition
  is raised
 *****************************************************************************/
void CallBocIfConditionArises(condnum, fn_ptr, bocNum)
    int condnum; FUNCTION_PTR fn_ptr; int bocNum;
{
  CallBocStuff *newEntry;
  
  if((newEntry = (CallBocStuff *) CmiAlloc(sizeof(CallBocStuff))) == NULL) 
    {
      CkMemError(newEntry);
      return;
    }
  else {
    newEntry->bocNum   = bocNum;
    newEntry->fn_ptr   = fn_ptr;;
    CcdCallOnCondition(condnum, (CcdVoidFn)CallBocFn, (void *)newEntry);
    }
}

/*****************************************************************************
  This function adds a call that will send a message during a TimerChecks() 
  call after a minimum delay of deltaT
 *****************************************************************************/
void SendMsgAfter(deltaT, entry, msgToSend, size, pChareID)
    unsigned int deltaT; int entry; void *msgToSend;
    int size; ChareIDType *pChareID;
{
  SendMsgStuff *newEntry;

  if((newEntry = (SendMsgStuff *) CmiAlloc(sizeof(SendMsgStuff))) == NULL) 
    {
      CkMemError(newEntry);
      return;
    }
  else 
    {
      newEntry->entry     = entry;
      newEntry->msg       = msgToSend;
      newEntry->cid   = pChareID;        
      
      CpvAccess(outstanding_sends)++;
      
      CcdCallFnAfter((CcdVoidFn)SendMsgFn, (void *)newEntry, deltaT);
    }
} 

/*****************************************************************************
  This function adds a BOC call that will be made during a TimerChecks() call 
  after a minimum delay of deltaT
 *****************************************************************************/
void CallBocAfter(fn_ptr, bocNum, deltaT)
    FUNCTION_PTR fn_ptr; int bocNum; unsigned int deltaT;
{
  CallBocStuff *newEntry;

  if((newEntry = (CallBocStuff *) CmiAlloc(sizeof(CallBocStuff))) == NULL) 
    {
      CkMemError(newEntry);
      return;
    }
  else 
    {
      newEntry->bocNum = bocNum;  
      newEntry->fn_ptr = fn_ptr;  
      CcdCallFnAfter((CcdVoidFn)CallBocFn, (void *)newEntry, deltaT);
    } 
} 

/*****************************************************************************
  In reality, this function adds a BOC call that will be made during each 
  PeriodicChecks() call
 *****************************************************************************/
void CallBocOnCondition(fn_ptr, bocNum)
    FUNCTION_PTR fn_ptr; int bocNum;
{
  CallBocStuff *newEntry;

  if((newEntry = (CallBocStuff *) CmiAlloc(sizeof(CallBocStuff))) == NULL) 
    {
      CkMemError(newEntry);
      return;
    }
  else 
    {
      newEntry->bocNum = bocNum;  
      newEntry->fn_ptr = fn_ptr;  
      CcdPeriodicallyCall((CcdVoidFn)CallBocFn, (void *)newEntry);
    } 
} 

/*****************************************************************************
  Checks the static local variable outstanding_sends to see if all the 
  delayed sends have been indeed sent off 
  ****************************************************************************/
int  NoDelayedMsgs()
{
  if(CpvAccess(outstanding_sends))
    return 0;
  else
    return 1;
}


