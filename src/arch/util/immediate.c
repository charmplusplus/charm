/** @file
 * \brief Immediate message implementation
 * \ingroup Machine
 */

/**
 * \addtogroup Machine
*/
/*@{*/

/*
  support immediate message in Converse
*/

int _immediateReady = 0;

int _immRunning=0; /* if set, somebody's inside an immediate message */

/* _immediateLock and _immediateFlag declared in convcore.c 
   for machine layers with CMK_IMMEDIATE_MSG=0   */ 

#if CMK_IMMEDIATE_MSG

/* SMP: These variables are protected by immRecvLock. */
static void *currentImmediateMsg=NULL; /* immediate message currently being executed */

/*  push immediate messages into imm queue. Immediate messages can be pushed
    from both processor threads or comm. thread.
    
    The actual user handler is in Xhandler; the converse handler
    is still marked as the immediate message "handler".

SMP:  This routine does its own locking.
*/
void CmiPushImmediateMsg(void *msg)
{
  MACHSTATE(4,"pushing immediate message {");
  /* This lock check needs portable access to comm_flag, which is tough:
     MACHLOCK_ASSERT(_immRunning||comm_flag,"CmiPushImmediateMsg");
  */
  
  CmiLock(CsvAccess(NodeState).immSendLock);
  CMIQueuePush(CsvAccess(NodeState).immQ, (char *)msg);
  CmiUnlock(CsvAccess(NodeState).immSendLock);
  MACHSTATE(4,"} pushing immediate message");
}

/* In user's immediate handler, if the immediate message cannot be processed
   due to failure to acquire locks, etc, user program can call this function
   to postpone the immediate message. The immediate message will eventually
   be re-inserted into the imm queue.

SMP: This routine must be called holding immRecvLock
*/
void CmiDelayImmediate()
{
  MACHLOCK_ASSERT(_immRunning,"CmiDelayImmediate");

  CQdCreate(CpvAccess(cQdState),1);
  MACHSTATE(5,"Actually delaying an immediate message");
  CMIQueuePush(CsvAccess(NodeState).delayedImmQ, (char *)currentImmediateMsg);
}


/*
  Handle an immediate message, using the handler table of
  processor rank 0.  We can't call CmiHandleMessage for 
  immediate messages, because of CpvAccess and tracing.

SMP: This routine must be called holding immRecvLock
 */
void CmiHandleImmediateMessage(void *msg) {
/*  int handlerNo=CmiGetXHandler(msg); */
  int handlerNo=CmiImmediateHandler(msg);
  CmiHandlerInfo *h;
#if (defined(_FAULT_MLOG_) || defined(_FAULT_CAUSAL_))
        CmiAssert(0);
#endif
  MACHSTATE2(4,"immediate message handler %d %d", CmiGetHandler(msg), handlerNo)
/*  CmiHandlerInfo *h=&CpvAccessOther(CmiHandlerTable,0)[handlerNo]; */
  h = &CpvAccess(CmiHandlerTable)[handlerNo];
  CmiAssert(h && h->hdlr);

  MACHLOCK_ASSERT(_immRunning,"CmiHandleImmediateMessage");
  CQdProcess(CpvAccess(cQdState),1);
  (h->hdlr)(msg,h->userPtr);
}

/*
   Check for queued immediate messages and handle them.
   
   It is safe to call this routine from multiple threads, or from SIGIO.

SMP: This routine must be called holding no locks, not even the comm. lock.
 */
void CmiHandleImmediate()
{
   void *msg;

   /* converse init hasn't finish */
   if (!_immediateReady) return;
  
   /* If somebody else is checking the queue, we don't need to */
   if (CmiTryLock(CsvAccess(NodeState).immRecvLock)!=0) return;

   /* Make sure only one thread is running immediate messages:
   CmiLock(CsvAccess(NodeState).immRecvLock);
   */

   MACHLOCK_ASSERT(!_immRunning,"CmiHandleImmediate");
   _immRunning=1; /* prevents SIGIO reentrancy, and allows different send */
   MACHSTATE(2,"Entered handleImmediate {")

   /* Handle all pending immediate messages */
   while (NULL!=(msg=CMIQueuePop(CsvAccess(NodeState).immQ)))
   {
     currentImmediateMsg = msg;
     MACHSTATE(4,"calling immediate message handler {");
     CmiHandleImmediateMessage(msg);
     MACHSTATE(4,"} calling immediate message handler");
   }
   
   /* Take care of delayed immediate messages, which we have to handle next time */
   while (NULL!=(msg=CMIQueuePop(CsvAccess(NodeState).delayedImmQ)))
   	CmiPushImmediateMsg(msg);
   
   MACHSTATE(2,"} exiting handleImmediate")
   _immRunning = 0;
   
   CmiUnlock(CsvAccess(NodeState).immRecvLock);

   CmiClearImmediateFlag();
}

#endif

/*@}*/
