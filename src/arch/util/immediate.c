/*
  support immediate message in Converse
*/

#if CMK_IMMEDIATE_MSG

/*  push immediate messages into imm queue. Immediate messages can be pushed
    from both processor threads or comm. thread, the messages are stored
    and are processed only by the communication thread.
*/
void CmiPushImmediateMsg(void *msg)
{
  CmiCommLock();
  PCQueuePush(CsvAccess(NodeState).imm, (char *)msg);
  CmiCommUnlock();
}


static int immDone=1;

/* In user's imemdiate handler, if the immediate message cannot be processed
   due to failure to acquire locks, etc, user program can call this function
   to postpone the immediate message. The immediate message will be re-inserted
   into the imm queue.
*/
void CmiDelayImmediate()
{
  immDone = 0;
}

void CmiHandleImmediate()
{
   int qlen, i;
   if (PCQueueEmpty(CsvAccess(NodeState).imm)) return;
   qlen = PCQueueLength(CsvAccess(NodeState).imm);
   {
#ifdef CMK_CPV_IS_SMP
     CmiState cs = CmiGetState();
     int oldRank = cs->rank;
#endif
     for (i=0; i<qlen; i++) {
       void *msg = PCQueuePop(CsvAccess(NodeState).imm);
       if (msg == NULL) break;
#ifdef CMK_CPV_IS_SMP
       cs->rank = CMI_DEST_RANK(msg);    /* switch to the worker thread */
#endif
       immDone = 1;
       CmiHandleMessage(msg);
       if (!immDone) PCQueuePush(CsvAccess(NodeState).imm, msg);
     }
#ifdef CMK_CPV_IS_SMP
     cs->rank = oldRank;
#endif
   }
}

#endif
