/* This file is considered be used inside the machine layer, not to be used separately */
#if CMK_SMP && CMK_LEVERAGE_COMMTHREAD

/** Usage:
 * First: define a function that needs to be invoked on comm thread. E.g.
 *   void CommThdSayHello(int numParams, void *params) {
 *     int src = *(int *)params;
 *     CkPrintf("Notification from source %d\n", src);
 *   }
 * 
 * Second: create a notification msg and push it to comm thread. E.g.
 *   int *argv = (int *)malloc(sizeof(int)*1); *argv = SRCIDX;
 *   CmiNotifyCommThdMsg *one = CmiCreateNotifyCommThdMsg(CommThdSayHello, 1, (void *)(argv), 0);
 *   CmiNotifyCommThd(one);
 * Since we have set "toKeep" 0, the msg itself with the input arguments 
 * will be freed by the comm thread after the msg is processed. Otherwise,
 * the developer has to be responsible for freeing such message.
 */

/* This queue is only created on comm thread, and it's multi-producer-single-consumer */
CsvDeclare(PCQueue, notifyCommThdQ);

CmiNotifyCommThdMsg *CmiCreateNotifyCommThdMsg(CmiCommThdFnPtr fn, int numParams, void *params, int toKeep){
    CmiNotifyCommThdMsg *one = (CmiNotifyCommThdMsg *)malloc(sizeof(CmiNotifyCommThdMsg));
    one->fn = fn;
    one->numParams = numParams;
    one->params = params;
    one->toKeep = toKeep;
}

void CmiFreeNotifyCommThdMsg(CmiNotifyCommThdMsg *msg){
    free(msg->params);
    free(msg);
}

void CmiSetNotifyCommThdMsg(CmiNotifyCommThdMsg *msg, CmiCommThdFnPtr fn, int numParams, void *params, int toKeep){
    msg->fn = fn;
    msg->numParams = numParams;
    msg->params = params;
    msg->toKeep = toKeep;
}

void CmiNotifyCommThd(CmiNotifyCommThdMsg *msg){
    PCQueuePush(CsvAccess(notifyCommThdQ), (char *)msg);
}

void CmiPollCommThdNotificationQ(){
    PCQueue q = CsvAccess(notifyCommThdQ);
    CmiNotifyCommThdMsg *msg = (CmiNotifyCommThdMsg *)PCQueuePop(q);
    
    while(msg){
        /* execute the function indicated by the msg */
        (msg->fn)(msg->numParams, msg->params);
        if(!msg->toKeep) CmiFreeNotifyCommThdMsg(msg);
        msg = (CmiNotifyCommThdMsg *)PCQueuePop(q);
    }
}
#endif
