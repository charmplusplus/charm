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

/* This msg buffer pool is only created on comm thread, and it's multi-producer-single-consumer */
CsvDeclare(CMIQueue, notifyCommThdMsgBuffer);
CpvDeclare(int, notifyCommThdHdlr);

static void commThdHandleNotification(CmiNotifyCommThdMsg *msg){
    msg->fn(msg->numParams, msg->params);
    if(!msg->toKeep) CmiFreeNotifyCommThdMsg(msg);
}

/* Should be called in ConverseRunPE after ConverseCommonInit */
void CmiInitNotifyCommThdScheme(){
    CpvInitialize(int, notifyCommThdHdlr);
    CpvAccess(notifyCommThdHdlr) = CmiRegisterHandler((CmiHandler)commThdHandleNotification);;
    /* init the msg buffer */
    if(CmiMyRank() == CmiMyNodeSize()){
        int i;
        CsvAccess(notifyCommThdMsgBuffer) = CMIQueueCreate();
        CMIQueue q = CsvAccess(notifyCommThdMsgBuffer);
        /* create init buffer of 16 msgs */
        for(i=0; i<16; i++) {
            CmiNotifyCommThdMsg *one = (CmiNotifyCommThdMsg *)malloc(sizeof(CmiNotifyCommThdMsg));
            CmiSetHandler(one, CpvAccess(notifyCommThdHdlr));
            CmiBecomeImmediate(one);
            CMIQueuePush(q, (char *)one);
        }
    }
    CmiNodeAllBarrier();
}

/* ============ Beginning of implementation for user APIs ============= */
CmiNotifyCommThdMsg *CmiCreateNotifyCommThdMsg(CmiCommThdFnPtr fn, int numParams, void *params, int toKeep){
    CmiNotifyCommThdMsg *one = (CmiNotifyCommThdMsg *)CMIQueuePop(CsvAccess(notifyCommThdMsgBuffer));
    if(one == NULL) {
        one = (CmiNotifyCommThdMsg *)malloc(sizeof(CmiNotifyCommThdMsg));
        CmiSetHandler(one, CpvAccess(notifyCommThdHdlr));
        CmiBecomeImmediate(one);
    }
    one->fn = fn;
    one->numParams = numParams;
    one->params = params;
    one->toKeep = toKeep;
    return one;
}

void CmiFreeNotifyCommThdMsg(CmiNotifyCommThdMsg *msg){
    free(msg->params);
    /* Recycle the msg */
    CMIQueuePush(CsvAccess(notifyCommThdMsgBuffer), (char *)msg);
}

/* Note: the "msg" has to be created from function call "CmiCreateNotifyCommThdMsg" */
void CmiResetNotifyCommThdMsg(CmiNotifyCommThdMsg *msg, CmiCommThdFnPtr fn, int numParams, void *params, int toKeep){
    msg->fn = fn;
    msg->numParams = numParams;
    msg->params = params;
    msg->toKeep = toKeep;
}

void CmiNotifyCommThd(CmiNotifyCommThdMsg *msg){
   CmiPushImmediateMsg(msg);
}
/* ============ End of implementation for user APIs ============= */
#endif
