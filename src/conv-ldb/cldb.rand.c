#include "converse.h"

/* CldEnqueue is the load balancer.  It accepts a message, and
 * enqueues the message on some processor of its own choosing.
 * Processors are chosen in such a way as to help distribute the
 * workload.
 *
 * The user of the load balancer must set aside some space in the
 * message for load balancer to store temporary data.  The number of
 * bytes that must be reserved is CLD_FIELDSIZE.  There is no
 * stipulation about where in the message this field must be.
 *
 * If the message is to be prioritized, then the message must also
 * contain its priority.  This can be omitted if the message is to
 * have the null priority.  There is no stipulation about where in
 * the message the priority must be stored.
 *
 * The user of the load balancer must write an "info" function.
 * The load-balancer will call the info function when it needs to
 * know any of the following pieces of information:
 *
 *    - the length of the message, in bytes.
 *    - the queueing strategy of the message.
 *    - where the priority is, in the message.
 *    - where the ldb field is, in the message.
 *
 * The info function must be able to determine all this information.
 * The info function must be registered, much like a converse handler,
 * using CldRegisterInfoFn.  It must have this prototype:
 *
 *    void Info(void *msg, 
 *              int *len,
 *              void *ldbfield,
 *              int *queuing,
 *              int *priobits,
 *              int **prioptr);
 *
 * The user of the load balancer must also write a "pack" function.
 * When the load balancer is about to send the message across
 * processor boundaries, it will call the pack function.  The
 * pack function may modify the message in any way.  The pack function
 * must be registered using CldRegisterPackFn, and must have this
 * prototype:
 *
 *    void Pack(void **msg);
 *
 * Normally, CldEnqueue is used for load-balancing.  It can also be
 * used as a convenience that helps you enqueue a message on a processor
 * of your choosing.  The parameter 'pe' lets you specify which processor
 * the message is to go to.  If the value CLD_ANYWHERE is given, then
 * the message is load-balanced.  If it is CLD_BROADCAST, the message
 * is broadcast to all other processors.  If it is CLD_BROADCAST_ALL,
 * the message is broadcast to all processors.  If it is a processor
 * number, the message is sent to a specific location.
 *
 */

void CldHandler(char *msg)
{
  int len, queueing, priobits;
  unsigned int *prioptr; CldInfoFn ifn;
  CmiGrabBuffer((void **)&msg);
  CldRestoreHandler(msg);
  ifn = (CldInfoFn)CmiHandlerToFunction(CmiGetInfo(msg));
  ifn(msg, &len, &queueing, &priobits, &prioptr);
  CsdEnqueueGeneral(msg, queueing, priobits, prioptr);
}

CpvDeclare(int, CldHandlerIndex);

void CldEnqueue(int pe, void *msg, int infofn, int packfn)
{
  int len, queueing, priobits; unsigned int *prioptr;
  CldInfoFn ifn = (CldInfoFn)CmiHandlerToFunction(infofn);
  CldPackFn pfn = (CldPackFn)CmiHandlerToFunction(packfn);
  if (CmiGetHandler(msg) >= CpvAccess(CmiHandlerMax)) *((int*)0)=0;
  if (pe == CLD_ANYWHERE) pe = (((rand()+CmiMyPe())&0x7FFFFFFF)%CmiNumPes());
  if (pe == CmiMyPe()) {
    ifn(msg, &len, &queueing, &priobits, &prioptr);
    CsdEnqueueGeneral(msg, queueing, priobits, prioptr);
  } else {
    pfn(&msg);
    ifn(msg, &len, &queueing, &priobits, &prioptr);
    CldSwitchHandler(msg, CpvAccess(CldHandlerIndex));
    CmiSetInfo(msg,infofn);
    if (pe==CLD_BROADCAST) CmiSyncBroadcastAndFree(len, msg);
    else if (pe==CLD_BROADCAST_ALL) CmiSyncBroadcastAllAndFree(len, msg);
    else CmiSyncSendAndFree(pe, len, msg);
  }
}

void CldModuleInit()
{
  CpvInitialize(int, CldHandlerIndex);
  CpvAccess(CldHandlerIndex) = CmiRegisterHandler(CldHandler);
  CldModuleGeneralInit();
}
