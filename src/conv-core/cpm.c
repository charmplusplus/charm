#include <stdlib.h>
#include <string.h>
#include "queueing.h"
#include <converse.h>

static void CpmLSend(int pe, int len, void *msg)
{
  if (pe==CpmALL) CmiSyncBroadcastAllAndFree(len, msg);
  else if (pe==CpmOTHERS) CmiSyncBroadcastAndFree(len, msg);
  else CmiSyncSendAndFree(pe,len,msg);
}

/******************************************************************************
 *
 * Control for simple-send
 *
 *****************************************************************************/

typedef struct CpmDestinationSend
{ 
  void *(*sendfn)();
  int envsize;
  int pe;
}
*CpmDestinationSend;

typedef struct CpmDestinationSend DestinationSend;

void CpmSend1(CpmDestinationSend ctrl, int len, void *msg)
{
  CpmLSend(ctrl->pe, len, msg);
}

CpvStaticDeclare(DestinationSend, ctrlSend);

CpmDestination CpmSend(int pe)
{
  CpvAccess(ctrlSend).envsize = 0;
  CpvAccess(ctrlSend).sendfn = (CpmSender)CpmSend1;
  CpvAccess(ctrlSend).pe = pe;
  return (CpmDestination)&CpvAccess(ctrlSend);
}

/******************************************************************************
 *
 * Control for CpmEnqueue
 *
 *****************************************************************************/

typedef struct CpmDestinationEnq
{
  void *(*sendfn)();
  int envsize;
  int pe, qs, priobits;
  unsigned int *prioptr;
}
*CpmDestinationEnq;

typedef struct CpmDestinationEnq DestinationEnq;

CpvStaticDeclare(DestinationEnq, ctrlEnq);

CpvDeclare(int, CpmEnqueue2_Index);

void CpmEnqueue2(void *msg)
{
  int *env;
  env = (int *)CpmEnv(msg);
  CmiSetHandler(msg, env[0]);
  CsdEnqueueGeneral(msg, env[1], env[2], (unsigned int *)(env+3));
}

void *CpmEnqueue1(CpmDestinationEnq ctrl, int len, void *msg)
{
  int *env = (int *)CpmEnv(msg);
  int intbits = sizeof(int)*8;
  int priobits = ctrl->priobits;
  int prioints = (priobits+intbits-1) / intbits;
  env[0] = CmiGetHandler(msg);
  env[1] = ctrl->qs;
  env[2] = ctrl->priobits;
  memcpy(env+3, ctrl->prioptr, prioints*sizeof(int));
  CmiSetHandler(msg, CpvAccess(CpmEnqueue2_Index));
  CpmLSend(ctrl->pe, len, msg);
  return (void *) 0;
}

CpmDestination CpmEnqueue(int pe, int qs, int priobits,unsigned int *prioptr)
{
  int intbits = sizeof(int)*8;
  int prioints = (priobits+intbits-1) / intbits;
  CpvAccess(ctrlEnq).envsize = (3+prioints)*sizeof(int);
  CpvAccess(ctrlEnq).sendfn  = CpmEnqueue1;
  CpvAccess(ctrlEnq).pe = pe; CpvAccess(ctrlEnq).qs = qs; 
  CpvAccess(ctrlEnq).priobits = priobits; CpvAccess(ctrlEnq).prioptr = prioptr;
  return (CpmDestination)&CpvAccess(ctrlEnq);
}

CpvStaticDeclare(unsigned int, fiprio);

CpmDestination CpmEnqueueIFIFO(int pe, int prio)
{
  CpvAccess(fiprio) = prio;
  return CpmEnqueue(pe, CQS_QUEUEING_IFIFO, sizeof(int)*8, &CpvAccess(fiprio));
}

CpvStaticDeclare(unsigned int, liprio);

CpmDestination CpmEnqueueILIFO(int pe, int prio)
{
  CpvAccess(liprio) = prio;
  return CpmEnqueue(pe, CQS_QUEUEING_ILIFO, sizeof(int)*8, &CpvAccess(liprio));
}

CpmDestination CpmEnqueueBFIFO(int pe, int priobits,unsigned int *prioptr)
{
  return CpmEnqueue(pe, CQS_QUEUEING_BFIFO, priobits, prioptr);
}

CpmDestination CpmEnqueueBLIFO(int pe, int priobits,unsigned int *prioptr)
{
  return CpmEnqueue(pe, CQS_QUEUEING_BLIFO, priobits, prioptr);
}

CpmDestination CpmEnqueueLFIFO(int pe, int priobits,unsigned int *prioptr)
{
  return CpmEnqueue(pe, CQS_QUEUEING_LFIFO, priobits, prioptr);
}

CpmDestination CpmEnqueueLLIFO(int pe, int priobits,unsigned int *prioptr)
{
  return CpmEnqueue(pe, CQS_QUEUEING_LLIFO, priobits, prioptr);
}

/******************************************************************************
 *
 * Control for Enqueue-FIFO
 *
 *****************************************************************************/

CpvDeclare(int, CpmEnqueueFIFO2_Index);

void CpmEnqueueFIFO2(void *msg)
{
  int *env;
  env = (int *)CpmEnv(msg);
  CmiSetHandler(msg, env[0]);
  CsdEnqueueFifo(msg);
}

void *CpmEnqueueFIFO1(CpmDestinationSend ctrl, int len, void *msg)
{
  int *env = (int *)CpmEnv(msg);
  env[0] = CmiGetHandler(msg);
  CmiSetHandler(msg, CpvAccess(CpmEnqueueFIFO2_Index));
  CpmLSend(ctrl->pe, len, msg);
  return (void *) 0;
}

CpvStaticDeclare(DestinationSend, ctrlFIFO);

CpmDestination CpmEnqueueFIFO(int pe)
{
  CpvAccess(ctrlFIFO).envsize = sizeof(int);
  CpvAccess(ctrlFIFO).sendfn  = CpmEnqueueFIFO1;
  CpvAccess(ctrlFIFO).pe = pe;
  return (CpmDestination)&CpvAccess(ctrlFIFO);
}

/******************************************************************************
 *
 * Control for Enqueue-LIFO
 *
 *****************************************************************************/

CpvDeclare(int, CpmEnqueueLIFO2_Index);

void CpmEnqueueLIFO2(void *msg)
{
  int *env;
  env = (int *)CpmEnv(msg);
  CmiSetHandler(msg, env[0]);
  CsdEnqueueLifo(msg);
}

void *CpmEnqueueLIFO1(CpmDestinationSend ctrl, int len, void *msg)
{
  int *env = (int *)CpmEnv(msg);
  env[0] = CmiGetHandler(msg);
  CmiSetHandler(msg, CpvAccess(CpmEnqueueLIFO2_Index));
  CpmLSend(ctrl->pe, len, msg);
  return (void *) 0;
}

CpvStaticDeclare(DestinationSend, ctrlLIFO);

CpmDestination CpmEnqueueLIFO(int pe)
{
  CpvAccess(ctrlLIFO).envsize = sizeof(int);
  CpvAccess(ctrlLIFO).sendfn  = CpmEnqueueLIFO1;
  CpvAccess(ctrlLIFO).pe = pe;
  return (CpmDestination)&CpvAccess(ctrlLIFO);
}

/******************************************************************************
 *
 * Control for thread-creation
 *
 *****************************************************************************/

CpvDeclare(int, CpmThread2_Index);

void CpmThread3(void *msg)
{
  int *env = (int *)CpmEnv(msg);
  CmiHandlerInfo *h=&CmiHandlerToInfo(env[0]);
  (h->hdlr)(msg,h->userPtr);
  CthFree(CthSelf()); CthSuspend();
}

void CpmThread2(void *msg)
{
  CthThread t;
  t = CthCreate(CpmThread3, msg, 0);
  CthSetStrategyDefault(t); CthAwaken(t);
}

void CpmThread1(CpmDestinationSend ctrl, int len, void *msg)
{
  int *env = (int *)CpmEnv(msg);
  env[0] = CmiGetHandler(msg);
  CmiSetHandler(msg, CpvAccess(CpmThread2_Index));
  CpmLSend(ctrl->pe, len, msg);
}

CpvStaticDeclare(DestinationSend, ctrlThread);

CpmDestination CpmMakeThread(int pe)
{
  CpvAccess(ctrlThread).envsize = sizeof(int);
  CpvAccess(ctrlThread).sendfn = (CpmSender)CpmThread1;
  CpvAccess(ctrlThread).pe = pe;
  return (CpmDestination)&CpvAccess(ctrlThread);
}

/******************************************************************************
 *
 * Control for thread-creation with size parameter
 *
 *****************************************************************************/

CpvDeclare(int, CpmThreadSize2_Index);

typedef struct CpmDestinationThreadSize
{ 
  void *(*sendfn)();
  int envsize;
  int pe;
  int size;
}
*CpmDestinationThreadSize;

typedef struct CpmDestinationThreadSize DestinationThreadSize;

void CpmThreadSize2(void *msg)
{
  int *env = (int *)CpmEnv(msg);
  CthThread t;
  t = CthCreate(CpmThread3, msg, env[1]);
  CthSetStrategyDefault(t); CthAwaken(t);
}

void CpmThreadSize1(CpmDestinationThreadSize ctrl, int len, void *msg)
{
  int *env = (int *)CpmEnv(msg);
  env[0] = CmiGetHandler(msg);
  env[1] = ctrl->size;
  CmiSetHandler(msg, CpvAccess(CpmThreadSize2_Index));
  CpmLSend(ctrl->pe, len, msg);
}

CpvStaticDeclare(DestinationThreadSize, ctrlThreadSize);

CpmDestination CpmMakeThreadSize(int pe, int size)
{
  CpvAccess(ctrlThreadSize).envsize = 2*sizeof(int);
  CpvAccess(ctrlThreadSize).sendfn = (CpmSender)CpmThreadSize1;
  CpvAccess(ctrlThreadSize).pe = pe;
  CpvAccess(ctrlThreadSize).size = size;
  return (CpmDestination)&CpvAccess(ctrlThreadSize);
}

/******************************************************************************
 *
 * Cpm initialization
 *
 *****************************************************************************/

void CpmModuleInit()
{
  CpvInitialize(int, CpmThread2_Index);
  CpvAccess(CpmThread2_Index) = CmiRegisterHandler(CpmThread2);
  CpvInitialize(int, CpmThreadSize2_Index);
  CpvAccess(CpmThreadSize2_Index) = CmiRegisterHandler(CpmThreadSize2);
  CpvInitialize(int, CpmEnqueueFIFO2_Index);
  CpvAccess(CpmEnqueueFIFO2_Index) = CmiRegisterHandler(CpmEnqueueFIFO2);
  CpvInitialize(int, CpmEnqueueLIFO2_Index);
  CpvAccess(CpmEnqueueLIFO2_Index) = CmiRegisterHandler(CpmEnqueueLIFO2);
  CpvInitialize(int, CpmEnqueue2_Index);
  CpvAccess(CpmEnqueue2_Index) = CmiRegisterHandler(CpmEnqueue2);
  CpvInitialize(DestinationSend, ctrlSend);
  CpvInitialize(DestinationEnq, ctrlEnq);
  CpvInitialize(DestinationSend, ctrlFIFO);
  CpvInitialize(DestinationSend, ctrlLIFO);
  CpvInitialize(DestinationSend, ctrlThread);
  CpvInitialize(DestinationThreadSize, ctrlThreadSize);
  CpvInitialize(unsigned int, fiprio);
  CpvInitialize(unsigned int, liprio);
}

