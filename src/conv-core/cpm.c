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

void CpmSend1(CpmDestinationSend ctrl, int len, void *msg)
{
  CpmLSend(ctrl->pe, len, msg);
}

CpmDestination CpmSend(int pe)
{
  static struct CpmDestinationSend ctrl;
  ctrl.envsize = 0;
  ctrl.sendfn = (CpmSender)CpmSend1;
  ctrl.pe = pe;
  return (CpmDestination)&ctrl;
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
  int pe, qs, priobits, *prioptr;
}
*CpmDestinationEnq;

CpvDeclare(int, CpmEnqueue2_Index);

void CpmEnqueue2(void *msg)
{
  int *env;
  CmiGrabBuffer(&msg);
  env = (int *)CpmEnv(msg);
  CmiSetHandler(msg, env[0]);
  CsdEnqueueGeneral(msg, env[1], env[2], env+3);
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
}

CpmDestination CpmEnqueue(int pe, int qs, int priobits, int *prioptr)
{
  static struct CpmDestinationEnq ctrl;
  int intbits = sizeof(int)*8;
  int prioints = (priobits+intbits-1) / intbits;
  ctrl.envsize = (3+prioints)*sizeof(int);
  ctrl.sendfn  = CpmEnqueue1;
  ctrl.pe = pe; ctrl.qs = qs; ctrl.priobits = priobits; ctrl.prioptr = prioptr;
  return (CpmDestination)&ctrl;
}

CpmDestination CpmEnqueueIFIFO(int pe, int prio)
{
  static int iprio;
  iprio = prio;
  return CpmEnqueue(pe, CQS_QUEUEING_IFIFO, sizeof(int)*8, &iprio);
}

CpmDestination CpmEnqueueILIFO(int pe, int prio)
{
  static int iprio;
  iprio = prio;
  return CpmEnqueue(pe, CQS_QUEUEING_ILIFO, sizeof(int)*8, &iprio);
}

CpmDestination CpmEnqueueBFIFO(int pe, int priobits, int *prioptr)
{
  return CpmEnqueue(pe, CQS_QUEUEING_BFIFO, priobits, prioptr);
}

CpmDestination CpmEnqueueBLIFO(int pe, int priobits, int *prioptr)
{
  return CpmEnqueue(pe, CQS_QUEUEING_BLIFO, priobits, prioptr);
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
  CmiGrabBuffer(&msg);
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
}

CpmDestination CpmEnqueueFIFO(int pe)
{
  static struct CpmDestinationSend ctrl;
  ctrl.envsize = sizeof(int);
  ctrl.sendfn  = CpmEnqueueFIFO1;
  ctrl.pe = pe;
  return (CpmDestination)&ctrl;
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
  CmiGrabBuffer(&msg);
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
}

CpmDestination CpmEnqueueLIFO(int pe)
{
  static struct CpmDestinationSend ctrl;
  ctrl.envsize = sizeof(int);
  ctrl.sendfn  = CpmEnqueueLIFO1;
  ctrl.pe = pe;
  return (CpmDestination)&ctrl;
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
  CpvAccess(CmiHandlerTable)[env[0]](msg);
  CthFree(CthSelf()); CthSuspend();
}

void CpmThread2(void *msg)
{
  CthThread t;
  CmiGrabBuffer(&msg);
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

CpmDestination CpmMakeThread(int pe)
{
  static struct CpmDestinationSend ctrl;
  ctrl.envsize = sizeof(int);
  ctrl.sendfn = (CpmSender)CpmThread1;
  ctrl.pe = pe;
  return (CpmDestination)&ctrl;
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
  CpvInitialize(int, CpmEnqueueFIFO2_Index);
  CpvAccess(CpmEnqueueFIFO2_Index) = CmiRegisterHandler(CpmEnqueueFIFO2);
  CpvInitialize(int, CpmEnqueueLIFO2_Index);
  CpvAccess(CpmEnqueueLIFO2_Index) = CmiRegisterHandler(CpmEnqueueLIFO2);
  CpvInitialize(int, CpmEnqueue2_Index);
  CpvAccess(CpmEnqueue2_Index) = CmiRegisterHandler(CpmEnqueue2);
}

