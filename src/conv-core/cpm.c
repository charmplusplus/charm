
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

typedef struct CpmControlSend
{
  struct CpmControlStruct head;
  int pe;
}
*CpmControlSend;

void CpmSend1(CpmControlSend ctrl, int len, void *msg)
{
  CpmLSend(ctrl->pe, len, msg);
}

CpmControl CpmSend(int pe)
{
  static struct CpmControlSend ctrl;
  ctrl.head.envsize = 0;
  ctrl.head.sendfn = (CpmSender)CpmSend1;
  ctrl.pe = pe;
  return (CpmControl)&ctrl;
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

void CpmThread1(CpmControlSend ctrl, int len, void *msg)
{
  int *env = (int *)CpmEnv(msg);
  env[0] = CmiGetHandler(msg);
  CmiSetHandler(msg, CpvAccess(CpmThread2_Index));
  CpmLSend(ctrl->pe, len, msg);
}

CpmControl CpmMakeThread(int pe)
{
  static struct CpmControlSend ctrl;
  ctrl.head.envsize = sizeof(int);
  ctrl.head.sendfn = (CpmSender)CpmThread1;
  ctrl.pe = pe;
  return (CpmControl)&ctrl;
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
}

