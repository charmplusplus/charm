#include "converse.h"

typedef struct Cfuture_data_s
{
  CfutureValue value;
  int          ready;
  CthThread    waiters;
}
*futdata;

#define field_offset(type, field) ((int)(&(((type)0)->field)))

CpvDeclare(int, CfutureStoreIndex);

Cfuture CfutureCreate()
{
  futdata data = (futdata)malloc(sizeof(struct Cfuture_data_s));
  Cfuture result;
  data->value = 0;
  data->ready = 0;
  data->waiters = 0;
  result.pe = CmiMyPe();
  result.data = data;
  return result;
}

CfutureValue CfutureCreateValue(int bytes)
{
  int valsize = sizeof(struct CfutureValue_s) + bytes;
  CfutureValue m = (CfutureValue)CmiAlloc(valsize);
  CmiSetHandler(m, CpvAccess(CfutureStoreIndex));
  m->valsize = valsize;
  return m;
}

static void CfutureAwaken(futdata data, CfutureValue val)
{
  CthThread t;
  data->value = val;
  data->ready = 1;
  t = data->waiters;
  while (t) {
    CthAwaken(t);
    t = CthGetNext(t);
  }
  data->waiters=0;
}

static void CfutureStore(CfutureValue m)
{
  CmiGrabBuffer(&m);
  CfutureAwaken(m->data, m);
}

void CfutureSet(Cfuture f, CfutureValue m)
{
  futdata data;
  if (f.pe == CmiMyPe()) {
    CfutureAwaken(f.data, m);
  } else {
    m->data = f.data;
    CmiSyncSendAndFree(f.pe, m->valsize, m);
  }
}

CfutureValue CfutureWait(Cfuture f, int freeflag)
{
  CthThread self; CfutureValue value; futdata data;
  if (f.pe != CmiMyPe()) {
    CmiPrintf("error: CfutureWait: future not local.\n");
    exit(1);
  }
  data = f.data;
  if (data->ready == 0) {
    self = CthSelf();
    CthSetNext(self, data->waiters);
    data->waiters = self;
    CthSuspend();
  }
  value = data->value;
  if (freeflag) free(data);
  return value;
}

void CfutureDestroy(Cfuture f)
{
  if (f.pe != CmiMyPe()) {
    CmiPrintf("error: CfutureDestroy: future not local.\n");
    exit(1);
  }
  if (f.data->waiters) {
    CmiPrintf("error: CfutureDestroy: destroying an active future.\n");
    exit(1);
  }
  free(f);
}

void CfutureDestroyValue(CfutureValue v)
{
  CmiFree(v);
}

void CfutureInit()
{
  CpvInitialize(int, CfutureStoreIndex);
  CpvAccess(CfutureStoreIndex) = CmiRegisterHandler(CfutureStore);
}
