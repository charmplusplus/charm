#include <stdlib.h>
#include <string.h>
#include "converse.h"

typedef struct Cfuture_data_s
{
  void      *value;
  int        ready;
  CthThread  waiters;
}
*futdata;

typedef struct CfutureValue_s
{
  char core[CmiMsgHeaderSizeBytes];
  struct Cfuture_data_s *data;
  int valsize;
  double rest[1];
}
*CfutureValue;

#define field_offset(t, f) ((size_t)(((t)0)->f))
#define void_to_value(v) ((CfutureValue)(((char*)v)-field_offset(CfutureValue,rest)))

CpvDeclare(int, CfutureStoreIndex);

Cfuture CfutureCreate()
{
  futdata data = (futdata)malloc(sizeof(struct Cfuture_data_s));
  Cfuture result;
  _MEMCHECK(data);
  data->value = 0;
  data->ready = 0;
  data->waiters = 0;
  result.pe = CmiMyPe();
  result.data = data;
  return result;
}

static void CfutureAwaken(futdata data, CfutureValue val)
{
  CthThread t;
  data->value = val;
  data->ready = 1;
  for (t=data->waiters; t; t=CthGetNext(t))
    CthAwaken(t);
  data->waiters=0;
}

static void CfutureStore(CfutureValue m)
{
  CfutureAwaken(m->data, m);
}

void *CfutureCreateBuffer(int bytes)
{
  int valsize = sizeof(struct CfutureValue_s) + bytes;
  CfutureValue m = (CfutureValue)CmiAlloc(valsize);
  CmiSetHandler(m, CpvAccess(CfutureStoreIndex));
  m->valsize = valsize;
  return (void*)(m->rest);
}

void CfutureDestroyBuffer(void *v)
{
  CmiFree(v);
}

void CfutureStoreBuffer(Cfuture f, void *value)
{
  CfutureValue m = void_to_value(value);
  if (f.pe == CmiMyPe()) {
    CfutureAwaken(f.data, m);
  } else {
    m->data = f.data;
    CmiSyncSendAndFree(f.pe, m->valsize, m);
  }
}

void CfutureSet(Cfuture f, void *value, int len)
{
  void *copy = CfutureCreateBuffer(len);
  memcpy(copy, value, len);
  CfutureStoreBuffer(f, copy);
}

void *CfutureWait(Cfuture f)
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
  return (void*)(value->rest);
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
  if (f.data->value) CmiFree(f.data->value);
  free(f.data);
}

void CfutureModuleInit()
{
  CpvInitialize(int, CfutureStoreIndex);
  CpvAccess(CfutureStoreIndex) = CmiRegisterHandler((CmiHandler)CfutureStore);
}
