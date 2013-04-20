#include "converse.h"
#include "envelope.h"
#include <stdio.h>

void check_test(int argc, char** argv) {
  CmiInt2 int2;
  CmiInt4 int4;
  CmiInt8 int8;
  CmiUInt2 uint2;
  CmiUInt4 uint4;
  CmiUInt8 uint8;
  CmiFloat4 float4;
  CmiFloat8 float8;

  if (CkMyPe()!=0) return;
  if (sizeof(int2) != 2) {
    CmiPrintf("Error: sizeof(CmiInt2) is %d!\n",sizeof(int2));
    exit(1);
  }
  if (sizeof(int4) != 4) {
    CmiPrintf("Error: sizeof(CmiInt4) is %d!\n",sizeof(int4));
    exit(1);
  }
  if (sizeof(int8) != 8) {
    CmiPrintf("Error: sizeof(CmiInt8) is %d!\n",sizeof(int8));
    exit(1);
  }

  if (sizeof(uint2) != 2) {
    CmiPrintf("Error: sizeof(CmiUInt2) is %d!\n",sizeof(uint2));
    exit(1);
  }
  if (sizeof(uint4) != 4) {
    CmiPrintf("Error: sizeof(CmiUInt4) is %d!\n",sizeof(uint4));
    exit(1);
  }
  if (sizeof(uint8) != 8) {
    CmiPrintf("Error: sizeof(CmiUInt8) is %d!\n",sizeof(uint8));
    exit(1);
  }

  if (sizeof(float4) != 4) {
    CmiPrintf("Error: sizeof(CmiFloat4) is %d!\n",sizeof(float4));
    exit(1);
  }
  if (sizeof(float8) != 8) {
    CmiPrintf("Error: sizeof(CmiFloat8) is %d!\n",sizeof(float8));
    exit(1);
  }

  // Test for changes in CkEnvelopeType
  // If the test below fails, it means the CkEnvelopeType enum was modified.
  // BEFORE changing this test, make sure the CHARMDEBUG_MINOR version number is
  // incremented, and the CharmDebug correspondant enumeration (in
  // charm.debug.pdata.MsgInfo.java) is updated accordingly.
  if (LAST_CK_ENVELOPE_TYPE != 19) {
    CmiPrintf("Error: LAST_CK_ENVELOPE_TYPE changed. Update CharmDebug and fix this test:\n");
    CmiPrintf("       BEFORE changing this test, make sure the CHARMDEBUG_MINOR version number is incremented, and the CharmDebug correspondant enumeration (in charm.debug.pdata.MsgInfo.java) is updated accordingly.");
    exit(1);
  }

#if ! CMK_SMP
  const int s = 8*1024*1024;
  void *buf1 = CmiAlloc(s);
  memset(buf1, 1, s);
  CmiUInt8 mem_before = CmiMemoryUsage();
  void *buf2 = CmiAlloc(s);
  memset(buf2, 2, s);
  CmiUInt8 mem_after = CmiMemoryUsage();
  CmiFree(buf2);
  CmiFree(buf1);
  CmiPrintf("CmiMemoryUsage() reported %fMB (before) vs %fMB (after)!\n", mem_before/1E6, mem_after/1E6);
  if (mem_after - mem_before < s) {
    CmiPrintf("Error: CmiMemoryUsage() does not work %lld %lld!\n", mem_before, mem_after);
    CmiAbort("CmiMemoryUsage failed");
  }
#endif

  CmiPrintf("Info: converse header: %d envelope: %d\n", CmiReservedHeaderSize, sizeof(envelope));
  if (sizeof(envelope) % 8 != 0) {
    CmiPrintf("Error: size of envelope can not divide 8. \n");
    exit(1);
  }
  CmiPrintf("All tests passed\n");
}

int main(int argc, char **argv)
{
  ConverseInit(argc,argv,check_test,1,0);
}

