#include "converse.h"
#include "envelope.h"
#include <stdio.h>

void check_size_approx(size_t s) {
  size_t enc = pup_decodeSize(pup_encodeSize(s));
  double diff = s > enc ? s - enc : enc - s;
  if (diff / s > 0.005) {
    CmiPrintf("Failed to accurately encode size %lu as %lu: difference was %f\n",
              (unsigned long)s, enc, diff);
    exit(2);
  }
}

void test_CmiMemoryUsage() {
  // 1 G
  const CMK_TYPEDEF_UINT8 g = 1024 * 1024 * 1024;
  // 100 mbs
  const int m100 = 100 * 1024 * 1024;

  CMK_TYPEDEF_UINT8 mem_total = 0;
  CmiUInt8 mem_before = CmiMemoryUsage();

  void *buf1 = malloc(m100);
  if (buf1) {
    mem_total += m100;
  }

  void * buf2 = NULL;
  if (sizeof(void *) == 8) {
    // test malloc_info support for 64 bit, allocate 6 Gs
    buf2 = malloc(6*g);
    if (buf2) {
      mem_total += 6*g;
    } else {
      CmiPrintf("malloc of 6G failed\n");
    }
  }

  CmiUInt8 mem_after = CmiMemoryUsage();
  if (buf1) {free(buf1);}
  if (buf2) {free(buf2);}

  CmiPrintf("CmiMemoryUsage() reported %fMB (before) vs %fMB (after)\n", mem_before/1E6, mem_after/1E6);
  if (mem_after - mem_before < mem_total) {
    CmiPrintf("Error: CmiMemoryUsage() does not work %lld %lld!\n", mem_before, mem_after);
    CmiAbort("CmiMemoryUsage failed");
  }
}


// Test cases for the approximate compression provided by
// pup_{en,de}codeSize, as tested above by check_sizes_approx()
size_t check_size_values[] =
{
  0,
  1,
  256,
  1UL << 12,
  (1UL << 12) + 1,
  (1UL << 13) - 1,
  1UL << 13,
  (1UL << 13) + 1,
  (1UL << 14) - 1,
  1UL << 31
#if CMK_SIZET_64BIT
  , 1ULL << 32,
  1ULL << 33,
  1ULL << 34
#endif
};

void check_test(int argc, char** argv) {
  CmiInt2 int2;
  CmiInt4 int4;
  CmiInt8 int8;
  CmiUInt2 uint2;
  CmiUInt4 uint4;
  CmiUInt8 uint8;
  CmiFloat4 float4;
  CmiFloat8 float8;

  if (CmiMyPe()!=0) return;

  if (argc > 1) {
    const int expected_pes = atoi(argv[1]);
    if (CmiNumPes() != expected_pes) {
      char message[1000];
      sprintf(message, "PE count %d doesn't match expectation %d", CmiNumPes(), expected_pes);
      CmiAbort(message);
    }
  }

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
  if (LAST_CK_ENVELOPE_TYPE != 21) {
    CmiPrintf("Error: LAST_CK_ENVELOPE_TYPE changed. Update CharmDebug and fix this test:\n");
    CmiPrintf("       BEFORE changing this test, make sure the CHARMDEBUG_MINOR version number is incremented, and the CharmDebug correspondant enumeration (in charm.debug.pdata.MsgInfo.java) is updated accordingly.");
    exit(1);
  }

  test_CmiMemoryUsage();

  CmiPrintf("Info: converse header: %d envelope: %d\n", CmiReservedHeaderSize, sizeof(envelope));
  if (sizeof(envelope) % 8 != 0) {
    CmiPrintf("Error: size of envelope can not divide 8. \n");
    exit(1);
  }

  for (int i = 0; i < sizeof(check_size_values)/sizeof(check_size_values[0]); ++i)
    check_size_approx(check_size_values[i]);

  CmiPrintf("All tests passed\n");
}

int main(int argc, char **argv)
{
  ConverseInit(argc,argv,check_test,1,0);
}

