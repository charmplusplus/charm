#include "converse.h"
#include "charm++.h"
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
  CmiPrintf("All tests passed\n");
  CmiPrintf("Info: converse header: %d envelope: %d\n", CmiReservedHeaderSize, sizeof(envelope));
}

int main(int argc, char **argv)
{
  ConverseInit(argc,argv,check_test,1,0);
}

