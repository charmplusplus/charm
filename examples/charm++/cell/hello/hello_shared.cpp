#include <stdio.h>
#include <string.h>

#include "spert.h"
#include "hello_shared.h"


void funcLookup(int funcIndex,
                void* readWritePtr, int readWriteLen,
                void* readOnlyPtr, int readOnlyLen,
                void* writeOnlyPtr, int writeOnlyLen,
                DMAListEntry* dmaList
               ) {

  switch (funcIndex) {

    case FUNC_SAYHI: sayHi((char*)readWritePtr, (char*)readOnlyPtr); break;
    case FUNC_STRBUFS: strBufs(dmaList, readOnlyLen, readWriteLen, writeOnlyLen); break;

    default:
      //printf("!!! WARNING !!! :: SPE Received Invalid funcIndex (%d)... Ignoring...\n", funcIndex);
      break;
  }
}


void sayHi(char* readWritePtr, char* readOnlyPtr) {

  // Display a message for the user
  printf("I was told to say \"Hi\"... so \"Hi\"... ok... later...\n");

  // Display any strings in the readable buffers
  if (readWritePtr != NULL)
    printf("   readWritePtr -> \"%s\"\n", readWritePtr);
  if (readOnlyPtr != NULL)
    printf("    readOnlyPtr -> \"%s\"\n", readOnlyPtr);
}


void strBufs(DMAListEntry* dmaList, int numReadOnly, int numReadWrite, int numWriteOnly) {

  int numDMAEntries = numReadOnly + numReadWrite + numWriteOnly;

  // Zero out the write only buffers (this way the string functions don't go crazy if
  //   the original contents of the writeOnly buffer(s) happen to be non-zero)
  for (int i = numReadOnly + numReadWrite; i < numDMAEntries; i++)
    memset((void*)(dmaList[i].ea), 0x00, dmaList[i].size);

  // Display the contents of the dmaList passed into this function
  //for (int i = 0; i < numDMAEntries; i++) {
  //  printf("SPE :: dmaList[%d] = { size = %d, ea = 0x%08x }\n", i, dmaList[i].size, dmaList[i].ea);
  //}
  //printf("SPE :: numReadOnly = %d, numReadWrite = %d, numWriteOnly = %d\n", numReadOnly, numReadWrite, numWriteOnly);

  for (int i = 0; i < numDMAEntries; i++) {

    // Set the fill character based on the array type
    char fill = '-';
    char* typeStr = "RO";
    if (i >= numReadOnly) { fill = '='; typeStr = "RW"; }
    if (i >= (numReadOnly + numReadWrite)) { fill = '*'; typeStr = "WO"; }

    unsigned int strLen = strlen((char*)(dmaList[i].ea));

    // Print the array before modifying it
    //printf("SPE :: [%d before] (%d @ 0x%08x is %s) \"%s\"\n",
    //       i, dmaList[i].size, (void*)(dmaList[i].ea), typeStr, (char*)(dmaList[i].ea)
    //      );

    //// Fill in the array using the fill caracter (which depends on the type of buffer: RO, RW, WO)
    if (strLen > 0 && strLen < dmaList[i].size) {
      memset((void*)(dmaList[i].ea), fill, strLen);
    } else {
      for (int j = 0; j < dmaList[i].size - 1; j++)
        *((char*)(dmaList[i].ea) + j) = fill;
      *((char*)(dmaList[i].ea + dmaList[i].size - 1)) = '\0';
    }

    // Print the modified array
    //printf("SPE :: [%d  after] (%d @ 0x%08x is %s) \"%s\"\n",
    //       i, dmaList[i].size, (void*)(dmaList[i].ea), typeStr, (char*)(dmaList[i].ea)
    //      );
  }

}
