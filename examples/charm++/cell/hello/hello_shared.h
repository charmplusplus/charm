#ifndef __HELLO_SHARED_H__
#define __HELLO_SHARED_H__


/////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Defines

#define FUNC_SAYHI    1
#define FUNC_STRBUFS  2


/////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Function Prototypes

void funcLookup(int funcIndex, void* readWritePtr, int readWriteLen, void* readOnlyPtr, int readOnlyLen, void* writeOnlyPtr, int writeOnlyLen);
void sayHi(char* readWritePtr, char* readOnlyPtr);
void strBufs(DMAListEntry* dmaList, int numReadOnly, int numReadWrite, int numWriteOnly);


#endif //__HELLO_SHARED_H__
