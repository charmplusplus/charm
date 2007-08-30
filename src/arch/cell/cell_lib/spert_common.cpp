#ifndef __COMMON_H__
#define __COMMON_H__


#ifdef __cplusplus
extern "C" {
#endif
  #include <stdlib.h>
  #include <stdio.h>
  #include <string.h>
#ifdef __cplusplus
}
#endif

#include "spert_common.h"


///////////////////////////////////////////////////////////////////////////////////////////////////////////
// Defines

#define MALLOC_ALIGNED_ZERO_MEMORY   0
#define ALLOCA_ALIGNED_ZERO_MEMORY   0


///////////////////////////////////////////////////////////////////////////////////////////////////////////
// Function Bodies


extern "C" inline void* malloc_aligned_helper(size_t size, unsigned short alignment, int zeroFlag) {

  void* rtn = NULL;
  int tailPadding;
  unsigned short offset = 0;

  // Verify the parameters
  if (size <= 0 || alignment <= 0)
    return NULL;

  // Malloc memory of size equal to size + alignment + (alignment - (size % alignment)).  The
  //   last term 'alignment - (size % alignment)' ensures that there is enough tail padding
  //   so a DMA can be performed based on the alignment.  (I.e. - Start and end the memory
  //   region retured on an alignment boundry specified.)
  // NOTE: Since we need a byte long header, even if we "get lucky" and the malloc
  //   returns a pointer with the given alignment, we need to put in a byte
  //   preamble anyway.
  tailPadding = alignment - (size % alignment);
  if (tailPadding == alignment)
    tailPadding = 0;

  // Allocate the memory
  // NOTE : If the ZERO_MEMORY define for this function is set, force the memory to be zeroed.
  #if MALLOC_ALIGNED_ZERO_MEMORY != 0
    rtn = calloc(size + alignment + tailPadding, 1);
  #else
    rtn = ((zeroFlag != 0) ?
            (calloc(size + alignment + tailPadding, 1)) :
            (malloc(size + alignment + tailPadding)))   ;
  #endif

  // Calculate the offset into the returned memory chunk that has the required alignment
  offset = (char)(((size_t)rtn) % alignment);
  offset = alignment - offset;
  if (offset == 0) offset = alignment;

  // Write the offset into the byte before the address to be returned
  *((char*)rtn + offset - 1) = offset;

  // Return the address with offset
  return (void*)((char*)rtn + offset);
}

extern "C" void* malloc_aligned(size_t size, unsigned short alignment) {
  return malloc_aligned_helper(size, alignment, 0);
}
extern "C" void* calloc_aligned(size_t size, unsigned short alignment) {
  return malloc_aligned_helper(size, alignment, 1);
}


/*
extern "C" void* alloca_aligned(size_t size, char alignment, int zeroFlag) {

  void* rtn = NULL;
  int tailPadding;
  char offset = 0;

  // Verify the parameters
  if (size <= 0 || alignment <= 0)
    return NULL;

  // Malloc memory of size equal to size + (alignment - 1) + (alignment - (size % alignment)).
  //   size : The ammount of memory needed by the caller.
  //   'alignment - 1' : The number of bytes needed to ensure that the first byte returned
  //     to the caller is aligned properly.
  //   'alignment - (size % alignment)' : The number of bytes needed to ensure the end of the
  //     memory region is aligned properly for DMA transfers.
  tailPadding = alignment - (size % alignment);
  if (tailPadding == alignment)
    tailPadding = 0;

  // Allocate the memory
  // NOTE : If the ZERO_MEMORY define for this function is set, force the memory to be zeroed.
  rtn = alloca(size + alignment - 1 + tailPadding);
  #if ALLOCA_ALIGNED_ZERO_MEMORY != 0
    memset(rtn, 0, size + alignment - 1 + tailPadding);
  #endif

  // Return the address with offset
  return (void*)((char*)rtn + (((size_t)rtn) % alignment));
}
*/


extern "C" void free_aligned(void* ptr) {

  char offset;

  // Verify the parameter
  if (ptr == NULL) return;

  // Read the offset (byte before ptr)
  offset = *((char*)ptr - 1);

  // Free the memory
  free ((void*)((char*)ptr - offset));
}


#endif //__COMMON_H__
