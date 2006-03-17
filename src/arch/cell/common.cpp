#ifndef __COMMON_H__
#define __COMMON_H__


#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "general.h"
#include "common.h"


#define MALLOC_ALIGNED_ZERO_MEMORY  1

void* malloc_aligned(size_t size, char alignment) {

  // TODO : In the future, use posix_memalign() call instead (if available)

  void* rtn = NULL;
  int tailPadding;
  char offset = 0;

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
  rtn = malloc(size + alignment + tailPadding);
  #if MALLOC_ALIGNED_ZERO_MEMORY
    memset(rtn, 0, size + alignment + tailPadding);
  #endif

  // Calculate the offset into the returned memory chunk that has the required alignment
  offset = (char)(((size_t)rtn) % alignment);
  if (offset == 0) offset = alignment;

  // Write the offset into the byte before the address to be returned
  *((char*)rtn + offset - 1) = offset;

  // Return the address with offset
  return (void*)((char*)rtn + offset);
}


void free_aligned(void* ptr) {

  char offset;

  // Verify the parameter
  if (ptr == NULL) return;

  // Read the offset (byte before ptr)
  offset = *((char*)ptr - 1);

  // Free the memory
  free ((void*)((char*)ptr - offset));
}


#endif //__COMMON_H__
