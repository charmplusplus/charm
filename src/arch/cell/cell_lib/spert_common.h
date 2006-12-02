#ifndef __COMMON_H__
#define __COMMON_H__


#include <stdlib.h>


#ifdef __cplusplus
extern "C" {
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Defines

#ifndef TRUE
  #define TRUE (-1)
#endif

#ifndef FALSE
  #define FALSE (0)
#endif

#ifndef NULL
  #define NULL (0)
#endif

// SIZEOF_16 : Returns the size of the structure s rounded up to the nearest multiple of 16.
//   NOTE: All of the values in this macro are constants so a good compiler should be able to reduce all of this to a constant.

#define ROUNDUP_16(s)   (s + ((16 - (s & 15)) & 15))
#define SIZEOF_16(s)    (sizeof(s) + ((16 - (sizeof(s) & 15)) & 15))

#define ROUNDUP_128(s)  (s + ((128 - (s & 127)) & 127))
#define SIZEOF_128(s)   (sizeof(s) + ((128 - (sizeof(s) & 127)) & 127))

#define ROUNDUP(s,p2)   (s + ((p2 - (s & (p2 - 1))) & (p2 - 1)))
#define SIZEOF(s,p2)    (sizeof(s) + ((p2 - (sizeof(s) & (p2 - 1))) & (p2 - 1)))

//#define SIZEOF_16(s)   ( (((sizeof(s) & 0x0000000F)) == (0x00)) ? (int)(sizeof(s)) : (int)((sizeof(s) & 0xFFFFFFF0) + (0x10)) )
//#define ROUNDUP_16(s)  ( ((((s) & 0x0000000F)) == (0x00)) ? (int)(s) : (int)(((s) & 0xFFFFFFF0) + (0x10)) )


/////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Function Prototypes

// These memory functions ensure that both the start and the end of the returned memory region
//   are aligned (in bytes) on the specified byte boundries.
// NOTE: Like the alloca_aligned() function calls alloca and as such, there is no need to free
//   the memory returned by this function.  (See the man page for alloca for details.)
extern void* malloc_aligned(size_t size, char alignment);
extern void* calloc_aligned(size_t size, char alignment);
//extern void* alloca_aligned(size_t size, char alignment, int zeroFlag);
extern void free_aligned(void* ptr);


#ifdef __cplusplus
}
#endif

#endif //__COMMON_H__
