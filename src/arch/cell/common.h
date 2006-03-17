#ifndef __COMMON_H__
#define __COMMON_H__


#include "general.h"


// SIZEOF_16 : Returns the size of the structure s rounded up to the nearest multiple of 16.
//   NOTE: All of the values in this macro are constants so a good compiler should be able to reduce all of this to a constant.
#define SIZEOF_16(s)   ( ((sizeof(s) & 0xF) == 0) ? (sizeof(s)) : ((sizeof(s) & 0xFFFFFFF0) + (0x10)) )
#define ROUNDUP_16(s)  ( ((s & 0xF) == 0) ? (s) : ((s & 0xFFFFFFF0) + (0x10)) )

// This is a malloc function (and corresponding free) that will automatically align
//   the returned pointer on the alignment boundry specified (in bytes).
extern void* malloc_aligned(size_t size, char alignment);
extern void free_aligned(void* ptr);


/// Defines from bpa_map.def ///////////////////////////////////////////////////////////////////////////////

#define BP_BASE 0x20000000000ULL

#define DMA_LSA                 0x3004
#define DMA_EA                  0x3008
#define DMA_EA_HI               DMA_EA
#define DMA_EA_LO               (DMA_EA_HI + 4)
#define DMA_STBRC               0x3010  /* Addr for writing
                                           64-bit packed data */
#define DMA_Size                0x3010
#define DMA_Tag                 0x3012
#define DMA_BRCMD               0x3014  /* Addr for writing
                                           32-bit packed cmd */
#define DMA_RclassID            0x3015
#define DMA_CMD                 0x3016
#define DMA_CMDStatus           DMA_CMD
#define DMA_QStatus             0x3104
#define DMA_QueryType           0x3204
#define DMA_QueryMask           0x321C
#define DMA_TagStatus           0x322C
#define DMA_TAGSTATUS_INTR_ANY  1L
#define DMA_TAGSTATUS_INTR_ALL  2L


#endif //__COMMON_H__
