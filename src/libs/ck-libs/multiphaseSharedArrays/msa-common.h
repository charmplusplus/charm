// emacs mode line -*- mode: c++; tab-width: 4 -*-
#ifndef MSA_COMMON_H
#define MSA_COMMON_H

enum { MSA_INVALID_PAGE_NO = 0xFFFFFFFF };
enum { MSA_INVALID_PE = -1 };
typedef void* page_ptr_t;

typedef enum {
 Uninit_State = 0,
 Read_Fault = 1,
 Write_Fault = 2,
 Accumulate_Fault = 3
} MSA_Page_Fault_t;

/// Allow MSA_Page_Fault_t's to be pupped:
inline void operator|(PUP::er &p,MSA_Page_Fault_t &f) {
  int i=f;
  p|i;
  f=(MSA_Page_Fault_t)i;
}

enum { MSA_DEFAULT_ENTRIES_PER_PAGE = 1024 };
// Therefore, size of page == MSA_DEFAULT_ENTRIES_PER_PAGE * sizeof(element type)

// Size of cache on each PE
enum { MSA_DEFAULT_MAX_BYTES = 16*1024*1024 };

typedef enum { MSA_COL_MAJOR=0, MSA_ROW_MAJOR=1 } MSA_Array_Layout_t;

#define DEBUG_PRINTS

#endif
