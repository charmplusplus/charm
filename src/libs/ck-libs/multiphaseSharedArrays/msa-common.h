// emacs mode line -*- mode: c++; tab-width: 4 -*-
#ifndef MSA_COMMON_H
#define MSA_COMMON_H

const unsigned int MSA_INVALID_PAGE_NO = 0xFFFFFFFF;
const int MSA_INVALID_PE = -1;
typedef void* page_ptr_t;

const int Uninit_State = 0;
const int Read_Fault = 1;
const int Write_Fault = 2;
const int Accumulate_Fault = 3;

const unsigned int MSA_DEFAULT_ENTRIES_PER_PAGE = 1024;
// Therefore, size of page == MSA_DEFAULT_ENTRIES_PER_PAGE * sizeof(element type)

// Size of cache on each PE
const unsigned int MSA_DEFAULT_MAX_BYTES = 16*1024*1024;

#define DEBUG_PRINTS

#endif
