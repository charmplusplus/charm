#include "conv-config.h"

#if CMK_DLL_USE_DLOPEN && CMK_HAS_RTLD_NEXT

/* These macros are needed for:
 * dlfcn.h: RTLD_NEXT
 */
#ifndef _GNU_SOURCE
# define _GNU_SOURCE
#endif
#ifndef __USE_GNU
# define __USE_GNU
#endif

#include <sys/types.h>
#include <dlfcn.h>

struct mallinfo;

extern "C" void * (*mm_malloc)(size_t);
extern "C" void * (*mm_calloc)(size_t,size_t);
extern "C" void * (*mm_realloc)(void*,size_t);
extern "C" void * (*mm_memalign)(size_t,size_t);
extern "C" void * (*mm_valloc)(size_t);
extern "C" void (*mm_free)(void*);
extern "C" void (*mm_cfree)(void*);
extern "C" struct mallinfo (*mm_mallinfo)(void);

  
extern "C" void initialize_memory_wrapper() {
  mm_malloc = (void *(*)(size_t)) dlsym(RTLD_NEXT, "malloc");
  mm_realloc = (void *(*)(void*,size_t)) dlsym(RTLD_NEXT, "realloc");
  mm_calloc = (void *(*)(size_t,size_t)) dlsym(RTLD_NEXT, "calloc");
  mm_memalign = (void *(*)(size_t,size_t)) dlsym(RTLD_NEXT, "memalign");
  mm_valloc = (void *(*)(size_t)) dlsym(RTLD_NEXT, "valloc");
  mm_free = (void (*)(void*)) dlsym(RTLD_NEXT, "free");
  mm_cfree = (void (*)(void*)) dlsym(RTLD_NEXT, "cfree");
  mm_mallinfo = (struct mallinfo (*)(void)) dlsym(RTLD_NEXT, "mallinfo");
}

#endif
