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
extern "C" int (*mm_posix_memalign)(void **,size_t,size_t);
extern "C" void * (*mm_aligned_alloc)(size_t,size_t);
extern "C" void * (*mm_valloc)(size_t);
extern "C" void * (*mm_pvalloc)(size_t);
extern "C" void (*mm_free)(void*);
extern "C" void (*mm_cfree)(void*);
extern "C" struct mallinfo (*mm_mallinfo)(void);

  
extern char initialize_memory_wrapper_status;
char initialize_memory_wrapper_status;

extern "C" void initialize_memory_wrapper() {
  initialize_memory_wrapper_status = 1;

  // wait to install these all at once because dlsym calls them, and a mismatch would be bad
  void * (*os_malloc)(size_t) = (void *(*)(size_t)) dlsym(RTLD_NEXT, "malloc");
  void * (*os_calloc)(size_t,size_t) = (void *(*)(size_t,size_t)) dlsym(RTLD_NEXT, "calloc");
  void (*os_free)(void*) = (void (*)(void*)) dlsym(RTLD_NEXT, "free");

  mm_malloc = os_malloc;
  mm_calloc = os_calloc;
  mm_free = os_free;

  mm_realloc = (void *(*)(void*,size_t)) dlsym(RTLD_NEXT, "realloc");
  mm_memalign = (void *(*)(size_t,size_t)) dlsym(RTLD_NEXT, "memalign");
  mm_posix_memalign = (int (*)(void **,size_t,size_t)) dlsym(RTLD_NEXT, "posix_memalign");
  mm_aligned_alloc = (void *(*)(size_t,size_t)) dlsym(RTLD_NEXT, "aligned_alloc");
  mm_valloc = (void *(*)(size_t)) dlsym(RTLD_NEXT, "valloc");
  mm_pvalloc = (void *(*)(size_t)) dlsym(RTLD_NEXT, "pvalloc");
  mm_cfree = (void (*)(void*)) dlsym(RTLD_NEXT, "cfree");
  mm_mallinfo = (struct mallinfo (*)(void)) dlsym(RTLD_NEXT, "mallinfo");
}

#endif
