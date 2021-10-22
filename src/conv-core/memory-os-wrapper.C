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

extern void * (*mm_impl_malloc)(size_t);
extern void * (*mm_impl_calloc)(size_t,size_t);
extern void * (*mm_impl_realloc)(void*,size_t);
extern void * (*mm_impl_memalign)(size_t,size_t);
extern int (*mm_impl_posix_memalign)(void **,size_t,size_t);
extern void * (*mm_impl_aligned_alloc)(size_t,size_t);
extern void * (*mm_impl_valloc)(size_t);
extern void * (*mm_impl_pvalloc)(size_t);
extern void (*mm_impl_free)(void*);
extern void (*mm_impl_cfree)(void*);
extern struct mallinfo (*mm_impl_mallinfo)(void);

  
extern char initialize_memory_wrapper_status;
char initialize_memory_wrapper_status;

void initialize_memory_wrapper() {
  initialize_memory_wrapper_status = 1;

  // wait to install these all at once because dlsym calls them, and a mismatch would be bad
  void * (*os_malloc)(size_t) = (void *(*)(size_t)) dlsym(RTLD_NEXT, "malloc");
  void * (*os_calloc)(size_t,size_t) = (void *(*)(size_t,size_t)) dlsym(RTLD_NEXT, "calloc");
  void (*os_free)(void*) = (void (*)(void*)) dlsym(RTLD_NEXT, "free");

  mm_impl_malloc = os_malloc;
  mm_impl_calloc = os_calloc;
  mm_impl_free = os_free;

  mm_impl_realloc = (void *(*)(void*,size_t)) dlsym(RTLD_NEXT, "realloc");
  mm_impl_memalign = (void *(*)(size_t,size_t)) dlsym(RTLD_NEXT, "memalign");
  mm_impl_posix_memalign = (int (*)(void **,size_t,size_t)) dlsym(RTLD_NEXT, "posix_memalign");
  mm_impl_aligned_alloc = (void *(*)(size_t,size_t)) dlsym(RTLD_NEXT, "aligned_alloc");
  mm_impl_valloc = (void *(*)(size_t)) dlsym(RTLD_NEXT, "valloc");
  mm_impl_pvalloc = (void *(*)(size_t)) dlsym(RTLD_NEXT, "pvalloc");
  mm_impl_cfree = (void (*)(void*)) dlsym(RTLD_NEXT, "cfree");
  mm_impl_mallinfo = (struct mallinfo (*)(void)) dlsym(RTLD_NEXT, "mallinfo");
}

#endif
