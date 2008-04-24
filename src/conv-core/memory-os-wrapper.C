#include <sys/types.h>
#include <dlfcn.h>

extern "C" void * (*mm_malloc)(size_t);
extern "C" void * (*mm_calloc)(size_t,size_t);
extern "C" void * (*mm_realloc)(void*,size_t);
extern "C" void * (*mm_memalign)(size_t,size_t);
extern "C" void * (*mm_valloc)(size_t);
extern "C" void (*mm_free)(void*);
extern "C" void (*mm_cfree)(void*);

  
extern "C" void initialize_memory_wrapper() {
  mm_malloc = (void *(*)(size_t)) dlsym(RTLD_NEXT, "malloc");
  mm_realloc = (void *(*)(void*,size_t)) dlsym(RTLD_NEXT, "realloc");
  mm_calloc = (void *(*)(size_t,size_t)) dlsym(RTLD_NEXT, "calloc");
  mm_memalign = (void *(*)(size_t,size_t)) dlsym(RTLD_NEXT, "memalign");
  mm_valloc = (void *(*)(size_t)) dlsym(RTLD_NEXT, "valloc");
  mm_free = (void (*)(void*)) dlsym(RTLD_NEXT, "free");
  mm_cfree = (void (*)(void*)) dlsym(RTLD_NEXT, "cfree");
}
