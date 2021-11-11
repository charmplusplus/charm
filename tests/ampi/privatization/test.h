#ifndef TEST_H_
#define TEST_H_

#include "charm-api.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef int * (*int_ptr_accessor)();

#if defined test_globalvars
extern int extern_global_static;
#if defined test_staticvars
int * get_static_global_static();
int * get_scoped_global_static();
#endif
#endif
#if defined test_threadlocalvars
thread_local extern int extern_threadlocal_static;
#if defined test_staticvars
int * get_static_threadlocal_static();
int * get_scoped_threadlocal_static();
#endif
#endif

#if defined test_sharedlib
#if defined test_globalvars
extern CMI_EXPORT int extern_global_shared;
#if defined test_staticvars
CMI_EXPORT int * get_static_global_shared();
CMI_EXPORT int * get_scoped_global_shared();
#endif
#endif
#if defined test_threadlocalvars
thread_local extern CMI_EXPORT int extern_threadlocal_shared;
#if defined test_staticvars
CMI_EXPORT int * get_static_threadlocal_shared();
CMI_EXPORT int * get_scoped_threadlocal_shared();
#endif
#endif
#endif

#ifdef __cplusplus
}
#endif

#endif
