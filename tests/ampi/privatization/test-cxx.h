#ifndef TEST_CXX_H_
#define TEST_CXX_H_

#if defined test_using_tlsglobals
# define test_thread_local thread_local
#else
# define test_thread_local
#endif

typedef int * (*int_ptr_accessor)();

test_thread_local extern int extern_global_otherobject;
#if defined test_staticvars
extern int * get_static_global_otherobject();
extern int * get_static_local_otherobject();
#endif

test_thread_local extern int extern_global_staticlibrary;
#if defined test_staticvars
extern int * get_static_global_staticlibrary();
extern int * get_static_local_staticlibrary();
#endif

#if defined test_sharedlib
test_thread_local extern int extern_global_sharedlibrary;
#if defined test_staticvars
extern int * get_static_global_sharedlibrary();
extern int * get_static_local_sharedlibrary();
#endif
#endif

#endif
