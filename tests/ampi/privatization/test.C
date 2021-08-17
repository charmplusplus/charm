// Global Variable Privatization Test - C++

#include <stdio.h>
#include "mpi.h"
#include "framework.h"
#include "test.h"

#if defined test_dynamiclib
# include <dlfcn.h>
#endif


#if defined test_globalvars
static int * get_extern_global_static()
{
  return &extern_global_static;
}
#endif
#if defined test_threadlocalvars
static int * get_extern_threadlocal_static()
{
  return &extern_threadlocal_static;
}
#endif

#if defined test_sharedlib
#if defined test_globalvars
static int * get_extern_global_shared()
{
  return &extern_global_shared;
}
#endif
#if defined test_threadlocalvars
static int * get_extern_threadlocal_shared()
{
  return &extern_threadlocal_shared;
}
#endif
#endif


void perform_test_batch(int & failed, int & test, int & rank, int & my_wth, int & operation)
{
#if defined test_dynamiclib

#if defined test_globalvars
  int * extern_global_dynamic_ptr = nullptr;
#if defined test_staticvars
  int_ptr_accessor get_static_global_dynamic_ptr = nullptr;
  int_ptr_accessor get_scoped_global_dynamic_ptr = nullptr;
#endif
#endif
#if defined test_threadlocalvars
  int * extern_threadlocal_dynamic_ptr = nullptr;
#if defined test_staticvars
  int_ptr_accessor get_static_threadlocal_dynamic_ptr = nullptr;
  int_ptr_accessor get_scoped_threadlocal_dynamic_ptr = nullptr;
#endif
#endif

  void * dynamiclib = dlopen("lib" privatization_method_str "-vars-dynamic-cxx.so", RTLD_NOW);
  if (!dynamiclib)
  {
    fprintf(stderr, "dlopen failed: %s\n", dlerror());
  }
  else
  {
#if defined test_globalvars
    extern_global_dynamic_ptr = (int *)dlsym(dynamiclib, "extern_global_dynamic");
    if (!extern_global_dynamic_ptr)
      fprintf(stderr, "dlsym failed: %s\n", dlerror());
#if defined test_staticvars
    get_static_global_dynamic_ptr = (int_ptr_accessor)dlsym(dynamiclib, "get_static_global_dynamic");
    if (!get_static_global_dynamic_ptr)
      fprintf(stderr, "dlsym failed: %s\n", dlerror());
    get_scoped_global_dynamic_ptr = (int_ptr_accessor)dlsym(dynamiclib, "get_scoped_global_dynamic");
    if (!get_scoped_global_dynamic_ptr)
      fprintf(stderr, "dlsym failed: %s\n", dlerror());
#endif
#endif
#if defined test_threadlocalvars
    extern_threadlocal_dynamic_ptr = (int *)dlsym(dynamiclib, "extern_threadlocal_dynamic");
    if (!extern_threadlocal_dynamic_ptr)
      fprintf(stderr, "dlsym failed: %s\n", dlerror());
#if defined test_staticvars
    get_static_threadlocal_dynamic_ptr = (int_ptr_accessor)dlsym(dynamiclib, "get_static_threadlocal_dynamic");
    if (!get_static_threadlocal_dynamic_ptr)
      fprintf(stderr, "dlsym failed: %s\n", dlerror());
    get_scoped_threadlocal_dynamic_ptr = (int_ptr_accessor)dlsym(dynamiclib, "get_scoped_threadlocal_dynamic");
    if (!get_scoped_threadlocal_dynamic_ptr)
      fprintf(stderr, "dlsym failed: %s\n", dlerror());
#endif
#endif
  }
#endif


#if defined test_globalvars
  print_test(test, rank, "extern global, static linkage");
  test_privatization(failed, test, rank, my_wth, operation, *get_extern_global_static());
#if defined test_staticvars
  print_test(test, rank, "static global, static linkage");
  test_privatization(failed, test, rank, my_wth, operation, *get_static_global_static());
  print_test(test, rank, "scoped global, static linkage");
  test_privatization(failed, test, rank, my_wth, operation, *get_scoped_global_static());
#endif
#endif
#if defined test_threadlocalvars
  print_test(test, rank, "extern thread-local, static linkage");
  test_privatization(failed, test, rank, my_wth, operation, *get_extern_threadlocal_static());
#if defined test_staticvars
  print_test(test, rank, "static thread-local, static linkage");
  test_privatization(failed, test, rank, my_wth, operation, *get_static_threadlocal_static());
  print_test(test, rank, "scoped thread-local, static linkage");
  test_privatization(failed, test, rank, my_wth, operation, *get_scoped_threadlocal_static());
#endif
#endif

#if defined test_sharedlib
#if defined test_globalvars
  print_test(test, rank, "extern global, shared linkage");
  test_privatization(failed, test, rank, my_wth, operation, *get_extern_global_shared());
#if defined test_staticvars
  print_test(test, rank, "static global, shared linkage");
  test_privatization(failed, test, rank, my_wth, operation, *get_static_global_shared());
  print_test(test, rank, "scoped global, shared linkage");
  test_privatization(failed, test, rank, my_wth, operation, *get_scoped_global_shared());
#endif
#endif
#if defined test_threadlocalvars
  print_test(test, rank, "extern thread-local, shared linkage");
  test_privatization(failed, test, rank, my_wth, operation, *get_extern_threadlocal_shared());
#if defined test_staticvars
  print_test(test, rank, "static thread-local, shared linkage");
  test_privatization(failed, test, rank, my_wth, operation, *get_static_threadlocal_shared());
  print_test(test, rank, "scoped thread-local, shared linkage");
  test_privatization(failed, test, rank, my_wth, operation, *get_scoped_threadlocal_shared());
#endif
#endif
#endif

#if defined test_dynamiclib
#if defined test_globalvars
  print_test(test, rank, "extern global, dynamic linkage");
  if (extern_global_dynamic_ptr)
    test_privatization(failed, test, rank, my_wth, operation, *extern_global_dynamic_ptr);
  else
    test_skip(test, rank);
#if defined test_staticvars
  print_test(test, rank, "static global, dynamic linkage");
  if (get_static_global_dynamic_ptr)
    test_privatization(failed, test, rank, my_wth, operation, *get_static_global_dynamic_ptr());
  else
    test_skip(test, rank);
  print_test(test, rank, "scoped global, dynamic linkage");
  if (get_scoped_global_dynamic_ptr)
    test_privatization(failed, test, rank, my_wth, operation, *get_scoped_global_dynamic_ptr());
  else
    test_skip(test, rank);
#endif
#endif
#if defined test_threadlocalvars
  print_test(test, rank, "extern thread-local, dynamic linkage");
  if (extern_threadlocal_dynamic_ptr)
    test_privatization(failed, test, rank, my_wth, operation, *extern_threadlocal_dynamic_ptr);
  else
    test_skip(test, rank);
#if defined test_staticvars
  print_test(test, rank, "static thread-local, dynamic linkage");
  if (get_static_threadlocal_dynamic_ptr)
    test_privatization(failed, test, rank, my_wth, operation, *get_static_threadlocal_dynamic_ptr());
  else
    test_skip(test, rank);
  print_test(test, rank, "scoped thread-local, dynamic linkage");
  if (get_scoped_threadlocal_dynamic_ptr)
    test_privatization(failed, test, rank, my_wth, operation, *get_scoped_threadlocal_dynamic_ptr());
  else
    test_skip(test, rank);
#endif
#endif
#endif

#if defined test_dynamiclib
  dlclose(dynamiclib);
#endif
}


#if defined test_migration
#if defined test_globalvars
extern int global_myrank;
int global_myrank;
#endif
#if defined test_threadlocalvars
thread_local extern int threadlocal_myrank;
thread_local int threadlocal_myrank;
#endif
#endif

static void privatization_about_to_migrate()
{
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  printf("[%d] About to migrate.\n", rank);

#if defined test_migration
#if defined test_globalvars
  if (rank != global_myrank)
  {
    printf("[%d] Globals incorrect when about to migrate!\n", rank);
  }
#endif
#if defined test_threadlocalvars
  if (rank != threadlocal_myrank)
  {
    printf("[%d] Thread-locals incorrect when about to migrate!\n", rank);
  }
#endif
#endif
}
static void privatization_just_migrated()
{
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  printf("[%d] Just migrated.\n", rank);

#if defined test_migration
#if defined test_globalvars
  if (rank != global_myrank)
  {
    printf("[%d] Globals incorrect when just migrated!\n", rank);
  }
#endif
#if defined test_threadlocalvars
  if (rank != threadlocal_myrank)
  {
    printf("[%d] Thread-locals incorrect when just migrated!\n", rank);
  }
#endif
#endif
}

int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);

#if defined test_migration
#if defined test_globalvars
  MPI_Comm_rank(MPI_COMM_WORLD, &global_myrank);
#endif
#if defined test_threadlocalvars
  MPI_Comm_rank(MPI_COMM_WORLD, &threadlocal_myrank);
#endif
#endif

  AMPI_Register_about_to_migrate(privatization_about_to_migrate);
  AMPI_Register_just_migrated(privatization_just_migrated);

  privatization_test_framework();

  MPI_Finalize();

  return 0;
}
