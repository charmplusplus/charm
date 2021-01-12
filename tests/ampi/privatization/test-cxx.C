// Global Variable Privatization Test - C++

#include <stdio.h>
#include "mpi.h"
#include "test.h"
#include "test-cxx.h"

#if defined test_dynamiclib
# include <dlfcn.h>
#endif


test_thread_local extern int extern_global_sameobject;
test_thread_local int extern_global_sameobject;
static int * get_extern_global_sameobject()
{
  return &extern_global_sameobject;
}

#if defined test_staticvars
test_thread_local static int static_global_sameobject;
static int * get_static_global_sameobject()
{
  return &static_global_sameobject;
}

static int * get_static_local_sameobject()
{
  test_thread_local static int static_local_sameobject;
  return &static_local_sameobject;
}
#endif

static int * get_extern_global_otherobject()
{
  return &extern_global_otherobject;
}

static int * get_extern_global_staticlibrary()
{
  return &extern_global_staticlibrary;
}

#if defined test_sharedlib
static int * get_extern_global_sharedlibrary()
{
  return &extern_global_sharedlibrary;
}
#endif


void perform_test_batch(int & failed, int & test, int & rank, int & my_wth, int & operation)
{
#if defined test_dynamiclib
  int * extern_global_sharedlibrary_dynamic_ptr = nullptr;
#if defined test_staticvars
  int_ptr_accessor get_static_global_sharedlibrary_dynamic_ptr = nullptr;
  int_ptr_accessor get_static_local_sharedlibrary_dynamic_ptr = nullptr;
#endif
  void * dynamiclib = dlopen("libcxx-" privatization_method_str "-shared-library-dynamic.so", RTLD_NOW);
  if (!dynamiclib)
  {
    fprintf(stderr, "dlopen failed: %s\n", dlerror());
  }
  else
  {
    extern_global_sharedlibrary_dynamic_ptr = (int *)dlsym(dynamiclib, "extern_global_sharedlibrary_dynamic");
    if (!extern_global_sharedlibrary_dynamic_ptr)
      fprintf(stderr, "dlsym failed: %s\n", dlerror());
#if defined test_staticvars
    get_static_global_sharedlibrary_dynamic_ptr = (int_ptr_accessor)dlsym(dynamiclib, "get_static_global_sharedlibrary_dynamic");
    if (!get_static_global_sharedlibrary_dynamic_ptr)
      fprintf(stderr, "dlsym failed: %s\n", dlerror());
    get_static_local_sharedlibrary_dynamic_ptr = (int_ptr_accessor)dlsym(dynamiclib, "get_static_local_sharedlibrary_dynamic");
    if (!get_static_local_sharedlibrary_dynamic_ptr)
      fprintf(stderr, "dlsym failed: %s\n", dlerror());
#endif
  }
#endif


  print_test(test, rank, "extern global, in same object");
  test_privatization(failed, test, rank, my_wth, operation, *get_extern_global_sameobject());
#if defined test_staticvars
  print_test(test, rank, "static global, in same object");
  test_privatization(failed, test, rank, my_wth, operation, *get_static_global_sameobject());
  print_test(test, rank, "static local, in same object");
  test_privatization(failed, test, rank, my_wth, operation, *get_static_local_sameobject());
#endif

  print_test(test, rank, "extern global, in other object");
  test_privatization(failed, test, rank, my_wth, operation, *get_extern_global_otherobject());
#if defined test_staticvars
  print_test(test, rank, "static global, in other object");
  test_privatization(failed, test, rank, my_wth, operation, *get_static_global_otherobject());
  print_test(test, rank, "static local, in other object");
  test_privatization(failed, test, rank, my_wth, operation, *get_static_local_otherobject());
#endif

  print_test(test, rank, "extern global, in static library");
  test_privatization(failed, test, rank, my_wth, operation, *get_extern_global_staticlibrary());
#if defined test_staticvars
  print_test(test, rank, "static global, in static library");
  test_privatization(failed, test, rank, my_wth, operation, *get_static_global_staticlibrary());
  print_test(test, rank, "static local, in static library");
  test_privatization(failed, test, rank, my_wth, operation, *get_static_local_staticlibrary());
#endif

#if defined test_sharedlib
  print_test(test, rank, "extern global, in shared library");
  test_privatization(failed, test, rank, my_wth, operation, *get_extern_global_sharedlibrary());
#if defined test_staticvars
  print_test(test, rank, "static global, in shared library");
  test_privatization(failed, test, rank, my_wth, operation, *get_static_global_sharedlibrary());
  print_test(test, rank, "static local, in shared library");
  test_privatization(failed, test, rank, my_wth, operation, *get_static_local_sharedlibrary());
#endif
#endif

#if defined test_dynamiclib
  print_test(test, rank, "extern global, in shared library, linked dynamically");
  if (extern_global_sharedlibrary_dynamic_ptr)
    test_privatization(failed, test, rank, my_wth, operation, *extern_global_sharedlibrary_dynamic_ptr);
  else
    test_skip(test, rank);
#if defined test_staticvars
  print_test(test, rank, "static global, in shared library, linked dynamically");
  if (get_static_global_sharedlibrary_dynamic_ptr)
    test_privatization(failed, test, rank, my_wth, operation, *get_static_global_sharedlibrary_dynamic_ptr());
  else
    test_skip(test, rank);
  print_test(test, rank, "static local, in shared library, linked dynamically");
  if (get_static_local_sharedlibrary_dynamic_ptr)
    test_privatization(failed, test, rank, my_wth, operation, *get_static_local_sharedlibrary_dynamic_ptr());
  else
    test_skip(test, rank);
#endif
#endif

#if defined test_dynamiclib
  dlclose(dynamiclib);
#endif
}


#if defined test_migration
test_thread_local extern int global_myrank;
test_thread_local int global_myrank;
#endif

static void privatization_about_to_migrate()
{
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  printf("[%d] About to migrate.\n", rank);

#if defined test_migration
  if (rank != global_myrank)
  {
    printf("[%d] Globals incorrect when about to migrate!\n", rank);
  }
#endif
}
static void privatization_just_migrated()
{
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  printf("[%d] Just migrated.\n", rank);

#if defined test_migration
  if (rank != global_myrank)
  {
    printf("[%d] Globals incorrect when just migrated!\n", rank);
  }
#endif
}

int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);

#if defined test_migration
  MPI_Comm_rank(MPI_COMM_WORLD, &global_myrank);
#endif
  AMPI_Register_about_to_migrate(privatization_about_to_migrate);
  AMPI_Register_just_migrated(privatization_just_migrated);

  privatization_test_framework();

  MPI_Finalize();

  return 0;
}
