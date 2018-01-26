// Global Variable Privatization Test

#include "test.h"

#include <stdio.h>

#if defined test_dynamiclib
# include <dlfcn.h>
#endif

#include "mpi.h"


static int print_test_result(int rank, int my_wth, const char * name, bool result)
{
  printf(result_indent "[%d](%d) %s %s\n", rank, my_wth, name, result ? "passed" : "failed");
  return result ? 0 : 1;
}

static int test_privatization(int rank, int my_wth, int & global)
{
  int failed = 0;

  MPI_Barrier(MPI_COMM_WORLD);

  global = 0;

  MPI_Barrier(MPI_COMM_WORLD);

  if (rank == 0)
    global = 1;

  MPI_Barrier(MPI_COMM_WORLD);

  failed += print_test_result(rank, my_wth, "single write test", global == (rank == 0 ? 1 : 0));

  MPI_Barrier(MPI_COMM_WORLD);

  global = 0;

  MPI_Barrier(MPI_COMM_WORLD);

  global = rank;

  MPI_Barrier(MPI_COMM_WORLD);

  failed += print_test_result(rank, my_wth, "many write test", global == rank);

  MPI_Barrier(MPI_COMM_WORLD);

  return failed;
}


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


static int perform_test_batch(int rank, int my_wth)
{
  if (rank == 0) printf("Beginning round of testing.\n");

#if defined test_dynamiclib
  int * extern_global_sharedlibrary_dynamic_ptr = nullptr;
#if defined test_staticvars
  int_ptr_accessor get_static_global_sharedlibrary_dynamic_ptr = nullptr;
  int_ptr_accessor get_static_local_sharedlibrary_dynamic_ptr = nullptr;
#endif
  void * dynamiclib = dlopen("lib" privatization_method_str "-shared-library-dynamic.so", RTLD_NOW);
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

  int failed = 0;

  if (rank == 0) printf("Testing: extern global, in same object\n");
  failed += test_privatization(rank, my_wth, *get_extern_global_sameobject());
#if defined test_staticvars
  if (rank == 0) printf("Testing: static global, in same object\n");
  failed += test_privatization(rank, my_wth, *get_static_global_sameobject());
  if (rank == 0) printf("Testing: static local, in same object\n");
  failed += test_privatization(rank, my_wth, *get_static_local_sameobject());
#endif

  if (rank == 0) printf("Testing: extern global, in other object\n");
  failed += test_privatization(rank, my_wth, *get_extern_global_otherobject());
#if defined test_staticvars
  if (rank == 0) printf("Testing: static global, in other object\n");
  failed += test_privatization(rank, my_wth, *get_static_global_otherobject());
  if (rank == 0) printf("Testing: static local, in other object\n");
  failed += test_privatization(rank, my_wth, *get_static_local_otherobject());
#endif

  if (rank == 0) printf("Testing: extern global, in static library\n");
  failed += test_privatization(rank, my_wth, *get_extern_global_staticlibrary());
#if defined test_staticvars
  if (rank == 0) printf("Testing: static global, in static library\n");
  failed += test_privatization(rank, my_wth, *get_static_global_staticlibrary());
  if (rank == 0) printf("Testing: static local, in static library\n");
  failed += test_privatization(rank, my_wth, *get_static_local_staticlibrary());
#endif

#if defined test_sharedlib
  if (rank == 0) printf("Testing: extern global, in shared library\n");
  failed += test_privatization(rank, my_wth, *get_extern_global_sharedlibrary());
#if defined test_staticvars
  if (rank == 0) printf("Testing: static global, in shared library\n");
  failed += test_privatization(rank, my_wth, *get_static_global_sharedlibrary());
  if (rank == 0) printf("Testing: static local, in shared library\n");
  failed += test_privatization(rank, my_wth, *get_static_local_sharedlibrary());
#endif
#endif

#if defined test_dynamiclib
  if (rank == 0) printf("Testing: extern global, in shared library, linked dynamically\n");
  if (extern_global_sharedlibrary_dynamic_ptr)
    failed += test_privatization(rank, my_wth, *extern_global_sharedlibrary_dynamic_ptr);
  else
    if (rank == 0) printf(result_indent "Skipped.\n");
#if defined test_staticvars
  if (rank == 0) printf("Testing: static global, in shared library, linked dynamically\n");
  if (get_static_global_sharedlibrary_dynamic_ptr)
    failed += test_privatization(rank, my_wth, *get_static_global_sharedlibrary_dynamic_ptr());
  else
    if (rank == 0) printf(result_indent "Skipped.\n");
  if (rank == 0) printf("Testing: static local, in shared library, linked dynamically\n");
  if (get_static_local_sharedlibrary_dynamic_ptr)
    failed += test_privatization(rank, my_wth, *get_static_local_sharedlibrary_dynamic_ptr());
  else
    if (rank == 0) printf(result_indent "Skipped.\n");
#endif
#endif

#if defined test_dynamiclib
  dlclose(dynamiclib);
#endif

  if (rank == 0) printf("Round of testing complete.\n");

  return failed;
}


int main(int argc, char **argv)
{
  int rank;            /* process id */
  int p;               /* number of processes */

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &p);

  int my_wth, flag;
  MPI_Comm_get_attr(MPI_COMM_WORLD, AMPI_MY_WTH, &my_wth, &flag);


  int failed_before = perform_test_batch(rank, my_wth);

  if (rank == 0) printf("Requesting migration.\n");
  AMPI_Migrate(AMPI_INFO_LB_SYNC);

  int failed_after = perform_test_batch(rank, my_wth);

  if (failed_before != failed_after) printf("[%d](%d) Migration caused a test inconsistency.\n", rank, my_wth);

  int failed = failed_before + failed_after;
  int total_failures = 0;
  MPI_Reduce(&failed, &total_failures, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

  if (rank == 0)
  {
    if (total_failures > 0)
      printf("%d tests failed.\n", total_failures);
    else
      printf("All tests passed.\n");
  }


  if (total_failures > 0)
    MPI_Abort(MPI_COMM_WORLD, 1);

  MPI_Finalize();

  return total_failures > 0;
}
