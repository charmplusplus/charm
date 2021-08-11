// Global Variable Privatization Test - Framework

#ifndef __STDC_FORMAT_MACROS
# define __STDC_FORMAT_MACROS
#endif
#ifndef __STDC_LIMIT_MACROS
# define __STDC_LIMIT_MACROS
#endif
#include <stdint.h>
#include <inttypes.h>
#include <stdio.h>
#include "mpi.h"
#include "framework.h"



void print_test(int & test, int & rank, const char * name)
{
  if (rank == 0)
    printf("Test " test_format ": %s\n", test, name);
}

void print_test_fortran(int & test, int & rank, const char * name, int name_len)
{
  if (rank == 0)
    printf("Test " test_format ": %.*s\n", test, name_len, name);
}


static int print_test_result(int test, int rank, int my_wth, void * ptr, const char * name, int result)
{
  printf(test_format " - [%d](%d) - 0x%012" PRIxPTR " - %s - %s\n",
         test, rank, my_wth, (uintptr_t)ptr, name, result ? "passed" : "failed");
  return !result;
}

void test_privatization(int & failed, int & test, int & rank, int & my_wth, int & operation, int & global)
{
  MPI_Barrier(MPI_COMM_WORLD);

  if (operation == 0)
  {
    global = 0;

    MPI_Barrier(MPI_COMM_WORLD);

    global = rank;

    MPI_Barrier(MPI_COMM_WORLD);

    failed += print_test_result(test, rank, my_wth, &global, "privatization", global == rank);
  }
  else if (operation == 1)
  {
    failed += print_test_result(test, rank, my_wth, &global, "migration", global == rank);
  }

  MPI_Barrier(MPI_COMM_WORLD);

  ++test;
}

void test_skip(int & test, int & rank)
{
  MPI_Barrier(MPI_COMM_WORLD);

  if (rank == 0)
    printf(test_format " - skipped\n", test);

  MPI_Barrier(MPI_COMM_WORLD);

  ++test;
}


static void perform_test_batch_dispatch(int & failed, int & test, int & rank, int & my_wth, int & operation)
{
  if (rank == 0)
    printf("Beginning round of testing.\n");

  perform_test_batch(failed, test, rank, my_wth, operation);

  if (rank == 0)
    printf("Round of testing complete.\n");
}

void privatization_test_framework(void)
{
  int rank;            /* process id */
  int p;               /* number of processes */
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &p);

  int * my_wth_ptr;
  int flag;
  MPI_Comm_get_attr(MPI_COMM_WORLD, AMPI_MY_WTH, &my_wth_ptr, &flag);
  int my_wth = *my_wth_ptr;

  int test = 1;
  int operation;

  int failed_before = 0;
  operation = 0;
  perform_test_batch_dispatch(failed_before, test, rank, my_wth, operation);

#if defined test_migration
  if (rank == 0)
    printf("Requesting migration.\n");

  AMPI_Migrate(AMPI_INFO_LB_SYNC);

  int failed_migration = 0;
  operation = 1;
  perform_test_batch_dispatch(failed_migration, test, rank, my_wth, operation);
#endif

  int failed_after = 0;
  operation = 0;
  perform_test_batch_dispatch(failed_after, test, rank, my_wth, operation);

  if (failed_before != failed_after)
    printf("[%d](%d) Migration caused a test inconsistency.\n", rank, my_wth);

  int failed = failed_before + failed_after;
#if defined test_migration
  failed += failed_migration;
#endif
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
}
