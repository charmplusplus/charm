// Global Variable Privatization Test - Framework

#include <stdio.h>
#include "mpi.h"
#include "test.h"


static int print_test_result(int rank, int my_wth, const char * name, bool result)
{
  printf(result_indent "[%d](%d) %s %s\n", rank, my_wth, name, result ? "passed" : "failed");
  return result ? 0 : 1;
}

void test_privatization(int & failed, int & rank, int & my_wth, int & global)
{
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
}


void privatization_test_framework(void)
{
  int rank;            /* process id */
  int p;               /* number of processes */
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &p);

  int my_wth, flag;
  MPI_Comm_get_attr(MPI_COMM_WORLD, AMPI_MY_WTH, &my_wth, &flag);


  int failed_before = 0;
  perform_test_batch(failed_before, rank, my_wth);

  if (rank == 0) printf("Requesting migration.\n");
  AMPI_Migrate(AMPI_INFO_LB_SYNC);

  int failed_after = 0;
  perform_test_batch(failed_after, rank, my_wth);

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
}
