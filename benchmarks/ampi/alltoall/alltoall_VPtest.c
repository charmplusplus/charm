#include "mpi.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "converse.h" // for CmiMemoryUsage etc.

int max_msgs = 2;

char hash(int p, int i) { return (p * (100 - p) * 770 + i * i * 3) % 127; }

int main(int argc, char** argv)
{
  int my_id;     /* process id */
  int p;         /* number of processes */
  char* message; /* storage for the message */
  int i, j, k, msg_size;
  MPI_Status status; /* return status for receive */
  double startTime = 0;
  double elapsed_time_sec;
  char *sndbuf, *recvbuf;
  int memory_before, memory_after;
  int memory_diff, local_memory_max;
  int memory_min_small, memory_max_small, memory_min_medium, memory_max_medium,
      memory_min_normal, memory_max_normal, memory_min_large, memory_max_large;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_id);
  MPI_Comm_size(MPI_COMM_WORLD, &p);

  if (argc < 2)
  {
    fprintf(stderr, "need msg size as params\n");
    goto EXIT;
  }

  if (sscanf(argv[1], "%d", &msg_size) < 1)
  {
    fprintf(stderr, "need msg size as params\n");
    goto EXIT;
  }
  message = (char*)malloc(msg_size);

  if (argc > 2) sscanf(argv[2], "%d", &max_msgs);

  /* don't start timer until everybody is ok */
  MPI_Barrier(MPI_COMM_WORLD);

  if (my_id == 0)
  {
    int flag = 0;
  }
  sndbuf = (char*)malloc(msg_size * sizeof(char) * p);
  recvbuf = (char*)malloc(msg_size * sizeof(char) * p);

  for (j = 0; j < p; j++) memset(sndbuf + j * msg_size, hash(my_id, j), msg_size);
  memset(recvbuf, 0, msg_size * p);

  // Test Long
  if (1)
  {
    // warm up, not instrumented
    for (i = 0; i < max_msgs; i++)
    {
      AMPI_Alltoall_long(sndbuf, msg_size, MPI_CHAR, recvbuf, msg_size, MPI_CHAR,
                         MPI_COMM_WORLD);
    }

    memset(recvbuf, 0, msg_size * p);
    MPI_Barrier(MPI_COMM_WORLD);
    CmiResetMaxMemory();
    memory_before = CmiMemoryUsage();  // initial memory usage
    MPI_Barrier(MPI_COMM_WORLD);

    if (my_id == 0)
    {
      startTime = MPI_Wtime();
    }
    for (i = 0; i < max_msgs; i++)
    {
      AMPI_Alltoall_long(sndbuf, msg_size, MPI_CHAR, recvbuf, msg_size, MPI_CHAR,
                         MPI_COMM_WORLD);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (my_id == 0)
    {
      elapsed_time_sec = (MPI_Wtime() - startTime) / max_msgs;
    }
    memory_after = CmiMemoryUsage();

    if (CmiMaxMemoryUsage() < memory_before)
      local_memory_max = 0;
    else
      local_memory_max = CmiMaxMemoryUsage() - memory_before;

    // Reduce MAX here
    MPI_Reduce(&local_memory_max, &memory_max_large, 1,
               MPI_UNSIGNED_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_memory_max, &memory_min_large, 1,
               MPI_UNSIGNED_LONG, MPI_MIN, 0, MPI_COMM_WORLD);

    if (my_id == 0)
      printf("Large Mem Max Usage=%8d Kb\tMin Usage=%8d Kb\tVP=%d\tMsgSize=%d\t Elapsed Time for AlltoAll_long=%8.3f us\n",
             (memory_max_large) / 1024, (memory_min_large) / 1024, p, msg_size, elapsed_time_sec * 1e6);

    for (j = 0; j < p; j++)
      for (k = 0; k < msg_size; k++)
      {
        assert(*(recvbuf + j * msg_size + k) == hash(j, my_id));
      }
  }

// Test Short
#if 0
  {
    // warm up, not instrumented
    for(i=0; i<max_msgs; i++) {
      AMPI_Alltoall_short(sndbuf, msg_size, MPI_CHAR, recvbuf, msg_size, MPI_CHAR, MPI_COMM_WORLD);
    }

    memset(recvbuf,0,msg_size*p);
    MPI_Barrier(MPI_COMM_WORLD);
    CmiResetMaxMemory();
    memory_before = CmiMemoryUsage();  // initial memory usage
    MPI_Barrier(MPI_COMM_WORLD);

    if(my_id == 0){
      startTime = MPI_Wtime();
    }
    for(i=0; i<max_msgs; i++) {
      AMPI_Alltoall_short(sndbuf, msg_size, MPI_CHAR, recvbuf, msg_size, MPI_CHAR, MPI_COMM_WORLD);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (my_id == 0)
    {
      elapsed_time_sec = (MPI_Wtime() - startTime) / max_msgs;
    }
    memory_after = CmiMemoryUsage();

    if (CmiMaxMemoryUsage() < memory_before)
      local_memory_max = 0;
    else
      local_memory_max = CmiMaxMemoryUsage() - memory_before;

    // Reduce MAX here
    MPI_Reduce(&local_memory_max, &memory_max_small, 1, MPI_UNSIGNED_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_memory_max, &memory_min_small, 1, MPI_UNSIGNED_LONG, MPI_MIN, 0, MPI_COMM_WORLD);

    if(my_id==0)
      printf("Small Mem Max Usage=%8d Kb\tMin Usage=%8d Kb\tVP=%d\tMsgSize=%d\t Elapsed Time for AlltoAll_short=%8.3f us\n",
          (memory_max_small) / 1024, (memory_min_small) / 1024, p, msg_size, elapsed_time_sec * 1e6);

    for(j=0;j<p;j++)
      for(k=0;k<msg_size;k++)
      {
        assert(*(recvbuf+j*msg_size+k) == hash(j,my_id) );
      }
  }
#endif

  // Test Medium
  if (1)
  {
    // warm up, not instrumented
    for (i = 0; i < max_msgs; i++)
    {
      AMPI_Alltoall_medium(sndbuf, msg_size, MPI_CHAR, recvbuf, msg_size, MPI_CHAR,
                           MPI_COMM_WORLD);
    }

    memset(recvbuf, 0, msg_size * p);
    MPI_Barrier(MPI_COMM_WORLD);
    CmiResetMaxMemory();
    memory_before = CmiMemoryUsage();  // initial memory usage
    MPI_Barrier(MPI_COMM_WORLD);

    if (my_id == 0)
    {
      startTime = MPI_Wtime();
    }
    for (i = 0; i < max_msgs; i++)
    {
      AMPI_Alltoall_medium(sndbuf, msg_size, MPI_CHAR, recvbuf, msg_size, MPI_CHAR,
                           MPI_COMM_WORLD);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (my_id == 0)
    {
      elapsed_time_sec = (MPI_Wtime() - startTime) / max_msgs;
    }
    memory_after = CmiMemoryUsage();

    if (CmiMaxMemoryUsage() < memory_before)
      local_memory_max = 0;
    else
      local_memory_max = CmiMaxMemoryUsage() - memory_before;

    // Reduce MAX here
    MPI_Reduce(&local_memory_max, &memory_max_medium, 1,
               MPI_UNSIGNED_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_memory_max, &memory_min_medium, 1,
               MPI_UNSIGNED_LONG, MPI_MIN, 0, MPI_COMM_WORLD);

    if (my_id == 0)
      printf("Medium Mem Max Usage=%8d Kb\tMin Usage=%8d Kb\tVP=%d\tMsgSize=%d\tElapsed Time for AlltoAll_medium=%8.3f us\n",
             (memory_max_medium) / 1024, (memory_min_medium) / 1024, p, msg_size, elapsed_time_sec * 1e6);

    for (j = 0; j < p; j++)
      for (k = 0; k < msg_size; k++)
      {
        assert(*(recvbuf + j * msg_size + k) == hash(j, my_id));
      }
  }

  // Test standard version
  {
    // warm up, not instrumented
    for (i = 0; i < max_msgs; i++)
    {
      MPI_Alltoall(sndbuf, msg_size, MPI_CHAR, recvbuf, msg_size, MPI_CHAR,
                   MPI_COMM_WORLD);
    }

    memset(recvbuf, 0, msg_size * p);
    MPI_Barrier(MPI_COMM_WORLD);
    CmiResetMaxMemory();
    memory_before = CmiMemoryUsage();  // initial memory usage
    MPI_Barrier(MPI_COMM_WORLD);

    if (my_id == 0)
    {
      startTime = MPI_Wtime();
    }
    for (i = 0; i < max_msgs; i++)
    {
      MPI_Alltoall(sndbuf, msg_size, MPI_CHAR, recvbuf, msg_size, MPI_CHAR,
                   MPI_COMM_WORLD);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (my_id == 0)
    {
      elapsed_time_sec = (MPI_Wtime() - startTime) / max_msgs;
    }
    memory_after = CmiMemoryUsage();

    if (CmiMaxMemoryUsage() < memory_before)
      local_memory_max = 0;
    else
      local_memory_max = CmiMaxMemoryUsage() - memory_before;

    // Reduce MAX here
    MPI_Reduce(&local_memory_max, &memory_max_normal, 1,
               MPI_UNSIGNED_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_memory_max, &memory_min_normal, 1,
               MPI_UNSIGNED_LONG, MPI_MIN, 0, MPI_COMM_WORLD);


  if (my_id == 0)
      printf("Startd Mem Max Usage=%8d Kb\tMin Usage=%8d Kb\tVP=%d\tMsgSize=%d\tElapsed Time for AlltoAll (standard version)=%8.3fus\n",
             (memory_max_normal) / 1024, (memory_min_normal) / 1024, p, msg_size, elapsed_time_sec * 1e6);

    for (j = 0; j < p; j++)
      for (k = 0; k < msg_size; k++)
      {
        assert(*(recvbuf + j * msg_size + k) == hash(j, my_id));
      }
  }

  free(sndbuf);
  free(recvbuf);

EXIT:
  MPI_Finalize();
  return 0;
}
