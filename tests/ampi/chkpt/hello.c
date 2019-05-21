#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define CHKPT_TO_FILE 1 // define to 0 for in-memory checkpointing

int main(int argc, char **argv) {
  int myrank,size,leftnbr,rightnbr;
  int step=0;
  int i;
  double a[2]={.1,.3},b[2]={.5,.7};
  MPI_Status sts;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
  MPI_Comm_size(MPI_COMM_WORLD,&size);

  MPI_Info chkpt_info;
  MPI_Info_create(&chkpt_info);
#if CHKPT_TO_FILE
  MPI_Info_set(chkpt_info, "ampi_checkpoint", "to_file=log");
#else
  MPI_Info_set(chkpt_info, "ampi_checkpoint", "in_memory");
#endif

  for(step=0;step<6;step++){
    leftnbr = (myrank+size-1)%size;
    rightnbr = (myrank+1)%size;
    MPI_Send(a,2,MPI_DOUBLE,rightnbr,0,MPI_COMM_WORLD);
    MPI_Recv(b,2,MPI_DOUBLE,leftnbr,0,MPI_COMM_WORLD,&sts);
    if(myrank==0) printf("[%d]step %d,a={%f,%f},b={%f,%f}\n",myrank,step,a[0],a[1],b[0],b[1]);
    if(step==2){
      AMPI_Migrate(chkpt_info);
    }
  }

  MPI_Info_free(&chkpt_info);
  MPI_Finalize();
  return 0;
}
