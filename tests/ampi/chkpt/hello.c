#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char **argv) {
  int myrank,size,leftnbr,rightnbr;
  int step=0;
  int i;
  double a[2]={.1,.3},b[2]={.5,.7};
  MPI_Status sts;
  MPI_Info hints;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
  MPI_Comm_size(MPI_COMM_WORLD,&size);

  /* Set up MPI_Info hints for AMPI_Migrate() */
  MPI_Info_create(&hints);
  MPI_Info_set(hints, "ampi_checkpoint", "to_file=log");

  for(step=0;step<6;step++){
    leftnbr = (myrank+size-1)%size;
    rightnbr = (myrank+1)%size;
    MPI_Send(a,2,MPI_DOUBLE,rightnbr,0,MPI_COMM_WORLD);
    MPI_Recv(b,2,MPI_DOUBLE,leftnbr,0,MPI_COMM_WORLD,&sts);
    if(myrank==0) printf("[%d]step %d,a={%f,%f},b={%f,%f}\n",myrank,step,a[0],a[1],b[0],b[1]);
    if(step==2){
      AMPI_Migrate(hints);
    }
  }

  MPI_Finalize();
  return 0;
}
