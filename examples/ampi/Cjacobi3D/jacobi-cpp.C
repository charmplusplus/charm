#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"
#include "charm++.h"

#if CMK_BLUEGENE_CHARM
extern void BgPrintf(char *);
#define BGPRINTF(x)  if (thisIndex == 0) BgPrintf(x);
#else
#define BGPRINTF(x)
#endif

int DIM, NX, NY, NZ;

class chunk {
 public:
  int dim;
  int xidx, yidx, zidx;
  int xm, xp, ym, yp, zm, zp;
  double ***t; //[DIM+2][DIM+2][DIM+2];
  double *sbxm; //[DIM*DIM];
  double *sbxp; //[DIM*DIM];
  double *sbym; //[DIM*DIM];
  double *sbyp; //[DIM*DIM];
  double *sbzm; //[DIM*DIM];
  double *sbzp; //[DIM*DIM];
  double *rbxm; //[DIM*DIM];
  double *rbxp; //[DIM*DIM];
  double *rbym; //[DIM*DIM];
  double *rbyp; //[DIM*DIM];
  double *rbzm; //[DIM*DIM];
  double *rbzp; //[DIM*DIM];

 private:
  void create(){
    t = new double** [dim+2];
    for(int i=0;i<dim+2;i++){
      t[i] = new double* [dim+2];
      for(int j=0;j<dim+2;j++){
	t[i][j] = new double [dim+2];
      }
    }
    sbxm = new double [dim*dim];
    sbxp = new double [dim*dim];
    sbym = new double [dim*dim];
    sbyp = new double [dim*dim];
    sbzm = new double [dim*dim];
    sbzp = new double [dim*dim];
    rbxm = new double [dim*dim];
    rbxp = new double [dim*dim];
    rbym = new double [dim*dim];
    rbyp = new double [dim*dim];
    rbzm = new double [dim*dim];
    rbzp = new double [dim*dim];
  }
  void destroy(){
    for(int i=0;i<dim+2;i++){
      for(int j=0;j<dim+2;j++){
	delete [] t[i][j];
      }
      delete [] t[i];
    }
    delete [] t;
    delete [] sbxm;
    delete [] sbxp;
    delete [] sbym;
    delete [] sbyp;
    delete [] sbzm;
    delete [] sbzp;
    delete [] rbxm;
    delete [] rbxp;
    delete [] rbym;
    delete [] rbyp;
    delete [] rbzm;
    delete [] rbzp;
  }
  inline int index1d(int ix, int iy, int iz){
    return NY*NZ*ix + NZ*iy + iz;
  }
  
  inline void index3d(int index, int& ix, int& iy, int& iz){
    ix = index/(NY*NZ);
    iy = (index%(NY*NZ))/NZ;
    iz = index%NZ;
  }
  
 public:
  chunk(int d){
    dim = d;
    create();
  }
  ~chunk(){
    destroy();
  }
  void pup(PUP::er& p){
    p|dim;
    p|xidx;p|yidx;p|zidx;
    p|xp;p|xm;p|yp;p|ym;p|zp;p|zm;
    if(p.isUnpacking())
      create();
    for(int i=0;i<dim+2;i++){
      for(int j=0;j<dim+2;j++){
	p(t[i][j],dim+2);
      }
    }
    p(sbxm,dim*dim);
    p(sbxp,dim*dim);
    p(sbym,dim*dim);
    p(sbyp,dim*dim);
    p(sbzm,dim*dim);
    p(sbzp,dim*dim);
    p(rbxm,dim*dim);
    p(rbxp,dim*dim);
    p(rbym,dim*dim);
    p(rbyp,dim*dim);
    p(rbzm,dim*dim);
    p(rbzp,dim*dim);
    if(p.isDeleting())
      destroy();
  }
  void copyin(){
    int i, j;
    int l = 0;
    for(i=1;i<=dim;i++)
      for(j=1;j<=dim;j++,l++){
	t[0][i][j] = sbxm[l];
	t[dim+1][i][j] = sbxp[l];
	t[i][0][j] = sbym[l];
	t[i][dim+1][j] = sbyp[l];
	t[i][j][0] = sbzm[l];
	t[i][j][dim+1] = sbzp[l];
      }
  }

  void copyout(){
    int i, j;
    int l = 0;
    for(i=1;i<=dim;i++)
      for(j=1;j<=dim;j++,l++){
	sbxm[l] = t[1][i][j];
	sbxp[l] = t[dim][i][j];
	sbym[l] = t[i][1][j];
	sbyp[l] = t[i][dim][j];
	sbzm[l] = t[i][j][1];
	sbzp[l] = t[i][j][dim];
      }
  }

  void indexing(int rank){
    index3d(rank,xidx,yidx,zidx);
    xp = index1d((xidx+1)%NX,yidx,zidx);
    xm = index1d((xidx+NX-1)%NX,yidx,zidx);
    yp = index1d(xidx,(yidx+1)%NY,zidx);
    ym = index1d(xidx,(yidx+NY-1)%NY,zidx);
    zp = index1d(xidx,yidx,(zidx+1)%NZ);
    zm = index1d(xidx,yidx,(zidx+NZ-1)%NZ);
  }
};

#define abs(x) ((x)<0.0 ? -(x) : (x))

int main(int ac, char** av)
{
  int i,j,k,m,cidx;
  int iter, niter;
  MPI_Status status;
  double error, tval, maxerr, tmpmaxerr, starttime, endtime, itertime;
  chunk *cp;
  int thisIndex, ierr, nblocks;

  MPI_Init(&ac, &av);
  MPI_Comm_rank(MPI_COMM_WORLD, &thisIndex);
  MPI_Comm_size(MPI_COMM_WORLD, &nblocks);

  if (ac < 5) {
    if (thisIndex == 0)
      printf("Usage: jacobi DIM X Y Z [nIter].\n");
    MPI_Finalize();
  }
  DIM = atoi(av[1]);
  NX = atoi(av[2]);
  NY = atoi(av[3]);
  NZ = atoi(av[4]);
  if (NX*NY*NZ != nblocks) {
    if (thisIndex == 0) 
      printf("%d x %d x %d != %d\n", NX,NY,NZ, nblocks);
    MPI_Finalize();
  }
  if (ac == 6)
    niter = atoi(av[5]);
  else
    niter = 10;

  MPI_Bcast(&niter, 1, MPI_INT, 0, MPI_COMM_WORLD);

  cp = new chunk(DIM);

  cp->indexing(thisIndex);

  for(i=1; i<=DIM; i++)
    for(j=1; j<=DIM; j++)
      for(k=1; k<=DIM; k++)
        cp->t[k][j][i] = DIM*DIM*(i-1) + DIM*(j-2) + (k-1);

  MPI_Barrier(MPI_COMM_WORLD);
  starttime = MPI_Wtime();

  maxerr = 0.0;
  for(iter=1; iter<=niter; iter++) {
    BGPRINTF("interation starts at %f\n");
    maxerr = 0.0;

    cp->copyout();

    MPI_Request rreq[6];
    MPI_Status rsts[6];

    MPI_Irecv(cp->rbxp, DIM*DIM, MPI_DOUBLE, cp->xp, 0, MPI_COMM_WORLD, &rreq[0]);
    MPI_Irecv(cp->rbxm, DIM*DIM, MPI_DOUBLE, cp->xm, 1, MPI_COMM_WORLD, &rreq[1]);
    MPI_Irecv(cp->rbyp, DIM*DIM, MPI_DOUBLE, cp->yp, 2, MPI_COMM_WORLD, &rreq[2]);
    MPI_Irecv(cp->rbym, DIM*DIM, MPI_DOUBLE, cp->ym, 3, MPI_COMM_WORLD, &rreq[3]);
    MPI_Irecv(cp->rbzm, DIM*DIM, MPI_DOUBLE, cp->zm, 5, MPI_COMM_WORLD, &rreq[4]);
    MPI_Irecv(cp->rbzp, DIM*DIM, MPI_DOUBLE, cp->zp, 4, MPI_COMM_WORLD, &rreq[5]);

    MPI_Send(cp->sbxm, DIM*DIM, MPI_DOUBLE, cp->xm, 0, MPI_COMM_WORLD);
    MPI_Send(cp->sbxp, DIM*DIM, MPI_DOUBLE, cp->xp, 1, MPI_COMM_WORLD);
    MPI_Send(cp->sbym, DIM*DIM, MPI_DOUBLE, cp->ym, 2, MPI_COMM_WORLD);
    MPI_Send(cp->sbyp, DIM*DIM, MPI_DOUBLE, cp->yp, 3, MPI_COMM_WORLD);
    MPI_Send(cp->sbzm, DIM*DIM, MPI_DOUBLE, cp->zm, 4, MPI_COMM_WORLD);
    MPI_Send(cp->sbzp, DIM*DIM, MPI_DOUBLE, cp->zp, 5, MPI_COMM_WORLD);

    MPI_Waitall(6, rreq, rsts);

    cp->copyin();
 
    if(iter > 25 &&  iter < 85 && thisIndex == 35)
      m = 9;
    else
      m = 1;
    for(; m>0; m--)
      for(i=1; i<=DIM; i++)
        for(j=1; j<=DIM; j++)
          for(k=1; k<=DIM; k++) {
            tval = (cp->t[k][j][i] + cp->t[k][j][i+1] +
                 cp->t[k][j][i-1] + cp->t[k][j+1][i]+ 
                 cp->t[k][j-1][i] + cp->t[k+1][j][i] + cp->t[k-1][j][i])/7.0;
            error = abs(tval-cp->t[k][j][i]);
            cp->t[k][j][i] = tval;
            if (error > maxerr) maxerr = error;
          }
    MPI_Allreduce(&maxerr, &tmpmaxerr, 1, MPI_DOUBLE, MPI_MAX, 
                   MPI_COMM_WORLD);
    maxerr = tmpmaxerr;
    endtime = MPI_Wtime();
    itertime = endtime - starttime;
    double  it;
    MPI_Allreduce(&itertime, &it, 1, MPI_DOUBLE, MPI_SUM,
                   MPI_COMM_WORLD);
    itertime = it/nblocks;
    if (thisIndex == 0)
      printf("iter %d time: %lf maxerr: %lf\n", iter, itertime, maxerr);
    starttime = MPI_Wtime();

    if(iter%20 == 10) {
      //      MPI_Migrate();
    }

  }
  MPI_Finalize();
  return 0;
}
