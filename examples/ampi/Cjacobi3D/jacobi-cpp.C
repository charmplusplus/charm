#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"
#ifdef AMPI
#include "charm++.h"
#endif

#if CMK_BIGSIM_CHARM
extern void BgPrintf(char *);
#define BGPRINTF(x)  if (thisIndex == 0) BgPrintf(x);
#else
#define BGPRINTF(x)
#endif

int DIM, DIMX, DIMY, DIMZ, NX, NY, NZ;

class chunk {
 public:
  int dimx, dimy, dimz;
  int xidx, yidx, zidx;
  int xm, xp, ym, yp, zm, zp;
  double ***t; 
  double *sbxm;
  double *sbxp;
  double *sbym;
  double *sbyp;
  double *sbzm;
  double *sbzp;
  double *rbxm;
  double *rbxp;
  double *rbym;
  double *rbyp;
  double *rbzm;
  double *rbzp;

 private:
  void create(){
    t = new double** [dimx+2];
    for(int i=0;i<dimx+2;i++){
      t[i] = new double* [dimy+2];
      for(int j=0;j<dimy+2;j++){
	t[i][j] = new double [dimz+2];
      }
    }
    sbxm = new double [dimy*dimz];
    sbxp = new double [dimy*dimz];
    sbym = new double [dimx*dimz];
    sbyp = new double [dimx*dimz];
    sbzm = new double [dimx*dimy];
    sbzp = new double [dimx*dimy];
    rbxm = new double [dimy*dimz];
    rbxp = new double [dimy*dimz];
    rbym = new double [dimx*dimz];
    rbyp = new double [dimx*dimz];
    rbzm = new double [dimx*dimy];
    rbzp = new double [dimx*dimy];
  }
  void destroy(){
    for(int i=0;i<dimx+2;i++){
      for(int j=0;j<dimy+2;j++){
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
  chunk(int x, int y, int z, int rank){
    int i,j,k;
    dimx = x; dimy = y; dimz = z;
    create();
    indexing(rank);
    for(i=1; i<=DIMX; i++)
      for(j=1; j<=DIMY; j++)
	for(k=1; k<=DIMZ; k++)
	  t[i][j][k] = DIMY*DIMZ*(i-1) + DIMZ*(j-1) + (k-1);
  }
  ~chunk(){
    destroy();
  }
#ifdef AMPI
  void pup(PUP::er& p){
    p|dimx;p|dimy;p|dimz;
    p|xidx;p|yidx;p|zidx;
    p|xp;p|xm;p|yp;p|ym;p|zp;p|zm;
    if(p.isUnpacking())
      create();
    for(int i=0;i<dimx+2;i++){
      for(int j=0;j<dimy+2;j++){
	p(t[i][j],dimz+2);
      }
    }
    p(sbxm,dimy*dimz);
    p(sbxp,dimy*dimz);
    p(sbym,dimx*dimz);
    p(sbyp,dimx*dimz);
    p(sbzm,dimx*dimy);
    p(sbzp,dimx*dimy);
    p(rbxm,dimy*dimz);
    p(rbxp,dimy*dimz);
    p(rbym,dimx*dimz);
    p(rbyp,dimx*dimz);
    p(rbzm,dimx*dimy);
    p(rbzp,dimx*dimy);
    if(p.isDeleting())
      destroy();
  }
#endif
  void copyin(){
    int i, j;
    int l = 0;
    for(i=1;i<=dimy;i++)
      for(j=1;j<=dimz;j++,l++){
	t[0][i][j] = sbxm[l];
	t[dimx+1][i][j] = sbxp[l];
      }
    l = 0;
    for(i=1;i<=dimx;i++)
      for(j=1;j<=dimz;j++,l++){
	t[i][0][j] = sbym[l];
	t[i][dimy+1][j] = sbyp[l];
      }
    l = 0;
    for(i=1;i<=dimx;i++)
      for(j=1;j<=dimy;j++,l++){
	t[i][j][0] = sbzm[l];
	t[i][j][dimz+1] = sbzp[l];
      }
  }

  void copyout(){
    int i, j;
    int l = 0;
    for(i=1;i<=dimy;i++)
      for(j=1;j<=dimz;j++,l++){
	sbxm[l] = t[1][i][j];
	sbxp[l] = t[dimx][i][j];
      }
    l = 0;
    for(i=1;i<=dimx;i++)
      for(j=1;j<=dimz;j++,l++){
	sbym[l] = t[i][1][j];
	sbyp[l] = t[i][dimy][j];
      }
    l = 0;
    for(i=1;i<=dimx;i++)
      for(j=1;j<=dimy;j++,l++){
	sbzm[l] = t[i][j][1];
	sbzp[l] = t[i][j][dimz];
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

  double calc(){
    double error = 0.0, maxerr = 0.0, tval;
    int i,j,k;
    for(k=1; k<=DIMX; k++) 
      for(j=1; j<=DIMY; j++)
	for(i=1; i<=DIMZ; i++){
	  tval = (t[k][j][i] + t[k][j][i+1] + t[k][j][i-1] + t[k][j+1][i]+ 
		  t[k][j-1][i] + t[k+1][j][i] + t[k-1][j][i])/7.0;
	  error = abs(tval-t[k][j][i]);
	  t[k][j][i] = tval;
	  if (error > maxerr) maxerr = error;
	}
    return maxerr;
  }
};

#define abs(x) ((x)<0.0 ? -(x) : (x))

int main(int ac, char** av)
{
  int i,j,k,m,cidx;
  int iter, niter;
  MPI_Status status;
  MPI_Info hints;
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

  /* Set up MPI_Info hints for AMPI_Migrate() */
  MPI_Info_create(&hints);
  MPI_Info_set(hints, "ampi_load_balance", "sync");

  DIMX = DIM/NX;
  DIMY = DIM/NY;
  DIMZ = DIM/NZ;

  MPI_Bcast(&niter, 1, MPI_INT, 0, MPI_COMM_WORLD);

  cp = new chunk(DIMX,DIMY,DIMZ,thisIndex);

  MPI_Barrier(MPI_COMM_WORLD);
  starttime = MPI_Wtime();

  for(iter=1; iter<=niter; iter++) {
    BGPRINTF("interation starts at %f\n");
    maxerr = 0.0;

    cp->copyout();

    MPI_Request rreq[6];
    MPI_Status rsts[6];

    MPI_Irecv(cp->rbxp, DIMY*DIMZ, MPI_DOUBLE, cp->xp, 0, MPI_COMM_WORLD, &rreq[0]);
    MPI_Irecv(cp->rbxm, DIMY*DIMZ, MPI_DOUBLE, cp->xm, 1, MPI_COMM_WORLD, &rreq[1]);
    MPI_Irecv(cp->rbyp, DIMX*DIMZ, MPI_DOUBLE, cp->yp, 2, MPI_COMM_WORLD, &rreq[2]);
    MPI_Irecv(cp->rbym, DIMX*DIMZ, MPI_DOUBLE, cp->ym, 3, MPI_COMM_WORLD, &rreq[3]);
    MPI_Irecv(cp->rbzp, DIMX*DIMY, MPI_DOUBLE, cp->zp, 4, MPI_COMM_WORLD, &rreq[4]);
    MPI_Irecv(cp->rbzm, DIMX*DIMY, MPI_DOUBLE, cp->zm, 5, MPI_COMM_WORLD, &rreq[5]);

    MPI_Send(cp->sbxm, DIMY*DIMZ, MPI_DOUBLE, cp->xm, 0, MPI_COMM_WORLD);
    MPI_Send(cp->sbxp, DIMY*DIMZ, MPI_DOUBLE, cp->xp, 1, MPI_COMM_WORLD);
    MPI_Send(cp->sbym, DIMX*DIMZ, MPI_DOUBLE, cp->ym, 2, MPI_COMM_WORLD);
    MPI_Send(cp->sbyp, DIMX*DIMZ, MPI_DOUBLE, cp->yp, 3, MPI_COMM_WORLD);
    MPI_Send(cp->sbzm, DIMX*DIMY, MPI_DOUBLE, cp->zm, 4, MPI_COMM_WORLD);
    MPI_Send(cp->sbzp, DIMX*DIMY, MPI_DOUBLE, cp->zp, 5, MPI_COMM_WORLD);

    MPI_Waitall(6, rreq, rsts);

    cp->copyin();

    maxerr = cp->calc(); 

//    MPI_Allreduce(&maxerr, &tmpmaxerr, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

#ifdef AMPI
    if(iter%20 == 10) {
      AMPI_Migrate(hints);
    }
#endif
  }
  MPI_Barrier(MPI_COMM_WORLD);
  endtime = MPI_Wtime();
  if (thisIndex == 0){
    itertime = (endtime - starttime) / niter;
    printf("iteration time: %lf s\n", itertime);
  }

  delete cp;
  MPI_Finalize();
  return 0;
}
