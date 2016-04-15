#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"

#if CMK_BLUEGENE_CHARM
extern void BgPrintf(char *);
#define BGPRINTF(x)  if (thisIndex == 0) BgPrintf(x);
#else
#define BGPRINTF(x)
#endif

#define DIMX 128
#define DIMY 128
#define DIMZ 128
#define CKPT_FREQ 100
#define NO_PUP

int NX, NY, NZ;

class chunk {
  public:
    double t[DIMX+2][DIMY+2][DIMZ+2];
    int xidx, yidx, zidx;
    int xm, xp, ym, yp, zm, zp;
    double sbxm[DIMY*DIMZ];
    double sbxp[DIMY*DIMZ];
    double sbym[DIMX*DIMZ];
    double sbyp[DIMX*DIMZ];
    double sbzm[DIMX*DIMY];
    double sbzp[DIMX*DIMY];
    double rbxm[DIMY*DIMZ];
    double rbxp[DIMY*DIMZ];
    double rbym[DIMX*DIMZ];
    double rbyp[DIMX*DIMZ];
    double rbzm[DIMX*DIMY];
    double rbzp[DIMX*DIMY];
};

#ifdef AMPI
void chunk_pup(pup_er p, void *d)
{
  chunk **cpp = (chunk **) d;
  if(pup_isUnpacking(p))
    *cpp = new chunk;
  chunk *cp = *cpp;
  pup_doubles(p, &cp->t[0][0][0], (DIMX+2)*(DIMY+2)*(DIMZ+2));
  pup_int(p, &cp->xidx);
  pup_int(p, &cp->yidx);
  pup_int(p, &cp->zidx);
  pup_int(p, &cp->xp);
  pup_int(p, &cp->xm);
  pup_int(p, &cp->yp);
  pup_int(p, &cp->ym);
  pup_int(p, &cp->zp);
  pup_int(p, &cp->zm);
  pup_doubles(p, cp->sbxm, (DIMY*DIMZ));
  pup_doubles(p, cp->sbxp, (DIMY*DIMZ));
  pup_doubles(p, cp->rbxm, (DIMY*DIMZ));
  pup_doubles(p, cp->rbxp, (DIMY*DIMZ));
  pup_doubles(p, cp->sbym, (DIMX*DIMZ));
  pup_doubles(p, cp->sbyp, (DIMX*DIMZ));
  pup_doubles(p, cp->rbym, (DIMX*DIMZ));
  pup_doubles(p, cp->rbyp, (DIMX*DIMZ));
  pup_doubles(p, cp->sbzm, (DIMX*DIMY));
  pup_doubles(p, cp->sbzp, (DIMX*DIMY));
  pup_doubles(p, cp->rbzm, (DIMX*DIMY));
  pup_doubles(p, cp->rbzp, (DIMX*DIMY));
  if(pup_isDeleting(p))
    delete cp;
}
#endif

#define abs(x) ((x)<0.0 ? -(x) : (x))

int index1d(int ix, int iy, int iz)
{
  return NY*NZ*ix + NZ*iy + iz;
}

void index3d(int index, int& ix, int& iy, int& iz)
{
  ix = index/(NY*NZ);
  iy = (index%(NY*NZ))/NZ;
  iz = index%NZ;
}

static void copyout(double *d, double t[DIMX+2][DIMY+2][DIMZ+2],
                    int sx, int ex, int sy, int ey, int sz, int ez)
{
  int i, j, k;
  int l = 0;
  for(i=sx; i<=ex; i++)
    for(j=sy; j<=ey; j++)
      for(k=sz; k<=ez; k++, l++)
        d[l] = t[i][j][k];
}

static void copyin(double *d, double t[DIMX+2][DIMY+2][DIMZ+2],
                    int sx, int ex, int sy, int ey, int sz, int ez)
{
  int i, j, k;
  int l = 0;
  for(i=sx; i<=ex; i++)
    for(j=sy; j<=ey; j++)
      for(k=sz; k<=ez; k++, l++)
        t[i][j][k] = d[l];
}

int main(int ac, char** av)
{
  int i,j,k,m,cidx;
  int iter, niter, cp_idx;
  MPI_Status status;
  MPI_Info hints;
  double error, tval, maxerr, tmpmaxerr, starttime, endtime, itertime;
  chunk *cp;
  int thisIndex, ierr, nblocks;

  MPI_Init(&ac, &av);
  MPI_Comm_rank(MPI_COMM_WORLD, &thisIndex);
  MPI_Comm_size(MPI_COMM_WORLD, &nblocks);

  if (ac < 4) {
    if (thisIndex == 0)
      printf("Usage: jacobi X Y Z [nIter].\n");
    MPI_Finalize();
  }
  NX = atoi(av[1]);
  NY = atoi(av[2]);
  NZ = atoi(av[3]);
  if (NX*NY*NZ != nblocks) {
    if (thisIndex == 0) 
      printf("%d x %d x %d != %d\n", NX,NY,NZ, nblocks);
    MPI_Finalize();
  }
  if (ac == 5)
    niter = atoi(av[4]);
  else
    niter = 20;

  /* Set up MPI_Info hints for AMPI_Migrate() */
  MPI_Info_create(&hints);
#ifdef CMK_MEM_CHECKPOINT
  MPI_Info_set(hints, "ampi_checkpoint", "in_memory");
#elif CMK_MESSAGE_LOGGING
  MPI_Info_set(hints, "ampi_checkpoint", "message_logging");
#endif

  MPI_Bcast(&niter, 1, MPI_INT, 0, MPI_COMM_WORLD);

#if CMK_AIX
  cp = (chunk*)malloc(sizeof(chunk));
#else
  cp = new chunk;
#endif
#if defined(AMPI) && ! defined(NO_PUP)
  AMPI_Register_pup((MPI_PupFn)chunk_pup, (void*)&cp, &cp_idx);
#endif

  index3d(thisIndex, cp->xidx, cp->yidx, cp->zidx);
  cp->xp = index1d((cp->xidx+1)%NX,cp->yidx,cp->zidx);
  cp->xm = index1d((cp->xidx+NX-1)%NX,cp->yidx,cp->zidx);
  cp->yp = index1d(cp->xidx,(cp->yidx+1)%NY,cp->zidx);
  cp->ym = index1d(cp->xidx,(cp->yidx+NY-1)%NY,cp->zidx);
  cp->zp = index1d(cp->xidx,cp->yidx,(cp->zidx+1)%NZ);
  cp->zm = index1d(cp->xidx,cp->yidx,(cp->zidx+NZ-1)%NZ);
  for(i=1; i<=DIMZ; i++)
    for(j=1; j<=DIMY; j++)
      for(k=1; k<=DIMX; k++)
        cp->t[k][j][i] = DIMY*DIMX*(i-1) + DIMX*(j-2) + (k-1);

  MPI_Barrier(MPI_COMM_WORLD);
  starttime = MPI_Wtime();

  maxerr = 0.0;
  for(iter=1; iter<=niter; iter++) {
    BGPRINTF("interation starts at %f\n");
    maxerr = 0.0;
    copyout(cp->sbxm, cp->t, 1, 1, 1, DIMY, 1, DIMZ);
    copyout(cp->sbxp, cp->t, DIMX, DIMX, 1, DIMY, 1, DIMZ);
    copyout(cp->sbym, cp->t, 1, DIMX, 1, 1, 1, DIMZ);
    copyout(cp->sbyp, cp->t, 1, DIMX, DIMY, DIMY, 1, DIMZ);
    copyout(cp->sbzm, cp->t, 1, DIMX, 1, DIMY, 1, 1);
    copyout(cp->sbzp, cp->t, 1, DIMX, 1, DIMY, DIMZ, DIMZ);

    MPI_Request rreq[6];
    MPI_Status rsts[6];

    MPI_Irecv(cp->rbxp, DIMY*DIMZ, MPI_DOUBLE, cp->xp, 0, MPI_COMM_WORLD, &rreq[0]);
    MPI_Irecv(cp->rbxm, DIMY*DIMZ, MPI_DOUBLE, cp->xm, 1, MPI_COMM_WORLD, &rreq[1]);
    MPI_Irecv(cp->rbyp, DIMX*DIMZ, MPI_DOUBLE, cp->yp, 2, MPI_COMM_WORLD, &rreq[2]);
    MPI_Irecv(cp->rbym, DIMX*DIMZ, MPI_DOUBLE, cp->ym, 3, MPI_COMM_WORLD, &rreq[3]);
    MPI_Irecv(cp->rbzm, DIMX*DIMY, MPI_DOUBLE, cp->zm, 5, MPI_COMM_WORLD, &rreq[4]);
    MPI_Irecv(cp->rbzp, DIMX*DIMY, MPI_DOUBLE, cp->zp, 4, MPI_COMM_WORLD, &rreq[5]);

    MPI_Send(cp->sbxm, DIMY*DIMZ, MPI_DOUBLE, cp->xm, 0, MPI_COMM_WORLD);
    MPI_Send(cp->sbxp, DIMY*DIMZ, MPI_DOUBLE, cp->xp, 1, MPI_COMM_WORLD);
    MPI_Send(cp->sbym, DIMX*DIMZ, MPI_DOUBLE, cp->ym, 2, MPI_COMM_WORLD);
    MPI_Send(cp->sbyp, DIMX*DIMZ, MPI_DOUBLE, cp->yp, 3, MPI_COMM_WORLD);
    MPI_Send(cp->sbzm, DIMX*DIMY, MPI_DOUBLE, cp->zm, 4, MPI_COMM_WORLD);
    MPI_Send(cp->sbzp, DIMX*DIMY, MPI_DOUBLE, cp->zp, 5, MPI_COMM_WORLD);

    MPI_Waitall(6, rreq, rsts);

    copyin(cp->sbxm, cp->t, 0, 0, 1, DIMY, 1, DIMZ);
    copyin(cp->sbxp, cp->t, DIMX+1, DIMX+1, 1, DIMY, 1, DIMZ);
    copyin(cp->sbym, cp->t, 1, DIMX, 0, 0, 1, DIMZ);
    copyin(cp->sbyp, cp->t, 1, DIMX, DIMY+1, DIMY+1, 1, DIMZ);
    copyin(cp->sbzm, cp->t, 1, DIMX, 1, DIMY, 0, 0);
    copyin(cp->sbzp, cp->t, 1, DIMX, 1, DIMY, DIMZ+1, DIMZ+1);
    if(iter > 25 &&  iter < 85 && thisIndex == 35)
      m = 9;
    else
      m = 1;
    for(; m>0; m--)
      for(i=1; i<=DIMZ; i++)
        for(j=1; j<=DIMY; j++)
          for(k=1; k<=DIMX; k++) {
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
      printf("iter %d elapsed time: %f time: %lf maxerr: %lf\n", iter, MPI_Wtime(), itertime, maxerr);
    starttime = MPI_Wtime();
#ifdef AMPI
    if(iter%CKPT_FREQ == 50) {
		AMPI_Migrate(hints);
    }
#endif
  }
  MPI_Finalize();
  return 0;
}
