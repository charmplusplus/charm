#ifndef _MPIF_H
#define _MPIF_H

  integer, parameter :: MPI_COMM_WORLD=0
  integer, parameter :: MPI_STATUS_SIZE=3

  integer, parameter :: MPI_DOUBLE_PRECISION=0
  integer, parameter :: MPI_INTEGER=1
  integer, parameter :: MPI_REAL=2
  integer, parameter :: MPI_COMPLEX=3
  integer, parameter :: MPI_LOGICAL=4
  integer, parameter :: MPI_CHARACTER=5
  integer, parameter :: MPI_BYTE=6
  integer, parameter :: MPI_PACKED=7

  integer, parameter :: MPI_MAX=1
  integer, parameter :: MPI_MIN=2
  integer, parameter :: MPI_SUM=3
  integer, parameter :: MPI_PROD=4

#endif
