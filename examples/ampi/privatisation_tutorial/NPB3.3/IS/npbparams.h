#define CLASS 'W'
#define NUM_PROCS 32
/*
   This file is generated automatically by the setparams utility.
   It sets the number of processors and the class of the NPB
   in this directory. Do not modify it by hand.   */
   
#define COMPILETIME "10 Jun 2019"
#define NPBVERSION "3.3.1"
#define MPICC "$(CHARMBASE)/ampicc"
#define CFLAGS "-O0  -g -mcmodel=medium -fopenmp #-tlsglobals"
#define CLINK "$(MPICC)"
#define CLINKFLAGS "-O0 -fopenmp -mcmodel=medium #-tlsglobals"
#define CMPI_LIB "#-L/home/ankit/mpich-install/lib -lmpi"
#define CMPI_INC "-I/usr/local/include"
