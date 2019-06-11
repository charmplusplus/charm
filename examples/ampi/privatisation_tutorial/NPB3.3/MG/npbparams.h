c NPROCS = 32 CLASS = S
c  
c  
c  This file is generated automatically by the setparams utility.
c  It sets the number of processors and the class of the NPB
c  in this directory. Do not modify it by hand.
c  
        integer nprocs_compiled
        parameter (nprocs_compiled = 32)
        integer nx_default, ny_default, nz_default
        parameter (nx_default=32, ny_default=32, nz_default=32)
        integer nit_default, lm, lt_default
        parameter (nit_default=4, lm = 4, lt_default=5)
        integer debug_default
        parameter (debug_default=0)
        integer ndim1, ndim2, ndim3
        parameter (ndim1 = 4, ndim2 = 3, ndim3 = 3)
        logical  convertdouble
        parameter (convertdouble = .false.)
        character*11 compiletime
        parameter (compiletime='10 Jun 2019')
        character*5 npbversion
        parameter (npbversion='3.3.1')
        character*20 cs1
        parameter (cs1='$(CHARMBASE)/ampif77')
        character*9 cs2
        parameter (cs2='$(MPIF77)')
        character*37 cs3
        parameter (cs3='-L/home/ankit/mpich-install/lib -lmpi')
        character*20 cs4
        parameter (cs4='-I/usr/local/include')
        character*44 cs5
        parameter (cs5='-O0  -g -tlsglobals -fopenmp -mcmodel=medium')
        character*40 cs6
        parameter (cs6='-O0 -tlsglobals -fopenmp -mcmodel=medium')
        character*6 cs7
        parameter (cs7='randi8')
