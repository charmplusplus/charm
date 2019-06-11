c NPROCS = 9 CLASS = S
c  
c  
c  This file is generated automatically by the setparams utility.
c  It sets the number of processors and the class of the NPB
c  in this directory. Do not modify it by hand.
c  
        integer maxcells, problem_size, niter_default
        parameter (maxcells=3, problem_size=12, niter_default=100)
        double precision dt_default
        parameter (dt_default = 0.015d0)
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
