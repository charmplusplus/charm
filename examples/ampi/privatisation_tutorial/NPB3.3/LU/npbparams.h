c NPROCS = 9 CLASS = S
c  
c  
c  This file is generated automatically by the setparams utility.
c  It sets the number of processors and the class of the NPB
c  in this directory. Do not modify it by hand.
c  

c number of nodes for which this version is compiled
        integer nnodes_compiled, nnodes_xdim
        parameter (nnodes_compiled=9, nnodes_xdim=3)

c full problem size
        integer isiz01, isiz02, isiz03
        parameter (isiz01=12, isiz02=12, isiz03=12)

c sub-domain array size
        integer isiz1, isiz2, isiz3
        parameter (isiz1=4, isiz2=4, isiz3=isiz03)

c number of iterations and how often to print the norm
        integer itmax_default, inorm_default
        parameter (itmax_default=50, inorm_default=50)
        double precision dt_default
        parameter (dt_default = 0.5d0)
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
