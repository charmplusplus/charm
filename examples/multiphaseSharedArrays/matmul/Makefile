-include ../../common.mk

HEADERS=nepp.h params.h 
PGM=t2d
#PGM=matmul2D
include ../Makefile_common

# test : test.C test.ci Makefile
# 	rm -f test
# 	$(CHARMC) test.ci
# 	$(CHARMC) -o test -module msa test.C -lm

all: opt

tt: opt
	$(call run, ++nodegroup one +p1 t2d 1 200000000 2000 5000 300 1 1)
	$(call run, ++nodegroup one +p1 t2d 1 200000000 2000 5000 300 4 1)

t: opt
	w
	$(call run, ++nodegroup one +p1 t2d 1 200000000 2000 5000 300 4 1)

lem: opt
	prun -n1 ./t2d 1 200000000 2000 5000 300 4 1
	prun -n1 ./t2d 1 200000000 2000 5000 300 1 1
	prun -n2 ./t2d 2 200000000 2000 5000 300 4 1
	prun -n4 ./t2d 4 200000000 2000 5000 300 4 1
	prun -n1 ./t2d 4 200000000 2000 5000 300 4 1
	prun -n2 ./t2d 8 200000000 2000 5000 300 4 1
	prun -n4 ./t2d 16 200000000 2000 5000 300 4 1

tung: opt
	$(HOME)/gennodelist.csh
	$(call run, +p1 t2d 1 200000000 2000 5000 300 4 1)
	$(call run, +p2 t2d 2 200000000 2000 5000 300 4 1)
	$(call run, +p1 t2d 4 200000000 2000 5000 300 4 1)
	$(call run, +p2 t2d 8 200000000 2000 5000 300 4 1)

tung832:
	$(call run, +p8 t2d 32 200000000 2000 5000 300 4 1)

tung816:
	$(call run, +p8 t2d 16 200000000 2000 5000 300 4 1)

t1:
	#make OPTS="-DOLD" c opt
	w
	$(call run, ++nodegroup one +p1 t2d 1 200000000 2000 5000 300 1 1)

# quick test on 1 cpu
test: opt
	$(call run, +p1 $(PGM) 2 1048576 100 500 100 1)
	$(call run, +p1 $(PGM) 4 1048576 100 500 100 2)
	$(call run, +p1 $(PGM) 8 1048576 100 500 100 3)
	$(call run, +p1 $(PGM) 2 1048576 100 500 100 4)
	$(call run, +p1 $(PGM) 4 1048576 100 500 100 5)
	$(call run, +p1 $(PGM) 8 1048576 100 500 100 6)

# quick test on 2 cpus
test2: opt
	$(call run, +p2 $(PGM) 2 1048576 100 500 100 1)
	$(call run, +p2 $(PGM) 4 1048576 100 500 100 2)
	$(call run, +p2 $(PGM) 8 1048576 100 500 100 3)
	$(call run, +p2 $(PGM) 4 1048576 100 500 100 4)
	$(call run, +p2 $(PGM) 16 1048576 100 500 100 5)
	$(call run, +p2 $(PGM) 64 1048576 100 500 100 6)

#bigtest
bt: opt
	$(call run, +p1 $(PGM) 1 200000000 2000 5000 300)
#	 ./charmrun +p1 t2d 1 200000000 2000 5000 300 1 1
#opteron	 ./charmrun ++local +p1 t2d 1 200000000 2000 5000 300 4 1

bt2d: opt
	$(call run, +p1 $(PGM) 1 200000000 2000 5000 300 2)

bt3d: opt
	$(call run, +p1 $(PGM) 1 200000000 2000 5000 300 3)

bt4:
	$(call run, +p1 $(PGM) 1 200000000 2000 5000 300 4 1)

seq:
	g++ -o mm_sequential -O3 mm_sequential.C

seqs-icc:
	icc -no_cpprt -DBIGGER_ALGO -DGLOBAL_VAR_VERSION -USKIP_INIT -o mmgii -O3 mm_sequential.c
	icc -no_cpprt -DBIGGER_ALGO -DGLOBAL_VAR_VERSION -DSKIP_INIT -o mmgsi -O3 mm_sequential.c
	icc -no_cpprt -DBIGGER_ALGO -UGLOBAL_VAR_VERSION -USKIP_INIT -o mmmii -O3 mm_sequential.c
	icc -no_cpprt -DBIGGER_ALGO -UGLOBAL_VAR_VERSION -DSKIP_INIT -o mmmsi -O3 mm_sequential.c

seqs:
	g++ -DBIGGER_ALGO -DGLOBAL_VAR_VERSION -DSKIP_INIT -o mmgs -O3 mm_sequential.C
	g++ -DBIGGER_ALGO -DGLOBAL_VAR_VERSION -USKIP_INIT -o mmgi -O3 mm_sequential.C
	g++ -DBIGGER_ALGO -UGLOBAL_VAR_VERSION -DSKIP_INIT -o mmms -O3 mm_sequential.C
	g++ -DBIGGER_ALGO -UGLOBAL_VAR_VERSION -USKIP_INIT -o mmmi -O3 mm_sequential.C

smallseqs:
	g++ -UBIGGER_ALGO -DGLOBAL_VAR_VERSION -DSKIP_INIT -o mmgs_s -O3 mm_sequential.C
	g++ -UBIGGER_ALGO -DGLOBAL_VAR_VERSION -USKIP_INIT -o mmgi_s -O3 mm_sequential.C

runseqs:
	 ./mmgs
	 ./mmgi
	 ./mmms
	 ./mmmi

runsmallseqs:
	 ./mmgs_s
	 ./mmgi_s
