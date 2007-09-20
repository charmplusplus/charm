
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
	time ./charmrun ++nodegroup one +p1 t2d 1 200000000 2000 5000 300 1 1
	time ./charmrun ++nodegroup one +p1 t2d 1 200000000 2000 5000 300 4 1

t: opt
	w
	time ./charmrun ++nodegroup one +p1 t2d 1 200000000 2000 5000 300 4 1

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
	./charmrun +p1 t2d 1 200000000 2000 5000 300 4 1
	./charmrun +p2 t2d 2 200000000 2000 5000 300 4 1
	./charmrun +p1 t2d 4 200000000 2000 5000 300 4 1
	./charmrun +p2 t2d 8 200000000 2000 5000 300 4 1

tung832:
	./charmrun +p8 t2d 32 200000000 2000 5000 300 4 1

tung816:
	./charmrun +p8 t2d 16 200000000 2000 5000 300 4 1

t1:
	#make OPTS="-DOLD" c opt
	w
	time ./charmrun ++nodegroup one +p1 t2d 1 200000000 2000 5000 300 1 1

# quick test on 1 cpu
test: opt
	./charmrun +p1 $(PGM) 2 1048576 100 500 100 1
	./charmrun +p1 $(PGM) 4 1048576 100 500 100 2
	./charmrun +p1 $(PGM) 8 1048576 100 500 100 3
	./charmrun +p1 $(PGM) 2 1048576 100 500 100 4
	./charmrun +p1 $(PGM) 4 1048576 100 500 100 5
	./charmrun +p1 $(PGM) 8 1048576 100 500 100 6

# quick test on 2 cpus
test2: opt
	./charmrun +p2 $(PGM) 2 1048576 100 500 100 1
	./charmrun +p2 $(PGM) 4 1048576 100 500 100 2
	./charmrun +p2 $(PGM) 8 1048576 100 500 100 3
	./charmrun +p2 $(PGM) 4 1048576 100 500 100 4
	./charmrun +p2 $(PGM) 16 1048576 100 500 100 5
	./charmrun +p2 $(PGM) 64 1048576 100 500 100 6

#bigtest
bt: opt
	time ./charmrun +p1 $(PGM) 1 200000000 2000 5000 300
#	time ./charmrun +p1 t2d 1 200000000 2000 5000 300 1 1
#opteron	time ./charmrun ++local +p1 t2d 1 200000000 2000 5000 300 4 1

bt2d: opt
	time ./charmrun +p1 $(PGM) 1 200000000 2000 5000 300 2

bt3d: opt
	time ./charmrun +p1 $(PGM) 1 200000000 2000 5000 300 3

bt4:
	time ./charmrun +p1 $(PGM) 1 200000000 2000 5000 300 4 1

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
	time ./mmgs
	time ./mmgi
	time ./mmms
	time ./mmmi

runsmallseqs:
	time ./mmgs_s
	time ./mmgi_s
