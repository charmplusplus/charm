CHARMC=../../../bin/charmc $(OPTS)

OBJS = LB_Bench.o
PROJ= -tracemode projections -tracemode summary

all: LB_Bench LB_Bench_Proj LB_Bench_Proj_LB

LB_Bench: $(OBJS)
	$(CHARMC) -language charm++ -o LB_Bench $(OBJS)

LB_Bench_Proj: $(OBJS)
	$(CHARMC) -language charm++ -o LB_Bench_Proj $(PROJ) $(OBJS)

LB_Bench_Proj_LB: $(OBJS)
	$(CHARMC) -language charm++ -o LB_Bench_Proj_LB -module CommonLBs $(PROJ) $(OBJS)

LB_Bench.decl.h: LB_Bench.ci
	$(CHARMC)  LB_Bench.ci

test: LB_Bench_Proj_LB
	rm logs/* 
	srun  --sockets-per-node=2 LB_Bench_Proj_LB 16 +traceroot logs/ +LBPeriod 2

clean:
	rm -f *.decl.h *.def.h conv-host *.o LB_Bench charmrun *~ *.core LB_Bench_Proj *.projrc *.sts logs/*

LB_Bench.o: LB_Bench.C LB_Bench.decl.h
	$(CHARMC) -c LB_Bench.C
