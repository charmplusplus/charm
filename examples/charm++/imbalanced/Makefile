CHARMC=../../../bin/charmc $(OPTS)

OBJS = LB_Bench.o
PROJ= -tracemode projections -tracemode summary

all: LB_Bench LB_Bench_Proj

LB_Bench: $(OBJS)
	$(CHARMC) -language charm++ -o LB_Bench $(OBJS)

LB_Bench_Proj: $(OBJS)
	$(CHARMC) -language charm++ -o LB_Bench_Proj $(PROJ) $(OBJS)

LB_Bench.decl.h: LB_Bench.ci
	$(CHARMC)  LB_Bench.ci

clean:
	rm -f *.decl.h *.def.h conv-host *.o LB_Bench charmrun *~ *.core LB_Bench_Proj

LB_Bench.o: LB_Bench.C LB_Bench.decl.h
	$(CHARMC) -c LB_Bench.C
