#CHARMC=/expand/home/sarood/Work/charm/bin/charmc  $(OPTS)
#CHARMC=/expand/home/sarood/Work/charmOrg/bin/charmc  $(OPTS)
CHARMC=../../../bin/charmc $(OPTS)
#CHARMC=/expand/home/sarood/Work/201012_TempAwareLDB/charm/bin/charmc  $(OPTS)
#CHARMC=/expand/home/sarood/201012_TempAwareLDB/charm/bin/charmc  $(OPTS)


OBJS = jacobi2d.o

all: jacobi2d

jacobi2d: $(OBJS)
	$(CHARMC) -language charm++ -o jacobi2d -balancer TempAwareGreedyLB -tracemode projections -lcpufreq $(OBJS)
#	$(CHARMC) -language charm++ -o jacobi2d -balancer GreedyLB -tracemode projections $(OBJS)
#	$(CHARMC) jacobi2d.o  -o jacobi2d -language charm++

jacobi2d.decl.h: jacobi2d.ci
	$(CHARMC)  jacobi2d.ci

clean:
	rm -f *.decl.h *.def.h conv-host *.o jacobi2d charmrun *~

jacobi2d.o: jacobi2d.C jacobi2d.decl.h
	$(CHARMC) -c jacobi2d.C
