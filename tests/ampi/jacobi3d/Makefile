-include ../../common.mk
CHARMC=../../../bin/ampicxx $(OPTS) $(MOPTS)

OBJS = jacobi.o

all: jacobi

jacobi: $(OBJS)
	$(CHARMC) -o jacobi $(OBJS) -lm -memory isomalloc -module DummyLB

jacobi.o: jacobi.C
	$(CHARMC) -language ampic -c -swapglobals jacobi.C 

syncfttest causalfttest: jacobi
	$(call run, ./jacobi 2 2 2 200 +vp8 +p8 +balancer DummyLB +isomalloc_sync +killFile kill_01.txt )
	$(call run, ./jacobi 4 2 2 200 +vp16 +p8 +balancer DummyLB +isomalloc_sync +killFile kill_02.txt )

clean:
	rm -f *.o jacobi jacobi.o *~ moduleinit.C charmrun conv-host jacobi-cpp jacobi.iso jacobi-get jacobi.tls
	rm -rf 40 80 120
