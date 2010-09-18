CHARMBASE=../../../
CHARMC=../../../bin/ampicxx $(OPTS)
TOKENS=6

all: ampi

ampi: jacobi jacobi.iso jacobi.tls jacobi-get

mpi: jacobi.C
	mpiCC -o jacobi jacobi.C  $(OPTS)

jacobi-cpp: jacobi-cpp.C
	$(CHARMC) -c jacobi-cpp.C 
	$(CHARMC) -o jacobi-cpp jacobi-cpp.o -module EveryLB -lm

jacobi: jacobi.C
	$(CHARMC) -c jacobi.C 
	$(CHARMC) -o jacobi jacobi.o -module EveryLB -lm

jacobi.iso: jacobi.C
	$(CHARMC) -c -DNO_PUP jacobi.C -o jacobi.iso.o
	-$(CHARMC) -o jacobi.iso jacobi.iso.o -module EveryLB -memory isomalloc

jacobi.tls: jacobi.C
	$(CHARMC) -c -DNO_PUP -tlsglobal jacobi.C -o jacobi.tls.o 
	-$(CHARMC) -o jacobi.tls jacobi.tls.o -tlsglobal -Wl,--allow-multiple-definition -thread context-tls -module EveryLB #-memory isomalloc

jacobi-get: jacobi-get.C
	$(CHARMC) -c jacobi-get.C
	$(CHARMC) -o jacobi-get jacobi-get.o -module EveryLB -lm

jacobi-get-tokens:
	cd $(CHARMBASE)/tmp; touch machine.c; touch ckfutures.C; make AMPI OPTS="-DIGET_FLOWCONTROL=1 -DIGET_TOKENNUM=$(TOKENS)"; cd -; 
	make jacobi-get;

test: ampi
	./charmrun +p3 ./jacobi 2 2 2 +vp8 +balancer RotateLB +LBDebug 1 $(TESTOPTS)
	-./charmrun +p3 ./jacobi.iso 2 2 2 +vp8 +balancer RotateLB +LBDebug 1 $(TESTOPTS) &  \
 pid=$$!; (sleep 30; echo killall jacobi.iso; killall -9 jacobi.iso; kill -9 $$pid)

bgtest: jacobi
	./charmrun +p2 ./jacobi 2 2 2 10 +vp8 +balancer RotateLB +x2 +y2 +z1 $(TESTOPTS)
	-./charmrun +p2 ./jacobi.iso 2 2 2 10 +vp8 +balancer RotateLB +x2 +y2 +z1 $(TESTOPTS)

clean:
	rm -f *.o jacobi *~ moduleinit.C charmrun conv-host jacobi-cpp jacobi.iso jacobi-get
	rm -rf 40 80 120
