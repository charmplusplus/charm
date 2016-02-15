-include ../../common.mk
CHARMBASE=../../../
CHARMC=../../../bin/ampicxx $(OPTS)
TOKENS=6

all: ampi

ampi: jacobi jacobi.iso jacobi.tls jacobi-get

mpi: jacobi.C
	mpiCC -o jacobi jacobi.C  $(OPTS)

jacobi-cpp: jacobi-cpp.C
	$(CHARMC) -c jacobi-cpp.C 
	$(CHARMC) -o jacobi-cpp jacobi-cpp.o -module CommonLBs -lm

jacobi: jacobi.C
	$(CHARMC) -c jacobi.C 
	$(CHARMC) -o jacobi jacobi.o -module CommonLBs -lm

jacobi.iso: jacobi.C
	$(CHARMC) -c -DNO_PUP jacobi.C -o jacobi.iso.o
	-$(CHARMC) -o jacobi.iso jacobi.iso.o -module CommonLBs -memory isomalloc

jacobi.tls: jacobi.C
	-$(CHARMC) -c -tlsglobal jacobi.C -o jacobi.tls.o 
	-$(CHARMC) -o jacobi.tls jacobi.tls.o -tlsglobal -module CommonLBs #-memory isomalloc

jacobi.rose: jacobi.C
	$(CHARMC) -roseomptlsglobals -o jacobi.rose.o -c $<
	$(CHARMC) -roseomptlsglobals -o $@ jacobi.rose.o -module CommonLBs

jacobi-get: jacobi-get.C
	$(CHARMC) -c jacobi-get.C
	$(CHARMC) -o jacobi-get jacobi-get.o -module CommonLBs -lm

jacobi-get-tokens:
	cd $(CHARMBASE)/tmp; touch machine.c; touch ckfutures.C; make AMPI OPTS="-DIGET_FLOWCONTROL=1 -DIGET_TOKENNUM=$(TOKENS)"; cd -; 
	make jacobi-get;

test: ampi
	$(call run, +p3 ./jacobi 2 2 2 +vp8 +balancer RotateLB +LBDebug 1 )
	-$(call run, +p3 ./jacobi.iso 2 2 2 +vp8 +balancer RotateLB +LBDebug 1)  &  \
 pid=$$!; (sleep 30; echo killall jacobi.iso; killall -9 jacobi.iso; kill -9 $$pid)

bgtest: jacobi
	$(call run, +p2 ./jacobi 2 2 2 10 +vp8 +balancer RotateLB +x2 +y2 +z1 )
	-$(call run, +p2 ./jacobi.iso 2 2 2 10 +vp8 +balancer RotateLB +x2 +y2 +z1 )

clean:
	rm -f *.o jacobi *~ moduleinit.C charmrun conv-host jacobi-cpp jacobi.iso jacobi-get jacobi.tls
	rm -rf 40 80 120
