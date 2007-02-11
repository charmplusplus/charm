CHARMBASE=../../../
CHARMC=../../../bin/charmc $(OPTS)
TOKENS=6

ampi: jacobi jacobi.iso jacobi-get

mpi: jacobi.C
	mpiCC -o jacobi jacobi.C  $(OPTS)

jacobi-cpp: jacobi-cpp.C
	$(CHARMC) -c jacobi-cpp.C 
	$(CHARMC) -o jacobi-cpp jacobi-cpp.o -language ampi -module EveryLB -lm

jacobi: jacobi.C
	$(CHARMC) -c jacobi.C 
	$(CHARMC) -o jacobi jacobi.o -language ampi -module EveryLB -lm

jacobi.iso: jacobi.C
	$(CHARMC) -c -DNO_PUP jacobi.C
	-$(CHARMC) -o jacobi.iso jacobi.o -language ampi -module EveryLB -memory isomalloc
jacobi-get: jacobi-get.C
	$(CHARMC) -c jacobi-get.C
	$(CHARMC) -o jacobi-get jacobi-get.o -language ampi -module EveryLB -lm

jacobi-get-tokens:
	cd $(CHARMBASE)/tmp; touch machine.c; touch ckfutures.C; make AMPI OPTS="-DIGET_FLOWCONTROL=1 -DIGET_TOKENNUM=$(TOKENS)"; cd -; 
	make jacobi-get;

test: ampi
	./charmrun +p3 jacobi 2 2 2 +vp8 +balancer RotateLB +LBDebug $(TESTOPTS)
	-./charmrun +p3 jacobi.iso 2 2 2 +vp8 +balancer RotateLB +LBDebug $(TESTOPTS)

bgtest: jacobi
	./charmrun +p2 jacobi 2 2 2 10 +vp8 +balancer RotateLB +x2 +y2 +z1 $(TESTOPTS)
	-./charmrun +p2 jacobi.iso 2 2 2 10 +vp8 +balancer RotateLB +x2 +y2 +z1 $(TESTOPTS)

clean:
	rm -f *.o jacobi *~ moduleinit.C charmrun conv-host jacobi-cpp jacobi.iso jacobi-get
	rm -rf 40 80 120
