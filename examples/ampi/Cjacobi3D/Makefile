CHARMC=../../../bin/charmc $(OPTS)

ampi: jacobi

mpi: jacobi.C
	mpiCC -o jacobi jacobi.C  $(OPTS)

jacobi: jacobi.C
	$(CHARMC) -c jacobi.C
	$(CHARMC) -o jacobi jacobi.o -language ampi -module EveryLB

jacobi.iso: jacobi.C
	$(CHARMC) -c -DNO_PUP jacobi.C
	-$(CHARMC) -o jacobi.iso jacobi.o -language ampi -module EveryLB -memory isomalloc

test: ampi  jacobi.iso
	./charmrun +p3 jacobi 2 2 2 +vp8 +balancer GreedyCommLB
	-./charmrun +p3 jacobi.iso 2 2 2 +vp8 +balancer GreedyCommLB

bgtest: jacobi
	./charmrun +p2 jacobi 2 2 2 10 +vp8 +balancer GreedyCommLB +x2 +y2 +z1

clean:
	rm -f *.o jacobi *~ moduleinit.C charmrun conv-host
	rm -rf 40 80 120
