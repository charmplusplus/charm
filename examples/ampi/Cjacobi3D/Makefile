CHARMC=../../../bin/charmc $(OPTS)

ampi: jacobi

mpi: jacobi.C
	mpiCC -o jacobi jacobi.C  $(OPTS)

jacobi: jacobi.C
	$(CHARMC) -c jacobi.C
	$(CHARMC) -o jacobi jacobi.o -language ampi -module EveryLB

test: ampi
	./charmrun +p3 jacobi 2 2 2 +vp8 +balancer GreedyCommLB

bgtest: jacobi
	./charmrun +p2 jacobi 2 2 2 10 +vp8 +bgconfig ./bg_config

clean:
	rm -f *.o jacobi *~ moduleinit.C charmrun conv-host
	rm -rf 40 80 120
