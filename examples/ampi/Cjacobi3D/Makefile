CHARMC=../../../bin/charmc $(OPTS)

ampi: jacobi.C
	$(CHARMC) -c jacobi.C
	$(CHARMC) -o jacobi jacobi.o -language ampi -module EveryLB

mpi: jacobi.C
	mpiCC -o jacobi jacobi.C  $(OPTS)

test: ampi
	./charmrun +p3 jacobi 2 2 2 +vp8 +balancer GreedyCommLB

clean:
	rm -f *.o jacobi *~ moduleinit.C charmrun conv-host
	rm -rf 40 80 120
