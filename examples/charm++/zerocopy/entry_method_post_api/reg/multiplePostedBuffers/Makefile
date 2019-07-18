-include ../../../../../common.mk
CHARMC=../../../../../../bin/charmc $(OPTS)

all: multiplePostedBuffers

multiplePostedBuffers:  multiplePostedBuffers.o
	$(CHARMC) multiplePostedBuffers.o -o multiplePostedBuffers -language charm++ -module CommonLBs

cifiles: multiplePostedBuffers.ci
	$(CHARMC) -c multiplePostedBuffers.ci
	touch cifiles

multiplePostedBuffers.o : multiplePostedBuffers.C cifiles
	$(CHARMC) -c multiplePostedBuffers.C

test: all
	$(call run, +p4 ./multiplePostedBuffers 32 +balancer RotateLB)
	$(call run, +p4 ./multiplePostedBuffers 32 +balancer GreedyLB)
	$(call run, +p4 ./multiplePostedBuffers 32 +balancer RotateLB +noCMAForZC )
	$(call run, +p4 ./multiplePostedBuffers 32 +balancer GreedyLB +noCMAForZC )

clean:
	rm -f *.def.h *.decl.h *.o *~ *.exe cifiles charmrun multiplePostedBuffers
