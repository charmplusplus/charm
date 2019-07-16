-include ../../../../../common.mk
CHARMC=../../../../../../bin/charmc $(OPTS)

all: simpleBcastPost

simpleBcastPost:  simpleBcastPost.o
	$(CHARMC) simpleBcastPost.o -o simpleBcastPost -language charm++ -module CommonLBs

cifiles: simpleBcastPost.ci
	$(CHARMC) -c simpleBcastPost.ci
	touch cifiles

simpleBcastPost.o : simpleBcastPost.C cifiles
	$(CHARMC) -c simpleBcastPost.C

test: all
	$(call run, +p4 ./simpleBcastPost)
	$(call run, +p6 ./simpleBcastPost)
	$(call run, +p4 ./simpleBcastPost +noCMAForZC)
	$(call run, +p6 ./simpleBcastPost +noCMAForZC)

clean:
	rm -f *.def.h *.decl.h *.o *~ *.exe cifiles charmrun simpleBcastPost
