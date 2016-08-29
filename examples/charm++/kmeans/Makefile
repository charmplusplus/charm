-include ../../common.mk
CHARMC=../../../bin/charmc $(OPTS)

OBJS = kmeans.o

all: kmeans

kmeans: $(OBJS)
	$(CHARMC) -language charm++ -o kmeans $(OBJS)

kmeans.o: kmeans.C kmeans.decl.h
	$(CHARMC) -c kmeans.C

kmeans.decl.h: kmeans.ci
	$(CHARMC)  kmeans.ci

clean:
	rm -f *.decl.h *.def.h conv-host *.o kmeans  charmrun *~ *log *projrc *sts

test: all
	$(call run, +p4 ./kmeans 32 4 4 )

