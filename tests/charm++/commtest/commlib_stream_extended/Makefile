CHARMC=../../../../bin/charmc $(OPTS)

OBJS = streaming.o

all: streaming

streaming: $(OBJS)
	$(CHARMC) -language charm++ -o streaming $(OBJS) -module comlib

streaming.decl.h: streaming.ci
	$(CHARMC)  streaming.ci

clean:
	rm -f *.decl.h *.def.h conv-host *.o streaming charmrun *.log *.sum *.sts *~

streaming.o: streaming.C streaming.decl.h
	$(CHARMC) -c -g streaming.C

test: all
	./charmrun ./streaming +p2 

debug: all
	./charmrun ./streaming +p2 ++debug

bgtest: all
	./charmrun ./streaming +p2 +x2 +y2 +z2 +cth1 +wth1