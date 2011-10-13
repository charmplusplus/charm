CHARMC=../../../../bin/charmc $(OPTS)

OBJS = streaming.o

all: streaming

streaming: $(OBJS)
	$(CHARMC) -language charm++ -module comlib -tracemode projections -o streaming $(OBJS) 

streaming.decl.h: streaming.ci
	$(CHARMC)  streaming.ci

clean:
	rm -f *.decl.h *.def.h conv-host *.o streaming charmrun *.log *.sum *.sts *~

streaming.o: streaming.C streaming.decl.h
	$(CHARMC) -c -O3 streaming.C

test: all
	./charmrun ./streaming +p2 

debug: all
	./charmrun ./streaming +p2 ++debug