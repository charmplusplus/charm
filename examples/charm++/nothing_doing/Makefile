CHARMC=../../../bin/charmc $(OPTS)

OBJS = NothingDoing.o
PROJ= -tracemode projections -tracemode summary

all: NothingDoing 

NothingDoing: $(OBJS)
	$(CHARMC) -language charm++ -o NothingDoing $(OBJS)

NothingDoing.decl.h: NothingDoing.ci
	$(CHARMC)  NothingDoing.ci

clean:
	rm -f *.decl.h *.def.h conv-host *.o NothingDoing charmrun *~ *.core 

NothingDoing.o: NothingDoing.C NothingDoing.decl.h
	$(CHARMC) -c NothingDoing.C
