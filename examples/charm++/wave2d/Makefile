CHARMC=../../../bin/charmc -module liveViz $(OPTS) -g

OBJS = wave2d.o

all: wave2d

wave2d: $(OBJS)
	$(CHARMC) -language charm++ -o wave2d $(OBJS)

wave2d.decl.h: wave2d.ci
	$(CHARMC)  wave2d.ci

wave2d.def.h: wave2d.decl.h

clean:
	rm -f *.decl.h *.def.h conv-host *.o wave2d charmrun *~

wave2d.o: wave2d.C wave2d.decl.h wave2d.def.h
	$(CHARMC) -c wave2d.C
