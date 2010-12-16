CHARMC=../../../../bin/charmc $(OPTS)

INCS=-I.
LIBS=-lgsl -lgslcblas -lm

all: interpolatelog

interpolatelog: interpolatelog.o EventInterpolator.o
	$(CHARMC) -seq -o interpolatelog interpolatelog.o EventInterpolator.o -lconv-bigsim-logs -lconv-util -lblue-standalone -language c++  $(LIBS)

interpolatelog.o: interpolatelog.C
	$(CHARMC) -seq -c interpolatelog.C $(INCS)

EventInterpolator.o: EventInterpolator.C
	$(CHARMC) -c EventInterpolator.C $(INCS)

clean:
	rm -f *.o *~ moduleinit.C charmrun conv-host
	rm -rf *.decl.h *.def.h core *.log newtraces interpolatelog
