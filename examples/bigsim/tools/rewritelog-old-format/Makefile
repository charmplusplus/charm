CHARMC=../../../../bin/charmc $(OPTS) -g

INCS=-I/usr/include/ -I.
LIBS=-lgsl -lgslcblas -lm -L/usr/lib

all: interpolatelog

interpolatelog: interpolatelog.o EventInterpolator.o
	$(CHARMC) -seq -o interpolatelog interpolatelog.o EventInterpolator.o -lconv-bigsim-logs -lconv-util -lblue-standalone -language c++  $(LIBS)

interpolatelog.o: interpolatelog.C
	$(CHARMC) -seq -c interpolatelog.C $(INCS)

EventInterpolator.o: EventInterpolator.C
	$(CHARMC) -c EventInterpolator.C $(INCS)

rewritelog: rewritelog.o
	$(CHARMC) -seq -o rewritelog rewritelog.o -lconv-bigsim-logs -lconv-util -lblue-standalone -language c++

rewritelog.o: rewritelog.C
	$(CHARMC) -seq -c rewritelog.C

clean:
	rm -f *.o *~ moduleinit.C charmrun conv-host rewritelog
	rm -rf *.decl.h *.def.h core *.log newtraces interpolatelog

test: rewritelog
	./rewritelog < sample-input
