CHARMC=../../../../bin/charmc $(OPTS) -O2

INCS=-I/SW/include
LIBS=-lgsl -lgslcblas -lm -L/sw/lib
 
interpolatelog: interpolatelog.o 
	$(CHARMC) -seq -o interpolatelog interpolatelog.o -lconv-bigsim-logs -lconv-util -lblue-standalone -language c++  $(LIBS)
 
interpolatelog.o: interpolatelog.C 
	$(CHARMC) -seq -c interpolatelog.C $(INCS) $(LIBS)


rewritelog: rewritelog.o
	$(CHARMC) -seq -o rewritelog rewritelog.o -lconv-bigsim-logs -lconv-util -lblue-standalone -language c++

rewritelog.o: rewritelog.C
	$(CHARMC) -seq -c rewritelog.C

clean:
	rm -f *.o *~ moduleinit.C charmrun conv-host rewritelog
	rm -rf *.decl.h *.def.h core *.log newtraces interpolatelog

test: rewritelog
	./rewritelog < sample-input
