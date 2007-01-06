CHARMC=../../../../bin/charmc $(OPTS) -O2


rewritelog: rewritelog.o
	$(CHARMC) -seq -o rewritelog rewritelog.o -lconv-bigsim-logs -lconv-util -lblue-standalone -language c++


rewritelog.o: rewritelog.C
	$(CHARMC) -seq -c rewritelog.C

clean:
	rm -f *.o *~ moduleinit.C charmrun conv-host rewritelog
	rm -rf *.decl.h *.def.h core *.log newtraces

test: rewritelog
	./rewritelog < sample-input
