CHARMC=../../../bin/charmc

pgm: iotest.ci iotest.C
	$(CHARMC) iotest.ci
	$(CHARMC) -g iotest.C -o $@ -module CkIO

clean:
	rm -f *.o *.decl.h *.def.h pgm test*
