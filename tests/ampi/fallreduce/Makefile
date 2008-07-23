CHARMC=../../../bin/charmc $(OPTS)

pgm: test.o
	$(CHARMC) -o pgm test.o -language ampif

test.o: test.f90
	$(CHARMC) -c test.f90

#
# clean up .o, .mod, .exe and EMACS backup files
#
clean:
	rm -f *.o *.mod pgm *~ conv-host charmrun
