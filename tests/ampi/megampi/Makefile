CHARMC=../../../bin/charmc $(OPTS)

pgm: test.o
	$(CHARMC) -o pgm test.o -language ampi -balancer RandCentLB

test.o: test.C
	$(CHARMC) -c test.C

#
#
# clean up .o, .mod, .exe and EMACS backup files
#
clean:
	rm -f *.o *.mod pgm *~ conv-host charmrun

test: pgm
	./charmrun ./pgm +p1 +vp1
	./charmrun ./pgm +p1 +vp2
	./charmrun ./pgm +p1 +vp4
	./charmrun ./pgm +p2 +vp2
	./charmrun ./pgm +p2 +vp4
	./charmrun ./pgm +p2 +vp1

