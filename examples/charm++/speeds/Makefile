CHARMC=../../../../bin/charmc $(OPTS)

pgm:  speed.o
	$(CHARMC) speed.o  -o pgm -language charm++

speed.o : speed.C speed.h speed.ci
	$(CHARMC) -c speed.ci speed.C

clean:
	rm -f conv-host pgm *.def.h *.decl.h *.o *~ charmrun
