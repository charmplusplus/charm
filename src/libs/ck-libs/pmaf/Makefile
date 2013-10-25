OBJECTS=ref.o element.o chunk.o pgm.o
CHARMC=/expand8/home/wilmarth/charm/net-linux/bin/charmc $(OPTS)

all:	$(OBJECTS)
	$(CHARMC) $(OBJECTS) -o pgm -language charm++

pgm.o:	pgm.C pgm.h Pgm.def.h Pgm.decl.h 
	$(CHARMC) -c pgm.C

Pgm.def.h Pgm.decl.h:	pgm.ci
	$(CHARMC) pgm.ci

chunk.o:chunk.C chunk.h PMAF.def.h PMAF.decl.h
	$(CHARMC) -c chunk.C

PMAF.def.h PMAF.decl.h:	chunk.ci
	$(CHARMC) chunk.ci

element.o:	element.C element.h
	$(CHARMC) -c element.C

ref.o:	ref.C ref.h PMAF.def.h PMAF.decl.h
	$(CHARMC) -c ref.C

clean:	
	rm -f *.*de*.h *.o pgm 
