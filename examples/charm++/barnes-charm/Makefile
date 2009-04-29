OPTS=-g -DPRINT_TREE -DPARTITION -DFORCES
CHARMC=../../../multicore-linux64/bin/charmc $(OPTS)

OBJS = TreePiece.o ParticleChunk.o Main.o grav.o 
#PROJ= -tracemode projections -tracemode summary

all: barnes 

barnes: $(OBJS)
	$(CHARMC) -language charm++ -o barnes $(OBJS) $(PROJ) 

grav.o: grav.C barnes.h
	$(CHARMC) -c grav.C

ParticleChunk.o: ParticleChunk.cpp barnes.decl.h barnes.h
	$(CHARMC) -c ParticleChunk.cpp

TreePiece.o: TreePiece.cpp barnes.decl.h barnes.h
	$(CHARMC) -c TreePiece.cpp

Main.o: Main.cpp barnes.decl.h barnes.h
	$(CHARMC) -c Main.cpp

barnes.decl.h: barnes.ci
	$(CHARMC) -c barnes.ci

clean:
	rm -f *.decl.h *.def.h conv-host *.o barnes charmrun *~ *.core 

realclean: clean
	rm -f out.* *.dot *.ps
