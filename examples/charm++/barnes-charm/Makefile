OPTS=-g -DPRINT_TREE
CHARMC=../../../multicore-linux64/bin/charmc $(OPTS)

OBJS = TreePiece.o ParticleChunk.o Main.o 
#PROJ= -tracemode projections -tracemode summary

all: barnes 

barnes: $(OBJS)
	$(CHARMC) -language charm++ -o barnes $(OBJS) $(PROJ) 

ParticleChunk.o: ParticleChunk.cpp barnes.decl.h
	$(CHARMC) -c ParticleChunk.cpp

TreePiece.o: TreePiece.cpp barnes.decl.h
	$(CHARMC) -c TreePiece.cpp

Main.o: Main.cpp barnes.decl.h
	$(CHARMC) -c Main.cpp

barnes.decl.h: barnes.ci
	$(CHARMC) -c barnes.ci

clean:
	rm -f *.decl.h *.def.h conv-host *.o barnes charmrun *~ *.core 
