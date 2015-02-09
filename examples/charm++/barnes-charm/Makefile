-include ../../common.mk
# to disable various phases of the simulation:
# -DNO_PARTITION
# -DNO_FORCES
# -DNO_ADVANCE
# -DNO_LB
#
# to control verbosity of program:
# -DVERBOSE_MAIN 
# -DVERBOSE_CHUNKS
# -DVERBOSE_PIECES
#
# to print out a graphical representation of the constructed tree:
# -DPRINT_TREE
#
# to disable timing 
# -DNO_TIME
#
# to enable heap memory corruption checks
# -DMEMCHECK

OPTS= -g -O3 #-DVERBOSE_PIECES -DVERBOSE_MAIN -DPRINT_TREE -DOUTPUT_ACC -DMEMCHECK  -DNO_LB #-DVERBOSE_CHUNKS -DVERBOSE_PIECES -DVERBOSE_MAIN
CHARMC=../../../bin/charmc $(OPTS)

OBJS = TreePiece.o ParticleChunk.o Main.o grav.o util.o code_io.o 
PROJ= #-tracemode projections #-tracemode summary

TARGET = BARNES
all: $(TARGET) 

$(TARGET): $(OBJS)
	$(CHARMC) -language charm++ -module GreedyLB -module RotateLB -module GreedyCommLB -o $(TARGET) $(OBJS) $(PROJ) -memory os

$(TARGET).%: $(TARGET)
	mv $(TARGET) $@
	mv charmrun charmrun.$*

test: $(TARGET)
	$(call run, ./$(TARGET) +LBPeriod 0.0001 +balancer NullLB -in=input_100000 -idepth=2 -fat=100 -tstop=0.225 -NPROC=$(NPROC) $(BARNES_OPTS) +p$(P) ++ppn$(P)) 

grav.o: grav.C barnes.decl.h
	$(CHARMC) -c grav.C

ParticleChunk.o: ParticleChunk.cpp barnes.decl.h barnes.h
	$(CHARMC) -c ParticleChunk.cpp

TreePiece.o: TreePiece.cpp barnes.decl.h barnes.h
	$(CHARMC) -c TreePiece.cpp

Main.o: Main.cpp barnes.decl.h barnes.h
	$(CHARMC) -c Main.cpp

barnes.decl.h: barnes.ci
	$(CHARMC) -c barnes.ci

util.o: util.C
	$(CHARMC) -c util.C

code_io.o: code_io.C barnes.decl.h
	$(CHARMC) -c code_io.C

clean:
	rm -f *.decl.h *.def.h conv-host *.o BARNES charmrun *~ *.core 

realclean: clean
	rm -f out.* *.dot *.ps
