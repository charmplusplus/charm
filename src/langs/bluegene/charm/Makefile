#OPTS=-O -DCMK_OPTIMIZE=1
#OPTS=-g -pg -DCMK_OPTIMIZE=1
OPTS=-g

#CHARMC=/rmount/paci/uiuc/ux452814/charm/mpi-sp/bin/charmc $(OPTS)
#CHARMC=/expand6/software/converse/charm/net-sol-cc/bin/charmc $(OPTS)
#CHARMC=/home/cheelee/work/cvs-projects/charm/net-linux/bin/charmc $(OPTS)
#CHARMC=$(HOME)/charm/net-sol-cc/bin/charmc $(OPTS)
CHARMC=$(HOME)/charm/net-linux/bin/charmc $(OPTS)

RING_OBJ=BlueGene.o ring.o
LINE_OBJ=BlueGene.o line.o
PRIME_OBJ=BlueGene.o prime.o

all: ring line prime

BlueGene.o: BlueGene.C BlueGene.h BlueGene.ci
	$(CHARMC) BlueGene.ci
	$(CHARMC) -c BlueGene.C

ring: $(RING_OBJ)
	$(CHARMC) $(RING_OBJ) -o ring -language charm++

prime: $(PRIME_OBJ)
	$(CHARMC) $(PRIME_OBJ) -o prime -language charm++

line: $(LINE_OBJ)
	$(CHARMC) $(LINE_OBJ) -o line -language charm++

ring.o: ring.C BlueGene.h BlueGene.ci
	$(CHARMC) ring.C

prime.o: prime.C BlueGene.h BlueGene.ci
	$(CHARMC) prime.C

line.o: line.C BlueGene.h BlueGene.ci
	$(CHARMC) line.C

clean:
	$(RM) -r *def.h *decl.h *.o charmrun conv-host \
	ring maxReduceNV line octo jacobi3D ping time prime
