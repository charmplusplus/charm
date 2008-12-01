CHARMC=../../../../bin/charmc $(OPTS)

OBJS = matmul2d.o

all: matmul2d

matmul2d: $(OBJS)
	$(CHARMC) -language charm++ -o matmul2d $(OBJS)

projections: $(OBJS)
	$(CHARMC) -language charm++ -tracemode projections -o matmul2d.prj $(OBJS)

summary: $(OBJS)
	$(CHARMC) -language charm++ -tracemode summary -lz -o matmul2d.sum $(OBJS)

matmul2d.decl.h: matmul2d.ci
	$(CHARMC)  matmul2d.ci

clean:
	rm -f *.decl.h *.def.h conv-host *.o matmul2d matmul2d.prj charmrun *~ mat.*

matmul2d.o: matmul2d.C matmul2d.h matmul2d.decl.h
	$(CHARMC) -c matmul2d.C 
