CHARMC=../../../../bin/charmc $(OPTS)

OBJS = matmul3d.o

all: matmul3d

matmul3d: $(OBJS)
	$(CHARMC) -language charm++ -o matmul3d $(OBJS)

projections: $(OBJS)
	$(CHARMC) -language charm++ -tracemode projections -o matmul3d.prj $(OBJS)

summary: $(OBJS)
	$(CHARMC) -language charm++ -tracemode summary -lz -o matmul3d.sum $(OBJS)

matmul3d.decl.h: matmul3d.ci
	$(CHARMC)  matmul3d.ci

clean:
	rm -f *.decl.h *.def.h conv-host *.o matmul3d matmul3d.prj charmrun *~

matmul3d.o: matmul3d.C matmul3d.h matmul3d.decl.h
	$(CHARMC) -c matmul3d.C
