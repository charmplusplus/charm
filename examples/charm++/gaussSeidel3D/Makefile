OPTS	= -O3 
CHARMC	= ../../../bin/charmc $(OPTS)

OBJS = gaussSeidel3d.o

all: gaussSeidel3d gaussSeidel3d.prj

gaussSeidel3d: $(OBJS)
	$(CHARMC) -language charm++ -o gaussSeidel3d $(OBJS)

gaussSeidel3d.prj: $(OBJS)
	$(CHARMC) -language charm++ -tracemode projections -o gaussSeidel3d.prj $(OBJS)

gaussSeidel3d.sum: $(OBJS)
	$(CHARMC) -language charm++ -tracemode summary -o gaussSeidel3d.sum $(OBJS)

gaussSeidel3d.decl.h: gaussSeidel3d.ci
	$(CHARMC)  gaussSeidel3d.ci

gaussSeidel3d.o: gaussSeidel3d.C gaussSeidel3d.decl.h
	$(CHARMC) -c gaussSeidel3d.C


clean:
	rm -f *.decl.h *.def.h conv-host *.o gaussSeidel3d gaussSeidel3d.prj charmrun *~ gaussSeidel3d.prj.*.log gaussSeidel3d.prj.projrc  gaussSeidel3d.prj.sts