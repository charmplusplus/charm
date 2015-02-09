-include ../../../common.mk
OPTS	= -O3
CHARMC	= ../../../../bin/charmc $(OPTS)

OBJS = stencil3d.o

all: stencil3d

stencil3d: $(OBJS)
	$(CHARMC) -language charm++ -module CommonLBs -o stencil3d $(OBJS)

projections: $(OBJS)
	$(CHARMC) -language charm++ -module CommonLBs -tracemode projections -lz -o stencil3d.prj $(OBJS)

summary: $(OBJS)
	$(CHARMC) -language charm++ -module CommonLBs -tracemode summary -lz -o stencil3d.sum $(OBJS)

stencil3d.decl.h: stencil3d.ci
	$(CHARMC)  stencil3d.ci

stencil3d.o: stencil3d.C stencil3d.decl.h
	$(CHARMC) -c stencil3d.C

clean:
	rm -f *.decl.h *.def.h conv-host *.o stencil3d stencil3d.prj charmrun *~

test: stencil3d
	$(call run, +p4 ./stencil3d 64 32 +balancer RefineLB )
	$(call run, +p4 ./stencil3d 64 32 +balancer GreedyLB )

bgtest: stencil3d
	$(call run, +p4 ./stencil3d 32 16 +balancer CommLB +x2 +y2 +z1 +cth1 +wth1 )
