OPTS=-g
CHARMC=../../../bin/charmc $(OPTS)
SRC=../../../src

OBJS = stencil3d.o

all: stencil3d

stencil3d: $(OBJS)
	$(CHARMC) -language charm++ -o stencil3d $(OBJS)

stencil3d.decl.h: stencil3d.ci
	$(CHARMC)  stencil3d.ci

clean:
	rm -f *.decl.h *.def.h conv-host *.o stencil3d charmrun

stencil3d.o: stencil3d.C stencil3d.decl.h
	$(CHARMC) -I$(SRC)/ck-core -I$(SRC)/conv-core -USE_CKDIRECT -c stencil3d.C

test: all
	./charmrun stencil3d +p4 20 5 100
