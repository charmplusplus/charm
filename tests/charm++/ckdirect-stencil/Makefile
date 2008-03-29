OPTS=-O3
CHARMC=../../../bin/charmc $(OPTS)
SRC=../../../src

OBJS = stencil3d.o

all: stencil3d-ckdirect stencil3d-msg

stencil3d-ckdirect: stencil3d-ckdirect.o
	$(CHARMC) -language charm++ -o stencil3d-ckdirect  stencil3d-ckdirect.o

stencil3d-msg: stencil3d-msg.o
	$(CHARMC) -language charm++ -o stencil3d-msg  stencil3d-msg.o



stencil3d.decl.h: stencil3d.ci
	$(CHARMC)  stencil3d.ci

clean:
	rm -f *.decl.h *.def.h conv-host *.o stencil3d charmrun stencil3d-ckdirect stencil3d-msg

stencil3d-ckdirect.o: stencil3d.C stencil3d.decl.h
	$(CHARMC) -o stencil3d-ckdirect.o -DUSE_CKDIRECT  stencil3d.C 

stencil3d-msg.o: stencil3d.C stencil3d.decl.h
	$(CHARMC) -o stencil3d-msg.o -DUSE_MESSAGES  stencil3d.C 

test: all
	./charmrun stencil3d +p4 20 5 100
