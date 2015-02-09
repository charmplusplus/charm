-include ../../../common.mk
OPTS	= -O3
CHARMC	= ../../../../bin/charmc $(OPTS)
SRC	= ../../../../src

OBJS = stencil3d.o

all: stencil3d-ckdirect stencil3d-msg

stencil3d-ckdirect: stencil3d-ckdirect.o
	$(CHARMC) -language charm++ -o stencil3d-ckdirect stencil3d-ckdirect.o

stencil3d-msg: stencil3d-msg.o
	$(CHARMC) -language charm++ -o stencil3d-msg stencil3d-msg.o

stencil3d.decl.h: stencil3d.ci
	$(CHARMC)  stencil3d.ci

clean:
	rm -f *.decl.h *.def.h conv-host *.o stencil3d charmrun stencil3d-ckdirect stencil3d-msg

stencil3d-ckdirect.o: stencil3d.C stencil3d.decl.h
	$(CHARMC) -o stencil3d-ckdirect.o -DUSE_CKDIRECT stencil3d.C 

stencil3d-msg.o: stencil3d.C stencil3d.decl.h
	$(CHARMC) -o stencil3d-msg.o -DUSE_MESSAGES stencil3d.C 

# these test values are for sanity checking.  Wider and larger favors RDMA
test: all
	$(call run, ./stencil3d-msg +p4 20 10 100 2 2 2 10 )
	$(call run, ./stencil3d-ckdirect +p4 20 10 100 2 2 2 10 )
