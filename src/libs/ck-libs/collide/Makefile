include ../common.mk

HEADERS=collide_util.h bbox.h collide_cfg.h collide_buffers.h \
	collidecharm.h collidec.h collidef.h collidecharm.decl.h 
HEADDEP=$(HEADERS) collidecharm_impl.h \
	collide_serial.h collide_buffers.h collide_cfg.h \
	collide.decl.h collidecharm.decl.h headers

COBJS=collide_util.o collide_serial.o collidecharm.o collide_buffers.o 
CLIB=libmodulecollidecharm
CDEST=$(LIBDIR)/$(CLIB).a

OBJS=$(COBJS) threadCollide.o
LIB=libmodulecollide
DEST=$(LIBDIR)/$(LIB).a

all: $(DEST) $(CDEST) headers

$(DEST): $(OBJS) $(COMPAT) 
	$(CHARMC) $(OBJS) $(COMPAT) -o $@
	cp $(LIB).dep $(LIBDIR)/$(LIB).dep

$(CDEST): $(COBJS) $(COMPAT) 
	$(CHARMC) $(COBJS) $(COMPAT) -o $@
#	cp $(CLIB).dep $(LIBDIR)/$(CLIB).dep

headers: $(HEADERS)
	cp $(HEADERS) $(CDIR)/include/
	touch headers

collide_util.o: collide_util.C $(HEADDEP)
	$(CHARMC) -c collide_util.C

collide_serial.o: collide_serial.C $(HEADDEP)
	$(CHARMC) -c collide_serial.C

collidecharm.o: collidecharm.C $(HEADDEP)
	$(CHARMC) -c collidecharm.C

collide_buffers.o: collide_buffers.C $(HEADDEP)
	$(CHARMC) -c collide_buffers.C

threadCollide.o: threadCollide.C $(HEADDEP)
	$(CHARMC) -c threadCollide.C

collide.decl.h collide.def.h: INTERFACE.threadCollide

INTERFACE.threadCollide: threadCollide.ci
	$(CHARMC) threadCollide.ci
	touch $@

collidecharm.decl.h collidecharm.def.h: INTERFACE.collidecharm

INTERFACE.collidecharm: collidecharm.ci
	$(CHARMC) collidecharm.ci
	touch $@

call_init.o: call_init.c
	$(CHARMC) -c call_init.c

clean:
	rm -rf *.a *.def.h *.decl.h *.o SunWS_cache $(DEST) headers INTERFACE.*

