CDIR=../../../..
CHARMC=$(CDIR)/bin/charmc $(OPTS)

HEADERS=collision.h collidef.h \
	col_aggregate.h parCollide.decl.h collide.decl.h
OBJS = collision.o parCollide.o threadCollide.o \
	aggregate.o 

DEST=$(CDIR)/lib/libmodulecollide.a

all: $(DEST)

$(DEST): $(OBJS) $(CDIR)/include/collision.h
	$(CHARMC) $(OBJS) -o $@

$(CDIR)/include/collision.h: $(HEADERS)
	cp $(HEADERS) $(CDIR)/include/

model_corr.o: model_corr.C $(HEADERS)
	$(CHARMC) $(CFLAGS) -c $<

collision.o: collision.C $(HEADERS)
	$(CHARMC) $(CFLAGS) -c $<

aggregate.o: aggregate.C $(HEADERS)
	$(CHARMC) $(CFLAGS) -c $<

parCollide.o: parCollide.C $(HEADERS)
	$(CHARMC) $(CFLAGS) -c $<

threadCollide.o: threadCollide.C $(HEADERS)
	$(CHARMC) $(CFLAGS) -c $<

parCollide.decl.h: parCollide.ci
	$(CHARMC) parCollide.ci

collide.decl.h: threadCollide.ci
	$(CHARMC) threadCollide.ci


test_par.o: test_par.C $(HEADERS)
	$(CHARMC) $(CFLAGS) -c $<

test_par.decl.h: test_par.ci
	$(CHARMC) test_par.ci

clean: 
	-rm -fr *.o *~ *.decl.h *.def.h gmon.out $(DEST) conv-host charmrun

