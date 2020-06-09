CDIR=../../../..
CHARMC=$(CDIR)/bin/charmc $(OPTS)
DEST=$(CDIR)/lib/libmodulechannel.a

all: $(DEST) $(CDIR)/include/channel.h $(CDIR)/include/channel.decl.h $(CDIR)/include/channel.def.h

$(DEST): channel.o
	$(CHARMC) -o $(DEST) channel.o

channel.o: channel.C channel.h channel.decl.h channel.def.h
	$(CHARMC) -c channel.C

channel.decl.h channel.def.h: channel.ci
	$(CHARMC) channel.ci

$(CDIR)/include/channel.def.h: channel.def.h
	/bin/cp channel.def.h $(CDIR)/include

$(CDIR)/include/channel.decl.h: channel.decl.h
	/bin/cp channel.decl.h $(CDIR)/include

$(CDIR)/include/channel.h: channel.h
	/bin/cp channel.h $(CDIR)/include

clean:
	rm -f *.decl.h *.def.h *.o $(DEST)
