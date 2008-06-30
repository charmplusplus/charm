CDIR=../../../..
CHARMC=$(CDIR)/bin/charmc $(OPTS)

OBJS=netfem.o netfem_data.o
HEADERS=netfem.h netfemf.h netfem_data.h
DEST=$(CDIR)/lib/libmodulenetfem.a

all: $(DEST) headers

$(DEST): $(OBJS)
	$(CHARMC) -o $(DEST) $(OBJS)

netfem.o: netfem.C netfem.decl.h $(HEADERS)
	$(CHARMC) -c netfem.C

netfem_data.o: netfem_data.C $(HEADERS)
	$(CHARMC) -c netfem_data.C

netfem.decl.h: netfem.ci
	$(CHARMC) netfem.ci

headers: $(HEADERS)
	/bin/cp $(HEADERS) $(CDIR)/include

clean:
	rm -rf *.a *.def.h *.decl.h *.o SunWS_cache

# Text-only client
textclient: textclient.o $(DEST)
	$(CHARMC) textclient.o $(DEST) -o $@ -lccs-client

textclient.o: textclient.C
	$(CHARMC) -c textclient.C
