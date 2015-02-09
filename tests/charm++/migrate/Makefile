-include ../../common.mk
CHARMC=../../../bin/charmc $(OPTS)

OBJS = migrateHello.o

all: migrateHello

migrateHello: $(OBJS)
	$(CHARMC) -language charm++ -o migrateHello $(OBJS)

migrateHello.decl.h: migrateHello.ci
	$(CHARMC)  migrateHello.ci

clean:
	rm -f *.decl.h *.def.h conv-host *.o migrateHello charmrun

migrateHello.o: migrateHello.C migrateHello.decl.h
	$(CHARMC) -c migrateHello.C

test: all
	$(call run, ./migrateHello +p2 10000 1 )
	$(call run, ./migrateHello +p2 10000 4096 )
	$(call run, ./migrateHello +p2 10000 65536 )

bgtest: all
	$(call run, ./migrateHello +p2 10000 +x2 +y1 +z1 )
