CHARMC=../../../bin/charmc $(OPTS) $(MOPTS) 

OBJS = RedExample.o

all: RedExample

RedExample: $(OBJS)
	$(CHARMC) -language charm++ -o RedExample $(OBJS)

RedExample.decl.h: RedExample.ci
	$(CHARMC)  RedExample.ci

clean:
	rm -f *.decl.h *.def.h conv-host *.o RedExample charmrun *.log *.sum *.sts

RedExample.o: RedExample.C RedExample.decl.h RedExample.h
	$(CHARMC) -c RedExample.C

test: all
	./charmrun ./RedExample 20 4.5 10.3 +p4

bgtest: all
	./charmrun ./RedExample 20 4.5 10.3 +p4 +x2 +y2 +z2
