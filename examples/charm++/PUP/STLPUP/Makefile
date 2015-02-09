-include ../../../common.mk
CHARMC=../../../../bin/charmc $(OPTS)

OBJS = SimplePUP.o

all: SimplePUP

SimplePUP: $(OBJS)
	$(CHARMC) -language charm++ -o SimplePUP $(OBJS)

SimplePUP.decl.h: SimplePUP.ci
	$(CHARMC)  SimplePUP.ci

clean:
	rm -f *.decl.h *.def.h conv-host *.o SimplePUP charmrun

SimplePUP.o: SimplePUP.C SimplePUP.h HeapObjectSTL.h SimplePUP.decl.h
	$(CHARMC) -c SimplePUP.C

test: all
	$(call run, ./SimplePUP +p4  )
	$(call run, ./SimplePUP +p4  )

bgtest: all
	$(call run, ./SimplePUP +p4 +x2 +y2 +z1)
