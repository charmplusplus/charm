-include ../../common.mk
CHARMC=../../../bin/charmc $(OPTS)

OBJS = grouptest.o

all: grouptest

grouptest: $(OBJS)
	$(CHARMC) -language charm++ -o grouptest $(OBJS)

grouptest.decl.h: grouptest.ci
	$(CHARMC) grouptest.ci

clean:
	rm -f *.decl.h *.def.h conv-host *.o grouptest charmrun *.log *.sum *.sts grouptest.exe grouptest.pdb grouptest.ilk

grouptest.o: grouptest.C grouptest.decl.h
	$(CHARMC) -c grouptest.C

# Test on a single PE and multiple PEs
test: all
	$(call run, ./grouptest +p1)
	$(call run, ./grouptest +p3)

# In SMP mode also test a single process and multiple processes
smptest: all
	$(call run, ./grouptest +p2 ++ppn 2 )
	$(call run, ./grouptest +p4 ++ppn 2 )
