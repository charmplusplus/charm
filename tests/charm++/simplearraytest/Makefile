-include ../../common.mk
CHARMC=../../../bin/charmc $(OPTS)

OBJS = arraytest.o

all: arraytest

arraytest: $(OBJS)
	$(CHARMC) -language charm++ -o arraytest $(OBJS)

arraytest.decl.h: arraytest.ci
	$(CHARMC) arraytest.ci

clean:
	rm -f *.decl.h *.def.h conv-host *.o arraytest charmrun *.log *.sum *.sts arraytest.exe arraytest.pdb arraytest.ilk

arraytest.o: arraytest.C arraytest.decl.h
	$(CHARMC) -c arraytest.C

# Test on a single PE and multiple PEs
test: all
	$(call run, ./arraytest +p1)
	$(call run, ./arraytest +p3)

# In SMP mode also test a single process and multiple processes
smptest: all
	$(call run, ./arraytest +p2 ++ppn 2 )
	$(call run, ./arraytest +p4 ++ppn 2 )

bgtest: all
	$(call run, ./arraytest +p4 +x2 +y2 +z2 +cth1 +wth1 )
