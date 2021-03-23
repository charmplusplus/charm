-include ../../common.mk
CHARMC=../../../bin/charmc $(OPTS)

all: whenidle

whenidle:  idlework.o
	$(CHARMC) idlework.o  -o whenidle -language charm++

idlework.o : idlework.C idlework.h idlework.ci
	$(CHARMC) -c idlework.ci idlework.C

test: whenidle
	$(call run, +p4 ./whenidle )

clean:
	rm -f conv-host whenidle *.def.h *.decl.h *.o *~ charmrun charmrun.exe whenidle.exe whenidle.pdb whenidle.ilk
