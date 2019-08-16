-include ../../common.mk
CHARMC=../../../bin/ampicxx $(OPTS)

all: datatype

datatype: datatype.o
	$(CHARMC) -o datatype datatype.o

datatype.o: datatype.C
	$(CHARMC) -c datatype.C

test: datatype
	$(call run, ./datatype +p1 +vp4)

clean:
	rm -f *.o *.mod datatype *~ conv-host charmrun ampirun datatype.o charmrun.exe datatype.exe datatype.pdb datatype.ilk
