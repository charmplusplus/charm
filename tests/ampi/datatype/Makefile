-include ../../common.mk
CHARMC=../../../bin/ampicxx $(OPTS)

all: datatype addresses

datatype: datatype.o
	$(CHARMC) -o datatype datatype.o

datatype.o: datatype.C
	$(CHARMC) -c datatype.C

addresses: addresses.o
	$(CHARMC) -o addresses addresses.o

addresses.o: addresses.C
	$(CHARMC) -c addresses.C

test: datatype
	$(call run, ./datatype +p1 +vp4)

test_addresses: addresses
	$(call run, ./addresses +p1 +vp1 ++local ++quiet)

bgtest: datatype
	$(call run, ./datatype +p1 +vp4 +x1 +y1 +z1 )

clean:
	rm -f *.o *.mod datatype *~ conv-host charmrun ampirun datatype.o charmrun.exe datatype.exe datatype.pdb datatype.ilk
