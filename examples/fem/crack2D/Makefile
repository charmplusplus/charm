CHARMC=../../../../bin/charmc $(OPTS)
all: getmesh pgm

getmesh: getmesh.C
	$(CHARMC) -seq -o getmesh getmesh.C -language c++

pgm: driver.o readfile.o volume.o lst_NL.o lst_coh2.o
	$(CHARMC) -o pgm readfile.o volume.o lst_NL.o lst_coh2.o driver.o \
	  -language fem 

readfile.o: readfile.C crack.h
	$(CHARMC) -c readfile.C

driver.o: driver.C crack.h
	$(CHARMC) -c driver.C

volume.o: volume.C crack.h
	$(CHARMC) -c volume.C

lst_NL.o: lst_NL.C crack.h
	$(CHARMC) -c lst_NL.C

lst_coh2.o: lst_coh2.C crack.h
	$(CHARMC) -c lst_coh2.C

clean:
	rm -rf *.o pgm *.sts *.log *.bak *~ SunWS_cache ii_files ti_files
	rm -f conv-host getmesh *.mesh meshdata.*
