-include ../../common.mk
CHARMC=../../../bin/ampif90 $(OPTS)

all: migrateluntest migrateluntest-pie

migrateluntest:  AMPI_LUN_Migratable.o luntest.o pgm.o
	$(CHARMC) -tlsglobals -fopenmp -o migrateluntest AMPI_LUN_Migratable.o luntest.o pgm.o 

pgm.o: pgm.f90
	$(CHARMC) -tlsglobals -fopenmp -c pgm.f90

luntest.o: luntest.f90
	$(CHARMC) -tlsglobals -fopenmp -c luntest.f90 -o luntest.o

AMPI_LUN_Migratable.o: AMPI_LUN_Migratable.f90
	$(CHARMC) -tlsglobals -fopenmp -c AMPI_LUN_Migratable.f90 -o AMPI_LUN_Migratable.o


migrateluntest-pie:  AMPI_LUN_Migratable-pie.o luntest-pie.o pgm-pie.o
	$(CHARMC) -pieglobals -o migrateluntest-pie AMPI_LUN_Migratable-pie.o luntest-pie.o pgm-pie.o 

pgm-pie.o: pgm.f90
	$(CHARMC) -pieglobals -standalone -c pgm.f90 -o pgm-pie.o

luntest-pie.o: luntest.f90
	$(CHARMC) -pieglobals -standalone  -c luntest.f90 -o luntest-pie.o

AMPI_LUN_Migratable-pie.o: AMPI_LUN_Migratable.f90
	$(CHARMC) -pieglobals -standalone -c AMPI_LUN_Migratable.f90 -o AMPI_LUN_Migratable-pie.o

#
# clean up .o, .mod, .exe and EMACS backup files
#
clean:
	rm -f *.o *.mod migrateluntest *~ conv-host charmrun ampirun

test: migrateluntest
	$(call run, ./migrateluntest +p2 +vp4 +balancer GreedyRefineLB)

testp: migrateluntest
	$(call run, ./migrateluntest +p$(P) +vp$(P) +balancer GreedyRefineLB)
	$(call run, ./migrateluntest +p$(P) +vp$$(( $(P) * 2 )) +balancer GreedyRefineLB)
	$(call run, ./migrateluntest +p$(P) +vp$$(( $(P) * 4 )) +balancer GreedyRefineLB)

test-pie: migrateluntest-pie
	$(call run, ./migrateluntest-pie +p2 +vp4 +balancer GreedyRefineLB)

test-all: test test-pie
