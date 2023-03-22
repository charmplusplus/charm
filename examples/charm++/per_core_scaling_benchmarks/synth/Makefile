CHARMC=/dcsdata/home/kchndrs2/Research/lb/power/bilge_work/charm_energy/bin/charmc -O3 -march=native
#CHARMC=/scratch/bilge/charm/multicore-linux64/bin/charmc -O3 -march=native #comparison
#CHARMC=/scratch/bilge/charm/netlrts-linux-x86_64/bin/charmc -O3 -march=native #projections
#CHARMC=/scratch/bilge/per_core_scaling/charm_per_core/multicore-linux64_energyOpt/bin/charmc -O3 -march=native

CXX=$(CHARMC)
CXXFLAGS=$(OPTS)
LIBS= #-lpapi /opt/cray/papi/5.4.0.1/lib/libpapi.a
INC=#-I/opt/cray/papi/5.4.0.1/include
BINARY=main

all: $(BINARY)

$(BINARY): main.o
	#$(CHARMC) $(INC) -language charm++ -module energyOpt -o main main.o $(LIBS) $(OPTS) 
	$(CHARMC) $(INC) -language charm++ -o main main.o $(LIBS) $(OPTS) 
	#$(CHARMC) $(INC) -language charm++ -tracemode projections -o main main.o $(LIBS) $(OPTS) 

main.o: main.C main.decl.h
	$(CHARMC) $(INC) -c main.C

main.decl.h: main.ci
	$(CHARMC) $(INC) main.ci

test: $(BINARY)
	sudo ./charmrun +p4 ./main 128 1000 +pemap 0-3 ++local ++ppn 4
	#./charmrun +p4 ./main 50 5 $(TESTOPTS)

clean:
	rm -f *.o *.decl.h *.def.h
	rm -f *.log.gz *.sts *.projrc *.topo
	rm -rf *.log.*
