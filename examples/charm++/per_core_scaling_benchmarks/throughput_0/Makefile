CHARMC=../charm_energy/bin/charmc -O3 -march=native

CXX=$(CHARMC)
CXXFLAGS=$(OPTS)
LIBS= -lpapi #/opt/cray/papi/5.4.0.1/lib/libpapi.a
INC=#-I/opt/cray/papi/5.4.0.1/include
BINARY=throughput

all: $(BINARY)

$(BINARY): throughput.o
	$(CHARMC) $(INC) -language charm++ -o throughput throughput.o $(LIBS) $(OPTS) 

throughput.o: throughput.C throughput.decl.h
	$(CHARMC) $(INC) -c throughput.C

throughput.decl.h: throughput.ci
	$(CHARMC) $(INC) throughput.ci

test: $(BINARY)
	./charmrun +p4 ./throughput 50 5 $(TESTOPTS)

clean:
	rm -f *.o *.decl.h *.def.h
