CDIR=$(CharmDIR)
CHARMC=$(CDIR)/bin/charmc $(OPTS)
INC= -I../src -I../build/src -DCHARM_OMP
MODULE = OmpCharm
LIB = $(CDIR)/lib/libmodule$(MODULE).a
LIBOBJ= OmpCharm.o 
CIFILES= OmpCharm.ci
OMPOBJ=$(shell find ../build/src/CMakeFiles/omp.dir -name "*.o")

all: $(LIB)  

$(LIB): $(LIBOBJ) $(OMPOBJ)
	$(CHARMC) -language charm++ $(INC) -o $@ $^
$(MODULE).o:  $(MODULE).C $(MODULE).decl.h
	$(CHARMC) $(INC) -c $<  
$(MODULE).decl.h: OmpCharm.ci
	$(CHARMC) -c $<

clean:
	rm -f *.o *.decl.h *.def.h $(LIB) 
