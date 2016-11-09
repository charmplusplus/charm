CDIR=$(shell readlink -f ../../../..)
CHARMC=$(CDIR)/bin/charmc 
CFLAGS=$(OPTS)
export CharmDIR=$(CDIR)
all: llvm_omp ompcharm 

llvm_omp: omp

omp:
	if [ ! -d runtime/build ]; then mkdir runtime/build; fi;\
	cd runtime/build;\
	if [ ! -f ./Makefile ]; then cmake ../ -DCMAKE_C_COMPILER=$(CHARMC) -DCMAKE_CXX_COMPILER=$(CHARMC) -DLIBOMP_CPPFLAGS="$(OPTS)" -DLIBOMP_CFLAGS="$(OPTS)"; fi; cd ../../; \
	make $(MFLAGS) -C runtime/build

ompcharm: llvm_omp
	make $(MFLAGS) -C runtime/$@

clean:
	rm -rf runtime/build runtime/ompcharm/*.o runtime/ompcharm/*.def.h runtime/ompcharm/*.decl.h
