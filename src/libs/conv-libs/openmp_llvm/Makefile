CDIR=$(shell ./readlink.sh ../../../..)
CHARMC="$(CDIR)/bin/charmc"
override OPTS+=-seq
CFLAGS=$(OPTS)
CXXFLAGS=$(OPTS)
export CharmDIR=$(CDIR)
all: llvm_omp ompcharm

llvm_omp: omp

omp:
	if [ ! -d runtime/build ]; then mkdir runtime/build; fi;\
	cd runtime/build;\
	if [ ! -f ./Makefile ]; then cmake ../ -DCMAKE_C_COMPILER="$(CHARMC)" -DCMAKE_CXX_COMPILER="$(CHARMC)" -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_FLAGS="$(OPTS)" -DCMAKE_CXX_FLAGS="$(OPTS)"; fi; cd ../../; \
	make $(MFLAGS) -C runtime/build

ompcharm: llvm_omp
	make $(MFLAGS) -C runtime/$@

clean:
	rm -rf runtime/build runtime/ompcharm/*.o runtime/ompcharm/*.def.h runtime/ompcharm/*.decl.h
