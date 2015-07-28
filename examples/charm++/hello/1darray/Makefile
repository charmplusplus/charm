-include ../../../common.mk
LD=-ld clang++
CXX_OPT=$(shell clamp-config --build --cxxflags | tr ' ' "\n" |  sed -e 's/^/\-c\+\+\-option\ /g' | xargs)
LDXX_OPT=$(shell clamp-config --build --ldflags | tr ' ' "\n" |  sed -e 's/^/\-ld\+\+\-option\ /g' | xargs)
OPTS+=$(CXX_OPT) $(LD) $(LDXX_OPT)
CHARMC=../../../../bin/charmc $(OPTS)

OBJS = hello.o

all: hello

hello: $(OBJS)
	$(CHARMC) -language charm++ -o hello $(OBJS)

hello.decl.h: hello.ci
	$(CHARMC)  hello.ci

clean:
	rm -f *.decl.h *.def.h conv-host *.o hello charmrun

hello.o: hello.C hello.decl.h
	$(CHARMC) -c hello.C

test: all
	$(call run, ./hello +p4 10 )

bgtest: all
	$(call run, ./hello +p4 10 +x2 +y2 +z1 )
