CHARM_BASE_DIR = ../../../..
CHARM_BIN_DIR = $(CHARM_BASE_DIR)/bin
CHARMC = $(CHARM_BIN_DIR)/charmc $(OPTS)

OBJS = hello.o

all: hello


hello: $(OBJS)
	$(CHARMC) -language charm++ -o hello $(OBJS)

hello.decl.h hello.def.h: hello.ci
	$(CHARMC)  hello.ci

hello.o: hello.C hello.decl.h hello.def.h
	$(CHARMC) -c hello.C


clean:
	rm -f *.decl.h *.def.h conv-host *.o hello charmrun hello_shared *genSPECode*
