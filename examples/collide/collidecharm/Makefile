-include ../../common.mk
CHARMC=../../../bin/charmc $(OPTS)

OBJS = hello.o

all:	cifiles hello

hello: $(OBJS)
	$(CHARMC) -language charm++ -module collidecharm -o hello $(OBJS)

cifiles: hello.ci
	$(CHARMC)  hello.ci

clean:
	rm -f *.decl.h *.def.h conv-host *.o hello charmrun

hello.o: hello.C 
	$(CHARMC) -c hello.C

test: all
	$(call run, ./hello +p4 10 )
