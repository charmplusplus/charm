-include ../../../common.mk
CHARMC=../../../../bin/charmc $(OPTS)

OBJS = hello.o TestUserDataLB.o
LDFLAGS = -L. -language charm++ -module TestUserDataLB

all: hello

hello: $(OBJS) libmoduleTestUserDataLB.a
	$(CHARMC) -language charm++ -o hello $(OBJS) $(LDFLAGS)

hello.decl.h: hello.ci
	$(CHARMC)  hello.ci

libmoduleTestUserDataLB.a: TestUserDataLB.o
	$(CHARMC)  -o libmoduleTestUserDataLB.a TestUserDataLB.o

TestUserDataLB.decl.h: TestUserDataLB.ci
	$(CHARMC)  TestUserDataLB.ci

TestUserDataLB.o: TestUserDataLB.C TestUserDataLB.decl.h
	$(CHARMC) -c TestUserDataLB.C

clean:
	rm -f *.decl.h *.def.h conv-host *.o *.a hello charmrun

hello.o: hello.C hello.decl.h
	$(CHARMC) -c hello.C

test: all
	$(call run, ./hello +p4 10 100 10 +balancer TestUserDataLB )

bgtest: all
	$(call run, ./hello +p4 10 +x2 +y2 +z1 )
