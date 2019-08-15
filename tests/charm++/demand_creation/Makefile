-include ../../common.mk
CHARMC=../../../bin/charmc $(OPTS)

OBJS = hello.o

all: hello

hello: $(OBJS)
	$(CHARMC) -language charm++ -o hello $(OBJS)

hello.decl.h: hello.ci
	$(CHARMC) hello.ci

clean:
	rm -f *.decl.h *.def.h conv-host *.o hello charmrun *.log *.sum *.sts

hello.o: hello.C hello.decl.h
	$(CHARMC) -c hello.C

test: all
	$(call run, ./hello +p4 10 )

smptest: all
	$(call run, ./hello +p2 10 ++ppn 2)
	$(call run, ./hello +p4 10 ++ppn 2)

bgtest: all
	$(call run, ./hello +p4 10 +x2 +y2 +z2 +cth1 +wth1 +bglog )
