CHARMC=../../../../bin/charmc $(OPTS)

OBJS = hello.o

all: hello

hello: $(OBJS)
	$(CHARMC) -module comlib -language charm++ -o hello $(OBJS)

hello.decl.h: hello.ci
	$(CHARMC)  hello.ci

clean:
	rm -f *.decl.h *.def.h conv-host *.o hello charmrun *.log *.sum *.sts

hello.o: hello.C hello.decl.h
	$(CHARMC) -c hello.C

test: all
	./charmrun hello +p1 1
	./charmrun hello +p1 2
	./charmrun hello +p2 2
	./charmrun hello +p4 10 2
	./charmrun hello +p4 10 3
	./charmrun hello +p4 10 4
	./charmrun hello +p4 10 5
	./charmrun hello +p2 1

bgtest: all
	./charmrun hello +p1 1 +x1 +y1 +z1
	./charmrun hello +p1 2 +x1 +y1 +z1
	./charmrun hello +p2 2 +x2 +y1 +z1
	./charmrun hello +p4 10 2 +x2 +y2 +z1
	./charmrun hello +p4 10 3 +x2 +y2 +z1
	./charmrun hello +p4 10 4 +x2 +y2 +z1
	./charmrun hello +p4 10 5 +x2 +y2 +z1
	./charmrun hello +p2 1 +x2 +y1 +z1

