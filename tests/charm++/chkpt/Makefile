CHARMC=../../../bin/charmc $(OPTS)

OBJS = hello.o

all: hello
	rm -rf log/

hello: $(OBJS)
	$(CHARMC) -language charm++ -o hello hello.o $(LIBS)

hello.decl.h: hello.ci
	$(CHARMC)  hello.ci

clean:
	rm -f *~ *.decl.h *.def.h conv-host *.o hello charmrun log/

hello.o: hello.C hello.decl.h
	$(CHARMC) -c hello.C

test: all
	-rm -fr log
	./charmrun hello +p3
	./charmrun hello +p3 +restart log 
	./charmrun hello +p4 +restart log 
