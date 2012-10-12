CHARMC=../../../bin/charmc $(OPTS)

OBJS = hello.o

all:	cifiles hello

hello: $(OBJS)
	$(CHARMC) -language ampi -module collide -o hello $(OBJS)

cifiles: hello.ci
	$(CHARMC)  hello.ci

clean:
	rm -f *.decl.h *.def.h conv-host *.o hello charmrun

hello.o: hello.C 
	$(CHARMC) -c hello.C

test: all
	./charmrun hello +p4 +vp10
