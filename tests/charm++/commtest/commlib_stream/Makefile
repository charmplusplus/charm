CHARMC=../../../../bin/charmc $(OPTS)

all: hello linear manytomany

hello.decl.h: hello.ci
	$(CHARMC)  hello.ci

clean:
	rm -f *.decl.h *.def.h conv-host *.o hello linear \
	charmrun *~ manytomany

hello : hello.C hello.decl.h
	$(CHARMC) -o hello hello.C -module comlib

linear: linear.C hello.decl.h
	$(CHARMC) -o linear linear.C -module comlib

manytomany : manytomany.ci manytomany.C
	$(CHARMC) manytomany.ci
	$(CHARMC) -o manytomany manytomany.C -module comlib

test: all
	./charmrun hello +p4 10 $(TESTOPTS)
	./charmrun linear +p2 2 $(TESTOPTS)

bgtest: all
	./charmrun hello +p4 10 +x2 +y2 +z1
