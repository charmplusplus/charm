CHARMC=../../../../bin/charmc $(OPTS)
LIBDIR=../../../../lib

all: $(LIBDIR)/libcms.a

$(LIBDIR)/libcms.a: cms.o
	$(CHARMC) -o $(LIBDIR)/libcms.a cms.o
	rm -f ../../../../include/cms.h
	cp cms.h ../../../../include

cms.o: cms.c
	$(CHARMC) -c cms.c

clean:
	rm -f *.o $(LIBDIR)/libcms.a
