CHARMC=../../../../bin/charmc $(OPTS)

all: libcms.a

libcms.a: cms.o
	$(CHARMC) -o ../../../../lib/libcms.a cms.o
	rm -f ../../../../include/cms.h
	cp cms.h ../../../../include

cms.o: cms.c
	$(CHARMC) -c cms.c

clean:
	rm -f *.o libcms.a
