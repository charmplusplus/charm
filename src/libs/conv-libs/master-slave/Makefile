COPTS=-g
CHARMC=../../../../bin/charmc $(COPTS)

all: libcms.a

libcms.a: cms.o
	$(CHARMC) -cp ../../../../lib -o libcms.a cms.o
	rm -f ../../../../include/cms.h
	cp cms.h ../../../../include

cms.o: cms.c
	$(CHARMC) -c cms.c
