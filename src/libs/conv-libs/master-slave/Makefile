CHARMC=../../../../bin/charmc $(OPTS)

all: libcms.a

libcms.a: cms.o
	@echo 'buidling libcms.a...'
	@$(CHARMC) -cp ../../../../lib -o libcms.a cms.o
	@rm -f ../../../../include/cms.h
	@cp cms.h ../../../../include

cms.o: cms.c
	@echo '.... cms.c...'
	@$(CHARMC) -c cms.c
