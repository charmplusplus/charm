CHARMC= ../../../../bin/charmc $(OPTS)
OBJS=packc.o packf77.o
MYINCPATH= ../../../../include

all: libpacklib.a
	@/bin/rm -f $(MYINCPATH)/PackLib.h
	@/bin/cp PackLib.h $(MYINCPATH)
	@/bin/rm -f $(MYINCPATH)/cpacklib.h
	@/bin/cp cpacklib.h $(MYINCPATH)

libpacklib.a: $(OBJS)
	@echo 'building libpacklib.a...'
	@$(CHARMC) -cp ../../../../lib -o libpacklib.a $(OBJS)

packc.o: packc.C PackLib.h
	@echo '.... packc.C...'
	@$(CHARMC) -c packc.C

packf77.o: packf77.C PackLib.h
	@echo '.... packf77.C...'
	@$(CHARMC) -c packf77.C
