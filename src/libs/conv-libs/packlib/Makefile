CHARMC= ../../../../bin/charmc $(OPTS)
OBJS=packc.o packf77.o
MYINCPATH= ../../../../include

all: libpacklib.a
	/bin/rm -f $(MYINCPATH)/PackLib.h
	/bin/cp PackLib.h $(MYINCPATH)
	/bin/rm -f $(MYINCPATH)/cpacklib.h
	/bin/cp cpacklib.h $(MYINCPATH)

libpacklib.a: $(OBJS)
	$(CHARMC) -cp ../../../../lib -o libpacklib.a $(OBJS)

packc.o: packc.C PackLib.h
	$(CHARMC) -c packc.C

packf77.o: packf77.C PackLib.h
	$(CHARMC) -c packf77.C

clean:
	rm -f *.o libpacklib.a 
