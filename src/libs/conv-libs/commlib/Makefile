CHARMC= ../../../../bin/charmc $(OPTS)
OBJS=commlib.o gridrouter.o de.o treerouter.o petable.o overlapper.o \
     rsend.o bcast.o

all: libcomm.a
	rm -f ../../../../include/commlib.h
	cp commlib.h ../../../../include

libcomm.a: $(OBJS)
	$(CHARMC) -cp ../../../../lib -o libcomm.a $(OBJS)

charm_bind.o: charm_bind.C commlib.h
	$(CHARMC)  -c charm_bind.C

commlib.o:  commlib.C commlib_pvt.h
	$(CHARMC)  -c commlib.C

overlapper.o:  overlapper.C overlapper.h newtypes.h
	$(CHARMC)  -c overlapper.C

convfns.o: convfns.c
	$(CHARMC)  -c convfns.c

treerouter.o:  treerouter.C  treerouter.h commlib.h
	$(CHARMC)  -c treerouter.C

gridrouter.o:  gridrouter.C commlib.h gridrouter.h 
	$(CHARMC)  -c gridrouter.C

de.o: de.C commlib.h de.h
	$(CHARMC)  -c de.C

bcast.o:  bcast.C commlib.h bcast.h
	$(CHARMC)  -c bcast.C

rsend.o:  rsend.C commlib.h rsend.h
	$(CHARMC)  -c rsend.C

petable.o: petable.C petable.h commlib_pvt.h
	$(CHARMC)  -c petable.C

clean:
	rm -f *.o *.a
