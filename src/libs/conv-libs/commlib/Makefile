CHARMC= ../../../../bin/charmc $(OPTS)
OBJS=commlib.o gridrouter.o de.o treerouter.o petable.o overlapper.o \
     rsend.o bcast.o charm_bind.o

all: libcomm.a
	@rm -f ../../../../include/commlib.h
	@cp commlib.h ../../../../include

libcomm.a: $(OBJS)
	@echo 'building libcomm.a...'
	@$(CHARMC) -cp ../../../../lib -o libcomm.a $(OBJS)

charm_bind.o: charm_bind.C commlib.h
	@echo '.... charm_bind.c...'
	@$(CHARMC)  -c -I../../.. charm_bind.C

commlib.o:  commlib.C commlib_pvt.h
	@echo '.... commlib.C...'
	@$(CHARMC)  -c commlib.C

overlapper.o:  overlapper.C overlapper.h newtypes.h
	@echo '.... overlapper.C...'
	@$(CHARMC)  -c overlapper.C

convfns.o: convfns.c
	@echo '.... convfns.c...'
	@$(CHARMC)  -c convfns.c

treerouter.o:  treerouter.C  treerouter.h commlib.h
	@echo '.... treerouter.C...'
	@$(CHARMC)  -c treerouter.C

gridrouter.o:  gridrouter.C commlib.h gridrouter.h 
	@echo '.... gridrouter.C...'
	@$(CHARMC)  -c gridrouter.C

de.o: de.C commlib.h de.h
	@echo '.... de.C...'
	@$(CHARMC)  -c de.C

bcast.o:  bcast.C commlib.h bcast.h
	@echo '.... bcast.C...'
	@$(CHARMC)  -c bcast.C

rsend.o:  rsend.C commlib.h rsend.h
	@echo '.... rsend.C...'
	@$(CHARMC)  -c rsend.C

petable.o: petable.C petable.h commlib_pvt.h
	@echo '.... petable.C...'
	@$(CHARMC)  -c petable.C

clean:
	rm -f *.o *.a
