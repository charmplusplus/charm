OBJS=pvmc_buf.o pvmc_comm.o pvmc_conv.o pvmc_main.o pvmc_pack.o pvmc_groups.o

# CC=cc -Aa -D_HPUX_SOURCE -g -DPVM_DEBUG
CC=cc -Aa -D_HPUX_SOURCE
CFLAGS=-O -I$(CHARM)/include
LIBS=-L. -lpvmc

all:	libpvmc.a

pvmctest:     	pvmctest.o libpvmc.a
	$(CHARM)/bin/charmc -save -verbose -o pvmctest pvmctest.o pvmc_main.o $(LIBS) -language converse

purify:     	pvmctest.o libpvmc.a
	$(CHARM)/bin/charmc -verbose -cc "purify cc" -o pvmctest pvmctest.o pvmc_main.o $(LIBS) -language converse 

libpvmc.a: $(OBJS)
	rm -f libpvmc.a
	ar r libpvmc.a $(OBJS)

clean:
	/bin/rm -f *.o *.a pvmctest

pvmc_buf.o:	pvmc_buf.c pvmc.h

pvmc_conv.o:	pvmc_conv.c

pvmc_pack.o:	pvmc_pack.c pvmc.h

pvmc_comm.o:	pvmc_comm.c

pvmc_main.o:	pvmc_main.c

pvmctest.o:	pvmctest.c pvm3.h
