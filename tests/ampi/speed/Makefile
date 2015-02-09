-include ../../common.mk
CHARMC=../../../bin/charmc $(OPTS)

OBJS = hello.o msgspeed.o test_mpi.o

all: hello

hello: $(OBJS)
	$(CHARMC) -language charm++ -module ampi -o hello $(OBJS)

hello.decl.h: hello.ci
	$(CHARMC)  hello.ci

clean:
	rm -f *.decl.h *.def.h conv-host *.o hello charmrun *.log *.sum *.sts charmrun.exe hello.exe hello.pdb hello.ilk

hello.o: hello.C hello.decl.h
	$(CHARMC) -c hello.C

test_mpi.o: test_mpi.c
	$(CHARMC) -c $<

msgspeed.o: msgspeed.c
	$(CHARMC) -c $<

test: all
	$(call run, hello +p2)


# Various stand-alone (non-charm) API calls:

standalone_mpi: standalone_mpi.c test_mpi.c msgspeed.c
	mpicc -o $@ standalone_mpi.c test_mpi.c msgspeed.c
