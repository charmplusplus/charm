CHARMC=../../../../bin/charmc $(OPTS)

OBJS = vectorAdd.o

all: vectorAdd

vectorAdd: $(OBJS)
	$(CHARMC) -language charm++ -o vectorAdd $(OBJS) $(LD_LIBS)

vectorAdd.decl.h: vectorAdd.ci
	$(CHARMC)  vectorAdd.ci

clean:
	rm -f *.decl.h *.def.h conv-host *.o vectorAdd charmrun vectorAdd.genCUDACode.* __triggered_main.genCUDACode.*

vectorAdd.o: vectorAdd.C vectorAdd.decl.h
	$(CHARMC)  -c vectorAdd.C


test: all
	./charmrun ./vectorAdd +p4 64 256  +accelDeviceOnly
