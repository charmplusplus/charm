CHARMC=../../../../bin/charmc $(OPTS)

OBJS = multiply.o

all: multiply

multiply: $(OBJS)
	$(CHARMC) -language charm++ -o multiply $(OBJS) $(LD_LIBS)

multiply.decl.h: multiply.ci
	$(CHARMC)  multiply.ci

clean:
	rm -f *.decl.h *.def.h conv-host *.o multiply charmrun multiply.genCUDACode.* __triggered_main.genCUDACode.*

multiply.o: multiply.C multiply.decl.h
	$(CHARMC)  -c multiply.C


test: all
	./charmrun ./multiply +p2 4 8  +accelPercentDevice 0.5      
