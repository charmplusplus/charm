CHARMC=$(HOME)/projects/charm/net-hp-cc/bin/charmc -g -save -verbose

OBJS = hello.o Array1D.o ArrayMap.o ArrayElement.o RRMap.o HelloRing.o

all:	hello

hello: $(OBJS)
	$(CHARMC) -language charm++ -o hello $(OBJS)

cifiles:
	$(CHARMC)  hello.ci Array1D.ci ArrayMap.ci ArrayElement.ci \
RRMap.ci HelloRing.ci

clean:
	rm -f *.o hello

hello.o: hello.C Array1D.h
	$(CHARMC) -c hello.C

Array1D.o: Array1D.C Array1D.h
	$(CHARMC) -c Array1D.C

ArrayMap.o: ArrayMap.C ArrayMap.h
	$(CHARMC) -c ArrayMap.C

RRMap.o: RRMap.C RRMap.h ArrayMap.h
	$(CHARMC) -c RRMap.C

ArrayElement.o: ArrayElement.C ArrayElement.h
	$(CHARMC) -c ArrayElement.C

HelloRing.o: HelloRing.C HelloRing.h ArrayElement.h
	$(CHARMC) -c HelloRing.C


