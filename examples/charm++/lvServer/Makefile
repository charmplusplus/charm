-include ../../common.mk
LIBS = -lpython2.7 -lpthread -lutil -ldl -lm

CHARMC=../../../bin/charmc -module liveViz -module PythonCCS $(OPTS)

OBJS = lvServer.o

all: lvServer

lvServer: $(OBJS)
	$(CHARMC) -language charm++ -o lvServer $(OBJS) $(LIBS)

lvServer.decl.h: lvServer.ci
	$(CHARMC)  lvServer.ci

clean:
	rm -f *.decl.h *.def.h conv-host *.o lvServer charmrun *.log *.sum *.sts *~

lvServer.o: lvServer.C lvServer.decl.h
	$(CHARMC) -c lvServer.C

test: all
	$(call run, ./lvServer +p5 ++server ++server-port 1234)
