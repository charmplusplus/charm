CHARMC = ../../../../bin/charmc -g
INCL = -I/usr/include/python/
LIBS = /usr/lib/python/config/libpython2.3.a -lpthread -lutil -ldl -lm
INCLC = -I../../../../include/
MODULES = -module PythonCCS
PYMODULE = ../../../../lib/libmodulePythonCCS.a

OBJS = server.o


all: server client

server: $(OBJS)
	$(CHARMC) $(INCL) -language charm++ -o server $(OBJS) $(LIBS) $(MODULES)

server.decl.h: server.ci $(PYMODULE) ../../../../bin/charmxi
	$(CHARMC) $<

client: client.C
	g++ $(INCLC) -o $@ $<

clean:
	rm -f *.decl.h *.def.h *.o server client charmrun *~


server.o: server.C server.decl.h server.h
	$(CHARMC) -c $<

test: all
	./charmrun ./server ++remote-shell ssh ++server ++server-port 1234 +p2 $(OPTS) &
	./client localhost 1234
	sleep 2
	./client localhost 1234
	sleep 2
	./client localhost 1234 kill
