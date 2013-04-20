CHARMC=../../../../bin/charmc $(OPTS)

OBJS = seek_block.o

all: seek_block

seek_block: $(OBJS)
	$(CHARMC) -language charm++ -o seek_block $(OBJS)

seek_block.decl.h: seek_block.ci
	$(CHARMC)  seek_block.ci

clean:
	rm -f *.decl.h *.def.h conv-host *.o seek_block charmrun

seek_block.o: seek_block.C seek_block.h seek_block.decl.h
	$(CHARMC) -c $<

test: all
	./charmrun ./seek_block +p4  $(TESTOPTS)
	./charmrun ./seek_block +p4  $(TESTOPTS)

bgtest: all
	./charmrun ./seek_block +p4 +x2 +y2 +z1
