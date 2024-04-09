-include ../../common.mk
CHARMC=../../../bin/charmc $(OPTS)

FILENAME = "readtest.txt"
FILESIZE = 121
N_BUFFER_CHARES = 3
N_READERS = 11

all: iotest

iotest: iotest.ci iotest.C
	$(CHARMC) iotest.ci
	$(CHARMC) iotest.C -o $@ -module CkIO

test: iotest
	#dd if=/dev/urandom of=large_test.txt bs=64M count=16 iflag=fullblock
	$(call run, ./iotest +p2 $(N_BUFFER_CHARES) $(FILESIZE) $(N_READERS) $(FILENAME))
	$(call run, ./iotest +p4 $(N_BUFFER_CHARES) $(FILESIZE) $(N_READERS) $(FILENAME))

testp: iotest
	$(call run, ./iotest +p$(P) $(N_BUFFER_CHARES) $(FILESIZE) $(N_READERS) $(FILENAME))
	$(call run, ./iotest +p$(P) $$((2 * $(N_BUFFER_CHARES))) $(FILESIZE) $$((4 * $(N_READERS))) $(FILENAME))

smptest: iotest
	$(call run, ./iotest +p2 ++ppn 2 $(N_BUFFER_CHARES) $(FILESIZE) $(N_READERS) $(FILENAME))
	$(call run, ./iotest +p4 ++ppn 2 $(N_BUFFER_CHARES) $(FILESIZE) $(N_READERS) $(FILENAME))

clean:
	rm -f *.o *.decl.h *.def.h iotest test*

