-include ../../common.mk
CHARMBASE=../../../
CHARMC=../../../bin/ampicxx $(OPTS)

all: wave2d

wave2d: wave2d.C
	$(CHARMC) -c wave2d.C
	-$(CHARMC) -o wave2d wave2d.o -module liveViz -memory isomalloc

test: wave2d
	[ ! -s "wave2d" ] || $(call run, +p4 ./wave2d +vp64 )

clean:
	rm -f *.o wave2d *~ moduleinit.C charmrun conv-host ampirun
