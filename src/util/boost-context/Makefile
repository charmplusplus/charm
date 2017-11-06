include ../uFcontext.mk

CDIR=../../
CHARMC=../../bin/charmc

all:
	if [ $(shell echo ${TARGET}| grep -c 'windows') -gt 0 ]; then	make -f Makefile_windows;\
	else make -f Makefile_default;	fi

clean:
	rm *.o
