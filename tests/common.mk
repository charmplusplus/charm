ifneq ($(wildcard ../../bin/.),)
	run = ../../bin/testrun $(1) $(TESTOPTS)
else
ifneq ($(wildcard ../../../bin/.),)
	run = ../../../bin/testrun $(1) $(TESTOPTS)
else
	run = ../../../../bin/testrun $(1) $(TESTOPTS)
endif
endif
