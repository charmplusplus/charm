ifneq ($(wildcard ../../bin/.),)
	run = ../../bin/testrun $(1) $(TESTOPTS)
else
ifneq ($(wildcard ../../../bin/.),)
	run = ../../../bin/testrun $(1) $(TESTOPTS)
else
	run = ../../../../bin/testrun $(1) $(TESTOPTS)
endif
endif

default: all

test:
ifeq ($(CONV_DAEMON),1)
ifeq ($(CHARMRUN_SET),)
ifneq ($(wildcard commands.txt), )
	/bin/rm commands.txt
endif
ifneq ($(wildcard summary.txt), )
	/bin/rm summary.txt
endif
	$(eval export COMMANDS_FILE=$(shell pwd)/commands.txt)
	$(eval export CHARMRUN_SET=1)
endif
endif
	make test-local
ifeq ($(CONV_DAEMON),1)
ifeq ($(CHARMRUN_SET),)
	@echo "Running charmrun from $(notdir $(shell pwd))"
	$(CHARMRUN) commands.txt
endif
endif
