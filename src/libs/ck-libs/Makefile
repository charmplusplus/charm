CHARMC=../../bin/charmc $(OPTS)

all:
	@echo 'building barrier library...'
	@cd barrier; make all OPTS='$(OPTS)'
	@echo 'building irecv library...'
	@cd irecv; make all OPTS='$(OPTS)'
	@echo 'building fem library...'
	@cd fem; make -k all OPTS='$(OPTS)'
