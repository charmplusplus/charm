CHARMC=../../../bin/charmc $(OPTS)

all:
	@echo 'buidling converse master-slave library...'
	@cd master-slave; make all OPTS='$(OPTS)'
	@echo 'buidling converse collective comm library...'
	@cd commlib; make all OPTS='$(OPTS)'
	@echo 'buidling converse message packing library...'
	@cd packlib; make all OPTS='$(OPTS)'
	@echo 'buidling metis...'
	@cd metis; make all OPTS='$(OPTS)'
	@echo 'buidling ddt...'
	@cd ddt; make all OPTS='$(OPTS)'

