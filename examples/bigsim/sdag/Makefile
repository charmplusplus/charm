DIRS=  jacobi-no-redn

test:
	for d in $(DIRS); do \
                (cd $$d; make test OPTS='$(OPTS)' || exit 1) || exit 1; \
        done

bgtest:
	for d in $(DIRS); do \
                (cd $$d; make test OPTS='$(OPTS)' || exit 1) || exit 1; \
        done

clean:
	for d in $(DIRS); do (cd $$d; make clean OPTS='$(OPTS)'); done
	rm -f TAGS #*#
	rm -f core *~

