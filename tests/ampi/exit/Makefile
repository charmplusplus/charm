-include ../../common.mk
CHARMC=../../../bin/ampicxx $(OPTS) $(MOPTS)

# Other tests that could be here:
# - Fortran EXIT()
# - C exit(), return
# - Other main() prototypes

all: exit_explicit exit_return

exit_explicit: exit_explicit.C
	$(CHARMC) -o $@ $<

exit_return: exit_return.C
	$(CHARMC) -o $@ $<

# Check that all applications exit with exit code 42.
test: all
	$(call run, ./exit_explicit 2>&1); [ $$? -eq 42 ] && echo "exit_explicit test passed." || (echo "exit_explicit test failed (exit code was $$?, but expected 42)."; false)
	$(call run, ./exit_return 2>&1); [ $$? -eq 42 ] && echo "exit_return test passed." || (echo "exit_return test failed (exit code was $$?, but expected 42)."; false)

clean:
	rm -f *.o exit_explicit exit_return charmrun ampirun
