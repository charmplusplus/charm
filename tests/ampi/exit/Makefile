-include ../../common.mk
CHARMC=../../../bin/ampicxx $(OPTS) $(MOPTS)

all: test_exit

test_exit: test_exit.C
	$(CHARMC) -o $@ $<

# Check that application exits correctly with exit code 42.
test: test_exit
	$(call run, test_exit 2>&1); [ $$? -eq 42 ] && echo "Exit test passed." || (echo "Exit test failed."; false)

clean:
	rm -f *.o test_exit charmrun ampirun
