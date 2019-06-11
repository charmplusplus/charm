# Assume that charm is installed in ../charm/
THIS_DIR:=$(dir $(CURDIR)/$(word $(words $(MAKEFILE_LIST)),$(MAKEFILE_LIST)))
CHARMBASE:=$(abspath $(THIS_DIR)/../../../multicore-linux-gcc-x86_64)

AMPICC:=$(abspath $(CHARMBASE)/bin/ampicc) $(OPTS)
AMPICXX:=$(abspath $(CHARMBASE)/bin/ampicxx) $(OPTS)
AMPIF90:=$(abspath $(CHARMBASE)/bin/ampif90) $(OPTS)
