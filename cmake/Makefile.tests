# This Makefile contains the rules to build & test Charm++

# The rules below are for compatibility with the CMake build system
#
# On making changes to this file, ensure that the same changes are ported
# to src/scripts/Makefile in order to maintain consistency with the old
# non-cmake based build system
#
# Changes to cmake/Makefile.tests will be reflected in <charm-arch>/tmp/Makefile
# for a new cmake based build. If you want these changes to reflect in the existing
# charm builds, you would have to manually copy them to <charm-arch>/tmp/Makefile

# Figure out whether make's silent mode was requested, in which case
# $(MAKEFLAGS) will contain "s". $(SILENT) is used in Makefile.tests.common.
SILENT:=$(findstring s,$(word 1, $(MAKEFLAGS)))

# Determine the full path of the build directory
mkfile_path:=$(abspath $(lastword $(MAKEFILE_LIST)))
mkfile_dir:=$(dir $(mkfile_path))
build_dir:=$(abspath $(mkfile_dir)/..)


all: charm++

charm++:
	@echo "=== Note: this make will not check if Charm++ source files have changed."
	@echo "=== Please run 'make' in $(build_dir) instead."
	@$(MAKE) -C $(build_dir) -q lib/libck.a || $(MAKE) -C $(build_dir)

AMPI: charm++
	@$(MAKE) -C $(build_dir) -q lib/libmoduleampi.a || { echo "=== You are missing the AMPI targets. Please build Charm++ with './build AMPI <...>'. ==="; exit 1; }

LIBS: charm++
	@$(MAKE) -C $(build_dir) -q lib/libmoduleCkCache.a || { echo "=== You are missing the LIBS targets. Please build Charm++ with './build LIBS <...>'. ==="; exit 1; }

converse: charm++

include Makefile.tests.common

