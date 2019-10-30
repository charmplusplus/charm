#!/usr/bin/env python3

import os
import sys
import platform
import multiprocessing as mp
import pyctest.pyctest as pyctest
import pyctest.helpers as helpers

import urllib
import urllib.parse

parser = helpers.ArgumentParser("ProjectName", source_dir=os.getcwd()+'/../', binary_dir=os.getcwd()+'/../',
                                vcs_type="git", submit=True)
args = parser.parse_args()

pyctest.MODEL = "Experimental"

# Project settings
pyctest.PROJECT_NAME = urllib.parse.quote("Charm++")
pyctest.set("CTEST_DROP_SITE", "my.cdash.org")
pyctest.DROP_LOCATION = "/cdash/submit.php?project=" + pyctest.PROJECT_NAME
pyctest.BUILD_NAME = "netlrts-darwin-x86_64"
pyctest.BUILD_COMMAND = "./build LIBS netlrts-darwin-x86_64 -j5 -g"

# Define test for tests/ directory
tests = pyctest.test()
tests.SetName("tests/")
tests.SetCommand(["make", "-C", "tests/", "test"])
tests.SetProperty("TIMEOUT", "1200")

# Define test for examples/ directory
examples = pyctest.test()
examples.SetName("examples/")
examples.SetCommand(["make", "-C", "examples/", "test"])
examples.SetProperty("TIMEOUT", "1200")

# Define test for benchmarks/ directory
benchmarks = pyctest.test()
benchmarks.SetName("benchmarks/")
benchmarks.SetCommand(["make", "-C", "benchmarks/", "test"])
benchmarks.SetProperty("TIMEOUT", "1200")

# Run all tests
pyctest.run()
