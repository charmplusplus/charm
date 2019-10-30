#!/usr/bin/env python3

# This scripts recreates the *.cmake files in this directory.

import os
import sys
import platform
import multiprocessing as mp
import pyctest.pyctest as pyctest
import pyctest.helpers as helpers

import urllib
import urllib.parse

parser = helpers.ArgumentParser("ProjectName", source_dir=os.getcwd(), binary_dir=os.getcwd(),
                                vcs_type="git", submit=True)
args = parser.parse_args()

pyctest.MODEL = "Experimental"

# Project settings
pyctest.PROJECT_NAME = urllib.parse.quote("Charm++")
pyctest.set("CTEST_DROP_SITE", "my.cdash.org")
pyctest.DROP_LOCATION = "/cdash/submit.php?project=" + pyctest.PROJECT_NAME
pyctest.BUILD_NAME = "undefined_BUILD_NAME"
pyctest.BUILD_COMMAND = "sh -c 'cd .. && ./build LIBS ${CTEST_BUILD_NAME} -j8 -g --with-production'"

# Define test for tests/ directory
tests = pyctest.test()
tests.SetName("tests/")
tests.SetProperty("WORKING_DIRECTORY", "..")
tests.SetCommand(["make", "-C", "./tests/", "test"])
tests.SetProperty("TIMEOUT", "1200")

# Define test for examples/ directory
examples = pyctest.test()
examples.SetName("examples/")
examples.SetProperty("WORKING_DIRECTORY", "..")
examples.SetCommand(["make", "-C", "./examples/", "test"])
examples.SetProperty("TIMEOUT", "1200")

# Define test for benchmarks/ directory
benchmarks = pyctest.test()
benchmarks.SetName("benchmarks/")
benchmarks.SetProperty("WORKING_DIRECTORY", "..")
benchmarks.SetCommand(["make", "-C", "./benchmarks/", "test"])
benchmarks.SetProperty("TIMEOUT", "1200")

# Run all tests
# pyctest.run()

pyctest.generate_config()
pyctest.generate_test_file()

