# The CMake build system

This document provides an overview of the CMake build system for Charm++.

## Requirements

CMake version 3.4+ is required for all builds.

No further software, apart from the usual Charm++ requirements (including
autoconf/automake for the embedded hwloc), should be necessary.

## Files

The build system is comprised of the following files (filenames are relative
to the Charm++ root directory):

| Filename                              | Purpose                                        |
|---------------------------------------|------------------------------------------------|
| CMakeLists.txt                        | Main file; option parsing                      |
| cmake/ci-files.cmake                  | Targets to create .decl.h files from .ci files |
| cmake/converse.cmake                  | Targets for Converse                           |
| cmake/detect-features.cmake           | Main system/compiler feature detection         |
| cmake/detect-features-c.cmake         | Compiler feature detection for C               |
| cmake/detect-features-cxx.cmake       | Compiler feature detection for C++             |
| cmake/detect-features-fortran.cmake   | Compiler feature detection for Fortran         |
| src/ck-core/CMakeLists.txt            | Targets for ck-core                            |
| src/ck-perf/CMakeLists.txt            | Targets for ck-perf                            |
| src/ck-pics/CMakeLists.txt            | Targets for ck-pics                            |
| src/conv-core/CMakeLists.txt          | Targets for conv-core                          |
| src/libs/ck-libs/ampi/CMakeLists.txt  | AMPI targets                                   |
| src/libs/ck-libs/CMakeLists.txt       | Misc Ck libs targets                           |
| src/libs/conv-libs/CMakeLists.txt     | Misc. Conv libs targets                        |
| src/QuickThreads/CMakeLists.txt       | QuickThreads targets                           |
| src/util/boost-context/CMakeLists.txt | Boost context library targets                  |
| src/util/charmrun-src/CMakeLists.txt  | Charmrun target                                |
| src/xlat-i/CMakeLists.txt             | Charmxi target                                 |

There are other CMakeLists.txt files in the Charm++ distribution, but these are unrelated
to the Charm++ build system.

Note that in most projects that use CMake, there is one CMakeLists.txt file per source
directory. This doesn't work well in Charm++, as there are many targets that use files
from multiple source directories.


## Limitations

Currently, the CMake build system has the following limitations.

- No Windows support. 
- No support for `gni-*` and `*-crayx?` targets.
- Not all options available in the old build system are supported.

See https://github.com/charmplusplus/charm/issues/2839 for details.
