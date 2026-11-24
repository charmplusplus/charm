# Charm++

[![reconverse-ci](https://github.com/charmplusplus/charm/actions/workflows/reconverse-ci.yaml/badge.svg?branch=recharm)](https://github.com/charmplusplus/charm/actions/workflows/reconverse-ci.yaml)
[![Documentation Status](https://readthedocs.org/projects/charm/badge/?version=latest)](https://charm.readthedocs.io/en/latest/?badge=latest)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3370873.svg)](https://doi.org/10.5281/zenodo.3370873)


## Introduction

Charm++ is a message-passing parallel language and runtime system.
It is implemented as a set of libraries for C++, is efficient,
and is portable to a wide variety of parallel machines.
Source code is provided, and non-commercial use is free.

Charm++ 9.0 runs on reconverse, the runtime layer in the
`contrib/reconverse` submodule; this line is the branch `recharm`. The
classic Converse runtime is on the branch `classic`, and 8.1.0 is its
last release.


## Getting the Latest Source

You can use anonymous Git access to obtain the latest Charm++ source
code, as follows:

     $ git clone --recurse-submodules https://github.com/charmplusplus/charm

Reconverse is the `contrib/reconverse` submodule, which is what
`--recurse-submodules` picks up. If you already cloned without it, run

     $ git submodule update --init -- contrib/reconverse

The build does this for you when the submodule is missing. To experiment with
a different reconverse, check the submodule out to the branch or commit
you want and rebuild:

     $ git -C contrib/reconverse fetch origin
     $ git -C contrib/reconverse checkout <branch-or-sha>

To jump straight to the tip of reconverse `main` instead, use

     $ git submodule update --remote -- contrib/reconverse

Note that this is deliberately not what a plain clone or
`--init` does: those check out the exact commit Charm++ records, so a
given Charm++ commit always builds against a known reconverse. Use
`--remote` when you want the latest reconverse, not when you want a
reproducible build.

Committing the moved submodule pointer is what changes the pairing for
everyone else. To build against a reconverse tree kept outside the
Charm++ checkout, pass `--with-reconverse-dir=<path>` to `./build`.

Reconverse uses LCI2 as its network layer. The build uses an installed
LCI when CMake finds one, and otherwise fetches and builds LCI from
source; `--without-reconverse-fetch-lci` turns the fetch off. The LCI it
fetches needs Python 3.8 or newer and CMake 3.21 or newer. With an older
Python, configure fails with "Cannot find source file
.../src/binding/generated/binding.cpp"; with an older CMake, it fails
with "C_STANDARD is set to invalid value '17'". If your system's
defaults are older, load a newer Python module and install a recent
CMake with `pip install cmake`.


## Build Configuration

The `build` script takes the following arguments:

     $ ./build <target> <version> [options ...]
                                  [--basedir=dir] [--libdir=dir] [--incdir=dir]
                                  [charmc-options ...]

The recommended build on a 64-bit Linux machine is

     $ ./build charm++ reconverse-linux-x86_64 --with-production -j8

which builds Charm++ in the directory `reconverse-linux-x86_64/`.

### `<target>`

`<target>` specifies the parts of Charm++ to compile:

* `charm++`  The key Charm++ executables and runtime libraries
* `AMPI`     Adaptive MPI, an implementation of MPI on top of Charm++
             (see [README.ampi](README.ampi))
* `LIBS`     Charm++, AMPI, and the other libraries built on top of them
* `charm4py` The shared library used by Charm4py
             (see [README.charm4py](README.charm4py))

### `<version>`

`<version>` names the runtime, the operating system and the CPU
architecture. Every version is a reconverse version; the list is the
set of directories under `src/arch/`:

| Charm++ Version           | OS      | CPU            | Default Compiler                       |
|---------------------------|---------|----------------|----------------------------------------|
| `reconverse-linux-x86_64` | Linux   | x86-64         | GNU compiler                           |
| `reconverse-linux-arm8`   | Linux   | ARM64 (AArch64)| GNU compiler                           |
| `reconverse-darwin-arm8`  | macOS   | Apple silicon  | Clang C++ compiler                     |
| `reconverse-win-x86_64`   | Windows | x86-64         | MS Visual C++; see [README.windows.md](README.windows.md) |

Every reconverse build runs in SMP mode: one OS process per logical
node, with the PEs as threads of that process. There is no
communication thread; the PEs themselves drive the network, so a process
with N PEs uses N cores. A Windows build runs as a single process.

### `<options>`

`<options>` select compilers, optional features, and flags passed to
the `charmc` compiler script. Compiler words select a compiler other
than the default, for example `gcc`, `clang`, `icx`, `nvhpc` or
`mpicxx`, and `gfortran`, `flang`, `ifx` or `nvfortran` for Fortran; a
version suffix such as `gcc-13` selects a specific version. Other
option words:

* `cuda`, `hip` - GPU support through CUDA or HIP/ROCm.
* `omp` - Integrated OpenMP support.
* `tsan` - Compile Charm++ with support for ThreadSanitizer.
* `perftools`, `craype` - Flags for Cray perftools and the Cray
  programming environment.

Flags of the form `--enable-<feature>`, `--with-<feature>` and others:

* `--with-production` - Turn on optimizations and turn off most of the
  run-time checking in Charm++ and reconverse. Tracing is also off
  unless `--enable-tracing` is given. Use this option for performance
  runs, after the program has been debugged.
* `--enable-tracing` - Build the tracing modules, for analysis of
  performance with Projections.
* `--enable-error-checking` - Keep run-time checking in a production
  build.
* `--enable-ccs`, `--enable-replay` - Converse client-server and
  deterministic record-replay.
* `--coverage` - Instrument for clang source-based coverage (see
  [coverage/README.md](coverage/README.md)).
* `--disable-fortran` - Do not build the Fortran interfaces.
* `--with-reconverse-dir=<path>`, `--without-reconverse-fetch-lci` -
  See "Getting the Latest Source" above.
* `-j<N>` - Build in parallel with N jobs.
* `-g`, `-O`, `-Ipath`, `-Lpath`, `-llib` and other unrecognized
  arguments are passed to `charmc` and from there to the compiler.
* `--force` - Delete and reconfigure an existing build directory.
* `[--libdir=dir]` specify additional lib paths for building Charm++.
* `[--incdir=dir]` specify additional include paths for building Charm++.
* `[--basedir=dir]` a shortcut to specify additional include and library paths
  for building Charm++, the include path is `dir/include` and library path
  is `dir/lib`.

The build directory is named after the version followed by the
selected option words and compilers, for example
`reconverse-linux-x86_64-cuda-clang/`.


## Building the Source

Once you have chosen a target, a version and options, `cd` into
`charm/` and run

     $ ./build <target> <version> <opts>

For example, on a Linux machine, you would run

     $ ./build charm++ reconverse-linux-x86_64 --with-production -j8

and on a Mac with Apple silicon

     $ ./build charm++ reconverse-darwin-arm8 --with-production -j8

To rebuild after changing the source, run `make` in the build directory.
When Charm++ is built successfully, the directory structure under the
build directory looks like:
```
reconverse-linux-x86_64/
   |
   ---  benchmarks/      # benchmark programs
   |
   ---  bin/             # all executables: charmc, charmxi, lcrun
   |
   ---  include/         # header files
   |
   ---  lib/             # static libraries
   |
   ---  lib_so/          # shared libraries (with --build-shared)
   |
   ---  examples/        # example programs
   |
   ---  tests/           # test programs
```

The top-level `bin/`, `include/` and `lib/` of the checkout are links
to the most recent build.

## Building a Program

To make a sample program, `cd` into `examples/charm++/NQueen/`.
This program solves the _n_ queens problem-- find how many ways there
are to arrange _n_ queens on an _n_ x _n_ chess board such that none may
attack another.

To build the program, type `make`.  You should get an
executable named `nqueen`.


## Running a Program

A Charm++ program is an ordinary executable. The number of processing
elements (PEs) it runs on is given on its own command line with `+pe`.
Following the previous example, to run the program on two PEs in one
process, type

     $ ./nqueen +pe 2 12 6

This should run for a few seconds, and print out:
`There are 14200 Solutions to 12 queens. Time=0.109440 End time=0.112752`

`+pe` is always the total number of PEs across all processes. To run
several processes on one machine, use LCI's `lcrun` launcher, which the
build puts in `<build>/bin` next to `charmc`:

     $ lcrun -n 2 ./nqueen +pe 4 12 6      # two processes, 4 PEs in total

Inside a batch allocation, start the program with the site's launcher,
such as `srun`, `mpirun` or `mpiexec`, even for a single-process run:

     $ srun -N 2 --ntasks-per-node=1 -c 16 ./pgm +pe 32

There is no `charmrun`. The manual's "Running Charm++ Programs" chapter
describes the run-time options.


## Building Dynamic Libraries

The `--build-shared` option to `./build` compiles Charm++ into dynamic
libraries in the `lib_so/` directory, and `charmc`'s `-charm-shared`
option links an application against them:

     $ charmc -o pgm pgm.o -charm-shared

On reconverse versions this option does not work yet: the link fails
because `libreconverse` is placed in `lib_so/` while `charmc` looks for
it in `lib/` (issue #4049). The `charm4py` target builds its own
shared library and does not use this option.


## Contributing

The recommended way to contribute to Charm++ development is to open a pull request (PR) on GitHub.
To open a pull request, create a fork of the Charm++ repo in your own space
(if you already have a fork, make sure is it up-to-date), and then create a new branch off of the
`recharm` branch.

GitHub provides a detailed tutorial on creating pull requests 
(https://docs.github.com/en/pull-requests/collaborating-with-pull-requests). 

Each pull request must pass code review and CI tests before it can be merged by someone on
the core development team. A pull request that changes how a subsystem works also updates
that subsystem's chapter in [doc/internals/](doc/internals/README.md), or adds the chapter
to the list there if none exists yet.
Our wiki contains additional information about pull requests
(https://github.com/charmplusplus/charm/wiki/Working-with-Pull-Requests).


## For More Information

The Charm++ documentation is at https://charm.readthedocs.io/; its
source is in [doc/](doc/). The [CHANGES](CHANGES) file lists the
significant changes in each release.

How the runtime works inside, for people who work on the runtime itself,
is described chapter by chapter in [doc/internals/](doc/internals/README.md).

The Charm++ web page, with more information, more programs,
and the latest version of Charm++, is at https://charmplusplus.org

The UIUC Parallel Programming Laboratory web page, with information
on past and present research, is at https://charm.cs.illinois.edu

For questions, comments, suggestions, improvements, or bug reports,
please create an issue or discussion on our GitHub, https://github.com/charmplusplus/charm


## Authors

Charm++ was created and is maintained by the Parallel Programming Lab,
in the Computer Science department at the University of Illinois at
Urbana-Champaign.  Our managing professor is Dr. L.V. Kale; students
and staff have included (in rough time order) Wennie Shu, Kevin
Nomura, Wayne Fenton, Balkrishna Ramkumar, Vikram Saletore, Amitabh
B. Sinha, Manish Gupta, Attila Gursoy, Nimish Shah, Sanjeev Krishnan,
Jayant DeSouza, Parthasarathy Ramachandran, Jeff Wright, Michael Lang,
Jackie Wang, Fang Hu, Michael Denardo, Joshua Yelon, Narain
Jagathesan, Zehra Sura, Krishnan Varadarajan, Sameer Paranjpye, Milind
Bhandarkar, Robert Brunner, Terry Wilmarth, Gengbin Zheng, Orion
Lawlor, Celso Mendes, Karthik Mahesh, Neelam Saboo, Greg Koenig,
Sameer Kumar, Sayantan Chakravorty, Chao Huang, Chee Lee, Fillipo
Gioachin, Isaac Dooley, Abhinav Bhatele, Aaron Becker, Ryan Mokos,
Ramprasad Venkataraman, Gagan Gupta, Pritish Jetley, Lukasz
Wesolowski, Esteban Meneses, Chao Mei, David Kunzman, Osman Sarood,
Abhishek Gupta, Yanhua Sun, Ehsan Totoni, Akhil Langer, Cyril Bordage,
Harshit Dokania, Prateek Jindal, Jonathan Lifflander, Xiang Ni,
Harshitha Menon, Nikhil Jain, Vipul Harsh, Bilge Acun, Phil Miller,
Seonmyeong Bak, Karthik Senthil, Juan Galvez, Michael Robson,
Raghavendra Kanakagiri, Venkatasubrahmanian Narayanan. Nitin Bhat, and
Justin Szaday. Current developers include: Eric Bohm, Ronak Buch, Eric
Mikida, Sam White, Kavitha Chandrasekar, Jaemin Choi, Matthias Diener,
Evan Ramos, Zane Fink, Pathikrit Ghosh, Maya Taylor, Aditya Bhosale,
Mathew Jacob, Tom Vander Aa, Thomas Quinn, and Ritvik Rao.

Copyright (C) 1989-2024 University of Illinois Board of Trustees
