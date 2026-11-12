Building and Running AMPI Programs
==================================

Installing AMPI
---------------

AMPI is included in the source distribution of Charm++, which is at
https://github.com/charmplusplus/charm. Clone it, then build Charm++ and
AMPI from source.

The build script for Charm++ is called ``build``. The syntax for this
script is:

.. code-block:: bash

   $ ./build <target> <version> <opts>

Users who are interested only in AMPI and not any other component of
Charm++ should specify ``<target>`` to be ``AMPI-only``. This will build
Charm++ and other libraries needed by AMPI in a mode configured and
tuned exclusively for AMPI. To fully build Charm++ underneath AMPI for
use with either paradigm, or for interoperation between the two, specify
``<target>`` to be ``AMPI``.

``<opts>`` are command line options passed to the ``charmc`` compile
script. Common compile time options such as
``-g, -O, -Ipath, -Lpath, -llib`` are accepted.

To build a debugging version of AMPI, use the option: ``-g``. To build a
production version of AMPI, use the option: ``--with-production``.

``<version>`` depends on the operating system and processor:
``reconverse-linux-x86_64``, ``reconverse-linux-arm8``,
``reconverse-darwin-arm8`` or ``reconverse-win-x86_64``. See the
Reconverse section of the Charm++ installation instructions
(:ref:`sec:reconverseinstall`) for details. Here is an example of how to
build a debug version of AMPI on Linux:

.. code-block:: bash

   $ ./build AMPI reconverse-linux-x86_64 -g

And the following is an example of how to build a production version of
AMPI, with MPI-level error checking in AMPI turned off:

.. code-block:: bash

   $ ./build AMPI-only reconverse-linux-x86_64 --with-production --disable-ampi-error-checking

Reconverse builds always support multithreaded parallelism (several
worker threads per process), so the ``smp`` option is not needed.

AMPI ranks are implemented as user-level threads with a stack size
default of 1MB. If the default is not correct for your program, you can
specify a different default stack size (in bytes) at build time. The
following build command illustrates this:

.. code-block:: bash

   $ ./build AMPI-only reconverse-linux-x86_64 --with-production -DTCHARM_STACKSIZE_DEFAULT=16777216

The same can be done for AMPI’s RDMA messaging threshold using
``AMPI_RDMA_THRESHOLD_DEFAULT`` and, for messages sent within the same
address space (ranks on the same worker thread or ranks on different
worker threads in the same process), using
``AMPI_SMP_RDMA_THRESHOLD_DEFAULT``. Contiguous messages with sizes
larger than the threshold are sent via RDMA on communication layers that
support this capability. You can also set the environment variables
``AMPI_RDMA_THRESHOLD`` and ``AMPI_SMP_RDMA_THRESHOLD`` before running a
job to override the default specified at build time.

Building AMPI Programs
----------------------

AMPI provides compiler wrappers such as ``ampicc``, ``ampif90``, and
``ampicxx`` in the ``bin`` subdirectory of Charm++ installations. You can
use them to build your AMPI program using the same syntax as other
compilers like ``gcc``. They are intended as drop-in replacements for
``mpicc`` wrappers provided by most conventional MPI implementations.
These scripts automatically handle the details of building and linking
against AMPI and the Charm++ runtime system. This includes launching the
compiler selected during the Charm++ build process, passing any toolchain
parameters important for proper function on the selected build target,
supplying the include and link paths for the runtime system, and linking
with Charm++ components important for AMPI, including Isomalloc heap
interception and commonly used load balancers.

.. _tab:toolchain:
.. table:: Full list of AMPI toolchain wrappers.

   ============ ==============
   Command Name Purpose
   ============ ==============
   ampicc       C
   ampiCC       C++
   ampicxx      C++
   ampic++      C++
   ampif77      Fortran 77
   ampif90      Fortran 90
   ampifort     Fortran 90
   ampirun      Program Launch
   ampiexec     Program Launch
   ============ ==============

All command line flags that you would use for other compilers can be used
with the AMPI compilers the same way. For example:

.. code-block:: bash

   $ ampicc -c pgm.c -O3
   $ ampif90 -c pgm.f90 -O0 -g
   $ ampicc -o pgm pgm.o -lm -O3

For consistency with other MPI implementations, these wrappers are also
provided using their standard names with the suffix ``.ampi``:

.. code-block:: bash

   $ mpicc.ampi -c pgm.c -O3
   $ mpif90.ampi -c pgm.f90 -O0 -g
   $ mpicc.ampi -o pgm pgm.o -lm -O3

Additionally, the ``bin/ampi`` subdirectory of Charm++ installations
contains the wrappers with their exact standard names, allowing them to
be given precedence as shell commands in a ``module``-like fashion by
adding this directory to the ``$PATH`` environment variable:

   $ export PATH=/home/user/charm/reconverse-linux-x86_64/bin/ampi:$PATH
   $ mpicc -c pgm.c -O3
   $ mpif90 -c pgm.f90 -O0 -g
   $ mpicc -o pgm pgm.o -lm -O3

These wrappers also allow the user to configure AMPI and Charm++-specific
functionality.
For example, to automatically select a Charm++ load balancer at program
launch without passing the ``+balancer`` runtime parameter, specify a
strategy at link time with ``-balancer <LB>``:

.. code-block:: bash

   $ ampicc pgm.c -o pgm -O3 -balancer GreedyRefineLB

Internally, the toolchain wrappers call the Charm runtime's general
toolchain script, ``charmc``. By default, they will specify ``-memory
isomalloc`` and ``-module CommonLBs``. Advanced users can disable
Isomalloc heap interception by passing ``-memory default``. For
diagnostic purposes, the ``-verbose`` option will print all parameters
passed to each stage of the toolchain. Refer to the Charm++ manual for
information about the full set of parameters supported by ``charmc``.

Running AMPI Programs
---------------------

An AMPI program can be run directly, with ``lcrun`` or the site
launcher like any other Charm++ program, or with ``ampirun``.

Running directly
~~~~~~~~~~~~~~~~

An AMPI program is an ordinary executable. It takes the total number of
PEs (``+pe``), AMPI options (such as the number of ranks to create, and
the stack size of every user-level thread) and the program arguments. A
typical invocation of an AMPI program ``pgm`` is:

.. code-block:: bash

   $ ./pgm +pe 16 +vp64

Here, the AMPI program ``pgm`` is run on 16 PEs with 64 total virtual
ranks (which will be mapped 4 per PE initially). Several processes are
started with ``lcrun -n 4 ./pgm +pe 16 +vp64`` on one host, or with the
site launcher under a batch system, as described below for ``ampirun``.

To run with load balancing, specify a load balancing strategy.

You can also specify the size of user-level thread’s stack
using the ``+tcharm_stacksize`` option, which can be used to decrease
the size of the stack that must be migrated, as in the following
example:

.. code-block:: bash

   $ ./pgm +pe 16 +vp128 +tcharm_stacksize 32K +balancer RefineLB

Running with ampirun
~~~~~~~~~~~~~~~~~~~~

For compliance with the MPI standard and simpler execution, AMPI ships
with the ``ampirun`` script that is similar to ``mpirun`` provided by
other MPI runtimes. ``ampirun`` is copied
automatically to the program directory when compiling an application
with ``ampicc``. Users with prior MPI experience may find ``ampirun`` the
simplest way to run AMPI programs.

The basic usage of ampirun is as follows:

.. code-block:: bash

   $ ./ampirun -np 16 ./pgm

This command will create 16 (non-virtualized) ranks, one per PE.

When using the ``-vr`` option, AMPI will create the number of ranks
specified by the ``-np`` parameter as virtual ranks, and will run one PE
per process:

.. code-block:: bash

   $ ./ampirun -np 16 --procs 4 -vr ./pgm

Other options (such as the load balancing strategy), can be specified as
program arguments:

.. code-block:: bash

   $ ./ampirun -np 16 ./pgm +balancer RefineLB

ampirun on reconverse builds
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Charm++ built on the reconverse runtime has no ``charmrun``; ``ampirun``
runs the program directly, and ``--host`` is not supported. Without
further options the program runs as one process with one PE per rank:

.. code-block:: bash

   $ ./ampirun -np 16 ./pgm

The ``--procs`` option runs several processes on the local host through
LCI's ``lcrun`` launcher. This command runs four processes with four PEs
each (``lcrun -n 4 ./pgm +pe 16``):

.. code-block:: bash

   $ ./ampirun -np 16 --procs 4 ./pgm

Under a batch system, use the site launcher and give the program its
total number of PEs with ``+pe``, putting ``setarch $(uname -m) -R``
inside the launch, after the launcher's own arguments:

.. code-block:: bash

   $ srun -n 4 setarch $(uname -m) -R ./pgm +pe 16 +vp64

The ``setarch -R`` prefix turns off address space randomization. A rank
that migrates to another process, or is restored from a checkpoint by a
new process, resumes on a stack that holds return addresses into the
program and into the runtime library. These addresses are valid only if
the code is loaded at the same address in every process, which requires
randomization to be off. The processes must also have the same stack
size limit (``ulimit -s``): a limit above about 128 MB, or an unlimited
one, changes where Linux loads the program and its libraries even with
randomization off. Restore a checkpoint under the same limit it was
written with. On Linux, ``ampirun`` and the ``testrun`` script used by
``make test`` add the prefix themselves; set
``AMPIRUN_VA_RANDOMIZATION=1`` or ``TESTRUN_VA_RANDOMIZATION=1`` to
leave randomization on. Where the system refuses the setting (a
container under Docker's default seccomp profile), both say so once and
run with randomization on. Under ``srun`` the prefix must be inside the
launch because the Slurm daemon on each compute node starts the tasks,
and they do not inherit settings of the ``srun`` process. macOS has no
``setarch``; randomization can be turned off only by the launcher, at
the moment it spawns the process (classic ``charmrun`` did), which
``ampirun`` and ``testrun`` do not do, so with them a rank cannot
migrate to another process or be restored from a checkpoint on macOS;
migration between the PEs of one process works.

Other options
~~~~~~~~~~~~~

Note that for AMPI programs compiled with gfortran, users may need to
set the following environment variable to see program output on stdout:

.. code-block:: bash

   $ export GFORTRAN_UNBUFFERED_ALL=1
