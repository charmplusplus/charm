
Appendix
========

.. _sec:install:

Installing Charm++
------------------

Charm++ can be installed manually from the source code or using a
precompiled binary package. We also provide a source code package for
the *Spack* package manager. Building from the source code provides more
flexibility, since one can choose the options as desired. However, a
precompiled binary may be slightly easier to get running.

Manual installation
~~~~~~~~~~~~~~~~~~~

Downloading Charm++
^^^^^^^^^^^^^^^^^^^

Charm++ can be downloaded using one of the following methods:

-  From Charm++ website - The current stable version (source code and
   binaries) can be downloaded from our website at
   *http://charm.cs.illinois.edu/software*.

-  From source archive - The latest development version of Charm++ can
   be downloaded from our source archive using *git clone
   http://charm.cs.illinois.edu/gerrit/charm*.

If you download the source code from the website, you will have to
unpack it using a tool capable of extracting gzip’d tar files, such as
tar (on Unix) or WinZIP (under Windows). Charm++ will be extracted to a
directory called ``charm``.

.. _sec:manualinstall:

Installation
^^^^^^^^^^^^

A typical prototype command for building Charm++ from the source code
is:

``./build <TARGET> <TARGET ARCHITECTURE> [OPTIONS]`` where,

TARGET
   is the framework one wants to build such as *charm++* or *AMPI*.

TARGET ARCHITECTURE
   is the machine architecture one wants to build for such as
   *netlrts-linux-x86_64*, *pamilrts-bluegeneq* etc.

OPTIONS
   are additional options to the build process, e.g. *smp* is used to
   build a shared memory version, *-j8* is given to build in parallel
   etc.

In Table :numref:`tab:buildlist`, a list of build
commands is provided for some of the commonly used systems. Note that,
in general, options such as *smp*, ``--with-production``, compiler
specifiers etc can be used with all targets. It is advisable to build
with ``--with-production`` to obtain the best performance. If one
desires to perform trace collection (for Projections),
``--enable-tracing --enable-tracing-commthread`` should also be passed
to the build command.

Details on all the available alternatives for each of the above
mentioned parameters can be found by invoking ``./build --help``. One
can also go through the build process in an interactive manner. Run
``./build``, and it will be followed by a few queries to select
appropriate choices for the build one wants to perform.

.. table:: Build command for some common cases
   :name: tab:buildlist

   ================================================================ =====================================================================
   Machine                                                          Build command
   ================================================================ =====================================================================
   Net with 32 bit Linux                                            ``./build charm++ netlrts-linux --with-production -j8``
   Multicore (single node, shared memory) 64 bit Linux              ``./build charm++ multicore-linux-x86_64 --with-production -j8``
   Net with 64 bit Linux                                            ``./build charm++ netlrts-linux-x86_64 --with-production -j8``
   Net with 64 bit Linux (intel compilers)                          ``./build charm++ netlrts-linux-x86_64 icc --with-production -j8``
   Net with 64 bit Linux (shared memory)                            ``./build charm++ netlrts-linux-x86_64 smp --with-production -j8``
   Net with 64 bit Linux (checkpoint restart based fault tolerance) ``./build charm++ netlrts-linux-x86_64 syncft --with-production -j8``
   MPI with 64 bit Linux                                            ``./build charm++ mpi-linux-x86_64 --with-production -j8``
   MPI with 64 bit Linux (shared memory)                            ``./build charm++ mpi-linux-x86_64 smp --with-production -j8``
   MPI with 64 bit Linux (mpicxx wrappers)                          ``./build charm++ mpi-linux-x86_64 mpicxx --with-production -j8``
   IBVERBS with 64 bit Linux                                        ``./build charm++ verbs-linux-x86_64 --with-production -j8``
   OFI with 64 bit Linux                                            ``./build charm++ ofi-linux-x86_64 --with-production -j8``
   Net with 64 bit Windows                                          ``./build charm++ netlrts-win-x86_64 --with-production -j8``
   MPI with 64 bit Windows                                          ``./build charm++ mpi-win-x86_64 --with-production -j8``
   Net with 64 bit Mac                                              ``./build charm++ netlrts-darwin-x86_64 --with-production -j8``
   Blue Gene/Q (bgclang compilers)                                  ``./build charm++ pami-bluegeneq --with-production -j8``
   Blue Gene/Q (bgclang compilers)                                  ``./build charm++ pamilrts-bluegeneq --with-production -j8``
   Cray XE6                                                         ``./build charm++ gni-crayxe --with-production -j8``
   Cray XK7                                                         ``./build charm++ gni-crayxe-cuda --with-production -j8``
   Cray XC40                                                        ``./build charm++ gni-crayxc --with-production -j8``
   ================================================================ =====================================================================

As mentioned earlier, one can also build Charm++ using the precompiled
binary in a manner similar to what is used for installing any common
software.

When a Charm++ build folder has already been generated, it is possible
to perform incremental rebuilds by invoking ``make`` from the ``tmp``
folder inside it. For example, with a *netlrts-linux-x86_64* build, the
path would be ``netlrts-linux-x86_64/tmp``. On Linux and macOS, the tmp
symlink in the top-level charm directory also points to the tmp
directory of the most recent build.

Alternatively, CMake can be used for configuring and building Charm++.
You can use ``cmake-gui`` or ``ccmake`` for an overview of available
options. Note that some are only effective when passed with ``-D`` from
the command line while configuring from a blank slate. To build with all
defaults, ``cmake .`` is sufficient, though invoking CMake from a
separate location (ex:
``mkdir mybuild && cd mybuild && cmake ../charm``) is recommended.

Installation through the Spack package manager
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Charm++ can also be installed through the Spack package manager
(https://spack.io/).

A basic command to install Charm++ through Spack is the following:

.. code-block:: bash

    $ spack install charmpp

By default, the netlrts network backend with SMP support is built. You
can specify other backends by providing the ``backend`` parameter to
spack. It is also possible to specify other options, as listed in
Section :numref:`sec:manualinstall`, by adding them to the Spack
command prepended by a ``+``. For example, to build the MPI
version of Charm++ with the integrated OpenMP support, you can use the
following command:

.. code-block:: bash

    $ spack install charmpp backend=mpi +omp

To disable an option, prepend it with a ``~``. For example, to
build Charm++ with SMP support disabled, you can use the following
command:

.. code-block:: bash

    $ spack install charmpp ~smp

By default, the newest released version of Charm++ is built. You can
select another version with the ``@`` option (for example,
``@6.8.1``). To build the current git version of Charm++, specify the
``@develop`` version:

.. code-block:: bash

    $ spack install charmpp@develop

Charm++ installation directories
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The main directories in a Charm++ installation are:

charm/bin
   Executables, such as charmc and charmrun, used by Charm++.

charm/doc
   Documentation for Charm++, such as this document. Distributed as
   LaTeX source code; HTML and PDF versions can be built or downloaded
   from our web site.

charm/include
   The Charm++ C++ and Fortran user include files (.h).

charm/lib
   The static libraries (.a) that comprise Charm++.

charm/lib_so
   The shared libraries (.so/.dylib) that comprise Charm++, if Charm++
   is compiled with the ``-build-shared`` option.

charm/examples
   Example Charm++ programs.

charm/src
   Source code for Charm++ itself.

charm/tmp
   Directory where Charm++ is built.

charm/tests
   Test Charm++ programs used by autobuild.

Reducing disk usage
~~~~~~~~~~~~~~~~~~~

The charm directory contains a collection of example-programs and
test-programs. These may be deleted with no other effects. You may also
``strip`` all the binaries in ``charm/bin``.

.. _sec:compile:

Compiling Charm++ Programs
--------------------------

The ``charmc`` program, located in “charm/bin”, standardizes compiling
and linking procedures among various machines and operating systems.
“charmc” is a general-purpose tool for compiling and linking, not only
restricted to Charm++ programs.

Charmc can perform the following tasks. The (simplified) syntax for each
of these modes is shown. Caution: in reality, one almost always has to
add some command-line options in addition to the simplified syntax shown
below. The options are described next.

.. code-block:: none

    * Compile C                            charmc -o pgm.o pgm.c
    * Compile C++                          charmc -o pgm.o pgm.C
    * Link                                 charmc -o pgm   obj1.o obj2.o obj3.o...
    * Compile + Link                       charmc -o pgm   src1.c src2.ci src3.C
    * Create Library                       charmc -o lib.a obj1.o obj2.o obj3.o...
    * Translate Charm++ Interface File     charmc file.ci

Charmc automatically determines where the charm lib and include
directories are — at no point do you have to configure this information.
However, the code that finds the lib and include directories can be
confused if you remove charmc from its normal directory, or rearrange
the directory tree. Thus, the files in the charm/bin, charm/include, and
charm/lib directories must be left where they are relative to each
other.

The following command-line options are available to users of charmc:

``-o`` *output-file*:
   Output file name. Note: charmc only ever produces one output file at
   a time. Because of this, you cannot compile multiple source files at
   once, unless you then link or archive them into a single output-file.
   If exactly one source-file is specified, then an output file will be
   selected by default using the obvious rule (e.g., if the input file
   if pgm.c, the output file is pgm.o). If multiple input files are
   specified, you must manually specify the name of the output file,
   which must be a library or executable.

``-c``:
   Ignored. There for compatibility with ``cc``.

``-Dsymbol[=value]``:
   Defines preprocessor variables from the command line at compile time.

``-I``:
   Add a directory to the search path for preprocessor include files.

``-g``:
   Causes compiled files to include debugging information.

``-L*``:
   Add a directory to the search path for libraries selected by the
   ``-l`` command.

``-l*``:
   Specifies libraries to link in.

``-module m1[,m2[,...]]``
   Specifies additional Charm++ modules to link in. Similar to ``-l``,
   but also registers Charm++ parallel objects. See the library’s
   documentation for whether to use ``-l`` or ``-module``.

``-optimize``:
   Causes files to be compiled with maximum optimization.

``-no-optimize``:
   If this follows ``-optimize`` on the command line, it turns
   optimization back off. This can be used to override options specified
   in makefiles.

``-production``:
   Enable architecture-specific production-mode features. For instance,
   use available hardware features more aggressively. It’s probably a
   bad idea to build some objects with this, and others without.

``-s``:
   Strip the executable of debugging symbols. Only meaningful when
   producing an executable.

``-verbose``:
   All commands executed by charmc are echoed to stdout.

``-seq``:
   Indicates that we’re compiling sequential code. On parallel machines
   with front ends, this option also means that the code is for the
   front end. This option is only valid with C and C++ files.

``-use-fastest-cc``:
   Some environments provide more than one C compiler (cc and gcc, for
   example). Usually, charmc prefers the less buggy of the two. This
   option causes charmc to switch to the most aggressive compiler,
   regardless of whether it’s buggy or not.

``-use-reliable-cc``:
   Some environments provide more than one C compiler (cc and gcc, for
   example). Usually, charmc prefers the less buggy of the two, but not
   always. This option causes charmc to switch to the most reliable
   compiler, regardless of whether it produces slow code or not.

``-language {converse|charm++|ampi|fem|f90charm}``:
   When linking with charmc, one must specify the “language”. This is
   just a way to help charmc include the right libraries. Pick the
   “language” according to this table:

   -  Charm++ if your program includes Charm++, C++, and C.

   -  Converse if your program includes C or C++.

   -  f90charm if your program includes f90 Charm interface.

``-balance`` *seed load-balance-strategy*:
   When linking any Converse program (including any Charm++ or sdag
   program), one must include a seed load-balancing library. There are
   currently three to choose from: ``rand``, ``test``, and ``neighbor``
   are supported. Default is ``-balance rand``.

   When linking with ``neighbor`` seed load balancer, one can also
   specify a virtual topology for constructing neighbors during run-time
   using ``+LBTopo topo``, where *topo* can be one of (a) ring, (b)
   mesh2d, (c) mesh3d and (d) graph. The default is mesh2d.

``-tracemode`` *tracing-mode*:
   Selects the desired degree of tracing for Charm++ programs. See the
   Charm++ manual and the Projections manuals for more information.
   Currently supported modes are ``none``, ``summary``, and
   ``projections``. Default is ``-tracemode none``.

``-memory`` *memory-mode*:
   Selects the implementation of malloc and free to use. Select a memory
   mode from the table below.

   -  os Use the operating system’s standard memory routines.

   -  gnu Use a set of GNU memory routines.

   -  paranoid Use an error-checking set of routines. These routines
      will detect common mistakes such as buffer overruns, underruns,
      double-deletes, and use-after-delete. The extra checks slow down
      programs, so this version should not be used in production code.

   -  leak Use a special set of memory routines and annotation functions
      for detecting memory leaks. Call CmiMemoryMark() to mark allocated
      memory as OK, do work which you suspect is leaking memory, then
      call CmiMemorySweep(const char \*tag) to check.

   -  verbose Use a tracing set of memory routines. Every memory-related
      call results in a line printed to standard out.

   -  default Use the default, which depends on the version of Charm++.

``-c++`` *C++ compiler*:
   Forces the specified C++ compiler to be used.

``-cc`` *C-compiler*:
   Forces the specified C compiler to be used.

``-cp`` *copy-file*:
   Creates a copy of the output file in *copy-file*.

``-cpp-option`` *options*:
   Options passed to the C pre-processor.

``-ld`` *linker*:
   Use this option only when compiling programs that do not include C++
   modules. Forces charmc to use the specified linker.

``-ld++`` *linker*:
   Use this option only when compiling programs that include C++
   modules. Forces charmc to use the specified linker.

``-ld++-option`` *options*:
   Options passed to the linker for ``-language charm++``.

``-ld-option`` *options*:
   Options passed to the linker for ``-language converse``.

``-ldro-option`` *options*:
   Options passes to the linker when linking ``.o`` files.

Other options that are not listed here will be passed directly to the
underlying compiler and/or linker.

.. _sec:run:

Running Charm++ Programs
------------------------

.. _charmrun:

Launching Programs with charmrun
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When compiling Charm++ programs, the charmc linker produces both an
executable file and an utility called ``charmrun``, which is used to
load the executable onto the parallel machine.

To run a Charm++ program named “pgm” on four processors, type:

.. code-block:: bash

   $ charmrun pgm +p4

Execution on platforms which use platform specific launchers, (i.e.,
**aprun**, **ibrun**), can proceed without charmrun, or charmrun can be
used in coordination with those launchers via the ``++mpiexec`` (see
Section :numref:`mpiexec`) parameter.

Programs built using the network version of Charm++ can be run alone,
without charmrun. This restricts you to using the processors on the
local machine, but it is convenient and often useful for debugging. For
example, a Charm++ program can be run on one processor in the debugger
using:

.. code-block:: bash

   $ gdb pgm

If the program needs some environment variables to be set for its
execution on compute nodes (such as library paths), they can be set in
.charmrunrc under home directory. charmrun will run that shell script
before running the executable.

Charmrun normally limits the number of status messages it prints to a
minimal level to avoid flooding the terminal with unimportant
information. However, if you encounter trouble launching a job, try
using the ``++verbose`` option to help diagnose the issue. (See the
``++quiet`` option to suppress output entirely.)

.. _command line options:

Command Line Options
~~~~~~~~~~~~~~~~~~~~

A Charm++ program accepts the following command line options:

``+auto-provision``
   Automatically determine the number of worker threads to launch in
   order to fully subscribe the machine running the program.

``+autoProvision``
   Same as above.

``+oneWthPerHost``
   Launch one worker thread per compute host. By the definition of
   standalone mode, this always results in exactly one worker thread.

``+oneWthPerSocket``
   Launch one worker thread per CPU socket.

``+oneWthPerCore``
   Launch one worker thread per CPU core.

``+oneWthPerPU``
   Launch one worker thread per CPU processing unit, i.e. hardware
   thread.

``+pN``
   Explicitly request exactly N worker threads. The default is 1.

``+ss``
   Print summary statistics about chare creation. This option prints the
   total number of chare creation requests, and the total number of
   chare creation requests processed across all processors.

``+cs``
   Print statistics about the number of create chare messages requested
   and processed, the number of messages for chares requested and
   processed, and the number of messages for branch office chares
   requested and processed, on a per processor basis. Note that the
   number of messages created and processed for a particular type of
   message on a given node may not be the same, since a message may be
   processed by a different processor from the one originating the
   request.

``user_options``
   Options that are be interpreted by the user program may be included
   mixed with the system options. However, ``user_options`` cannot start
   with +. The ``user_options`` will be passed as arguments to the user
   program via the usual ``argc/argv`` construct to the ``main`` entry
   point of the main chare. Charm++ system options will not appear in
   ``argc/argv``.

.. _network command line options:
.. _mpiexec:

Additional Network Options
^^^^^^^^^^^^^^^^^^^^^^^^^^

The following ``++`` command line options are available in the network
version.

First, commands related to subscription of computing resources:

``++auto-provision``
   Automatically determine the number of processes and threads to launch
   in order to fully subscribe the available resources.

``++autoProvision``
   Same as above.

``++processPerHost N``
   Launch N processes per compute host.

``++processPerSocket N``
   Launch N processes per CPU socket.

``++processPerCore N``
   Launch N processes per CPU core.

``++processPerPU N``
   Launch N processes per CPU processing unit, i.e. hardware thread.

The above options should allow sufficient control over process
provisioning for most users. If you need more control, the following
advanced options are available:

``++n N``
   Run the program with N processes. Functionally identical to ``+p`` in
   non-SMP mode (see below). The default is 1.

``++p N``
   Total number of processing elements to create. In SMP mode, this
   refers to worker threads (where
   :math:`\texttt{n} * \texttt{ppn} = \texttt{p}`), otherwise it refers
   to processes (:math:`\texttt{n} = \texttt{p}`). The default is 1. Use
   of ``++p`` is discouraged in favor of ``++processPer*`` (and
   ``++oneWthPer*`` in SMP mode) where desirable, or ``++n`` (and
   ``++ppn``) otherwise.

The remaining options cover details of process launch and connectivity:

``++local``
   Run charm program only on local machines. No remote shell invocation
   is needed in this case. It starts node programs right on your local
   machine. This could be useful if you just want to run small program
   on only one machine, for example, your laptop.

``++mpiexec``
   Use the cluster’s ``mpiexec`` job launcher instead of the
   built in ssh method.

   This will pass ``-n $P`` to indicate how many processes to launch. If
   ``-n $P`` is not required because the number of processes to launch
   is determined via queueing system environment variables then use
   ``++mpiexec-no-n`` rather than ``++mpiexec``. An executable named
   something other than ``mpiexec`` can be used with the additional
   argument ``++remote-shell`` *runmpi*, with ‘runmpi’ replaced by the
   necessary name.

   To pass additional arguments to ``mpiexec``, specify
   ``++remote-shell`` and list them as part of the value after the
   executable name as follows:

   .. code-block:: bash

      $ ./charmrun ++mpiexec ++remote-shell "mpiexec --YourArgumentsHere" ./pgm

   Use of this option can potentially provide a few benefits:

   -  Faster startup compared to the SSH approach charmrun would
      otherwise use

   -  No need to generate a nodelist file

   -  Multi-node job startup on clusters that do not allow connections
      from the head/login nodes to the compute nodes

   At present, this option depends on the environment variables for some
   common MPI implementations. It supports OpenMPI
   (``OMPI_COMM_WORLD_RANK`` and ``OMPI_COMM_WORLD_SIZE``), M(VA)PICH
   (``MPIRUN_RANK`` and ``MPIRUN_NPROCS`` or ``PMI_RANK`` and
   ``PMI_SIZE``), and IBM POE (``MP_CHILD`` and ``MP_PROCS``).

``++debug``
   Run each node under gdb in an xterm window, prompting the user to
   begin execution.

``++debug-no-pause``
   Run each node under gdb in an xterm window immediately (i.e. without
   prompting the user to begin execution).

   If using one of the ``++debug`` or ``++debug-no-pause`` options, the
   user must ensure the following:

   #. The ``DISPLAY`` environment variable points to your terminal.
      SSH’s X11 forwarding does not work properly with Charm++.

   #. The nodes must be authorized to create windows on the host machine
      (see man pages for ``xhost`` and ``xauth``).

   #. ``xterm``, ``xdpyinfo``, and ``gdb`` must be in the user’s path.

   #. The path must be set in the ``.cshrc`` file, not the ``.login``
      file, because ``ssh`` does not run the ``.login`` file.

``++scalable-start``
   Scalable start, or SMP-aware startup. It is useful for scalable
   process launch on multi-core systems since it creates only one ssh
   session per node and spawns all clients from that ssh session. This
   is the default startup strategy and the option is retained for
   backward compatibility.

``++batch``
   Ssh a set of node programs at a time, avoiding overloading Charmrun
   pe. In this strategy, the nodes assigned to a charmrun are divided
   into sets of fixed size. Charmrun performs ssh to the nodes in the
   current set, waits for the clients to connect back and then performs
   ssh on the next set. We call the number of nodes in one ssh set as
   batch size.

``++maxssh``
   Maximum number of ``ssh``\ ’s to run at a time. For backwards
   compatibility, this option is also available as ``++maxrsh``.

``++nodelist``
   File containing list of nodes.

``++numHosts``
   Number of nodes from the nodelist to use. If the value requested is
   larger than the number of nodes found, Charmrun will error out.

``++help``
   Print help messages

``++runscript``
   Script to run node-program with. The specified run script is invoked
   with the node program and parameter. For example:

   .. code-block:: bash

      $ ./charmrun +p4 ./pgm 100 2 3 ++runscript ./set_env_script

   In this case, the ``set_env_script`` is invoked on each node before
   launching ``pgm``.

``++xterm``
   Which xterm to use

``++in-xterm``
   Run each node in an xterm window

``++display``
   X Display for xterm

``++debugger``
   Which debugger to use

``++remote-shell``
   Which remote shell to use

``++useip``
   Use IP address provided for charmrun IP

``++usehostname``
   Send nodes our symbolic hostname instead of IP address

``++server-auth``
   CCS Authentication file

``++server-port``
   Port to listen for CCS requests

``++server``
   Enable client-server (CCS) mode

``++nodegroup``
   Which group of nodes to use

``++verbose``
   Print diagnostic messages

``++quiet``
   Suppress runtime output during startup and shutdown

``++timeout``
   Seconds to wait per host connection

``++timelimit``
   Seconds to wait for program to complete


.. _sec-smpopts:


SMP Options
^^^^^^^^^^^

SMP mode in Charm++ spawns one OS process per logical node. Within this
process there are two types of threads:

#. Worker Threads that have objects mapped to them and execute entry
   methods

#. Communication Thread that sends and receives data (depending on the
   network layer)

| Charm++ always spawns one communication thread per process when using
  SMP mode and as many worker threads as the user specifies (see the
  options below). In general, the worker threads produce messages and
  hand them to the communication thread, which receives messages and
  schedules them on worker threads.
| To use SMP mode in Charm++, build charm with the ``smp`` option, e.g.:

.. code-block:: bash

   $ ./build charm++ netlrts-linux-x86_64 smp

There are various trade-offs associated with SMP mode. For instance,
when using SMP mode there is no waiting to receive messages due to long
running entry methods. There is also no time spent in sending messages
by the worker threads and memory is limited by the node instead of per
core. In SMP mode, intra-node messages use simple pointer passing, which
bypasses the overhead associated with the network and extraneous copies.
Another benefit is that the runtime will not pollute the caches of
worker threads with communication-related data in SMP mode.

However, there are also some drawbacks associated with using SMP mode.
First and foremost, you sacrifice one core to the communication thread.
This is not ideal for compute bound applications. Additionally, this
communication thread may become a serialization bottleneck in
applications with large amounts of communication. Keep these trade-offs
in mind when evaluating whether to use SMP mode for your application or
deciding how many processes to launch per physical node when using SMP
mode. Finally, any library code the application may call needs to be
thread-safe.

Charm++ provides the following options to control the number of worker
threads spawned and the placement of both worker and communication
threads:

``++oneWthPerHost``
   Launch one worker thread per compute host.

``++oneWthPerSocket``
   Launch one worker thread per CPU socket.

``++oneWthPerCore``
   Launch one worker thread per CPU core.

``++oneWthPerPU``
   Launch one worker thread per CPU processing unit, i.e. hardware
   thread.

The above options (and ``++auto-provision``) should allow sufficient
control over thread provisioning for most users. If you need more
precise control over thread count and placement, the following options
are available:

``++ppn N``
   Number of PEs (or worker threads) per logical node (OS process). This
   option should be specified even when using platform specific
   launchers (e.g., aprun, ibrun).

``+pemap L[-U[:S[.R]+O]][,...]``
   Bind the execution threads to the sequence of cores described by the
   arguments using the operating system’s CPU affinity functions. Can be
   used outside SMP mode.

   A single number identifies a particular core. Two numbers separated
   by a dash identify an inclusive range (*lower bound* and *upper
   bound*). If they are followed by a colon and another number (a
   *stride*), that range will be stepped through in increments of the
   additional number. Within each stride, a dot followed by a *run* will
   indicate how many cores to use from that starting point. A plus
   represents the offset to the previous core number. Multiple
   ``+offset`` flags are supported, e.g., 0-7+8+16 equals 0,8,16,1,9,17.

   For example, the sequence ``0-8:2,16,20-24`` includes cores 0, 2, 4,
   6, 8, 16, 20, 21, 22, 23, 24. On a 4-way quad-core system, if one
   wanted to use 3 cores from each socket, one could write this as
   ``0-15:4.3``. ``++ppn 10 +pemap 0-11:6.5+12`` equals
   ``++ppn 10 +pemap 0,12,1,13,2,14,3,15,4,16,6,18,7,19,8,20,9,21,10,22``

``+commap p[,q,...]``
   Bind communication threads to the listed cores, one per process.

To run applications in SMP mode, we generally recommend using one
logical node per socket or NUMA domain. ``++ppn`` will spawn N threads
in addition to 1 thread spawned by the runtime for the communication
threads, so the total number of threads will be N+1 per node.
Consequently, you should map both the worker and communication threads
to separate cores. Depending on your system and application, it may be
necessary to spawn one thread less than the number of cores in order to
leave one free for the OS to run on. An example run command might look
like:

.. code-block:: bash

   $ ./charmrun ++ppn 3 +p6 +pemap 1-3,5-7 +commap 0,4 ./app <args>

This will create two logical nodes/OS processes (2 = 6 PEs/3 PEs per
node), each with three worker threads/PEs (``++ppn 3``). The worker
threads/PEs will be mapped thusly: PE 0 to core 1, PE 1 to core 2, PE 2
to core 3 and PE 4 to core 5, PE 5 to core 6, and PE 6 to core 7.
PEs/worker threads 0-2 compromise the first logical node and 3-5 are the
second logical node. Additionally, the communication threads will be
mapped to core 0, for the communication thread of the first logical
node, and to core 4, for the communication thread of the second logical
node.

Please keep in mind that ``+p`` always specifies the total number of PEs
created by Charm++, regardless of mode (the same number as returned by
``CkNumPes()``). The ``+p`` option does not include the communication
thread, there will always be exactly one of those per logical node.

Multicore Options
^^^^^^^^^^^^^^^^^

On multicore platforms, operating systems (by default) are free to move
processes and threads among cores to balance load. This however
sometimes can degrade the performance of Charm++ applications due to the
extra overhead of moving processes and threads, especially for Charm++
applications that already implement their own dynamic load balancing.

Charm++ provides the following runtime options to set the processor
affinity automatically so that processes or threads no longer move. When
cpu affinity is supported by an operating system (tested at Charm++
configuration time), the same runtime options can be used for all
flavors of Charm++ versions including network and MPI versions, smp and
non-smp versions.

``+setcpuaffinity``
   Set cpu affinity automatically for processes (when Charm++ is based
   on non-smp versions) or threads (when smp). This option is
   recommended, as it prevents the OS from unnecessarily moving
   processes/threads around the processors of a physical node.

``+excludecore <core #>``
   Do not set cpu affinity for the given core number. One can use this
   option multiple times to provide a list of core numbers to avoid.

.. _io buffer options:

IO buffering options
^^^^^^^^^^^^^^^^^^^^

There may be circumstances where a Charm++ application may want to take
or relinquish control of stdout buffer flushing. Most systems default to
giving the Charm++ runtime control over stdout but a few default to
giving the application that control. The user can override these system
defaults with the following runtime options:

``+io_flush_user``
   User (application) controls stdout flushing

``+io_flush_system``
   The Charm++ runtime controls flushing

Nodelist file
~~~~~~~~~~~~~

For network of workstations, the list of machines to run the program can
be specified in a file. Without a nodelist file, Charm++ runs the
program only on the local machine.

The format of this file allows you to define groups of machines, giving
each group a name. Each line of the nodes file is a command. The most
important command is:

.. code-block:: none

   host <hostname> <qualifiers>

which specifies a host. The other commands are qualifiers: they modify
the properties of all hosts that follow them. The qualifiers are:

| ``group <groupname>``   - subsequent hosts are members of specified
  group
| ``login <login>`` - subsequent hosts use the specified login
| ``shell <shell>`` - subsequent hosts use the specified remote shell
| ``setup <cmd>`` - subsequent hosts should execute cmd
| ``pathfix <dir1> <dir2>`` - subsequent hosts should replace dir1 with
  dir2 in the program path
| ``cpus <n>`` - subsequent hosts should use N light-weight processes
| ``speed <s>`` - subsequent hosts have relative speed rating
| ``ext <extn>`` - subsequent hosts should append extn to the pgm name

**Note:** By default, charmrun uses a remote shell “ssh” to spawn node
processes on the remote hosts. The ``shell`` qualifier can be used to
override it with say, “rsh”. One can set the ``CONV_RSH`` environment
variable or use charmrun option ``++remote-shell`` to override the
default remote shell for all hosts with unspecified ``shell`` qualifier.

All qualifiers accept “\*” as an argument, this resets the modifier to
its default value. Note that currently, the passwd, cpus, and speed
factors are ignored. Inline qualifiers are also allowed:

.. code-block:: none

   host beauty ++cpus 2 ++shell ssh

Except for “group”, every other qualifier can be inlined, with the
restriction that if the “setup” qualifier is inlined, it should be the
last qualifier on the “host” or “group” statement line.

Here is a simple nodes file:

.. code-block:: none

           group kale-sun ++cpus 1
             host charm.cs.illinois.edu ++shell ssh
             host dp.cs.illinois.edu
             host grace.cs.illinois.edu
             host dagger.cs.illinois.edu
           group kale-sol
             host beauty.cs.illinois.edu ++cpus 2
           group main
             host localhost

This defines three groups of machines: group kale-sun, group kale-sol,
and group main. The ++nodegroup option is used to specify which group of
machines to use. Note that there is wraparound: if you specify more
nodes than there are hosts in the group, it will reuse hosts. Thus,

.. code-block:: bash

   $ charmrun pgm ++nodegroup kale-sun +p6

uses hosts (charm, dp, grace, dagger, charm, dp) respectively as nodes
(0, 1, 2, 3, 4, 5).

If you don’t specify a ++nodegroup, the default is ++nodegroup main.
Thus, if one specifies

.. code-block:: bash

   $ charmrun pgm +p4

it will use “localhost” four times. “localhost” is a Unix trick; it
always find a name for whatever machine you’re on.

Using “ssh”, the user will have to setup password-less login to remote
hosts using public key authentication based on a key-pair and adding
public keys to “.ssh/authorized_keys” file. See “ssh” documentation for
more information. If “rsh” is used for remote login to the compute
nodes, the user is required to set up remote login permissions on all
nodes using the “.rhosts” file in their home directory.

In a network environment, ``charmrun`` must be able to locate the
directory of the executable. If all workstations share a common file
name space this is trivial. If they don’t, ``charmrun`` will attempt to
find the executable in a directory with the same path from the **$HOME**
directory. Pathname resolution is performed as follows:

#. The system computes the absolute path of ``pgm``.

#. If the absolute path starts with the equivalent of **$HOME** or the
   current working directory, the beginning part of the path is replaced
   with the environment variable **$HOME** or the current working
   directory. However, if ``++pathfix dir1 dir2`` is specified in the
   nodes file (see above), the part of the path matching ``dir1`` is
   replaced with ``dir2``.

#. The system tries to locate this program (with modified pathname and
   appended extension if specified) on all nodes.

.. _sec:keywords:

Reserved words in ``.ci`` files
-------------------------------

The following words are reserved for the Charm++ interface translator,
and cannot appear as variable or entry method names in a ``.ci`` file:

-  module

-  mainmodule

-  chare

-  mainchare

-  group

-  nodegroup

-  namespace

-  array

-  message

-  conditional

-  extern

-  initcall

-  initnode

-  initproc

-  readonly

-  PUPable

-  pupable

-  template

-  class

-  include

-  virtual

-  packed

-  varsize

-  entry

-  using

-  nocopy

-  Entry method attributes

   -  stacksize

   -  threaded

   -  migratable

   -  createhere

   -  createhome

   -  sync

   -  iget

   -  exclusive

   -  immediate

   -  expedited

   -  inline

   -  local

   -  aggregate

   -  nokeep

   -  notrace

   -  python

   -  accel

   -  readwrite

   -  writeonly

   -  accelblock

   -  memcritical

   -  reductiontarget

-  Basic C++ types

   -  int

   -  short

   -  long

   -  char

   -  float

   -  double

   -  unsigned

   -  void

   -  const

-  SDAG constructs

   -  atomic

   -  serial

   -  forward

   -  when

   -  while

   -  for

   -  forall

   -  if

   -  else

   -  overlap

   -  connect

   -  publishes

.. _sec:trace-projections:

Performance Tracing for Analysis
--------------------------------

Projections is a performance analysis/visualization framework that helps
you understand and investigate performance-related problems in the
(Charm++) applications. It is a framework with an event tracing
component which allows to control the amount of information generated.
The tracing has low perturbation on the application. It also has a
Java-based visualization and analysis component with various views that
help present the performance information in a visually useful manner.

Performance analysis with Projections typically involves two simple
steps:

#. Prepare your application by linking with the appropriate trace
   generation modules and execute it to generate trace data.

#. Using the Java-based tool to visually study various aspects of the
   performance and locate the performance issues for that application
   execution.

The Charm++ runtime automatically records pertinent performance data for
performance-related events during execution. These events include the
start and end of entry method execution, message send from entry methods
and scheduler idle time. This means *most* users do not need to manually
insert code into their applications in order to generate trace data. In
scenarios where special performance information not captured by the
runtime is required, an API (see section :numref:`sec::api_charm`) is
available for user-specific events with some support for visualization
by the Java-based tool. If greater control over tracing activities (e.g.
dynamically turning instrumentation on and off) is desired, the API also
allows users to insert code into their applications for such purposes.

The automatic recording of events by the Projections framework
introduces the overhead of an if-statement for each runtime event, even
if no performance analysis traces are desired. Developers of Charm++
applications who consider such an overhead to be unacceptable (e.g. for
a production application which requires the absolute best performance)
may recompile the Charm++ runtime with the ``--with-production`` flag,
which removes the instrumentation stubs. To enable the instrumentation
stubs while retaining the other optimizations enabled by
``--with-production``, one may compile Charm++ with both
``--with-production`` and ``--enable-tracing``, which explicitly enables
Projections tracing.

To enable performance tracing of your application, users simply need to
link the appropriate trace data generation module(s) (also referred to
as *tracemode(s)*). (see section :numref:`sec::trace modules_charm`)

.. _sec::trace modules_charm:

Enabling Performance Tracing at Link/Run Time
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Projections tracing modules dictate the type of performance data, data
detail and data format each processor will record. They are also
referred to as “tracemodes”. There are currently 2 tracemodes available.
Zero or more tracemodes may be specified at link-time. When no
tracemodes are specified, no trace data is generated.

.. _sec::trace module projections_charm:

Tracemode ``projections``
^^^^^^^^^^^^^^^^^^^^^^^^^

Link time option: ``-tracemode projections``

This tracemode generates files that contain information about all
Charm++ events like entry method calls and message packing during the
execution of the program. The data will be used by Projections in
visualization and analysis.

This tracemode creates a single symbol table file and :math:`p` ASCII
log files for :math:`p` processors. The names of the log files will be
NAME.#.log where NAME is the name of your executable and # is the
processor #. The name of the symbol table file is NAME.sts where NAME is
the name of your executable.

This is the main source of data needed by the performance visualizer.
Certain tools like timeline will not work without the detail data from
this tracemode.

The following is a list of runtime options available under this
tracemode:

-  ``+logsize NUM``: keep only NUM log entries in the memory of each
   processor. The logs are emptied and flushed to disk when filled.
   (defaults to 1,000,000)

-  ``+binary-trace``: generate projections log in binary form.

-  ``+gz-trace``: generate gzip (if available) compressed log files.

-  ``+gz-no-trace``: generate regular (not compressed) log files.

-  ``+checknested``: a debug option. Checks if events are improperly
   nested while recorded and issue a warning immediately.

-  ``+trace-subdirs NUM``: divide the generated log files among ``NUM``
   subdirectories of the trace root, each named ``PROGNAME.projdir.K``

Tracemode ``summary``
^^^^^^^^^^^^^^^^^^^^^

Compile option: ``-tracemode summary``

In this tracemode, execution time across all entry points for each
processor is partitioned into a fixed number of equally sized
time-interval bins. These bins are globally resized whenever they are
all filled in order to accommodate longer execution times while keeping
the amount of space used constant.

Additional data like the total number of calls made to each entry point
is summarized within each processor.

This tracemode will generate a single symbol table file and :math:`p`
ASCII summary files for :math:`p` processors. The names of the summary
files will be NAME.#.sum where NAME is the name of your executable and #
is the processor #. The name of the symbol table file is NAME.sum.sts
where NAME is the name of your executable.

This tracemode can be used to control the amount of output generated in
a run. It is typically used in scenarios where a quick look at the
overall utilization graph of the application is desired to identify
smaller regions of time for more detailed study. Attempting to generate
the same graph using the detailed logs of the prior tracemode may be
unnecessarily time consuming or impossible.

The following is a list of runtime options available under this
tracemode:

-  ``+bincount NUM``: use NUM time-interval bins. The bins are resized
   and compacted when filled.

-  ``+binsize TIME``: sets the initial time quantum each bin represents.

-  ``+version``: set summary version to generate.

-  ``+sumDetail``: Generates a additional set of files, one per
   processor, that stores the time spent by each entry method associated
   with each time-bin. The names of “summary detail” files will be
   NAME.#.sumd where NAME is the name of your executable and # is the
   processor #.

-  ``+sumonly``: Generates a single file that stores a single
   utilization value per time-bin, averaged across all processors. This
   file bears the name NAME.sum where NAME is the name of your
   executable. This runtime option currently overrides the
   ``+sumDetail`` option.

.. _sec::general options_charm:

General Runtime Options
^^^^^^^^^^^^^^^^^^^^^^^

The following is a list of runtime options available with the same
semantics for all tracemodes:

-  ``+traceroot DIR``: place all generated files in DIR.

-  ``+traceoff``: trace generation is turned off when the application is
   started. The user is expected to insert code to turn tracing on at
   some point in the run.

-  ``+traceWarn``: By default, warning messages from the framework are
   not displayed. This option enables warning messages to be printed to
   screen. However, on large numbers of processors, they can overwhelm
   the terminal I/O system of the machine and result in unacceptable
   perturbation of the application.

-  ``+traceprocessors RANGE``: Only output logfiles for PEs present in
   the range (i.e. ``0-31,32-999966:1000,999967-999999`` to record every
   PE on the first 32, only every thousanth for the middle range, and
   the last 32 for a million processor run).

End-of-run Analysis for Data Reduction
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

As applications are scaled to thousands or hundreds of thousands of
processors, the amount of data generated becomes extremely large and
potentially unmanageable by the visualization tool. At the time of
documentation, Projections is capable of handling data from 8000+
processors but with somewhat severe tool responsiveness issues. We have
developed an approach to mitigate this data size problem with options to
trim-off “uninteresting” processors’ data by not writing such data at
the end of an application’s execution.

This is currently done through heuristics to pick out interesting
extremal (i.e. poorly behaved) processors and at the same time using a
k-means clustering to pick out exemplar processors from equivalence
classes to form a representative subset of processor data. The analyst
is advised to also link in the summary module via ``+tracemode summary``
and enable the ``+sumDetail`` option in order to retain some profile
data for processors whose data were dropped.

-  ``+extrema``: enables extremal processor identification analysis at
   the end of the application’s execution.

-  ``+numClusters``: determines the number of clusters (equivalence
   classes) to be used by the k-means clustering algorithm for
   determining exemplar processors. Analysts should take advantage of
   their knowledge of natural application decomposition to guess at a
   good value for this.

This feature is still being developed and refined as part of our
research. It would be appreciated if users of this feature could contact
the developers if you have input or suggestions.

.. _sec::api_charm:

Controlling Tracing from Within the Program
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Selective Tracing
^^^^^^^^^^^^^^^^^

Charm++ allows users to start/stop tracing the execution at certain
points in time on the local processor. Users are advised to make these
calls on all processors and at well-defined points in the application.

Users may choose to have instrumentation turned off at first (by command
line option ``+traceoff`` - see section
:numref:`sec::general options_charm`) if some period of time in middle of
the applicationś execution is of interest to the user.

Alternatively, users may start the application with instrumentation
turned on (default) and turn off tracing for specific sections of the
application.

Again, users are advised to be consistent as the ``+traceoff`` runtime
option applies to all processors in the application.

-  ``void traceBegin()``

   Enables the runtime to trace events (including all user events) on
   the local processor where ``traceBegin`` is called.

-  ``void traceEnd()``

   Prevents the runtime from tracing events (including all user events)
   on the local processor where ``traceEnd`` is called.

Explicit Flushing
^^^^^^^^^^^^^^^^^

By default, when linking with ``-tracemode projections``, log files are
flushed to disk whenever the number of entries on a processor reaches
the logsize limit (see Section :numref:`sec::trace module projections_charm`).
However, this can occur at any time during the
execution of the program, potentially causing performance perturbations.
To address this, users can explicitly flush to disk using the
``traceFlushLog()`` function. Note that automatic flushing will still
occur if the logsize limit is reached, but sufficiently frequent
explicit flushes should prevent that from happening.

-  ``void traceFlushLog()``

   Explicitly flushes the collected logs to disk.

User Events
^^^^^^^^^^^

Projections has the ability to visualize traceable user specified
events. User events are usually displayed in the Timeline view as
vertical bars above the entry methods. Alternatively the user event can
be displayed as a vertical bar that vertically spans the timelines for
all processors. Follow these following basic steps for creating user
events in a charm++ program:

#. Register an event with an identifying string and either specify or
   acquire a globally unique event identifier. All user events that are
   not registered will be displayed in white.

#. Use the event identifier to specify trace points in your code of
   interest to you.

The functions available are as follows:

-  ``int traceRegisterUserEvent(char* EventDesc, int EventNum=-1)``

   This function registers a user event by associating ``EventNum`` to
   ``EventDesc``. If ``EventNum`` is not specified, a globally unique
   event identifier is obtained from the runtime and returned. The
   string ``EventDesc`` must either be a constant string, or it can be a
   dynamically allocated string that is **NOT** freed by the program. If
   the ``EventDesc`` contains a substring “\**\*” then the Projections
   Timeline tool will draw the event vertically spanning all PE
   timelines.

   ``EventNum`` has to be the same on all processors. Therefore use one
   of the following methods to ensure the same value for any PEs
   generating the user events:

   #. Call ``traceRegisterUserEvent`` on PE 0 in main::main without
      specifying an event number, and store returned event number into a
      readonly variable.

   #. Call ``traceRegisterUserEvent`` and specify the event number on
      processor 0. Doing this on other processors would have no effect.
      Afterwards, the event number can be used in the following user
      event calls.

   Eg. ``traceRegisterUserEvent("Time Step Begin", 10);``

   Eg. ``eventID = traceRegisterUserEvent(“Time Step Begin”);``

There are two main types of user events, bracketed and non bracketed.
Non-bracketed user events mark a specific point in time. Bracketed user
events span an arbitrary contiguous time range. Additionally, the user
can supply a short user supplied text string that is recorded with the
event in the log file. These strings should not contain newline
characters, but they may contain simple html formatting tags such as
``<br>``, ``<b>``, ``<i>``, ``<font color=#ff00ff>``, etc.

The calls for recording user events are the following:

-  ``void traceUserEvent(int EventNum)``

   This function creates a user event that marks a specific point in
   time.

   Eg. ``traceUserEvent(10);``

-  ``void traceBeginUserBracketEvent(int EventNum)``

   ``void traceEndUserBracketEvent(int EventNum)``

   These functions record a user event spanning a time interval. The
   tracing framework automatically associates the call with the time it
   was made, so timestamps are not explicitly passed in as they are with
   ``traceUserBracketEvent``.

-  ``void traceUserBracketEvent(int EventNum, double StartTime, double EndTime)``

   This function records a user event spanning a time interval from
   ``StartTime`` to ``EndTime``. Both ``StartTime`` and ``EndTime``
   should be obtained from a call to ``CmiWallTimer()`` at the
   appropriate point in the program.

   Eg.

   .. code-block:: c++

         traceRegisterUserEvent("Critical Code", 20); // on PE 0
         double critStart = CmiWallTimer();;  // start time
         // do the critical code
         traceUserBracketEvent(20, critStart,CmiWallTimer());

-  ``void traceUserSuppliedData(int data)``

   This function records a user specified data value at the current
   time. This data value can be used to color entry method invocations
   in Timeline, see :numref:`sec::timeline view`.

-  ``void traceUserSuppliedNote(char * note)``

   This function records a user specified text string at the current
   time.

-  ``void traceUserSuppliedBracketedNote(char *note, int EventNum, double StartTime, double EndTime)``

   This function records a user event spanning a time interval from
   ``StartTime`` to ``EndTime``. Both ``StartTime`` and ``EndTime``
   should be obtained from a call to ``CmiWallTimer()`` at the
   appropriate point in the program.

   Additionally, a user supplied text string is recorded, and the
   ``EventNum`` is recorded. These events are therefore displayed with
   colors determined by the ``EventNum``, just as those generated with
   ``traceUserBracketEvent`` are.

User Stats
^^^^^^^^^^

Charm++ allows the user to track the progression of any variable or
value throughout the program execution. These user specified stats can
then be plotted in Projections, either over time or by processor. To
enable this feature for Charm++, build Charm++ with the
``-enable-tracing`` flag.

Follow these steps to track user stats in a Charm++ program:

#. Register a stat with an identifying string and a globally unique
   integer identifier.

#. Update the value of the stat at points of interest in the code by
   calling the update stat functions.

#. Compile program with -tracemode projections flag.

The functions available are as follows:

-  ``int traceRegisterUserStat(const char * StatDesc, int StatNum)``

   This function is called once near the beginning the of the Charm++
   program. ``StatDesc`` is the identifying string and ``StatNum`` is
   the unique integer identifier.

-  ``void updateStat(int StatNum, double StatValue)``

   This function updates the value of a user stat and can be called many
   times throughout program execution. ``StatNum`` is the integer
   identifier corresponding to the desired stat. ``StatValue`` is the
   updated value of the user stat.

-  ``void updateStatPair(int StatNum, double StatValue, double Time)``

   This function works similar to ``updateStat()``, but also allows the
   user to store a user specified time for the update. In Projections,
   the user can then choose which time scale to use: real time, user
   specified time, or ordered.

Function-level Tracing for Adaptive MPI Applications
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Adaptive MPI (AMPI) is an implementation of the MPI interface on top of
Charm++. As with standard MPI programs, the appropriate semantic context
for performance analysis is captured through the observation of MPI
calls within C/C++/Fortran functions. Users can selectively begin and
end tracing in AMPI programs using the routines ``AMPI_Trace_begin`` and
``AMPI_Trace_end``.

Debugging
---------

Message Order Race Conditions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

While common Charm++ programs are data-race free due to a lack of shared
mutable state between threads, it is still possible to observe race
conditions resulting from variation in the order that messages are
delivered to each chare object. The Charm++ ecosystem offers a variety
of ways to attempt to reproduce these often non-deterministic bugs,
diagnose their causes, and test fixes.

Randomized Message Queueing
^^^^^^^^^^^^^^^^^^^^^^^^^^^

To facilitate debugging of applications and to identify race conditions
due to message order, the user can enable a randomized message scheduler
queue. Using the build-time configuration option
``--enable-randomized-msgq``, the charm message queue will be
randomized. Note that a randomized message queue is only available when
message priority type is not bit vector. Therefore, the user needs to
specify prio-type to be a data type long enough to hold the msg
priorities in your application for eg: ``--with-prio-type=int``.

CharmDebug
^^^^^^^^^^

The CharmDebug interactive debugging tool can be used to inspect the
messages in the scheduling queue of each processing element, and to
manipulate the order in which they’re delivered. More details on how to
use CharmDebug can be found in its manual.

Deterministic Record-Replay
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Charm++ supports recording the order in which messages are processed
from one run, to deterministically replay the same order in subsequent
runs. This can be useful to capture the infrequent undesirable message
order cases that cause intermittent failures. Once an impacted run has
been recorded, various debugging methods can be more easily brought to
bear, without worrying that they will perturb execution to avoid the
bug.

Support for record-replay is enabled in common builds of Charm++. Builds
with the ``--with-production`` option disable this support to reduce
overhead. To record traces, simply run the program with an additional
command line-flag ``+record``. The generated traces can be repeated with
the command-line flag ``+replay``. The full range of parallel and
sequential debugging techniques are available to apply during
deterministic replay.

The traces will work even if the application is modified and recompiled,
as long as entry method numbering and send/receive sequences do not
change. For instance, it is acceptable to add print statements or
assertions to aid in the debugging process.

Memory Access Errors
~~~~~~~~~~~~~~~~~~~~

Using Valgrind
^^^^^^^^^^^^^^

The popular Valgrind memory debugging tool can be used to monitor
Charm++ applications in both serial and parallel executions. For
single-process runs, it can be used directly:

.. code-block:: bash

   $ valgrind ...valgrind options... ./application_name ...application arguments...

When running in parallel, it is helpful to note a few useful adaptations
of the above incantation, for various kinds of process launchers:

.. code-block:: bash

   $ ./charmrun +p2 `which valgrind` --log-file=VG.out.%p --trace-children=yes ./application_name ...application arguments...
   $ aprun -n 2 `which valgrind` --log-file=VG.out.%p --trace-children=yes ./application_name ...application arguments...

The first adaptation is to use :literal:`\`which valgrind\`` to obtain a
full path to the valgrind binary, since parallel process launchers
typically do not search the environment ``$PATH`` directories for the
program to run. The second adaptation is found in the options passed to
valgrind. These will make sure that valgrind tracks the spawned
application process, and write its output to per-process logs in the
file system rather than standard error.

History
-------

The Charm software was developed as a group effort of the Parallel
Programming Laboratory at the University of Illinois at
Urbana-Champaign. Researchers at the Parallel Programming Laboratory
keep Charm++ updated for the new machines, new programming paradigms,
and for supporting and simplifying development of emerging applications
for parallel processing. The earliest prototype, Chare Kernel(1.0), was
developed in the late eighties. It consisted only of basic remote method
invocation constructs available as a library. The second prototype,
Chare Kernel(2.0), a complete re-write with major design changes. This
included C language extensions to denote Chares, messages and
asynchronous remote method invocation. Charm(3.0) improved on this
syntax, and contained important features such as information sharing
abstractions, and chare groups (called Branch Office Chares). Charm(4.0)
included Charm++ and was released in fall 1993. Charm++ in its initial
version consisted of syntactic changes to C++ and employed a special
translator that parsed the entire C++ code while translating the syntactic
extensions. Charm(4.5) had a major change that resulted from a
significant shift in the research agenda of the Parallel Programming
Laboratory. The message-driven runtime system code of the Charm++ was
separated from the actual language implementation, resulting in an
interoperable parallel runtime system called Converse. The
Charm++ runtime system was retargetted on top of Converse, and popular
programming paradigms such as MPI and PVM were also implemented on
Converse. This allowed interoperability between these paradigms and
Charm++. This release also eliminated the full-fledged
Charm++ translator by replacing syntactic extensions to C++ with C++ macros,
and instead contained a small language and a translator for describing
the interfaces of Charm++ entities to the runtime system. This version
of Charm++, which, in earlier releases was known as *Interface
Translator Charm++*, is the default version of Charm++ now, and hence
referred simply as **Charm++**. In early 1999, the runtime system of
Charm++  was rewritten in C++. Several new features were added. The
interface language underwent significant changes, and the macros that
replaced the syntactic extensions in original Charm++, were replaced by
natural C++ constructs. Late 1999, and early 2000 reflected several
additions to Charm++, when a load balancing framework and migratable
objects were added to Charm++.

Acknowledgements
----------------

-  Aaron Becker

-  Abhinav Bhatele

-  Abhishek Gupta

-  Akhil Langer

-  Amit Sharma

-  Anshu Arya

-  Artem Shvorin

-  Arun Singla

-  Attila Gursoy

-  Bilge Acun

-  Chao Huang

-  Chao Mei

-  Chee Wai Lee

-  Cyril Bordage

-  David Kunzman

-  Dmitriy Ofman

-  Edgar Solomonik

-  Ehsan Totoni

-  Emmanuel Jeannot

-  Eric Bohm

-  Eric Mikida

-  Eric Shook

-  Esteban Meneses

-  Esteban Pauli

-  Filippo Gioachin

-  Gengbin Zheng

-  Greg Koenig

-  Gunavardhan Kakulapati

-  Hari Govind

-  Harshit Dokania

-  Harshitha Menon

-  Isaac Dooley

-  Jaemin Choi

-  Jayant DeSouza

-  Jeffrey Wright

-  Jim Phillips

-  Jonathan Booth

-  Jonathan Lifflander

-  Joshua Unger

-  Josh Yelon

-  Juan Galvez

-  Kavitha Chandrasekar

-  Laxmikant Kale

-  Lixia Shi

-  Lukasz Wesolowski

-  Mani Srinivas Potnuru

-  Michael Robson

-  Milind Bhandarkar

-  Minas Charalambides

-  Narain Jagathesan

-  Neelam Saboo

-  Nihit Desai

-  Nikhil Jain

-  Nilesh Choudhury

-  Nitin Bhat

-  Orion Lawlor

-  Osman Sarood

-  Parthasarathy Ramachandran

-  Phil Miller

-  Prateek Jindal

-  Pritish Jetley

-  Puneet Narula

-  Rahul Joshi

-  Ralf Gunter

-  Ramkumar Vadali

-  Ramprasad Venkataraman

-  Rashmi Jyothi

-  Robert Blake

-  Robert Brunner

-  Ronak Buch

-  Rui Liu

-  Ryan Mokos

-  Sam White

-  Sameer Kumar

-  Sameer Paranjpye

-  Sanjeev Krishnan

-  Sayantan Chakravorty

-  Seonmyeong Bak

-  Sindhura Bandhakavi

-  Tarun Agarwal

-  Terry L. Wilmarth

-  Theckla Louchios

-  Tim Hinrichs

-  Timothy Knauff

-  Vikas Mehta

-  Viraj Paropkari

-  Vipul Harsh

-  Xiang Ni

-  Yanhua Sun

-  Yan Shi

-  Yogesh Mehta

-  Zheng Shao
