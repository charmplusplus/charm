=============
Jade Language
=============

.. contents::
   :depth: 3

Introduction
============

This manual describes Jade, which is a parallel programming language
developed over Charm++ and Java. Charm++ is a C++-based parallel
programming library developed by Prof. L. V. Kalé and his students over
the last 10 years at University of Illinois.

We first describe our philosophy behind this work (why we do what we
do). Later we give a brief introduction to Charm++ and rationale for
Jade. We describe Jade in detail. Appendices contain the details of
installing Jade, building and running Jade programs.

Our Philosophy
--------------

Terminology
-----------

Module
  A module refers to a named container which is the top-level construct
  in a program.

Thread
   A thread is a lightweight process that owns a stack and machine
   registers including program counter, but shares code and data with
   other threads within the same address space. If the underlying
   operating system recognizes a thread, it is known as kernel thread,
   otherwise it is known as user-thread. A context-switch between
   threads refers to suspending one thread’s execution and transferring
   control to another thread. Kernel threads typically have higher
   context switching costs than user-threads because of operating system
   overheads. The policy implemented by the underlying system for
   transferring control between threads is known as thread scheduling
   policy. Scheduling policy for kernel threads is determined by the
   operating system, and is often more inflexible than user-threads.
   Scheduling policy is said to be non-preemptive if a context-switch
   occurs only when the currently running thread willingly asks to be
   suspended, otherwise it is said to be preemptive. Jade threads are
   non-preemptive user-level threads.

Object
   An object is just a blob of memory on which certain computations can
   be performed. The memory is referred to as an object’s state, and the
   set of computations that can be performed on the object is called the
   interface of the object.

Charm++
=======

Charm++ is an object-oriented parallel programming library for C++. It
differs from traditional message passing programming libraries (such as
MPI) in that Charm++ is “message-driven”. Message-driven parallel
programs do not block the processor waiting for a message to be
received. Instead, each message carries with itself a computation that
the processor performs on arrival of that message. The underlying
runtime system of Charm++ is called Converse, which implements a
“scheduler” that chooses which message to schedule next
(message-scheduling in Charm++ involves locating the object for which
the message is intended, and executing the computation specified in the
incoming message on that object). A parallel object in Charm++ is a C++
object on which a certain computations can be asked to performed from
remote processors.

Charm++ programs exhibit latency tolerance since the scheduler always
picks up the next available message rather than waiting for a particular
message to arrive. They also tend to be modular, because of their
object-based nature. Most importantly, Charm++ programs can be
*dynamically load balanced*, because the messages are directed at
objects and not at processors; thus allowing the runtime system to
migrate the objects from heavily loaded processors to lightly loaded
processors. It is this feature of Charm++ that we utilize for Jade.

Since many CSE applications are originally written using MPI, one would
have to do a complete rewrite if they were to be converted to Charm++ to
take advantage of dynamic load balancing. This is indeed impractical.
However, Converse - the runtime system of Charm++ - came to our rescue
here, since it supports interoperability between different parallel
programming paradigms such as parallel objects and threads. Using this
feature, we developed Jade, an implementation of a significant subset of
MPI-1.1 standard over Charm++. Jade is described in the next section.

Jade
====

Every mainchare’s main is executed at startup.

threaded methods
----------------

::

   class C {
       public threaded void start(CProxy_CacheGroup cg) { ... }
   }

readonly
--------

::

   class C {
       public static readonly CProxy_TheMain mainChare;
       public static int readonly aReadOnly;
   }

The readonly variable can be accessed as ``C.aReadOnly``.

Must be initialized in the main of a mainchare. Value at the end of main
is propagated to all processors. Then execution begins.

msa
---

::

   arr1.enroll();
   int a = arr1[10]; // get
   arr1[10] = 122; // set
   arr1[10] += 2;  // accumulate
   arr1.sync();    // sync


Installing Jade
===============

Jade is included in the source distribution of Charm++. To get the
latest sources from PPL, visit: http://charm.cs.uiuc.edu/

and follow the download link. Now one has to build Charm++ and Jade from
source.

The build script for Charm++ is called ``build``. The syntax for this
script is:

::

   > build <target> <version> <opts>

For building Jade(which also includes building Charm++ and other
libraries needed by Jade), specify ``<target>`` to be ``jade``. And
``<opts>`` are command line options passed to the ``charmc`` compile
script. Common compile time options such as
``-g, -O, -Ipath, -Lpath, -llib`` are accepted.

To build a debugging version of Jade, use the option: ``-g``. To build
a production version of Jade, use the options:
``-O -DCMK_OPTIMIZE=1``.

``<version>`` depends on the machine, operating system, and the
underlying communication library one wants to use for running Jade
programs. See the ``charm/README`` file for details on picking the proper
version. Following is an example of how to build Jade under linux and
ethernet environment, with debugging info produced:

::

   > build jade netlrts-linux -g

Compiling and Running Jade Programs
===================================

Compiling Jade Programs
-----------------------

Charm++ provides a cross-platform compile-and-link script called
``charmc`` to compile C, C++, Fortran, Charm++ and Jade programs. This
script resides in the ``bin`` subdirectory in the Charm++ installation
directory. The main purpose of this script is to deal with the
differences of various compiler names and command-line options across
various machines on which Charm++ runs.

In spite of the platform-neutral syntax of ``charmc``, one may have to
specify some platform-specific options for compiling and building Jade
codes. Fortunately, if ``charmc`` does not recognize any particular
options on its command line, it promptly passes it to all the individual
compilers and linkers it invokes to compile the program.

You can use ``charmc`` to build your Jade program the same way as other
compilers like ``cc``. To build an Jade program, the command line option
*-language jade* should be specified. All the command line flags that
you would use for other compilers can be used with ``charmc`` the same
way. For example:

::

   > charmc -language jade -c pgm.java -O3
   > charmc -language jade -o pgm pgm.o -lm -O3

Running
-------

The Charm++ distribution contains a script called ``charmrun`` that
makes the job of running Jade programs portable and easier across all
parallel machines supported by Charm++. When compiling a Jade program,
``charmc`` copies ``charmrun`` to the directory where the Jade program
is built. ``charmrun`` takes a command line parameter specifying the
number of processors to run on, and the name of the program followed by
Jade options (such as TBD) and the program arguments. A typical
invocation of Jade program ``pgm`` with ``charmrun`` is:

::

   > charmrun pgm +p16 +vp32 +tcharm_stacksize 3276800

Here, the Jade program ``pgm`` is run on 16 physical processors with 32
chunks (which will be mapped 2 per processor initially), where each
user-level thread associated with a chunk has the stack size of
3,276,800 bytes.

Jade Developer documentation
============================

Files
-----

Jade source files are spread out across several directories of the
Charm++ CVS tree.

====================== =============================================
charm/doc/jade         Jade user documentation files
charm/src/langs/jade/  ANTLR parser files, Jade runtime library code
charm/java/charm/jade/ Jade java code
charm/java/bin/        Jade scripts
charm/pgms/jade/       Jade example programs and tests
====================== =============================================

After building Jade, files are installed in:

=============== =================================
charm/include/  Jade runtime library header files
charm/lib/      Jade runtime library
charm/java/bin/ ``jade.jar`` file
=============== =================================

Java packages
-------------

The way packages work in Java is as follows: There is a ROOT directory.
Within the ROOT, a subdirectory is used which also gives the package
name. Beneath the package directory all the ``.class`` files are stored.
The ROOT directory should be placed in the java CLASSPATH.

For Jade, the ROOT is charm/java/charm/.

The Jade package name is ``jade``, and is in charm/java/charm/jade.
Within here, all the jade Java files are placed, they are compiled to
``.class`` files, and then jar’d up into the ``jade.jar`` file, which is
placed in charm/java/bin for convenience.
