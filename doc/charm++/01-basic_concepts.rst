Basic Concepts
==============

Charm++ is a C++-based parallel programming system, founded on the
migratable-objects programming model, and supported by a novel and
powerful adaptive runtime system. It supports both irregular as well as
regular applications, and can be used to specify task-parallelism as
well as data parallelism in a single application. It automates dynamic
load balancing for task-parallel as well as data-parallel applications,
via separate suites of load-balancing strategies. Via its message-driven
execution model, it supports automatic latency tolerance, modularity and
parallel composition. Charm++ also supports automatic
checkpoint/restart, as well as fault tolerance based on distributed
checkpoints.

Charm++ is a production-quality parallel programming system used by
multiple applications in science and engineering on supercomputers as
well as smaller clusters around the world. Currently the parallel
platforms supported by Charm++ are the IBM BlueGene/Q and OpenPOWER
systems, Cray XE, XK, and XC systems, Omni-Path and Infiniband clusters,
single workstations and networks of workstations (including x86 (running
Linux, Windows, MacOS)), etc. The communication protocols and
infrastructures supported by Charm++ are UDP, MPI, OFI, Infiniband,
uGNI, and PAMI. Charm++ programs can run without changing the source on
all these platforms. Charm++ programs can also interoperate with MPI
programs (§ :numref:`sec:interop`). Please see the Installation and Usage
section for details about installing, compiling and running Charm++
programs (§ :numref:`sec:install`).

Programming Model
-----------------

The key feature of the migratable-objects programming model is
*over-decomposition*: The programmer decomposes the program into a large
number of work units and data units, and specifies the computation in
terms of creation of and interactions between these units, without any
direct reference to the processor on which any unit resides. This
empowers the runtime system to assign units to processors, and to change
the assignment at runtime as necessary. Charm++ is the main (and early)
exemplar of this programming model. AMPI is another example within the
Charm++ family of the same model.

.. _mainchare:

Execution Model
---------------

A basic unit of parallel computation in Charm++ programs is a *chare* .
A chare is similar to a process, an actor, an ADA task, etc. At its most
basic level, it is just a C++ object. A Charm++ computation consists of
a large number of chares distributed on available processors of the
system, and interacting with each other via asynchronous method
invocations. Asynchronously invoking a method on a remote object can
also be thought of as sending a “message” to it. So, these method
invocations are sometimes referred to as messages. (besides, in the
implementation, the method invocations are packaged as messages anyway).
Chares can be created dynamically.

Conceptually, the system maintains a “work-pool” consisting of seeds for
new chares, and messages for existing chares. The Charm++ runtime system
( *Charm RTS*) may pick multiple items, non-deterministically, from this
pool and execute them, with the proviso that two different methods
cannot be simultaneously executing on the same chare object (say, on
different processors). Although one can define a reasonable theoretical
operational semantics of Charm++ in this fashion, a more practical
description of execution is useful to understand Charm++. A Charm++
application’s execution is distributed among Processing Elements (PEs),
which are OS threads or processes depending on the selected Charm++
build options. (See section :numref:`sec:machine` for a
precise description.) On each PE, there is a scheduler operating with
its own private pool of messages. Each instantiated chare has one PE
which is where it currently resides. The pool on each PE includes
messages meant for Chares residing on that PE, and seeds for new Chares
that are tentatively meant to be instantiated on that PE. The scheduler
picks a message, creates a new chare if the message is a seed (i.e. a
constructor invocation) for a new Chare, and invokes the method
specified by the message. When the method returns control back to the
scheduler, it repeats the cycle. I.e. there is no pre-emptive scheduling
of other invocations.

When a chare method executes, it may create method invocations for other
chares. The Charm Runtime System (RTS, sometimes referred to as the
Chare Kernel in the manual) locates the PE where the targeted chare
resides, and delivers the invocation to the scheduler on that PE.

Methods of a chare that can be remotely invoked are called *entry*
methods. Entry methods may take serializable parameters, or a pointer to
a message object. Since chares can be created on remote processors,
obviously some constructor of a chare needs to be an entry method.
Ordinary entry methods [1]_ are completely non-preemptive- Charm++ will
not interrupt an executing method to start any other work, and all calls
made are asynchronous.

Charm++ provides dynamic seed-based load balancing. Thus location
(processor number) need not be specified while creating a remote chare.
The Charm RTS will then place the remote chare on a suitable processor.
Thus one can imagine chare creation as generating only a seed for the
new chare, which may *take root* on some specific processor at a later
time.

Chares can be grouped into collections. The types of collections of
chares supported in Charm++ are: *chare-arrays*, *chare-groups*, and
*chare-nodegroups*, referred to as *arrays*, *groups*, and *nodegroups*
throughout this manual for brevity. A Chare-array is a collection of an
arbitrary number of migratable chares, indexed by some index type, and
mapped to processors according to a user-defined map group. A group
(nodegroup) is a collection of chares, with exactly one member element
on each PE (“node”).

Charm++ does not allow global variables, except readonly variables (see
:numref:`readonly`). A chare can normally only access its own data
directly. However, each chare is accessible by a globally valid name.
So, one can think of Charm++ as supporting a *global object space*.

Every Charm++ program must have at least one mainchare. Each mainchare
is created by the system on processor 0 when the Charm++ program starts
up. Execution of a Charm++ program begins with the Charm Kernel
constructing all the designated mainchares. For a mainchare named X,
execution starts at constructor X() or X(CkArgMsg \*) which are
equivalent. Typically, the mainchare constructor starts the computation
by creating arrays, other chares, and groups. It can also be used to
initialize shared readonly objects.

Charm++ program execution is terminated by the CkExit call. Like the
exit system call, CkExit never returns, and it optionally accepts an
integer value to specify the exit code that is returned to the calling
shell. If no exit code is specified, a value of zero (indicating
successful execution) is returned. The Charm RTS ensures that no more
messages are processed and no entry methods are called after a CkExit.
CkExit need not be called on all processors; it is enough to call it
from just one processor at the end of the computation.

As described so far, the execution of individual Chares is “reactive”:
When method A is invoked the chare executes this code, and so on. But
very often, chares have specific life-cycles, and the sequence of entry
methods they execute can be specified in a structured manner, while
allowing for some localized non-determinism (e.g. a pair of methods may
execute in any order, but when they both finish, the execution continues
in a pre-determined manner, say executing a 3rd entry method). To
simplify expression of such control structures, Charm++ provides two
methods: the structured dagger notation (Sec :numref:`sec:sdag`), which
is the main notation we recommend you use. Alternatively, you may use
threaded entry methods, in combination with *futures* and *sync* methods
(See :numref:`threaded`). The threaded methods run in light-weight
user-level threads, and can block waiting for data in a variety of ways.
Again, only the particular thread of a particular chare is blocked,
while the PE continues executing other chares.

The normal entry methods, being asynchronous, are not allowed to return
any value, and are declared with a void return type. However, the *sync*
methods are an exception to this. They must be called from a threaded
method, and so are allowed to return (certain types of) values.

.. _proxies:

Proxies and the charm interface file
------------------------------------

To support asynchronous method invocation and global object space, the
RTS needs to be able to serialize (“marshall”) the parameters, and be
able to generate global “names” for chares. For this purpose,
programmers have to declare the chare classes and the signature of their
entry methods in a special “``.ci``” file, called an interface file.
Other than the interface file, the rest of a Charm++ program consists of
just normal C++ code. The system generates several classes based on the
declarations in the interface file, including “Proxy” classes for each
chare class. Those familiar with various component models (such as
CORBA) in the distributed computing world will recognize “proxy” to be a
dummy, standin entity that refers to an actual entity. For each chare
type, a “proxy” class exists. The methods of this “proxy” class
correspond to the remote methods of the actual class, and act as
“forwarders”. That is, when one invokes a method on a proxy to a remote
object, the proxy marshalls the parameters into a message, puts adequate
information about the target chare on the envelope of the message, and
forwards it to the remote object. Individual chares, chare array,
groups, node-groups, as well as the individual elements of these
collections have a such a proxy. Multiple methods for obtaining such
proxies are described in the manual. Proxies for each type of entity in
Charm++ have some differences among the features they support, but the
basic syntax and semantics remain the same - that of invoking methods on
the remote object by invoking methods on proxies.

The following sections provide detailed information about various
features of the Charm++ programming system. Part I, “Basic Usage”, is
sufficient for writing full-fledged applications. Note that only the
last two chapters of this part involve the notion of physical processors
(cores, nodes, ..), with the exception of simple query-type utilities
(Sec :numref:`basic utility fns`). We strongly suggest that all
application developers, beginners and experts alike, try to stick to the
basic language to the extent possible, and use features from the
advanced sections only when you are convinced they are essential. (They
are useful in specific situations; but a common mistake we see when we
examine programs written by beginners is the inclusion of complex
features that are not necessary for their purpose. Hence the caution).
The advanced concepts in the Part II of the manual support
optimizations, convenience features, and more complex or sophisticated
features.  [2]_

.. _machineModel:
.. _sec:machine:

Machine Model
-------------

At its basic level, Charm++ machine model is very simple:
Think of each chare as a separate processor by itself. The methods of
each chare can access its own instance variables (which are all private,
at this level), and any global variables declared as *readonly*. It also
has access to the names of all other chares (the “global object space”),
but all that it can do with that is to send asynchronous remote method
invocations towards other chare objects. (Of course, the instance
variables can include as many other regular C++ objects that it “has”;
but no chare objects. It can only have references to other chare
objects).

In accordance with this vision, the first part of the manual (up to and
including the chapter on load balancing) has almost no mention of
entities with physical meanings (cores, nodes, etc.). The runtime system
is responsible for the magic of keeping closely communicating objects on
nearby physical locations, and optimizing communications within chares
on the same node or core by exploiting the physically available shared
memory. The programmer does not have to deal with this at all. The only
exception to this pure model in the basic part are the functions used
for finding out which “processor” an object is running on, and for
finding how many total processors are there.

However, for implementing lower level libraries, and certain
optimizations, programmers need to be aware of processors. In any case,
it is useful to understand how the Charm++ implementation works under
the hood. So, we describe the machine model, and some associated
terminology here.

In terms of physical resources, we assume the parallel machine consists
of one or more *nodes*, where a node is a largest unit over which cache
coherent shared memory is feasible (and therefore, the maximal set of
cores per which a single process *can* run. Each node may include one or
more processor chips, with shared or private caches between them. Each
chip may contain multiple cores, and each core may support multiple
hardware threads (SMT for example).

Charm++ recognizes two logical entities: a PE (processing element) and a
logical node, or simply “node”. In a Charm++ program, a PE is a unit of
mapping and scheduling: each PE has a scheduler with an associated pool
of messages. Each chare is assumed to reside on one PE at a time. A
logical node is implemented as an OS process. In non-SMP mode there is
no distinction between a PE and a logical node. Otherwise, a PE takes
the form of an OS thread, and a logical node may contain one or more
PEs. Physical nodes may be partitioned into one or more logical nodes.
Since PEs within a logical node share the same memory address space, the
Charm++ runtime system optimizes communication between them by using
shared memory. Depending on the runtime command-line parameters, a PE
may optionally be associated with a subset of cores or hardware threads.

A Charm++ program can be launched with one or more (logical) nodes per
physical node. For example, on a machine with a four-core processor,
where each core has two hardware threads, common configurations in
non-SMP mode would be one node per core (four nodes/PEs total) or one
node per hardware thread (eight nodes/PEs total). In SMP mode, the most
common choice to fully subscribe the physical node would be one logical
node containing *seven* PEs-one OS thread is set aside per process for
network communications. (When built in the “multicore” mode that lacks
network support, a comm thread is unnecessary, and eight PEs can be used
in this case. A comm thread is also omitted when using some
high-performance network layers such as PAMI.) Alternatively, one can
choose to partition the physical node into multiple logical nodes, each
containing multiple PEs. One example would be *three* PEs per logical
node and two logical nodes per physical node, again reserving a comm
thread per logical node.

It is not a general practice in Charm++ to oversubscribe the underlying
physical cores or hardware threads on each node. In other words, a
Charm++ program is usually not launched with more PEs than there are
physical cores or hardware threads allocated to it. More information
about these launch time options are provided in
Appendix :numref:`sec:run`. And utility functions to retrieve the
information about those Charm++ logical machine entities in user
programs can be referred in section :numref:`basic utility fns`.
