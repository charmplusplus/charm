Charm++ Quickstart
==================

This section gives a concise overview of running your first Charm++ application.

Installing Charm++
------------------

To download the latest Charm++ release, run:

.. code-block:: console

   $ wget https://charm.cs.illinois.edu/distrib/charm-latest.tar.gz
   $ tar xzf charm-latest.tar.gz

To download the development version of Charm++, run:

.. code-block:: console

   $ git clone https://github.com/charmplusplus/charm


To build Charm++, use the following commands:

.. code-block:: console

   $ cd charm
   $ ./build charm++ netlrts-linux-x86_64 --with-production -j4

This is the recommended version to install Charm++ on Linux systems.
For macOS, substitute "linux" with "darwin", and on ARM64 machines,
replace "x86_64" with "arm8". For advanced compilation options,
please see Section :numref:`sec:install` of the manual.


Parallel "Hello World" with Charm++
-----------------------------------

The basic unit of computation in Charm++ is a **chare**, which is a C++
object. Chares have **entry methods** that can be invoked asynchronously.
A Charm++ application consists of collections of chares (such as chare arrays)
distributed among the processors of the system.

Each chare has a **proxy** associated to it, through which other chares can
invoke entry methods. This proxy is exposed through the **thisProxy** member variable,
which can be sent to other chares, allowing them to invoke entry methods on this chare.

Each Charm++ application consists of at least two files, a
*Charm interface* (`.ci`) file, and a normal C++ file. The interface
file describes the parallel interface of the application
(such as chares, chare arrays, and entry methods), while the C++ files
implement its behavior. Please see Section :numref:`programstructure`
of the manual for more information about the program structure.

In this section, we present a parallel *Hello World* example,
consisting of the files ``hello.ci`` and ``hello.cpp``.


The hello.ci File
'''''''''''''''''

The ``hello.ci`` file contains a mainchare, which starts and ends execution,
and a ``Hello`` chare array, whose elements print the "Hello World" message.
Compiling this file creates C++ header files (``hello.decl.h`` and ``hello.def.h``)
that can be included in your C++ files.

.. code-block:: charmci

   mainmodule hello {
     mainchare Main {
       // Main's entry methods
       entry Main(CkArgMsg *m);
       entry void done();
     };
     array [1D] Hello {
       // Hello's entry methods
       entry Hello();
       entry void SayHi();
     };            
   };


The hello.cpp File
''''''''''''''''''

The ``hello.cpp`` file contains the implementation of the mainchare and chare
array declared in the ``hello.ci`` file above.

.. code-block:: c++

   #include "hello.decl.h" // created from hello.ci file above

   /*readonly*/ CProxy_Main mainProxy;
   constexpr int nElem = 8;

   /*mainchare*/
   class Main : public CBase_Main
   {
   public:
     Main(CkArgMsg* m)
     {
       //Start computation
       CkPrintf("Running Hello on %d processors with %d elements.\n", CkNumPes(), nElem);
       CProxy_Hello arr = CProxy_Hello::ckNew(nElem); // Create a new chare array with nElem elements
       mainProxy = thisProxy;
       arr[0].SayHi(0);
     };

     void done()
     {
       // Finish computation
       CkPrintf("All done.\n");
       CkExit();
     };
   };

   /*array [1D]*/
   class Hello : public CBase_Hello 
   {
   public:
     Hello() {}

     void SayHi()
     {
       // thisIndex stores the element’s array index 
       CkPrintf("PE %d says: Hello world from element %d.\n", CkMyPe(), thisIndex);
       if (thisIndex < nElem - 1) {
         thisProxy[thisIndex + 1].SayHi(); // Pass the hello on
       } else {
         mainProxy.done(); // We've been around once -- we're done.
       }
     }
   };

   #include "hello.def.h" // created from hello.ci file above


Compiling the Example
'''''''''''''''''''''

Charm++ has a compiler wrapper, ``charmc``, to compile Charm++ applications. Please see
Section :numref:`sec:compile` for more information about ``charmc``.

.. code-block:: console

   $ charm/bin/charmc hello.ci # creates hello.def.h and hello.decl.h
   $ charm/bin/charmc hello.cpp -o hello


Running the Example
'''''''''''''''''''

Charm++ applications are started via ``charmrun``,
which is automatically created by the ``charmc`` command above.
Please see Section :numref:`sec:run` for more information about ``charmrun``.

To run the application on two processors, use the following command:

.. code-block:: console

   $ ./charmrun +p2 ./hello
   Charmrun> scalable start enabled.
   Charmrun> started all node programs in 1.996 seconds.
   Charm++> Running in non-SMP mode: 1 processes (PEs)
   Converse/Charm++ Commit ID: v6.9.0-172-gd31997cce
   Charm++> scheduler running in netpoll mode.
   CharmLB> Load balancer assumes all CPUs are same.
   Charm++> Running on 1 hosts (1 sockets x 4 cores x 2 PUs = 8-way SMP)
   Charm++> cpu topology info is gathered in 0.000 seconds.
   Running Hello on 2 processors with 8 elements.
   PE 0 says: Hello world from element 0.
   PE 0 says: Hello world from element 1.
   PE 0 says: Hello world from element 2.
   PE 0 says: Hello world from element 3.
   PE 1 says: Hello world from element 4.
   PE 1 says: Hello world from element 5.
   PE 1 says: Hello world from element 6.
   PE 1 says: Hello world from element 7.
   All done
   [Partition 0][Node 0] End of program


Where to go From Here
---------------------

- The ``tests/charm++/simplearrayhello`` folder in the Charm++ distribution has a more comprehensive example, from
  which the example in this file was derived.

- The main Charm++ manual (https://charm.readthedocs.io/) contains more information about developing
  and running Charm++ applications.

- Charm++ has lots of other features, such as chare migration, load balancing,
  and checkpoint/restart. The main manual has more information about them.

- AMPI (https://charm.readthedocs.io/en/latest/ampi/manual.html) is an implementation of MPI on top of Charm++, allowing
  MPI applications to run on the Charm++ runtime mostly unmodified.

- Charm4py (https://charm4py.readthedocs.io) is a Python package that enables development of Charm++ applications in Python.
