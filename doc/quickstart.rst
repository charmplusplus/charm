Charm++ Quickstart
==================


Installing Charm++
------------------

To download the latest Charm++ release, run:

.. code-block:: console

   $ wget http://charm.cs.illinois.edu/distrib/charm-6.9.0.tar.gz
   $ tar xzf charm-6.9.0.tar.gz

To download the development version of Charm++, run:

.. code-block:: console

   $ git clone https://github.com/UIUC-PPL/charm


To build Charm++, use the following commands:

.. code-block:: console

   $ cd charm
   $ ./build AMPI netlrts-linux-x86_64 --with-production -j4

This is the recommened version to install on Linux systems. For MacOS,
substitute "linux" with "darwin". For advanced options, please see
Section :numref:`sec:install` of the manual.


Parallel "Hello World" with Charm++
-----------------------------------

The .ci file
''''''''''''

.. code-block:: charmci
   mainmodule hello {
     mainchare Main {
       entry Main(CkArgMsg *m);
       entry void done();
     };
     array [1D] Hello {
       entry Hello();
       entry void SayHi();
     };            
   };


The .cpp file
'''''''''''''

.. code-block:: c++

   #include "hello.decl.h"

   /*readonly*/ CProxy_Main mainProxy;

   constexpr int nElements = 10;

   /*mainchare*/
   class Main : public CBase_Main
   {
   public:
     Main(CkArgMsg* m)
     {
       //Start the computation
       CkPrintf("Running Hello on %d processors with 10 elements\n", CkNumPes(), nElements);
       CProxy_Hello arr = CProxy_Hello::ckNew(nElements);
       mainProxy = thisProxy;
       arr[0].SayHi(0);
     };

     void done()
     {
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
       CkPrintf("PE %d says 'Hi from element %d'\n", CkMyPe(), thisIndex);
       if (thisIndex < nElements-1) {
         thisProxy[thisIndex+1].SayHi(); // Pass the hello on
       } else {
         mainProxy.done(); //We've been around once-- we're done.
       }
     }
   };

   #include "hello.def.h"


Compiling the example
''''''''''''''''''''

.. code-block:: bash

   $ charm/bin/charmc hello.ci
   $ charm/bin/charmc hello.cpp


Running the example
'''''''''''''''''''

.. code-block:: console

   $ ./charmrun +p2 ./hello
   Charm++: standalone mode (not using charmrun)
   Charm++> Running in non-SMP mode: 1 processes (PEs)
   Converse/Charm++ Commit ID: v6.9.0-172-gd31997cce
   Charm++> scheduler running in netpoll mode.
   CharmLB> Load balancer assumes all CPUs are same.
   Charm++> Running on 1 hosts (1 sockets x 4 cores x 2 PUs = 8-way SMP)
   Charm++> cpu topology info is gathered in 0.000 seconds.
   Running Hello on 1 processors with 10 elements
   PE 0 says Hi from element 0
   PE 0 says Hi from element 1
   PE 0 says Hi from element 2
   PE 0 says Hi from element 3
   PE 0 says Hi from element 4
   PE 1 says Hi from element 5
   PE 1 says Hi from element 6
   PE 1 says Hi from element 7
   PE 1 says Hi from element 8
   PE 1 says Hi from element 9
   All done
   [Partition 0][Node 0] End of program
