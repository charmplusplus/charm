Experimental Features
=====================

.. _sec:controlpoint:

Control Point Automatic Tuning
------------------------------

Charm++ currently includes an experimental automatic tuning framework
that can dynamically adapt a program at runtime to improve its
performance. The program provides a set of tunable knobs that are
adjusted automatically by the tuning framework. The user program also
provides information about the control points so that intelligent tuning
choices can be made. This information will be used to steer the program
instead of requiring the tuning framework to blindly search the possible
program configurations.

**Warning: this is still an experimental feature not meant for
production applications**

Exposing Control Points in a Charm++ Program
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The program should include a header file before any of its ``*.decl.h``
files:

.. code-block:: c++

       #include <controlPoints.h>

The control point framework initializes itself, so no changes need to be
made at startup in the program.

The program will request the values for each control point on PE 0.
Control point values are non-negative integers:

.. code-block:: c++

       my_var = controlPoint("any_name", 5, 10);
       my_var2 = controlPoint("another_name", 100,101);

To specify information about the effects of each control point, make
calls such as these once on PE 0 before accessing any control point
values:

.. code-block:: c++

       ControlPoint::EffectDecrease::Granularity("num_chare_rows");
       ControlPoint::EffectDecrease::Granularity("num_chare_cols");
       ControlPoint::EffectIncrease::NumMessages("num_chare_rows");
       ControlPoint::EffectIncrease::NumMessages("num_chare_cols");
       ControlPoint::EffectDecrease::MessageSizes("num_chare_rows");
       ControlPoint::EffectDecrease::MessageSizes("num_chare_cols");
       ControlPoint::EffectIncrease::Concurrency("num_chare_rows");
       ControlPoint::EffectIncrease::Concurrency("num_chare_cols");
       ControlPoint::EffectIncrease::NumComputeObjects("num_chare_rows");
       ControlPoint::EffectIncrease::NumComputeObjects("num_chare_cols");

For a complete list of these functions, see ``cp_effects.h`` in
``charm/include``.

The program, of course, has to adapt its behavior to use these new
control point values. There are two ways for the control point values
to change over time. The program can request that a new phase (with its
own control point values) be used whenever it wants, or the control
point framework can automatically advance to a new phase periodically.
The structure of the program will be slightly different in these to
cases. Sections :numref:`frameworkAdvancesPhases` and
:numref:`programAdvancesPhases` describe the additional changes to
the program that should be made for each case.

.. _frameworkAdvancesPhases:

Control Point Framework Advances Phases
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The program provides a callback to the control point framework in a
manner such as this:

.. code-block:: c++

       // Once early on in program, create a callback, and register it
       CkCallback cb(CkIndex_Main::granularityChange(NULL),thisProxy);
       registerCPChangeCallback(cb, true);

In the callback or after the callback has executed, the program should
request the new control point values on PE 0, and adapt its behavior
appropriately.

Alternatively, the program can specify that it wants to call
``gotoNextPhase();`` itself when it is ready. Perhaps the program wishes
to delay its adaptation for a while. To do this, it specifies ``false``
as the final parameter to ``registerCPChangeCallback`` as follows:

.. code-block:: c++

      registerCPChangeCallback(cb, false);

.. _programAdvancesPhases:

Program Advances Phases
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: c++

        registerControlPointTiming(duration); // called after each program iteration on PE 0
        gotoNextPhase(); // called after some number of iterations on PE 0
       // Then request new control point values

Linking With The Control Point Framework
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The control point tuning framework is now an integral part of the
Charm++ runtime system. It does not need to be linked in to an
application in any special way. It contains the framework code
responsible for recording information about the running program as well
as adjust the control point values. The trace module will enable
measurements to be gathered including information about utilization,
idle time, and memory usage.

Runtime Command Line Arguments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Various following command line arguments will affect the behavior of the
program when running with the control point framework. As this is an
experimental framework, these are subject to change.

The scheme used for tuning can be selected at runtime by the use of one
of the following options:

.. code-block:: none

        +CPSchemeRandom            Randomly Select Control Point Values
    +CPExhaustiveSearch            Exhaustive Search of Control Point Values
         +CPSimulAnneal            Simulated Annealing Search of Control Point Values
    +CPCriticalPathPrio            Use Critical Path to adapt Control Point Values
           +CPBestKnown            Use BestKnown Timing for Control Point Values
            +CPSteering            Use Steering to adjust Control Point Values
         +CPMemoryAware            Adjust control points to approach available memory

To intelligently tune or steer an application’s performance, performance
measurements ought to be used. Some of the schemes above require that
memory footprint statistics and utilization statistics be gathered. All
measurements are performed by a tracing module that has some overheads,
so is not enabled by default. To use any type of measurement based
steering scheme, it is necessary to add a runtime command line argument
to the user program to enable the tracing module:

.. code-block:: none

       +CPEnableMeasurements

The following flags enable the gathering of the different types of
available measurements.

.. code-block:: none

           +CPGatherAll            Gather all types of measurements for each phase
   +CPGatherMemoryUsage            Gather memory usage after each phase
   +CPGatherUtilization            Gather utilization & Idle time after each phase

The control point framework will periodically adapt the control point
values. The following command line flag determines the frequency at
which the control point framework will attempt to adjust things.

.. code-block:: none

        +CPSamplePeriod     number The time between Control Point Framework samples (in seconds)

The data from one run of the program can be saved and used in subsequent
runs. The following command line arguments specify that a file named
``controlPointData.txt`` should be created or loaded. This file contains
measurements for each phase as well as the control point values used in
each phase.

.. code-block:: none

            +CPSaveData            Save Control Point timings & configurations at completion
            +CPLoadData            Load Control Point timings & configurations at startup
        +CPDataFilename            Specify the data filename

It might be useful for example, to run once with
``+CPSimulAnneal +CPSaveData`` to try to find a good configuration for
the program, and then use ``+CPBestKnown +CPLoadData`` for all
subsequent production runs.

.. _sec:shrinkexpand:

Malleability: Shrink/Expand Number of Processors
------------------------------------------------

This feature enables a Charm++ application to dynamically shrink and
expand the number of processors that it is running on during the
execution. It internally uses three other features of Charm++: CCS
(Converse Client Server) interface, load balancing, and checkpoint
restart.

-  The CCS interface is used to send and receive the shrink and expand
   commands. These commands can be internal (i.e. application decides to
   shrink or expand itself) or external via a client. The runtime
   listens for the commands on a port specified at launch time.

-  The load balancing framework is used to evacuate the tasks before a
   shrink operation and distribute the load equally over the new set of
   processors after an expand operation.

-  The in-memory checkpoint restart mechanism is used to restart the
   application with the new processor count quickly and without leaving
   residual processes behind.

An example program with a CCS client to send shrink/expand commands can
be found in ``examples/charm++/shrink_expand`` in the charm
distribution.

To enable shrink expand, Charm++ needs to be built with the
``--enable-shrinkexpand`` option:

.. code-block:: bash

    $ ./build charm++ netlrts-linux-x86_64 --enable-shrinkexpand

An example application launch command needs to include a load balancer,
a nodelist file that contains all of the nodes that are going to be
used, and a port number to listen the shrink/expand commands:

.. code-block:: bash

    $ ./charmrun +p4 ./jacobi2d 200 20 +balancer GreedyLB ++nodelist ./mynodelistfile ++server ++server-port 1234

The CCS client to send shrink/expand commands needs to specify the
hostname, port number, the old(current) number of processor and the
new(future) number of processors:

.. code-block:: bash

    $ ./client <hostname> <port> <oldprocs> <newprocs>
    (./client valor 1234 4 8 //This will increase from 4 to 8 processors.)

To make a Charm++ application malleable, first, pup routines for all of
the constructs in the application need to be written. This includes
writing a pup routine for the ``mainchare`` and marking it migratable:

.. code-block:: c++

    mainchare [migratable]  Main { ... }

Second, the ``AtSync()`` and ``ResumeFromSync()`` functions need to be
implemented in the usual way of doing load balancing (See
Section :numref:`lbStrategy` for more info on load balancing).
Shrink/expand will happen at the next load balancing step after the
receipt of the shrink/expand command.

**NOTE:** If you want to shrink your application, for example, from two
physical nodes to one node where each node has eight cores, then you
should have eight entries in the nodelist file for each node, one per
processor. Otherwise, the application will shrink in a way that will use
four cores from each node, whereas what you likely want is to use eight
cores on only one of the physical nodes after shrinking. For example,
instead of having a nodelist like this:

.. code-block:: none

    host a
    host b

the nodelist should be like this:

.. code-block:: none

    host a
    host a
    host a
    host a
    host a
    host a
    host a
    host a
    host b
    host b
    host b
    host b
    host b
    host b
    host b
    host b

**Warning: this is an experimental feature and not supported in all
charm builds and applications.** Currently, it is tested on
``netlrts-{linux/darwin}-x86_64`` builds. Support for other Charm++
builds and AMPI applications are under development. It is only tested
with ``RefineLB`` and ``GreedyLB`` load balancing strategies; use other
strategies with caution.
