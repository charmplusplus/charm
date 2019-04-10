=======================================
The Charm++ Parallel Programming System
=======================================

.. contents::
   :depth: 3



.. [1]
   “Threaded” or “synchronous” methods are different. But even they do
   not lead to pre-emption; only to cooperative multi-threading

.. [2]
   For a description of the underlying design philosophy please refer
   to the following papers:

   - L. V. Kale and Sanjeev Krishnan, *“Charm++: Parallel Programming
     with Message-Driven Objects”*, in “Parallel Programming Using C++”,
     MIT Press, 1995.
   - L. V. Kale and Sanjeev Krishnan, *“Charm++: A Portable Concurrent
     Object Oriented System Based On C++”*, Proceedings of the Conference
     on Object Oriented Programming, Systems, Languages and Applications
     (OOPSLA), September 1993.

.. [3]
   Except when it is part of an array.

.. [4]
   Threaded methods augment this behavior since they execute in a
   separate user-level thread, and thus can block to wait for data.

.. [5]
   However, the element must eventually be created.

.. [6]
   AtSync() is a member function of class ArrayElement

.. [7]
   Currently not all load balancers recognize this setting though.

.. [8]
   Originally called *Branch Office Chare* or *Branched Chare*

.. [9]
   Older, deprecated syntax allows groups to inherit directly from the
   system-defined class Group

.. [10]
   As with groups, older syntax allows node groups to inherit from
   NodeGroup instead of a specific, generated “CBase\_” class.

.. [11]
   At present, this optimizes broadcasts to not save old messages for
   immigrating chares.

.. [12]
   This can enable a slightly faster default mapping scheme.

.. [13]
   See chapter :numref:`delegation` for general information on message
   delegation in Charm++.

.. [14]
   http://www.ks.uiuc.edu/Research/namd
