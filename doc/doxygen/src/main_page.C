/* This is the master Doxygen documentation links page--
   It contains no actual executable source code. */
/**
\mainpage Charm Source Code Documentation
<!-- This HTML is generated from charm/doc/doxygen/main_page.C -->

\section layers Major Runtime Layers:

<table border=2><tr><td>
<p>Libraries:
<p>Communication: Barrier, commlib, multicast, 
sparseReducer, sparseContiguousReducer

<P>Languages: ARMCI, AMPI, TCharm, taskGraph, search

<p>Frameworks: <a href="group__ParFUM.html">ParFUM</a>, IFEM, IDXL, Collide, MBlock, AMR

<p>Visualization: liveViz, liveViz3d, netFEM
 
</td></tr><tr><td>
Charm++:
<a href="group__CkArray.html">Arrays</a>,
<a href="group__CkLdb.html">Load balancer</a>,
<a href="group__CkPerf.html">Tracing</a>,
<a href="group__CkReduction.html">Reductions</a>,
<a href="group__CkArrayImpl.html">Array Implementation</a>

</td></tr><tr><td>
Charm Kernel:
<a href="group__Ck.html">Groups and Chares</a>,  
<a href="group__CkRegister.html">Registration</a>, 
<a href="group__CkQD.html">Quiescence detection</a>, 
<a href="group__CkFutures.html">Thread support</a>,
<a href="group__CkInit.html">Startup</a>,
<a href="group__CkEnvelope.html">Message Envelope</a>,
Translator, SDAG.

</td></tr><tr><td>
Converse:
<a href="group__Converse.html">core</a>,
scheduler, threads, memory allocation,
conditions, random numbers, converse client/server (CCS).

Converse tracing, parameter marshalling (CPM),
quiescence.

</td></tr><tr><td>
<a href="group__Machine.html">Converse Machine Layer</a>:
<ul>
<li><a href="group__NET.html">net</a>
  <ul>
  <li>smp
  <li>gm
  <li>tcp
  <li>udp
  </ul>
<li>mpi
<li>elan
<li>vmi
<li>shmem
<li>sp3
<li>t3e
<li>uth (user-level threads)
<li>ncube2
<li>origin
</ul>

</td></tr><tr><td>
Converse Configuration Layer:
build system, charmc, configure script, conv-mach.h/.sh.

</td></tr></table>

\section utilities Utility Code
PUP, CkBitVector, CkDll, CkHashtable, 
CkImage, CkVec, CkQ, CkStatistics, CkVector3d,
sockets.

*/
