=============================
ARMCI Interface under Charm++
=============================

.. contents::
   :depth: 3

Introduction
============

This manual describes the basic features and API of the Aggregate Remote
Memory Copy Interface (ARMCI) library implemented under Charm++. It is
meant for developers using ARMCI who desire the performance features of
the Charm++ run-time system (e.g. dynamic load balancing, fault
tolerance and scalability) applied transparently to their libraries
written using the ARMCI API.

ARMCI is a library that supports remote memory copy functionality. It
focuses on non-contiguous data transfers and is meant to be used by
other libraries as opposed to application development. Libraries that
the original ARMCI developers have targeted include Global Arrays,
P++/Overture and the Adlib PCRC run-time system.

ARMCI remote copy operations are one-sided and complete, regardless of
the actions taken by the remote process. For performance reasons,
polling can be helpful but should not be necessary to ensure progress.
The operations are ordered when referencing the same remote process.
Operations issued to different processes can complete in an arbitrary
order. Both blocking and non-blocking APIs are supported.

ARMCI supports three classes of operations: data transfer using *put*,
*get* and *accumulate* operations; synchronization with local and global
*fence* operations and atomic read-modify-write; and utility functions
for memory management and error handling. *Accumulate* and atomic
read-modify-write operations are currently not implemented for the
charmpp port.

A *get* operation transfers data from the remote process memory (source)
to the calling processing local memory (destination). A *put* operation
transfers data from the local memory of the calling process (source) to
the memory of the remote process (destination).

This manual will include several useful Charm++-specific extensions to
the ARMCI API. It will also list the functions that have not yet been
implemented but exists in the original ARMCI implementation. Readers of
this manual are advised to refer to the original ARMCI documentation
(See Section :numref:`sec::related doc`) for more complete information
and motivation for the development of this library.

.. _sec::charm build:

Building ARMCI Support under The Charm++ Runtime System
-------------------------------------------------------

Build charm target ARMCI (instead of charm or AMPI):

::

   > cd charm
   > ./build ARMCI netlrts-linux -O3

.. _sec::simple program:

Writing a Simple ARMCI Program
------------------------------

The following simple example has two processes place their own string
into the global array and then acquire the appropriate string from the
other’s global address space in order to print “hello world”.

The main function has to be compliant to ANSI C:

::

   #include <stdio.h>
   #include <stdlib.h>
   #include <string.h>

   #include <armci.h>

   #define MAX_PROCESSORS 2

   int main(int argc, char * argv[]) {
     void *baseAddress[MAX_PROCESSORS];
     char *myBuffer;
     int thisImage;

     // initialize
     ARMCI_Init();
     ARMCI_Myid(&thisImage);

     // allocate data (collective operation)
     ARMCI_Malloc(baseAddress, strlen("hello")+1);

     if (thisImage == 0) {
       sprintf((char *)baseAddress[0], "%s", "hello");
     } else if (thisImage == 1) {
       sprintf((char *)baseAddress[1], "%s", "world");
     }

     // allocate space for local buffer
     myBuffer = (char *)AMRCI_Malloc_local(strlen("hello")+1);

     ARMCI_Barrier();

     if (thisImage == 0) {
       ARMCI_Get(baseAddress[1], myBuffer, strlen("hello")+1, 1);
       printf("[%d] %s %s\n",thisImage, baseAddress[0], myBuffer);
     } else if (thisImage == 1) {
       ARMCI_Get(baseAddress[0], myBuffer, strlen("hello")+1, 0);
       printf("[%d] %s %s\n",thisImage, myBuffer, baseAddress[1]);
     }

     // finalize
     ARMCI_Finalize();
     return 0;
   }

.. _sec::armci build:

Building an ARMCI Binary and Execution
--------------------------------------

Compiling the code with:

.. code-block:: bash

   > charm/bin/charmc -c hello.c /$(OPTS)

Linking the program with:

.. code-block:: bash

   > charm/bin/charmc hello.o -o hello -swapglobals -memory isomalloc -language armci $(OPTS)

Run the program:

.. code-block:: bash

   > ./charmrun ./hello +p2 +vp8

.. _sec::data structures:

ARMCI Data Structures
=====================

ARMCI provides two formats to describe non-contiguous layouts of data in
memory.

The *generalized I/O vector* is the most general format intended for
multiple sets of equally sized data segments to be moved between
arbitrary local and remote memory locations. It uses two arrays of
pointers: one for source and one for destination addresses. The length
of each array is equal to the number of segments.

::

   typedef struct {
     void *src_ptr_ar;
     void *dst_ptr_ar;
     int bytes;
     int ptr_ar_len;
   } armci_giov_t;

Currently, there is no support for *generalized I/O vector* operations
in the charmpp implementation.

The *strided* format is an optimization of the generalized I/O vector
format. It is intended to minimize storage required to describe sections
of dense multi-dimensional arrays. Instead of including addresses for
all the segments, it specifies only an address of the first segment in
the set for source and destination. The addresses of the other segments
can be computed using the stride information.


Application Programmer’s Interface
==================================

The following is a list of functions supported on the Charm++ port of
ARMCI. The integer value returned by most ARMCI operations represents
the error code. The zero value is successful, other values represent
failure (See Section :numref:`sec::error codes` for details).

Startup, Cleanup and Status Functions
-------------------------------------

::

   int ARMCI_Init(void);

Initializes the ARMCI library. This function must be called before any
ARMCI functions may be used.

::

   int ARMCI_Finalize(void);

Shuts down the ARMCI library. No ARMCI functions may be called after
this call is made. It must be used before terminating the program
normally.

::

   void ARMCI_Cleanup(void);

Releases system resources that the ARMCI library might be holding. This
is intended to be used before terminating the program in case of error.

::

   void ARMCI_Error(char *msg, int code);

Combines the functionality of ARMCI_Cleanup and Charm++’s CkAbort call.
Prints to *stdout* and *stderr* ``msg`` followed by an integer ``code``.

::

   int ARMCI_Procs(int *procs);

The number of processes is stored in the address ``procs``.

::

   int ARMCI_Myid(int *myid);

The id of the process making this call is stored in the address
``myid``.

ARMCI Memory Allocation
-----------------------

::

   int ARMCI_Malloc(void* ptr_arr[], int bytes);

Collective operation to allocate memory that can be used in the context
of ARMCI copy operations. Memory of size ``bytes`` is allocated on each
process. The pointer address of each process’ allocated memory is stored
at ``ptr_arr[]`` indexed by the process’ id (see ``ARMCI_Myid``). Each
process gets a copy of ``ptr_arr``.

::

   int ARMCI_Free(void *ptr);

Collective operation to free memory which was allocated by
``ARMCI_Malloc``.

::

   void *ARMCI_Malloc_local(int bytes);

Local memory of size ``bytes`` allocated. Essentially a wrapper for
``malloc``.

::

   int ARMCI_Free_local(void *ptr);

Local memory address pointed to by ``ptr`` is freed. Essentially a
wrapper for ``free``.

Put and Get Communication
-------------------------

::

   int ARMCI_Put(void *src, void *dst, int bytes, int proc);

Transfer contiguous data of size ``bytes`` from the local process memory
(source) pointed to by ``src`` into the remote memory of process id
``proc`` pointed to by ``dst`` (remote memory pointer at destination).

::

   int ARMCI_NbPut(void *src, void* dst, int bytes, int proc,
                   armci_hdl_t *handle);

The non-blocking version of ``ARMCI_Put``. Passing a ``NULL`` value to
``handle`` makes this function perform an implicit handle non-blocking
transfer.

::

   int ARMCI_PutS(void *src_ptr, int src_stride_ar[],
                  void *dst_ptr, int dst_stride_ar[],
                  int count[], int stride_levels, int proc);

Transfer strided data from the local process memory (source) into remote
memory of process id ``proc``. ``src_ptr`` points to the first memory
segment in local process memory. ``dst_ptr`` is a remote memory address
that points to the first memory segment in the memory of process
``proc``. ``stride_levels`` represents the number of additional
dimensions of striding beyond 1. ``src_stride_ar`` is an array of size
``stride_levels`` whose values indicate the number of bytes to skip on
the local process memory layout. ``dst_stride_ar`` is an array of size
``stride_levels`` whose values indicate the number of bytes to skip on
process ``proc``\ ’s memory layout. ``count`` is an array of size
``stride_levels + 1`` whose values indicate the number of bytes to copy.

As an example, assume two 2-dimensional C arrays residing on different
processes.

::

             double A[10][20]; /* local process */
             double B[20][30]; /* remote process */

To put a block of data of 3x6 doubles starting at location (1,2) in
``A`` into location (3,4) in ``B``, the arguments to ``ARMCI_PutS`` will
be as follows (assuming C/C++ memory layout):

::

             src_ptr = &A[0][0] + (1 * 20 + 2); /* location (1,2) */
             src_stride_ar[0] = 20 * sizeof(double);
             dst_ptr = &B[0][0] + (3 * 30 + 4); /* location (3,4) */
             dst_stride_ar[0] = 30 * sizeof(double);
             count[0] = 6 * sizeof(double); /* contiguous data */
             count[1] = 3; /* number of rows of contiguous data */
             stride_levels = 1;
             proc = /*<B's id> */;

::

   int ARMCI_NbPutS(void *src_ptr, int src_stride_ar[],
                    void *dst_ptr, int dst_stride_ar[],
                    int count[], int stride_levels, int proc
                    armci_hdl_t *handle);

The non-blocking version of ``ARMCI_PutS``. Passing a ``NULL`` value to
``handle`` makes this function perform an implicit handle non-blocking
transfer.

::

   int ARMCI_Get(void *src, void *dst, int bytes, int proc);

Transfer contiguous data of size ``bytes`` from the remote process
memory at process ``proc`` (source) pointed to by ``src`` into the local
memory of the calling process pointed to by ``dst``.

::

   int ARMCI_NbGet(void *src, void *dst, int bytes, int proc,
                   armci_hdl_t *handle);

The non-blocking version of ``ARMCI_Get``. Passing a ``NULL`` value to
``handle`` makes this function perform an implicit handle non-blocking
transfer.

::

   int ARMCI_GetS(void *src_ptr, int src_stride_ar[],
                  void* dst_ptr, int dst_stride_ar[],
                  int count[], int stride_levels, int proc);

Transfer strided data segments from remote process memory on process
``proc`` to the local memory of the calling process. The semantics of
the parameters to this function are the same as that for ``ARMCI_PutS``.

::

   int ARMCI_NbGetS(void *src_ptr, int src_stride_ar[],
                    void* dst_ptr, int dst_stride_ar[],
                    int count[], int stride_levels, int proc,
                    armci_hdl_t *handle);

The non-blocking version of ``ARMCI_GetS``. Passing a ``NULL`` value to
``handle`` makes this function perform an implicit handle non-blocking
transfer.

Explicit Synchronization
------------------------

::

   int ARMCI_Wait(armci_hdl_t *handle);
   int ARMCI_WaitProc(int proc);
   int ARMCI_WaitAll();
   int ARMCI_Test(armci_hdl_t *handle);
   int ARMCI_Barrier();

::

   int ARMCI_Fence(int proc);

Blocks the calling process until all *put* or *accumulate* operations
the process issued to the remote process ``proc`` are completed at the
destination.

::

   int ARMCI_AllFence(void);

Blocks the calling process until all outstanding *put* or *accumulate*
operations it issued are completed on all remote destinations.

.. _sec::extensions:

Extensions to the Standard API
------------------------------

::

   void ARMCI_Migrate(void);
   void ARMCI_Async_Migrate(void);
   void ARMCI_Checkpoint(char* dirname);
   void ARMCI_MemCheckpoint(void);

   int armci_notify(int proc);
   int armci_notify_wait(int proc, int *pval);

List of Unimplemented Functions
===============================

The following functions are supported on the standard ARMCI
implementation but not yet supported in the Charm++ port.

::

   int ARMCI_GetV(...);
   int ARMCI_NbGetV(...);
   int ARMCI_PutV(...);
   int ARMCI_NbPutV(...);
   int ARMCI_AccV(...);
   int ARMCI_NbAccV(...);

   int ARMCI_Acc(...);
   int ARMCI_NbAcc(...);
   int ARMCI_AccS(...);
   int ARMCI_NbAccS(...);

   int ARMCI_PutValueLong(long src, void* dst, int proc);
   int ARMCI_PutValueInt(int src, void* dst, int proc);
   int ARMCI_PutValueFloat(float src, void* dst, int proc);
   int ARMCI_PutValueDouble(double src, void* dst, int proc);
   int ARMCI_NbPutValueLong(long src, void* dst, int proc, armci_hdl_t* handle);
   int ARMCI_NbPutValueInt(int src, void* dst, int proc, armci_hdl_t* handle);
   int ARMCI_NbPutValueFloat(float src, void* dst, int proc, armci_hdl_t* handle);
   int ARMCI_NbPutValueDouble(double src, void* dst, int proc, armci_hdl_t* handle);
   long ARMCI_GetValueLong(void *src, int proc);
   int ARMCI_GetValueInt(void *src, int proc);
   float ARMCI_GetValueFloat(void *src, int proc);
   double ARMCI_GetValueDouble(void *src, int proc);

   void ARMCI_SET_AGGREGATE_HANDLE (armci_hdl_t* handle);
   void ARMCI_UNSET_AGGREGATE_HANDLE (armci_hdl_t* handle);

   int ARMCI_Rmw(int op, int *ploc, int *prem, int extra, int proc);
   int ARMCI_Create_mutexes(int num);
   int ARMCI_Destroy_mutexes(void);
   void ARMCI_Lock(int mutex, int proc);
   void ARMCI_Unlock(int mutex, int proc);

.. _sec::error codes:

Error Codes
===========

As of this writing, attempts to locate the documented error codes have
failed because the release notes have not been found. Attempts are being
made to derive these from the ARMCI source directly. Currently Charm++
implementation does not implement any error codes.

.. _sec::related doc:

Related Manuals and Documents
=============================

ARMCI website: http://www.emsl.pnl.gov/docs/parsoft/armci/index.html
