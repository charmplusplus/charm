===========================================
CkDyn: Dynamic Chares for Language Bindings
===========================================

.. contents::
   :depth: 3

This manual is for authors of language bindings: a Python, Julia or Rust
package, or a C++ program, that defines Charm++ chare classes at run time
instead of in a ``.ci`` file. It describes the C API in
``src/ck-core/ckdyn.h`` from the binding's side: what to call, in which
order, and what the runtime expects of the binding's code. How the runtime
implements each feature is in the internals chapter,
``doc/internals/ckdyn.md``; this manual points to it where the mechanism
matters to a binding author.

.. _ckdyn-intro:

Introduction
============

What CkDyn is
-------------

CkDyn lets a program register chare classes at run time. A class is a
descriptor: a name, a kind (main chare, chare, group, nodegroup, array or
array map) and a table of entry methods, each with a name and attributes.
The runtime reaches the objects of these classes through one table of C
function pointers that the binding supplies, the *hook table*. Every entry
method of every dynamic class receives the same message type, ``CkDynMsg``:
a header followed by typed *slots* (integers, floats, byte strings, typed
n-dimensional buffers, futures, callbacks, section ids, proxies) and an
opaque *tail* that the binding fills with its own serialized data.

The objects behind a dynamic class are ordinary Charm++ objects: six fixed
C++ classes stand in for every dynamic class of their kind, and each holds
an opaque *handle* that the binding owns. Everything the runtime does for a
charmxi class it therefore does for a dynamic class:

- point-to-point sends, broadcasts, inline and expedited sends, message
  priorities;
- groups (also constrained to a PE list), nodegroups (with exclusive
  entries), arrays of one to six index dimensions, array maps, dynamic
  insertion;
- reductions with the built-in reducers and with reducers the binding
  defines;
- array and group sections over CkMulticast, with multicasts and section
  reductions;
- migration, load balancing (AtSync and anytime migration), checkpoint to
  disk and restart;
- quiescence detection;
- zero-copy transfer of host buffers and, in a CUDA or HIP build, device
  buffers;
- tracing: Projections shows the binding's class and method names and the
  index of each array element.

Who it is for
-------------

CkDyn is for code that cannot run charmxi: an interpreted or JIT-compiled
language whose classes exist only at run time, or a C++ program whose set of
chare classes is decided by its input. The API is C: plain structs with
fixed-width fields, function pointers and ``extern "C"`` functions.
``ckdyn.h`` includes no Charm++ header, so a binding builds without the
Charm++ include tree, and a language with a C foreign-function interface
(Julia ``ccall``, Rust ``extern "C"``, Python with any extension tool) can
call it directly.

What a binding provides and what it gets
----------------------------------------

A binding provides:

1. A **hook table** (``CkDynHooks``): functions that construct, destroy,
   deliver to, pack and unpack the binding's objects, plus hooks for
   futures, reductions with binding-defined reducers, array maps, load
   balancing resumption, section membership, readonly data and errors
   (:ref:`ckdyn-hooks`).
2. A **class descriptor** per class (``CkDynClassDesc``) with an entry
   table (``CkDynEntryDesc``), registered with ``CkDynRegisterClass``
   (:ref:`ckdyn-registration`).
3. A **message convention**: how its values map onto slots and the tail. CkDyn
   defines the slot types; which slots an entry method expects is up to the
   binding (:ref:`ckdyn-messages`).

It gets, per class, a Charm++ chare index and a contiguous range of entry
indices, and from then on the functions of ``ckdyn.h``: create objects,
send, broadcast, contribute, create sections, set futures, migrate,
checkpoint, trace.

The two existing bindings
-------------------------

- **The C++ test binding**, ``tests/charm++/ckdyn/cbinding.h`` and
  ``cbinding.C`` (about 450 lines). A class is a table of C++ function
  pointers; an object is a small struct. The acceptance test
  ``tests/charm++/ckdyn/ckdyntest.C`` uses it to exercise every feature
  without Python. It is the reference for this manual's examples
  (:ref:`ckdyn-walkthrough`).
- **charm4py 2**, the Python binding: the extension module
  ``src/ckdyn_py.cpp`` of the charm4py repository (branch ``thin``), built
  with nanobind. It shows the patterns a binding for an interpreted language
  needs: taking the interpreter lock in every hook, running entry methods on
  coroutines (greenlets), zero-copy numpy views over retained messages, and
  its own encodings inside the slot types.

Build requirements
------------------

CkDyn is compiled into ``libck`` on every Charm++ target built on
Reconverse; nothing in it runs until a program calls ``CkDynInit`` or
``CkDynRegisterClass``. It is not available on the frozen classic Converse
line.

Two build shapes are in use:

- **A C++ program** links CkDyn from ``libck`` and starts like any Charm++
  program. Any ``charm++`` build works, for example::

     ./build charm++ reconverse-linux-x86_64 --with-production -j8

  The program includes ``ckdyn.h`` and is linked with ``charmc``; it has a
  ``.ci`` file only for its own main module (the C++ test binding has one,
  with an ``initnode`` that registers the dynamic classes).
- **A shared-library binding** (Python, Julia, Rust) loads ``libcharm`` as a
  shared library and starts the runtime itself with ``CkDynStart``. The
  ``charm4py`` build target produces that library::

     ./build charm4py reconverse-linux-x86_64 -j8 --with-production \
         --disable-fortran --with-refnum-type="unsigned int"

Both shapes support every feature, including registration after start-up
(:ref:`ckdyn-late`).

**Reference number width.** Futures travel in the envelope's reference
number, whose type is ``CMK_REFNUM_TYPE``. The default build has a 16-bit
reference number, which limits future ids to 1 to 65534. A binding that
needs more builds Charm++ with ``--with-refnum-type="unsigned int"``, which
allows ids from 1 to 2\ :sup:`32`-2, and checks ``CkDynRefnumBytes()`` at
start-up (charm4py refuses to start unless it returns 4).

**Device support.** ``DEVICE`` slots, device posts and
``CkDynHapiAddCallback`` need a build with CUDA or HIP;
``CkDynHasDeviceSupport()`` reports it, and the device functions abort in
other builds.

API version and ABI rules
-------------------------

``CKDYN_API_VERSION`` (currently 2) is bumped on any incompatible change to
``ckdyn.h``. Three checks keep a binding and a ``libcharm`` consistent:

- ``CkDynApiVersion()`` returns the version the library was built with; a
  binding compares it with the ``CKDYN_API_VERSION`` it was compiled
  against before calling anything else.
- The hook table starts with ``version`` and ``size`` fields. The binding
  sets ``version = CKDYN_API_VERSION`` and ``size = sizeof(CkDynHooks)`` as
  it compiled it. ``CkDynInit`` aborts if either differs from the library's,
  so a binding compiled against a header with a different table is rejected
  even when the version number matches. A class's own hook table
  (``CkDynClassDesc.hooks``) is checked the same way at registration.
- The structs that CkDyn copies as bytes into messages have fixed-width
  integer fields only, and the library fixes their sizes at compile time:
  ``CkDynTarget`` 40 bytes, ``CkDynSectionID`` 16, ``CkDynCallback`` 88,
  ``CkDynHdr`` 24, ``CkDynSlot`` 24. The enums only supply constants; a
  binding in another language declares these structs with the same field
  widths and order and checks the sizes.

.. _ckdyn-concepts:

Concepts
========

Objects and handles
-------------------

An object of a dynamic class is a Charm++ object of one of six fixed C++
classes, one per kind, that holds the binding's *handle*: a ``void *``
returned by the ``construct``, ``mainchare`` or ``unpack`` hook. CkDyn
stores the handle and passes it back to every hook that concerns the object;
it never dereferences it. A handle must be non-NULL, because CkDyn uses
NULL to mark a group branch that was not constructed (a branch outside a
constrained group's PE list).

The binding decides what a handle points to: a C struct, a C++ object, a
reference to an interpreter object (charm4py passes a ``PyObject *`` that
holds a reference). The object's lifetime ends with the ``destruct`` hook,
which CkDyn calls when the object is deleted, when it migrates away from
the PE, and at exit.

Targets
-------

A ``CkDynTarget`` names one object of any kind. It is a plain 40-byte
struct that a binding can store, copy, compare and put into messages:

.. code-block:: c

   typedef struct CkDynTarget {
     int32_t kind;          /* CkDynKind */
     int32_t chare_idx;     /* the Charm++ chare index of its class */
     union {
       struct { int32_t pe; void *ptr; } chare;            /* MAINCHARE, CHARE */
       struct { int32_t gid; int32_t pe_or_node; } group;  /* GROUP, NODEGROUP, ARRAYMAP */
       struct { int32_t aid; int32_t ndims; int32_t index[CKDYN_MAX_DIMS]; } array;
     } u;
   } CkDynTarget;

- ``CKDYN_MAINCHARE``, ``CKDYN_CHARE``: the ``CkChareID`` (PE and object
  pointer on that PE). ``CkDynCreateChare`` returns a *virtual* id with a
  negative PE that is usable for sends and callbacks at once; the
  ``construct`` hook receives the real id. Both name the same chare.
- ``CKDYN_GROUP``, ``CKDYN_ARRAYMAP``: the group id and a PE;
  ``CKDYN_NODEGROUP``: the group id and a node.
- ``CKDYN_ARRAY``: the array id, the dimension count (1 to 6) and the index.
  Indices of four to six dimensions are stored as 16-bit values by Charm++,
  so each component must fit in 16 bits.

Every hook that creates an object (``construct``, ``mainchare``,
``unpack``) receives the object's target; a binding keeps it with the
object, because most CkDyn functions that act on a local object
(``CkDynContribute``, ``CkDynAtSync``, ``CkDynCoroResume``, ...) take the
target, not the handle. Fill unused fields with zeros (``memset``) before
setting the fields of a kind, so that targets compare equal byte for byte.

Class descriptors and entry descriptors
---------------------------------------

.. code-block:: c

   typedef struct CkDynEntryDesc {
     const char *name;     /* "Class.method"; kept by pointer for the program's lifetime */
     uint32_t    flags;    /* CKDYN_EP_* */
     uint32_t    user_id;  /* opaque to CkDyn; passed back on every delivery */
   } CkDynEntryDesc;

   typedef struct CkDynClassDesc {
     const char            *name;         /* "Class"; kept by pointer */
     CkDynKind              kind;
     int                    ndims;        /* arrays: 1..6, or 0 for any */
     int                    num_entries;  /* entries[0] is the constructor */
     const CkDynEntryDesc  *entries;
     void                  *binding_data; /* opaque; passed to every per-class hook */
     const struct CkDynHooks *hooks;      /* NULL: the table given to CkDynInit */
   } CkDynClassDesc;

``entries[0]`` is always the constructor. The names are the ones
Projections and error messages show; the ``"Class.method"`` form matches
what charmxi registers. ``user_id`` is any 32-bit value the binding wants
back on delivery: the C++ test binding uses the method's position in its
table, so ``deliver`` indexes the table with it.

``binding_data`` is the binding's per-class pointer (the C++ test binding
passes its ``CbClass *``); every per-class hook receives it as its first
argument. ``hooks`` selects a per-class hook table; NULL, the usual choice,
selects the table given to ``CkDynInit``. A per-class table is kept by
pointer and must outlive the program.

Entry attributes (``CkDynEntryDesc.flags``):

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - flag
     - meaning
   * - ``CKDYN_EP_THREADED``
     - the binding runs the method on a coroutine; CkDyn only passes it
       back in ``ep_flags``
   * - ``CKDYN_EP_INLINE``
     - a send to a local object runs the method at once, inside the send
   * - ``CKDYN_EP_EXPEDITED``
     - every send to the entry is expedited (bypasses the priority queue)
   * - ``CKDYN_EP_IMMEDIATE``
     - Charm++ immediate entry; nodegroups only
   * - ``CKDYN_EP_NOKEEP``
     - set by CkDyn on every entry; see :ref:`ckdyn-ownership`
   * - ``CKDYN_EP_NOTRACE``
     - the entry is not traced
   * - ``CKDYN_EP_APPWORK``
     - the entry is application work (``CK_EP_APPWORK``)
   * - ``CKDYN_EP_EXCLUSIVE``
     - nodegroups: one invocation at a time per branch
   * - ``CKDYN_EP_REDNTARGET``
     - documentation only: the entry may receive reduction results

.. _ckdyn-hooks:

The hook table
--------------

.. code-block:: c

   typedef struct CkDynHooks {
     uint32_t version;              /* CKDYN_API_VERSION */
     uint32_t size;                 /* sizeof(CkDynHooks) as compiled by the binding */
     void *global;                  /* passed to the hooks without a class */
     void (*register_classes)(void *global);
     void *(*construct)(void *binding_data, const CkDynTarget *t, const CkDynMsgView *ctor_msg);
     void  (*destruct)(void *binding_data, void *handle);
     void  (*deliver)(void *binding_data, void *handle, int ep, uint32_t user_id,
                      uint32_t ep_flags, const CkDynMsgView *msg);
     void  (*deliver_many)(void *binding_data, void **handles, int n, int ep, uint32_t user_id,
                           uint32_t ep_flags, const CkDynMsgView *msg);
     void  (*mainchare)(void *binding_data, void **handle_out, const CkDynTarget *t,
                        int argc, char **argv);
     size_t (*pack_size)(void *binding_data, void *handle, int reason);
     void   (*pack)(void *binding_data, void *handle, int reason, void *buf, size_t len);
     void  *(*unpack)(void *binding_data, const CkDynTarget *t, int reason,
                      const void *buf, size_t len);
     size_t (*pack_readonly_size)(void *global);
     void   (*pack_readonly)(void *global, void *buf, size_t len);
     void   (*unpack_readonly)(void *global, const void *buf, size_t len);
     void  (*reduce)(void *global, uint32_t reducer_id, const CkDynMsgView **contribs, int n,
                     CkDynMsgBuilder *result);
     void  (*future)(void *global, int pe, uint32_t id, const CkDynMsgView *value);
     void  (*section_joined)(void *binding_data, void *handle, const CkDynSectionID *sid);
     int   (*map_proc_num)(void *binding_data, void *handle, int aid, int ndims, const int *index);
     void  (*resume_from_sync)(void *binding_data, void *handle);
     void  (*error)(void *global, const char *what);
   } CkDynHooks;

Zero the table (``memset``) and set the hooks the binding uses. A hook that
is NULL is either optional or needed only when a situation calls for it; in
the second case CkDyn aborts with a message naming the hook when the
situation arises (for example a migration without ``pack_size``).

.. list-table::
   :header-rows: 1
   :widths: 20 45 35

   * - hook
     - called when
     - what it does
   * - ``register_classes``
     - at start-up, on rank 0 of each process, for a shared-library binding
       that called ``CkDynInit`` before ``CkDynStart``
     - registers the binding's classes (:ref:`ckdyn-registration`)
   * - ``construct``
     - an object of kind CHARE, GROUP, NODEGROUP, ARRAY or ARRAYMAP is
       created, on its PE
     - builds the object from the constructor message; returns the handle
   * - ``mainchare``
     - a MAINCHARE class's object is created, on PE 0, at start-up
     - stores the handle in ``*handle_out`` (before running user code that
       might send to the main chare), receives ``argc``/``argv``
   * - ``destruct``
     - an object is deleted, migrates away, or the program exits
     - releases the handle
   * - ``deliver``
     - an entry method is invoked
     - runs the method on the object
   * - ``deliver_many``
     - an array broadcast reaches a PE (optional)
     - runs the method on ``n`` local elements; see :ref:`ckdyn-bcast`
   * - ``pack_size``, ``pack``, ``unpack``
     - migration, checkpoint, restart
     - serialize an object; see :ref:`ckdyn-pup`
   * - ``pack_readonly_size``, ``pack_readonly``, ``unpack_readonly``
     - readonly broadcast at start-up, checkpoint, restart (optional)
     - serialize the binding's global read-only data
   * - ``reduce``
     - a reduction with a binding-defined reducer combines contributions
     - see :ref:`ckdyn-foreign`
   * - ``future``
     - a value for a binding future arrives (``CkDynFutureSet`` or a
       ``CKDYN_CB_FUTURE`` callback)
     - resolves the future
   * - ``section_joined``
     - a member receives its section's creation multicast (optional)
     - records membership
   * - ``map_proc_num``
     - the array manager asks an array map where an index lives
     - returns a PE
   * - ``resume_from_sync``
     - after a load balancing step, for array elements created with
       ``CKDYN_MSG_USESATSYNC``
     - resumes the element
   * - ``error``
     - CkDyn is about to abort because of something the binding did
     - reports the error in the binding's terms; CkDyn aborts after it
       returns

The per-class hooks (``construct``, ``destruct``, ``deliver``,
``deliver_many``, ``mainchare``, the object pup hooks, ``section_joined``,
``map_proc_num``, ``resume_from_sync``) receive the class's
``binding_data``. The hooks without a class (``register_classes``, the
readonly hooks, ``reduce``, ``future``, ``error``) receive ``global`` and are
always taken from the table given to ``CkDynInit``.

Where hooks run
~~~~~~~~~~~~~~~

Every hook runs inside the Charm++ scheduler on the PE that owns the object
(for ``future``, the PE that owns the future), except ``reduce``, which runs
inside the reduction manager on whichever PE combines contributions, with no
object context. A hook may call any CkDyn function: send, create objects,
contribute, set futures, create sections. Nested message building works
(a builder per nesting level is allocated as needed).

Rules that apply to every hook:

- **Views are valid only during the hook.** A ``CkDynMsgView`` and every
  pointer read from it (slots, data, tail, buffer pointers) are valid until
  the hook returns. To keep a pointer into a message longer, call
  ``CkDynMsgRetain`` before returning (:ref:`ckdyn-ownership`).
- **The runtime owns delivered messages.** A hook never frees the message it
  receives.
- **Do not unwind through CkDyn.** Hooks are called from C++ code; a binding
  in a language with exceptions catches them inside the hook. charm4py
  converts a Python exception in a hook into an abort with the traceback.

.. _ckdyn-threads:

Threading model
---------------

A Reconverse process can run several PEs, each a thread. CkDyn itself works
with several PEs per process; the C++ test runs four PEs in one process.
What a binding must assume:

- Hooks of different PEs of one process may run at the same time, on
  different threads. Per-PE binding state (future tables, coroutine
  schedulers) must be per thread: the C++ test binding keeps it in a
  ``Ckpv`` variable, charm4py in ``thread_local`` storage.
- The hook table and ``global`` are shared by all PEs of a process.
- A nodegroup branch is shared by the PEs of its process; its entries may
  run on any of them, concurrently, unless the entry is
  ``CKDYN_EP_EXCLUSIVE``.
- ``reduce`` may run on any PE and concurrently with other hooks.
- Registration happens once per process, on rank 0.

Two features need one PE per process: late registration
(:ref:`ckdyn-late`), and charm4py 2, whose interpreter lock serializes all
Python code in a process (it aborts at start-up when a process has more
than one PE; run several processes with one PE each).

.. _ckdyn-registration:

Start-up and Registration
=========================

Start-up
--------

A **shared-library binding** starts the runtime itself:

.. code-block:: c

   CkDynHooks h;
   memset(&h, 0, sizeof h);
   h.version = CKDYN_API_VERSION;
   h.size = sizeof(CkDynHooks);
   h.register_classes = my_register_classes;
   /* ... the other hooks ... */
   if (CkDynApiVersion() != CKDYN_API_VERSION) { /* refuse */ }
   CkDynInit(&h);             /* copies the table, checks version and size */
   CkDynStart(argc, argv);    /* ConverseInit; never returns */

``CkDynInit`` copies the table. ``CkDynStart`` copies ``argv`` into storage
it owns and calls ``ConverseInit``; it does not return. On every process,
the runtime then calls ``register_classes(global)`` on rank 0, where the
binding registers its classes. Then PE 0 constructs the main chares. A
binding for an interpreted language releases its interpreter lock around
``CkDynStart`` and takes it again in each hook (charm4py does this with
``PyEval_SaveThread`` and ``PyGILState_Ensure``).

A **C++ program** starts as any Charm++ program. It calls ``CkDynInit`` and
registers its classes from an ``initnode`` function declared in its ``.ci``
file; ``register_classes`` is not called for it. If its main chare is an
ordinary C++ main chare (not a dynamic MAINCHARE class), the main chare's
constructor calls ``CkDynCreateManagers()`` before using CkDyn, to create
CkDyn's per-PE manager group and its CkMulticast manager. A dynamic main
chare does this itself, before calling the ``mainchare`` hook.

Registering a class
-------------------

.. code-block:: c

   int first_ep;
   int chare_idx = CkDynRegisterClass(&desc, &first_ep);

``CkDynRegisterClass`` returns the class's chare index and stores the entry
index of ``entries[0]`` (the constructor) in ``first_ep``; ``entries[i]``
has entry index ``first_ep + i``. The binding computes entry indices by
addition and keeps ``chare_idx`` and ``first_ep`` with its class.

Rules:

- Call it on rank 0 of every process (from ``register_classes`` or an
  ``initnode``), and in the same order on every process, so that every
  process assigns the same indices.
- Call it before start-up completes. A class registered after that needs
  late registration.
- The class and entry names are kept by pointer: keep the strings valid for
  the program's lifetime (string literals, or strings the binding never
  frees). ``CkDynClassOf(chare_idx)`` returns the registered descriptor, or
  NULL for an index that is not a dynamic class.

A MAINCHARE class is constructed on PE 0 at start-up; a program may
register more than one (the C++ test has an ordinary main chare and a
dynamic one).

.. _ckdyn-late:

Late registration
-----------------

.. code-block:: c

   CkDynAllowLateRegistration();          /* once, before or during start-up */
   int idx = CkDynRegisterClassLate(&desc, &first_ep);

``CkDynRegisterClassLate`` registers a class after start-up, in every build.
Its limits:

- The process must have one PE; it aborts otherwise, because the runtime's
  entry and chare tables are shared by the PEs of a process.
- It must run on every process, in the same order, before any message for
  the class is sent. The C++ test registers from a group broadcast and waits
  for a reduction before it creates objects of the class.
- A MAINCHARE class registered late is never constructed.
- Readonly data is broadcast once at start-up; the readonly hooks are
  registered by ``CkDynInit``, so they are unaffected.
- With the trace-summary or trace-utilization modules, executing an entry of
  a late class aborts (their per-entry tables are sized at start-up);
  trace-projections works.
- A checkpoint holding objects of a late class restarts only if the
  restarted program registers the same classes, in the same order, at
  start-up.

Array dimensions
----------------

An array class has a fixed dimension count (``ndims`` 1 to 6) or accepts
any (``ndims`` 0): each creation and insert then gives its own count.
Languages whose arrays pick their dimension count at creation (charm4py)
register every array class with 0. ``CkDynCreateArray`` checks the count
only for a class registered with a fixed one.

For tracing, a class registered with 0 takes the dimension count of the
first index the process sees for it, so Projections records element indices
as for a charmxi array. A class used for arrays of different dimension
counts in one run keeps the first count.

Limits
------

- **Entry indices.** Entry indices are 16 bits: at most 65536 per process,
  the runtime's own included.
- **Message types.** CkDyn uses two of the runtime's 256 message types.
- **Array dimensions.** 1 to 6 (``CKDYN_MAX_DIMS``); components of 4- to
  6-dimensional indices fit in 16 bits.
- **Buffers.** At most 32 dimensions (``CKDYN_MAX_BUF_DIMS``).
- **Message size.** A slot's length is 32 bits; a payload is at most 2 GB;
  at most 65535 slots per message.

.. _ckdyn-messages:

Messages
========

Every entry method of every dynamic class receives a ``CkDynMsg``. Its
payload is laid out as:

1. the header, ``CkDynHdr`` (24 bytes): magic, entry index, flags, slot
   count, ``user_id``, tail length, data length, zero-copy buffer count;
2. ``nslots`` slot records, ``CkDynSlot`` (24 bytes each): type, element
   kind, dimension count, buffer flags, a 16-bit binding tag, length, and a
   value union (an inline scalar or an offset into the data region);
3. the data region, 8-byte aligned items: byte strings, buffer shapes and
   contents, and the bytes of callback, section and proxy values;
4. the tail, opaque bytes written by the binding.

A message with zero-copy slots also has a descriptor prefix before the
header; the view API hides it.

Slot types
----------

.. list-table::
   :header-rows: 1
   :widths: 22 30 48

   * - type
     - add with
     - read with
   * - ``CKDYN_SLOT_NONE``
     - ``CkDynMsgAddNone``
     - the type alone (the binding's null)
   * - ``BOOL``, ``I64``, ``U64``, ``F64``
     - ``CkDynMsgAddBool``, ``AddI64``, ``AddU64``, ``AddF64``
     - ``slots[i].v.i64``, ``.u64``, ``.f64`` (BOOL in ``i64``)
   * - ``CKDYN_SLOT_BYTES``
     - ``CkDynMsgAddBytes(b, p, len, tag)``
     - ``CkDynViewBytesPtr``; length ``slots[i].len``
   * - ``CKDYN_SLOT_BUFFER``
     - ``CkDynMsgAddBuffer(b, &desc)``
     - ``CkDynViewBufferPtr``, which also returns the shape
   * - ``CKDYN_SLOT_NCPY``
     - ``CkDynMsgAddNcpy``
     - ``CkDynViewBufferPtr`` (:ref:`ckdyn-zerocopy`)
   * - ``CKDYN_SLOT_DEVICE``
     - ``CkDynMsgAddDevice``
     - ``CkDynViewBufferPtr`` (device memory)
   * - ``CKDYN_SLOT_FUTURE``
     - ``CkDynMsgAddFuture(b, pe, id)``
     - ``v.u64 = (pe << 32) | id``
   * - ``CKDYN_SLOT_CALLBACK``
     - ``CkDynMsgAddCallback``
     - ``CkDynViewCallback`` (copies out)
   * - ``CKDYN_SLOT_SECTION``
     - ``CkDynMsgAddSection``
     - ``CkDynViewSection`` (copies out)
   * - ``CKDYN_SLOT_PROXY``
     - ``CkDynMsgAddProxy`` (a ``CkDynTarget``)
     - ``CkDynViewProxy`` (copies out)
   * - ``CKDYN_SLOT_TAIL``
     - ``CkDynMsgReserveTail``
     - ``view->tail``, ``view->tail_len``

Slots keep the order in which they were added; slot ``i`` of the view is the
``i``-th one added. The ``tag`` field of a BYTES, BUFFER, NCPY or DEVICE slot
is the binding's: CkDyn neither sets nor reads it. charm4py uses it to tell
a pickled argument from raw bytes and an ``array.array`` from a numpy array.

Element kinds (``CKDYN_ELEM_*``) describe buffer contents: signed and
unsigned integers of 8 to 64 bits, 16-, 32- and 64-bit floats, 64- and
128-bit complex, bool and byte.

The builder
-----------

.. code-block:: c

   CkDynMsgBuilder *b = CkDynMsgBegin(nslots_hint, data_hint, tail_hint, prio_bits, group_dep);
   CkDynMsgSetFlags(b, flags);           /* CKDYN_MSG_* */
   CkDynMsgSetUserId(b, user_id);        /* rarely needed: see foreign reducers */
   CkDynMsgAddI64(b, 42);
   CkDynMsgAddBuffer(b, &desc);
   void *t = CkDynMsgReserveTail(b, len);   /* write len bytes at t */
   CkDynMsg *m = CkDynMsgFinish(b, ep);  /* one allocation */

- The hints size the builder's scratch space; they are not limits.
- ``prio_bits`` reserves that many priority bits (0: none).
- ``group_dep`` is a group id the message must wait for on delivery, or -1.
  It must be given here because the envelope reserves space for it at
  allocation. The runtime honors it on chare, group, nodegroup and creation
  messages, not on point-to-point sends to array elements. Its main use is
  creating an array with a custom map: the array's constructor message
  depends on the map group (:ref:`ckdyn-arrays`).
- ``CkDynMsgAddBuffer`` copies the shape at once but the contents only at
  ``CkDynMsgFinish``, straight into the message, so ``desc.ptr`` must stay
  valid until ``Finish`` or ``Abandon``. Strides, if given, must describe a
  contiguous layout in C or Fortran order (``CKDYN_BUF_FORTRAN``); CkDyn
  aborts otherwise.
- ``CkDynMsgFinish(b, ep)`` allocates the message once and writes ``ep``
  into the header. The send functions write the destination entry into the
  header again, so for a plain send the ``ep`` given to ``Finish`` does not
  matter (the C++ test binding passes 0). For a constructor message it must
  be the class's constructor entry, ``first_ep``.
- ``CkDynMsgFinishMcast(b, ep)`` finishes the same payload as a section
  message, for ``CkDynSendSection``; it takes no group dependency.
- ``CkDynMsgAbandon(b)`` drops a builder that will not be finished. A binding
  calls it on every error path between ``Begin`` and ``Finish``; charm4py
  uses a guard object whose destructor abandons the builder. Abandoning
  runs the ``done`` functions of any zero-copy slots.
- ``CkDynMsgFree(m)`` frees a finished message that will not be sent.

Every send, creation, ``CkDynFutureSet`` and ``CkDynCallbackSend`` takes
ownership of the message: after the call the binding must not touch it.

Header flags
------------

``CkDynMsgSetFlags`` sets the header's flags:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - flag
     - meaning
   * - ``CKDYN_MSG_CREATION``
     - constructor message that acknowledges creation to its first FUTURE
       slot (:ref:`ckdyn-objects`)
   * - ``CKDYN_MSG_USESATSYNC``
     - array element constructor: the element uses AtSync load balancing
   * - ``CKDYN_MSG_CONSTRAINED``
     - group constructor: the last slot is a BYTES slot of ``int32`` PE
       numbers, and only branches on those PEs are constructed
   * - ``CKDYN_MSG_BCAST``
     - set by CkDyn on broadcasts
   * - ``CKDYN_MSG_BLOCK``, ``CKDYN_MSG_BCASTRET``, ``CKDYN_MSG_RAY``
     - defined for bindings and passed through unread. The convention of the
       existing bindings: with ``BLOCK``, a FUTURE slot receives the
       method's return value; ``BCASTRET`` gathers the return values of a
       broadcast
   * - ``CKDYN_MSG_BINDING_MASK``
     - bits 12 to 15 belong to the binding; CkDyn neither sets nor reads
       them

Priorities and queueing
-----------------------

Reserve priority bits at ``CkDynMsgBegin``, then after ``Finish`` write the
priority through ``CkDynMsgPriorityPtr(m)`` and set the queueing strategy
with ``CkDynMsgSetQueueing(m, q)``. The values of ``q`` are Converse's
``CQS_QUEUEING_*`` constants, which a binding without Charm++ headers
defines itself:

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - strategy
     - value
   * - FIFO, LIFO
     - 2, 3
   * - IFIFO, ILIFO (integer priority)
     - 4, 5
   * - BFIFO, BLIFO (bit-vector priority)
     - 6, 7
   * - LFIFO, LLIFO (64-bit priority)
     - 8, 9

From the C++ test (a smaller integer runs first):

.. code-block:: c++

   CkDynMsgBuilder *b = CkDynMsgBegin(1, 0, 0, 8 * sizeof(int), -1);
   CkDynMsgAddI64(b, p);
   CkDynMsg *m = CkDynMsgFinish(b, 0);
   *(int *)CkDynMsgPriorityPtr(m) = p;
   CkDynMsgSetQueueing(m, 4 /* IFIFO */);
   CkDynSendGroup(gid, CkDynMyPe(), ep, m, NULL);

Sending
-------

.. code-block:: c

   void CkDynSendChare     (const CkDynTarget *t, int ep, CkDynMsg *m, const CkDynSendOpts *o);
   void CkDynSendGroup     (int gid, int pe, int ep, CkDynMsg *m, const CkDynSendOpts *o);   /* pe -1: broadcast */
   void CkDynSendGroupMulti(int gid, int npes, const int *pes, int ep, CkDynMsg *m, const CkDynSendOpts *o);
   void CkDynSendNodeGroup (int gid, int node, int ep, CkDynMsg *m, const CkDynSendOpts *o); /* node -1: broadcast */
   void CkDynSendArray     (int aid, int ndims, const int *index, int ep, CkDynMsg *m,
                            const CkDynSendOpts *o);                                       /* index NULL: broadcast */
   void CkDynSendTarget    (const CkDynTarget *t, int ep, CkDynMsg *m, const CkDynSendOpts *o);

``CkDynSendTarget`` sends to whatever a target names, selecting the function
by its kind; a binding whose proxies hold a ``CkDynTarget`` uses it for
every point-to-point send. Groups and array maps use ``CkDynSendGroup``.
Array sends to an element that does not exist yet are buffered, as for any
Charm++ array.

``CkDynSendOpts.flags`` adds per-send options to the entry's own
attributes (NULL: none):

- ``CKDYN_SEND_INLINE``: a send to a local object runs the method inside
  the send.
- ``CKDYN_SEND_EXPEDITED``: the message bypasses the priority queue.
- ``CKDYN_SEND_IMMEDIATE``: Charm++ immediate delivery (nodegroups).
- ``CKDYN_SEND_NOTRACE``: the message is not recorded in the load
  balancer's communication statistics.

An inline send nests the receiver's ``deliver`` inside the sender's hook; a
binding whose methods send inline to each other in a loop should drive the
loop from one side, as the C++ test's inline ping-pong does, so that the
nesting stays shallow. A message with zero-copy slots is never sent inline.

Receiving: the view
-------------------

Every delivery hook receives a ``CkDynMsgView``:

.. code-block:: c

   typedef struct CkDynMsgView {
     CkDynMsgKind          kind;
     const CkDynHdr       *hdr;        /* NULL for REDRESULT and SYS */
     const CkDynSlot      *slots;
     int                   nslots;
     const char           *data;       /* data region */
     const char           *tail;       /* tail_len bytes, or NULL */
     uint32_t              tail_len;
     uint32_t              refnum;     /* the envelope's reference number, 0 if none */
     const CkDynSectionID *section;    /* non-NULL for a section delivery */
     void                 *msg;        /* the runtime message, for retain */
   } CkDynMsgView;

An entry method can receive four kinds of message:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - kind
     - what it is
   * - ``CKDYN_MSGKIND_DYN``
     - a ``CkDynMsg``: a send, broadcast, constructor message, future value,
       or the result of a binding-defined reducer
   * - ``CKDYN_MSGKIND_MCAST``
     - a section multicast, or a section callback's result; ``section``
       names the section
   * - ``CKDYN_MSGKIND_REDRESULT``
     - a built-in reduction result, shown as one BUFFER slot over the
       reduction data, plus a FUTURE slot ``(CkDynMyPe(), refnum)`` when the
       reference number is set
   * - ``CKDYN_MSGKIND_SYS``
     - a message from the runtime with no data (quiescence, checkpoint
       completion, a HAPI callback): no slots, except the same FUTURE slot

For DYN and MCAST views, ``hdr->ep`` is the entry index, and it is
authoritative (the envelope's copy can be stale on inline sends). The
``deliver`` hook also receives ``ep``, the entry's ``user_id`` and
``ep_flags`` (the entry's attributes, with ``CKDYN_EP_NOKEEP`` always set).

``refnum`` is the envelope's reference number; 0, and the all-ones value
(the default reduction user flag), read as none. A binding uses it as the
future id a result belongs to.

The slots of a REDRESULT or SYS view are made by CkDyn in the caller's stack
frame: the view, its slots and pointers read from them are valid only until
the hook returns, and retaining the message keeps the reduction data alive
but not those slots.

Reading slots:

.. code-block:: c

   const void *CkDynViewBufferPtr(const CkDynMsgView *v, int slot, const int64_t **shape_out);
   const void *CkDynViewBytesPtr(const CkDynMsgView *v, int slot);
   void        CkDynViewCallback(const CkDynMsgView *v, int slot, CkDynCallback *out);
   void        CkDynViewSection(const CkDynMsgView *v, int slot, CkDynSectionID *out);
   void        CkDynViewProxy(const CkDynMsgView *v, int slot, CkDynTarget *out);

Scalar slots are read from ``slots[i].v``. Accessors abort on a slot index
out of range or a slot of another type.

.. _ckdyn-ownership:

Ownership and retaining
-----------------------

Every CkDyn entry is registered "no keep": the runtime frees the message
after the hook returns. A binding that keeps a pointer into the message
beyond the hook (a zero-copy numpy view of a BUFFER slot, a received NCPY
buffer it forwards later) calls ``CkDynMsgRetain(view)`` before returning
and ``CkDynMsgRelease(view->msg)`` once per retain when done; the message is
freed when the last reference goes. Keep ``view->msg``, not the view: the
view itself is invalid after the hook.

From the C++ test, which echoes a received 16 MB NCPY buffer from the
retained message and releases it when the echo's transfer is done:

.. code-block:: c++

   CkDynMsgRetain(v);
   zSend(replyTo, mode, -1, p, seed, v->msg /* released by the done function */, ...);

charm4py wraps each retain in a Python object whose deallocation calls
``CkDynMsgRelease``, and gives numpy arrays over BUFFER slots that object as
their base.

Broadcasts
----------

``CkDynSendGroup`` with PE -1, ``CkDynSendNodeGroup`` with node -1 and
``CkDynSendArray`` with a NULL index broadcast. ``CkDynSendGroupMulti``
sends one message to a list of group branches. The receiver sees
``CKDYN_MSG_BCAST`` in the header.

.. _ckdyn-bcast:

Array broadcasts and ``deliver_many``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

An array broadcast arrives once per PE. Without a ``deliver_many`` hook,
CkDyn calls ``deliver`` once per local element, and traces and times each
call as for a charmxi element. With ``deliver_many``, CkDyn calls it once
with the handles of all local elements that should receive the broadcast;
the binding can then convert the message's arguments once and run the method
per element. In that case time is charged to the array manager and tracing
records no event per element, unless the binding brackets each element's
call with ``CkDynCoroResume``/``CkDynCoroSuspend`` (or
``CkDynCallstackPush``/``Pop`` for the load alone); charm4py does that:

.. code-block:: c++

   for (int i = 0; i < n; i++) {
     CkDynTarget t = target_of(handles[i]);
     CkDynCoroResume(&t, ep, 0);
     deliver_one(handles[i], ep, ep_flags, v, args);
     CkDynCoroSuspend(&t);
   }

A handler that destroys another local element makes that element's handle
invalid for the rest of the call.

.. _ckdyn-objects:

Objects
=======

Creating objects
----------------

Every creation takes a constructor message finished with the class's
constructor entry (``first_ep``); the ``construct`` hook receives its view.

.. code-block:: c

   void CkDynCreateChare       (int chare_idx, int on_pe, CkDynMsg *ctor, CkDynTarget *out);
   int  CkDynCreateGroup       (int chare_idx, CkDynMsg *ctor);       /* GROUP or ARRAYMAP */
   int  CkDynCreateNodeGroup   (int chare_idx, CkDynMsg *ctor);
   int  CkDynCreateArray       (int chare_idx, CkDynMsg *ctor, const CkDynArrayOpts *o);
   void CkDynArrayInsert       (int aid, int ndims, const int *index, int on_pe, CkDynMsg *ctor);
   void CkDynArrayDoneInserting(int aid);
   void CkDynArrayDestroyElement(int aid, int ndims, const int *index);

The creation functions check that the class has the right kind and abort
otherwise.

**Main chares.** A MAINCHARE class is constructed by the runtime on PE 0 at
start-up through the ``mainchare`` hook, with ``argc`` and ``argv``; there is
no constructor message. The C++ test binding runs the class's constructor
method with a NULL view.

**Singleton chares.** ``CkDynCreateChare(idx, on_pe, ctor, &out)`` creates
one chare on ``on_pe``, or, with ``CKDYN_PE_ANY``, where the seed balancer
places it. ``out`` receives a virtual chare id usable at once.

**Groups and constrained groups.** ``CkDynCreateGroup`` returns the group id;
a branch is constructed on every PE. To construct branches only on a list of
PEs, set ``CKDYN_MSG_CONSTRAINED`` and add the PE list as the last slot, a
BYTES slot of ``int32`` values:

.. code-block:: c

   CkDynMsgSetFlags(b, CKDYN_MSG_CONSTRAINED);
   /* ... constructor arguments ... */
   CkDynMsgAddBytes(b, pes, npes * sizeof(int32_t), 0);
   int gid = CkDynCreateGroup(idx, CkDynMsgFinish(b, first_ep));

Branches on other PEs exist with a NULL handle: they take part in
reductions, ignore broadcasts and callback results, and are skipped when
pupped. A point-to-point send to such a branch is a binding error.

**Nodegroups.** ``CkDynCreateNodeGroup`` constructs one branch per process.

.. _ckdyn-arrays:

**Arrays.** ``CkDynCreateArray`` takes options:

.. code-block:: c

   typedef struct CkDynArrayOpts {
     int        ndims;
     const int *dims;               /* NULL: empty array, insert later */
     int        map_gid;            /* -1: default map */
     int        anytime_migration;
     int        static_insertion;
     int        bind_to_aid;        /* -1, or an array to bind to */
   } CkDynArrayOpts;

With ``dims`` the array is created with all its elements; with NULL it starts
empty, elements are added with ``CkDynArrayInsert`` (on ``on_pe``, or -1 for
the map's choice), and ``CkDynArrayDoneInserting`` ends insertion. Every
element of an array is of the array's class, and the insert's constructor
message must be finished with that class's ``first_ep``.
``CKDYN_MSG_CREATION`` is for whole-array creation only: on an insert it
would start a reduction over one element.

**Array maps.** An array map is a class of kind ``CKDYN_ARRAYMAP``, created
with ``CkDynCreateGroup``. The runtime asks its local branch where an index
lives by calling ``map_proc_num(binding_data, handle, aid, ndims, index)``,
which returns a PE. An array that uses the map passes its id as ``map_gid``
and as the ``group_dep`` of the array's constructor message, so that no
element is created before the map exists on its PE:

.. code-block:: c

   CkDynMsgBuilder *b = CkDynMsgBegin(4, 64, 0, 0, map_gid /* group_dep */);
   /* ... */
   CkDynArrayOpts o = { 1, dims, map_gid, 0, 0, -1 };
   int aid = CkDynCreateArray(idx, CkDynMsgFinish(b, first_ep), &o);

Creation acknowledgement
------------------------

A constructor message with ``CKDYN_MSG_CREATION`` and a FUTURE slot
acknowledges creation without the binding's help: once the object exists (a
chare), or every branch or element exists (a group, nodegroup or array), the
future in the first FUTURE slot receives a value through the ``future``
hook. From the C++ test:

.. code-block:: c++

   aidPing = CkDynCreateArray(Ping.chareIdx,
       MsgB().flags(CKDYN_MSG_CREATION | CKDYN_MSG_USESATSYNC)
             .fut(0, created("Ping")).finish(Ping.ep(0)), &ao);

Destruction
-----------

``CkDynArrayDestroyElement`` destroys a local array element; it aborts for an
element that is not on the calling PE. CkDyn calls ``destruct`` for the
handle. The C API has no call to destroy a singleton chare or a group.

Local objects
-------------

CkDyn gives the binding its handles only through hooks. A binding that needs
the local object a target names outside a hook keeps its own table (the C++
test binding stores its object pointers where its methods find them;
charm4py keeps a map from targets to Python objects). Functions that act on
a local object take its target and abort when the object is not on the
calling PE.

.. _ckdyn-pup:

Migration and pup
-----------------

An array element migrates when the load balancer moves it or when the
binding calls ``CkDynMigrateMe(t, to_pe)``. ``CkDynSetMigratable(t, 0)``
pins an element. Both act on a local array element only.

**AtSync load balancing.** Create the elements with
``CKDYN_MSG_USESATSYNC`` in the constructor message. Each element calls
``CkDynAtSync(&target)``; after the load balancing step the runtime calls
``resume_from_sync(binding_data, handle)`` for each element, on its new PE.
The C++ test binding resumes by sending the element a message:

.. code-block:: c++

   static void hResumeFromSync(void *bd, void *h) {
     Obj *o = (Obj *)h;
     /* m: the class's "resumed" method */
     CkDynSendTarget(&o->t, c->ep(m), MsgB().finish(c->ep(m)), nullptr);
   }

``CkDynArrayOpts.anytime_migration`` allows the balancer to move elements at
any time.

**Pup hooks.** Packing an object runs two passes:

1. ``pack_size(bd, handle, reason | CKDYN_PUP_SIZING)`` returns the size
   ``n``;
2. ``pack_size(bd, handle, reason)`` is called again and must return the same
   ``n``, then ``pack(bd, handle, reason, buf, n)`` writes exactly ``n``
   bytes.

On the receiving PE, ``unpack(bd, target, reason, buf, n)`` returns a new
handle; the old object's ``destruct`` runs on the PE it left. ``reason`` is
``CKDYN_PUP_MIGRATE`` or ``CKDYN_PUP_CHECKPOINT``. An object that cannot be
packed in its current state (charm4py: one with a suspended coroutine)
returns ``CKDYN_PACK_REFUSED`` from ``pack_size``; CkDyn reports it through
``error`` and aborts. Section membership travels with the element; buffer
posts do not.

**Readonly data.** If ``pack_readonly_size`` or ``unpack_readonly`` is set,
``CkDynInit`` registers one readonly for the binding: its bytes are packed
on PE 0 and unpacked on every other PE at start-up (values set while the
main chares are constructed), and they are written to and read from a
checkpoint.

Checkpoint and restart
----------------------

.. code-block:: c

   void CkDynCheckpoint(const char *dirname, const CkDynCallback *done);

``CkDynCheckpoint`` writes a checkpoint of every object to ``dirname``
(main chares, chares, groups, nodegroups, arrays, through the pup hooks with
``CKDYN_PUP_CHECKPOINT``, and the readonly data). ``done`` is called when the
checkpoint is written and, in a program started with ``+restart dirname``,
once the restart completes. On restart:

- the program registers the same classes in the same order at start-up;
- the runtime restores the objects through ``unpack``; the main chare is
  restored too, and its ``mainchare`` hook is not called;
- ``CkDynIsRestarting()`` returns 1;
- ``done`` is delivered. It may be a TARGET callback to a group, an array
  element or a main chare entry: a callback to a main chare is routed by
  chare index, not by object pointer, so it reaches the restored main chare.
  A FUTURE callback does not survive a restart (the future belonged to the
  checkpointed run's coroutine).

Not restored: sections (a binding recreates them), buffer posts, futures and
pending messages. A program that continues after its own checkpoint tells
the two runs apart itself; the C++ test's dynamic main chare checks whether
its constructor ran in this process.

Quiescence
----------

``CkDynStartQD(&cb)`` starts quiescence detection; ``cb`` receives a SYS view
(with a FUTURE slot when the callback carries a reference number) once no
messages are in flight.

.. _ckdyn-futures:

Futures and Callbacks
=====================

Futures
-------

A future is the binding's: CkDyn knows it only as a pair ``(pe, id)``, the PE
whose binding holds it and a 32-bit id the binding chose. Any code can send a
value to a future:

.. code-block:: c

   void CkDynFutureSet(int pe, uint32_t id, CkDynMsg *value);

The message goes to CkDyn's manager on ``pe``, which calls
``future(global, pe, id, view)``. The binding finds the future by ``id`` and
resolves it, typically by resuming a coroutine that waits on it. Values also
reach futures from callbacks of kind ``CKDYN_CB_FUTURE`` (reductions,
quiescence, creation acknowledgements), as REDRESULT, SYS or DYN views.

Future ids travel as the message's reference number, so their range is
1 to 2\ :sup:`8 * CkDynRefnumBytes()`-2: 0 means "no reference number" and
the all-ones value is the default reduction flag. The C++ test binding keeps
ids below 65536 so that it works with the default 16-bit reference number;
charm4py requires 32 bits.

The ``future`` hook runs inside an untraced entry of CkDyn's manager. That
is where a binding resumes coroutines, bracketed with
``CkDynCoroResume``/``CkDynCoroSuspend`` (:ref:`ckdyn-coro`).

A method's return value as a future value (the ``CKDYN_MSG_BLOCK``
convention): the sender adds a FUTURE slot and sets ``CKDYN_MSG_BLOCK``; the
receiving binding runs the method and sends the result:

.. code-block:: c++

   int64_t r = c->methods[user_id].fn(o, v);
   if (v->hdr && (v->hdr->flags & CKDYN_MSG_BLOCK)) {
     int pe; uint32_t id;
     cbFindFuture(v, &pe, &id);            /* the first FUTURE slot */
     CkDynFutureSet(pe, id, MsgB().i64(r).finish(0));
   }

Callbacks
---------

A ``CkDynCallback`` is an 88-byte plain struct describing where a result
goes. Zero it, then set the fields of its kind:

.. list-table::
   :header-rows: 1
   :widths: 22 30 48

   * - kind
     - fields
     - delivers
   * - ``CKDYN_CB_TARGET``
     - ``t``, ``ep``, ``bcast``, ``refnum``
     - to entry ``ep`` of the element ``t``, or with ``bcast != 0`` to every
       member of the collection ``t`` names
   * - ``CKDYN_CB_SECTION``
     - ``sid``, ``ep``, ``refnum``
     - to entry ``ep`` of every member of a section
       (:ref:`ckdyn-sections`)
   * - ``CKDYN_CB_FUTURE``
     - ``pe``, ``id``
     - to the binding future ``(pe, id)``
   * - ``CKDYN_CB_IGNORE``
     -
     - nowhere
   * - ``CKDYN_CB_EXIT``
     -
     - exits the program

``refnum``, when not zero, becomes the reference number of the delivered
message, which the receiver reads as ``view->refnum``: a binding passes a
future id this way to an entry method that will forward the result. A
callback can travel in a message (``CkDynMsgAddCallback``), so an entry can
reply to a callback its caller chose.

``CkDynCallbackSend(&cb, m)`` sends a finished message through a callback;
for a TARGET callback CkDyn writes the entry into the header. Callbacks are
also accepted by ``CkDynContribute``, ``CkDynContributeSection``, section
creation, ``CkDynStartQD``, ``CkDynCheckpoint`` and
``CkDynHapiAddCallback``.

A TARGET callback to a main chare entry is routed by chare index through
CkDyn's manager on the main chare's PE, so it stays valid after a restart
from a checkpoint.

System callbacks
----------------

Callbacks that the runtime fires without data (quiescence, checkpoint
completion, HAPI completion, creation of a chare) deliver a SYS view or an
empty DYN message; a reduction delivers a REDRESULT view. A binding that
wants one function for all result kinds reads the view's ``kind`` and its
slots: the C++ test's ``resultOf`` accepts a BUFFER slot (a built-in
reduction), an I64 slot (its foreign reducer) or no slot.

Reduction numbers
-----------------

``CkDynGetReductionNumber(&target)`` returns the reduction number of a local
group branch, nodegroup branch, array map branch or array element: the number
of reductions it has contributed to. Every member computes the same value
for the same reduction, so a binding can derive from it a future id that
every member agrees on without communication.
