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
     const char *name;     /* "Class.method"; copied at registration */
     uint32_t    flags;    /* CKDYN_EP_* */
     uint32_t    user_id;  /* opaque to CkDyn; passed back on every delivery */
   } CkDynEntryDesc;

   typedef struct CkDynClassDesc {
     const char            *name;         /* "Class"; copied at registration */
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
     - the entry is application work (``CK_EP_APPWORK``); not applied to
       array broadcasts delivered through ``deliver_many``
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
- The class and entry names are copied at registration: the binding may
  free its strings once ``CkDynRegisterClass`` returns. ``CkDynClassOf(chare_idx)`` returns the registered descriptor, or
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
``CkDynRegisterClass`` after start-up aborts and names it, also in a
``charm4py`` build of Charm++, whose own registration tables accept late
entries. Its limits:

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

- The hints size the builder's scratch space; they are not limits. Each PE
  keeps one builder and reuses it, so its scratch space stays at the largest
  size any message on that PE needed; it is not returned until the run ends.
  A negative ``nslots_hint`` counts as 0.
- ``prio_bits`` reserves that many priority bits (0: none).
- ``group_dep`` is a group id the message must wait for on delivery, or -1.
  It must be given here because the envelope reserves space for it at
  allocation. The runtime honors it on chare, group, nodegroup and creation
  messages, not on point-to-point sends to array elements. Its main use is
  creating an array with a custom map: the array's constructor message
  depends on the map group (:ref:`array maps <ckdyn-arrays>`).
- ``CkDynMsgAddBuffer`` copies the shape at once but the contents only at
  ``CkDynMsgFinish``, straight into the message, so ``desc.ptr`` must stay
  valid until ``Finish`` or ``Abandon`` (except in the ``reduce`` hook's
  result builder, which copies the contents at once). A NULL ``desc.ptr``
  with a nonzero length aborts. Strides, if given, must describe a
  contiguous layout in C or Fortran order (``CKDYN_BUF_FORTRAN``); CkDyn
  aborts otherwise. An empty buffer (an extent of 0) is accepted whatever
  its strides.
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
with ``CkDynMsgSetQueueing(m, q)``, where ``q`` is one of the
``CKDYN_QUEUEING_*`` constants of ``ckdyn.h`` (their values equal Charm++'s
``CK_QUEUEING_*``). Without a call the message is queued FIFO.

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - constants
     - priority
   * - ``CKDYN_QUEUEING_FIFO``, ``CKDYN_QUEUEING_LIFO``
     - none
   * - ``CKDYN_QUEUEING_IFIFO``, ``CKDYN_QUEUEING_ILIFO``
     - an ``int``, lower first
   * - ``CKDYN_QUEUEING_BFIFO``, ``CKDYN_QUEUEING_BLIFO``
     - a bit vector
   * - ``CKDYN_QUEUEING_LFIFO``, ``CKDYN_QUEUEING_LLIFO``
     - a 64-bit integer

From the C++ test (a smaller integer runs first):

.. code-block:: c++

   CkDynMsgBuilder *b = CkDynMsgBegin(1, 0, 0, 8 * sizeof(int), -1);
   CkDynMsgAddI64(b, p);
   CkDynMsg *m = CkDynMsgFinish(b, 0);
   *(int *)CkDynMsgPriorityPtr(m) = p;
   CkDynMsgSetQueueing(m, CKDYN_QUEUEING_IFIFO);
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
  balancer's communication statistics (``CK_MSG_LB_NOTRACE``, which only
  array sends consult). It does not affect Projections tracing.

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
``CKDYN_MSG_BCAST`` in the header; so does the receiver of a message sent with
``CkDynCallbackSend`` through a broadcast TARGET callback.

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

Without ``deliver_many`` a handler may migrate or destroy other local
elements of the array: CkDyn looks each element up again before its call,
and an element migrated away before its turn receives the broadcast once, on
its new PE. With ``deliver_many`` the call covers every local element at
once, and each is counted as having received the broadcast before the call:
an element that a handler migrates away during the call does not receive it
on its new PE, and the binding must not deliver to its handle either.

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
message must be finished with that class's ``first_ep``; an insert with
another class's constructor aborts once the process has seen the array's
class (it created the array, or holds one of its elements).
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

``CkDynArrayOpts.anytime_migration`` does not decide whether elements can
migrate: AtSync, ``CkDynMigrateMe`` and the load balancer move them either
way. It is Charm++'s ``CkArrayOptions::setAnytimeMigration``, which only
matters together with ``static_insertion``: an array with static insertion
and without anytime migration is assumed to keep its elements where they
were created, and its broadcasts skip the bookkeeping for elements that
move. Anytime migration is on for every array unless the program runs with
``+noAnytimeMigration``, so ``anytime_migration = 0`` keeps that default and
a nonzero value turns it on for this array under ``+noAnytimeMigration``.

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
  checkpointed run's coroutine). A SECTION callback aborts in
  ``CkDynCheckpoint``: sections are not checkpointed, so it could not be
  delivered after a restart.

Not restored: sections (a binding recreates them; an old section id aborts on
use), buffer posts, futures and pending messages. A program that continues
after its own checkpoint tells the two runs apart itself; the C++ test's
dynamic main chare checks whether its constructor ran in this process.

A restart may use another PE count than the checkpointed run. Then:

- array elements are restored and spread over the new PEs;
- every group branch is unpacked from PE 0's bytes and every nodegroup
  branch from node 0's (the Charm++ checkpoint rule): ``unpack`` on every PE
  receives the bytes PE 0 packed, so per-branch state of the other branches
  is lost;
- singleton chares (``CkDynCreateChare``) are not restored at all, and
  nothing reports it: their targets name no object, and a send to one is an
  error the runtime does not detect.
  A program that holds singleton chares restarts with the same PE count;
- the main chare is restored on PE 0 as usual.

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

.. _ckdyn-reductions:

Reductions
==========

Built-in reducers
-----------------

``CkDynReducers(&table)`` returns the number of built-in reducers and points
``table`` at a static array of ``CkDynReducerInfo``: the reducer's name as in
``CkReduction::reducerType`` (``"sum_int"``, ``"max_double"``, ...), its type
value, and the element kind of its data. The table covers ``nop``; ``sum``,
``product``, ``max`` and ``min`` for every integer and floating type
(``_char`` to ``_ulong_long``, ``_float``, ``_double``); the ``logical_*``
and ``bitvec_*`` reducers; ``random``; and ``concat``, ``set``,
``statistics`` and ``tuple``, whose data is ``CKDYN_ELEM_BYTE`` and whose
format the binding handles. ``random`` passes one contribution on unchanged,
so its element kind is ``CKDYN_ELEM_BYTE`` too. The ``_char`` reducers
operate on the platform's plain ``char``: their element kind is
``CKDYN_ELEM_I8`` where ``char`` is signed (x86, macOS) and
``CKDYN_ELEM_U8`` where it is unsigned (aarch64 and POWER Linux); use
``_uchar`` or a binding-defined reducer for a type with a fixed sign. The
``_long`` reducers are 32-bit on Windows and 64-bit elsewhere. A binding
looks the types up by name once at start-up:

.. code-block:: c

   static int redType(const char *name) {
     const CkDynReducerInfo *t;
     int n = CkDynReducers(&t);
     for (int i = 0; i < n; i++) if (strcmp(t[i].name, name) == 0) return t[i].type;
     return -1;
   }

Contributing
------------

.. code-block:: c

   typedef struct CkDynRedData {
     const void *data;
     uint32_t    len;    /* bytes; 0 with data NULL for an empty contribution */
     int         type;   /* a reducer type, or CkDynForeignReducerType() */
   } CkDynRedData;

   void CkDynContribute(const CkDynTarget *me, const CkDynRedData *d, const CkDynCallback *cb);

``CkDynContribute`` contributes for the local group branch, nodegroup
branch or array element that ``me`` names (chares and main chares do not
contribute). The data is copied. Every member of the collection contributes
once per reduction, with the same reducer and callback; ``nop`` with no data
is a barrier.

.. _ckdyn-shaped:

Shaped contributions
~~~~~~~~~~~~~~~~~~~~

The built-in reducers combine flat arrays of elements. To get a result with
the contributors' array shape, contribute with the shape:

.. code-block:: c

   void CkDynContributeShaped(const CkDynTarget *me, const CkDynRedData *d, const CkDynCallback *cb,
                              int ndim, const int64_t *shape);
   void CkDynContributeSectionShaped(const CkDynTarget *me, const CkDynSectionID *sid, const CkDynRedData *d,
                                     const CkDynCallback *cb, int ndim, const int64_t *shape);

The elements are in C order, and the product of ``shape`` times the element
size of the reducer must equal ``d->len`` (CkDyn aborts otherwise). With
``ndim`` 2 or more and a built-in reducer on a numeric element kind, the
result's BUFFER slot (REDRESULT view, or the MCAST view of a SECTION
callback) has that ``ndim`` and shape: ``CkDynViewBufferPtr(v, 0, &shape)``
returns it. Every contribution to one reduction must have the same shape;
contributions of two shapes abort with "contributions to one reduction have
different shapes". With ``ndim`` 0 or 1, ``nop``, a binding-defined reducer,
or a reducer whose element kind is ``CKDYN_ELEM_BYTE`` or
``CKDYN_ELEM_NONE``, the shape is ignored and the call is
``CkDynContribute`` (or ``CkDynContributeSection``): the result slot is flat
(``ndim`` 0), as before. Both functions were added within API version 2;
the plain contribute functions are unchanged.

.. code-block:: c

   double a[2][3];                       /* this member's values */
   int64_t shape[2] = { 2, 3 };
   CkDynRedData d = { a, sizeof a, redType("sum_double") };
   CkDynContributeShaped(&o->t, &d, &cb, 2, shape);

   /* in the target entry: slot 0 has ndim 2 and shape (2, 3) */
   const int64_t *shp;
   const double *p = (const double *)CkDynViewBufferPtr(v, 0, &shp);

Results
-------

The callback receives the result:

- With a built-in reducer, a REDRESULT view: slot 0 is a BUFFER slot over
  the reduction data, with ``elem`` from the reducer table and ``len`` the
  data's byte count; ``ndim`` is 0, or the contributions' shape for
  :ref:`shaped contributions <ckdyn-shaped>`. When the reference number is set (a FUTURE callback, or
  a TARGET callback with ``refnum``), ``view->refnum`` holds it and a FUTURE
  slot ``(CkDynMyPe(), refnum)`` follows.
- With a binding-defined reducer, a DYN view of the payload the ``reduce``
  hook built; ``view->refnum`` carries the reference number, and there is
  no synthesized FUTURE slot.
- Through a SECTION callback, an MCAST view on every member of the section
  (:ref:`ckdyn-sections`).

A FUTURE callback hands the same view to the ``future`` hook. From the C++
test, an array reduction to element 0's entry:

.. code-block:: c++

   double x = myIndex(o) + 1.0;
   CkDynRedData d = { &x, sizeof x, redType("sum_double") };
   CkDynContribute(&o->t, &d, &cb);

   /* in the target entry */
   const double *p = (const double *)CkDynViewBufferPtr(v, 0, NULL);   /* slot 0 */

.. _ckdyn-foreign:

Binding-defined reducers
------------------------

CkDyn registers one Charm++ reducer for all binding-defined reducers;
``CkDynForeignReducerType()`` returns its type. A contribution to it is a
CkDyn payload (header, slots, data region, tail) whose header ``user_id`` is
the binding's own reducer id. Build it as a message, take its payload, and
free the message after contributing:

.. code-block:: c

   CkDynMsgBuilder *b = CkDynMsgBegin(1, 0, 0, 0, -1);
   CkDynMsgSetUserId(b, MY_REDUCER_SUM_I64);
   CkDynMsgAddI64(b, value);
   CkDynMsg *m = CkDynMsgFinish(b, 0);
   uint32_t len;
   const void *p = CkDynMsgPayload(m, &len);
   CkDynRedData d = { p, len, CkDynForeignReducerType() };
   CkDynContribute(&me, &d, &cb);
   CkDynMsgFree(m);

A delivered DYN message can also be contributed as is: its payload starts at
``view->hdr`` and is ``sizeof(CkDynHdr) + nslots * sizeof(CkDynSlot) +
data_len + tail_len`` bytes long.

The runtime combines contributions by calling
``reduce(global, reducer_id, contribs, n, result)`` with the ``n``
contributions as DYN views and an open builder. The hook adds the combined
value's slots (and tail) to ``result``; CkDyn writes the payload, with the
same ``user_id``, and passes it up the reduction tree, where it may be
combined again. The hook does not finish or abandon ``result``. It may set
header flags on ``result`` with ``CkDynMsgSetFlags``, for example a bit of
``CKDYN_MSG_BINDING_MASK`` that marks the message as a reduction result for
the receiving binding (charm4py does this). ``reduce`` runs on whichever PE
combines, with no object context, and can run concurrently with other hooks
of the process.

Everything the hook adds to ``result`` is copied before the add call
returns, BUFFER contents included: the hook may build its combined value
in a temporary and free it as soon as ``CkDynMsgAddBuffer`` returns. A
reduce result has the limits of any message (65535 slots, 2 GB with header
and envelope) and carries no NCPY or DEVICE slot; CkDyn aborts otherwise.
The C++ test binding's reducer, without the BUFFER slot it adds to check
this:

.. code-block:: c++

   static void hReduce(void *, uint32_t rid, const CkDynMsgView **contribs, int n,
                       CkDynMsgBuilder *result) {
     int64_t sum = 0;
     for (int i = 0; i < n; i++) sum += contribs[i]->slots[0].v.i64;
     CkDynMsgAddI64(result, sum);
   }

Zero-copy slots are not allowed in a contribution.

.. _ckdyn-sections:

Sections
========

A section is a subset of one array, of one group, or of several arrays (a
combined section), with multicasts and section reductions carried by
CkMulticast.

Creating and deleting
---------------------

.. code-block:: c

   void CkDynSectionCreateArray(int aid, int nelems, int ndims, const int *indices, int bfactor,
                                const CkDynCallback *done, CkDynSectionID *out);
   void CkDynSectionCreateGroup(int gid, int npes, const int *pes, int bfactor,
                                const CkDynCallback *done, CkDynSectionID *out);
   void CkDynSectionCombine(int nparts, const CkDynSectionID *parts, const CkDynCallback *done,
                            CkDynSectionID *out);
   void CkDynSectionDelete(const CkDynSectionID *sid);

Creation runs on the calling PE, which becomes the section's *root*.
``indices`` holds ``nelems`` indices of ``ndims`` components each; duplicate
members are removed. ``bfactor`` is the spanning tree's branching factor
(the C++ test passes 0, CkMulticast's default). ``out`` receives the id at
once, but the section is ready only when ``done`` fires: CkDyn multicasts
the id to every member, each member stores its section information (and the
``section_joined`` hook, if set, is called), and the members contribute an
empty reduction to ``done``. **Send to the section and contribute to it only
after ``done`` fires.**

``CkDynSectionCombine`` makes a section whose members are the union of the
parts, each member once. Parts may be array sections or combined sections,
from any root, all with one index dimension count; a group part aborts. A
combine takes each part's member list when the part's root receives the
request; a part that is itself a combined section not yet ready answers once
it is ready. Deleting such a part while a combine waits on it aborts, naming
the waiting combine: delete a part only after the combines that name it are
ready.

``CkDynSectionDelete`` deletes a section (from any PE; the request goes to
the root). Later use of the id aborts with a message about a stale
generation.

**Section operations from one PE are applied at the root in program order.**
A send (``CkDynSendSection``, also through a section callback), the member
query of a combine and a delete issued on one PE reach the root in any order
on the network, but the root applies them in the order that PE issued them,
so a send followed by a delete delivers the message. Operations from
different PEs have no order between them.

Section ids
-----------

.. code-block:: c

   typedef struct CkDynSectionID {
     int32_t  root_pe;    /* the PE that created the section */
     int32_t  idx;        /* index in the root's table */
     int32_t  coll_id;    /* array or group id (-1 for a combined section) */
     uint8_t  kind;       /* CKDYN_ARRAY, CKDYN_GROUP, or 0 for a combined section */
     uint8_t  reserved;   /* restarts before the id was made, modulo 256 */
     uint16_t gen;        /* generation */
   } CkDynSectionID;

The id is 16 bytes, travels in messages (``CkDynMsgAddSection``,
``CkDynViewSection``) and is valid on every PE. The generation makes a
deleted section's id distinct from a later section that reuses its table
slot; generation 0 is never valid. ``reserved`` holds the number of times
the program has been restarted from a checkpoint (modulo 256) when the id
was made. Sections are not checkpointed, and after ``+restart`` the first
sections a program creates get the same root, index and generation as
sections of the checkpointed run; every use of an id (send, contribute,
``CkDynSectionRedNo``, combine, delete) compares ``reserved`` with the
current count and aborts with "stale section id from before restart" when
they differ. A binding treats the id as 16 opaque bytes and copies all of
them.

Multicasts and section reductions
---------------------------------

.. code-block:: c

   CkDynMcastMsg *m = CkDynMsgFinishMcast(b, ep);
   CkDynSendSection(&sid, ep, m, NULL);

``CkDynSendSection`` sends from any PE (a send from another PE than the root
takes one extra hop to the root). Send options are ignored. Members receive
an MCAST view whose ``section`` field points at the id. Every member of a
combined section over arrays of different classes must have the entry: a
section message for an entry of another class than the receiving element
aborts.

.. code-block:: c

   void CkDynContributeSection(const CkDynTarget *me, const CkDynSectionID *sid,
                               const CkDynRedData *d, const CkDynCallback *cb);
   int  CkDynSectionRedNo(const CkDynTarget *me, const CkDynSectionID *sid);

A member contributes to a section reduction with ``CkDynContributeSection``;
built-in and binding-defined reducers both work. The member must be local
and must have received the creation multicast. ``CkDynSectionRedNo`` returns
the number of section reductions the member has contributed to, which every
member computes identically: a binding derives a collective future id from
it.

After ``CkDynSectionDelete`` the root multicasts the deletion to the members,
with the number of multicasts it sent to the section before it. A member
drops the section once that deletion and all those multicasts have reached
it; from then on ``CkDynContributeSection`` and ``CkDynSectionRedNo`` with
the id abort. The handler of a multicast sent before the deletion can still
use the id, whichever arrives first. A contribution made before the deletion
reaches the member is reduced as usual.

A **section callback** (``CKDYN_CB_SECTION``) multicasts a result to every
member of a section, as an MCAST view with ``refnum`` set. Unlike Charm++'s
``bcastSection`` callback it can be put into a message and pupped. A
built-in reduction result arrives as one BUFFER slot; a binding-defined
reducer's result keeps its payload; a SYS result (quiescence, checkpoint)
arrives empty.

A whole-collection reduction (``CkDynContribute`` from array elements,
group or nodegroup branches) may have a section callback, for example a
section of the same array: every contributor passes the same
``CkDynCallback``. CkDyn sends the contributions through an internal reducer
that carries the section, the entry and the reference number along with the
data and applies the reducer the binding gave (built-in or binding-defined),
so that every PE contributes with the same Charm++ callback; the result is
multicast from the section's root PE.

Migration, restart and restrictions
-----------------------------------

- A member that migrates keeps its section membership and its reduction
  numbers. After its next contribution, the next multicast reaches members
  point to point and CkMulticast rebuilds the spanning tree for the one
  after.
- Sections are not checkpointed: after ``+restart`` a binding recreates
  them. An id from before the restart, for example one held in an object's
  restored state, aborts on its next use (see `Section ids`_). The check
  repeats after 256 restarts.
- A deleted section's CkMulticast trees stay allocated. Deleting a section
  costs one multicast to its members.
- Section messages take no group dependency and no zero-copy slots.
- A group section must name a dynamic group; an empty member list aborts.
- Each PE keeps one relay entry per (section id, entry) pair used in a
  section callback created on it (``CkDynContribute`` needs none), plus one per distinct reference number of
  a quiescence or checkpoint section callback. They are never freed. With a
  16-bit reference number a PE can hold at most 65534.

.. _ckdyn-zerocopy:

Zero-copy and Devices
=====================

Host buffers: NCPY slots
------------------------

.. code-block:: c

   typedef void (*CkDynBufDone)(void *done_arg);
   void CkDynMsgAddNcpy(CkDynMsgBuilder *b, const CkDynBufDesc *d, CkDynBufDone done, void *arg);

An NCPY slot carries a host buffer that the runtime transfers without
copying it into the message: within a process the receiver's runtime copies
it once, across processes it is fetched with an RDMA get. ``d->ptr`` must
stay valid until ``done(arg)`` runs. ``done`` runs exactly once, from the
scheduler of the sending PE, after the data has left the source buffer, or
after ``CkDynMsgAbandon`` or ``CkDynMsgFree`` of an unsent message.

The receiver reads the data with ``CkDynViewBufferPtr``; the data lives as
long as the message, so a retain keeps it (:ref:`ckdyn-ownership`).

Restrictions: a message with zero-copy slots (header ``nbuf > 0``) is sent
point to point only (a send to a chare, a group or nodegroup branch, an
array element, a future, or through a TARGET callback to one element). A
broadcast, a section message, a multi-branch send, a constructor message, a
reduction contribution, or a broadcast or section callback aborts, naming
the slot. NCPY and DEVICE slots cannot share a message. A message holds at
most 4096 zero-copy slots. Sending such a message inline is not possible;
CkDyn drops the inline flag.

Posted buffers
--------------

.. code-block:: c

   void CkDynPostBuffer(const CkDynTarget *t, int ep, int slot, const CkDynBufDesc *d);
   void CkDynPostDeviceBuffer(const CkDynTarget *t, int ep, int slot, const CkDynDevDesc *d);
   int  CkDynUnpostBuffer(const CkDynTarget *t, int ep, int slot);
   #define CKDYN_POST_ARG 0x40000000

A post names, on the calling PE, the destination of slot ``slot`` of the
next message to the local object ``t``'s entry ``ep``. The posted length must
equal the slot's. One post per (object, entry, slot) at a time; posts are
dropped when the object migrates or is destroyed; nodegroups cannot post. A
host post may be made any time before the message is delivered to the
object, even after the send was issued: the runtime copies the slot from the
delivered message into the posted buffer (one copy on the receiving PE), and
``CkDynViewBufferPtr`` returns the posted buffer. A device post receives the
transfer directly.

``CkDynUnpostBuffer(t, ep, slot)`` withdraws the post for ``(t, ep, slot)``
if it is still outstanding and returns 1, or returns 0 when there is none. A
post is consumed only by a message whose slot is NCPY or DEVICE; a message
that carries that slot as a BUFFER or BYTES value leaves it in place, so a
binding that posted for a call whose argument then arrived by value
withdraws the post before posting again for the next call.

``slot`` in a post or unpost may be OR'd with ``CKDYN_POST_ARG``; it then
counts arguments, not slots: in a message whose header has
``CKDYN_MSG_BLOCK`` and whose slot 0 is a FUTURE slot, argument ``k`` is slot
``k + 1``, otherwise slot ``k``. The binding then need not know at post time
whether the sender will block on the call. An unpost names the slot in the
same form as its post.

Device buffers: DEVICE slots
----------------------------

.. code-block:: c

   typedef struct CkDynDevDesc {
     const void *ptr;      /* device memory, C-contiguous */
     uint64_t    len;      /* bytes */
     uint8_t     elem;
     uint8_t     ndim;
     uint16_t    tag;
     int         device;   /* device ordinal */
     uint64_t    stream;   /* the cudaStream_t / hipStream_t the data is valid on (0: default) */
     const int64_t *shape;
   } CkDynDevDesc;

   void CkDynMsgAddDevice(CkDynMsgBuilder *b, const CkDynDevDesc *d, CkDynBufDone done, void *arg);

A DEVICE slot is transferred device to device: through the stream within a
process, through the IPC buffer on one host, by RDMA otherwise. The rules of
NCPY slots apply, and ``done`` runs once the source may be reused. On the
receiver, ``CkDynViewBufferPtr`` returns device memory on the device named
by the slot's ``reserved`` field (the receiver's device). Unless the slot was
delivered into a posted buffer, that memory comes from the runtime's device
pool and belongs to the message: it is freed after the hook returns, or at
the last ``CkDynMsgRelease`` if the binding retained the message. A binding
that hands the data to user code either copies it or retains the message for
as long as the user holds it.

A device send needs to know the destination PE: a DEVICE slot to a singleton
chare created with ``CKDYN_PE_ANY`` that is not yet constructed aborts. For an
array element CkDyn uses the sending PE's last known location of the element
and chooses the transfer mode for that PE's process. If the element has moved
to another process by the time the message arrives (or the sender's location
was out of date), the receiver aborts ("Destination process does not match
the one the sender determined"); a move to another PE of the same process is
fine. A binding that sends DEVICE slots to array elements does so while
they do not migrate between processes (between load balancing steps, for
example).

``CkDynHasDeviceSupport()`` returns 1 in a CUDA or HIP build; every device
function aborts in other builds.

HAPI callbacks
--------------

``CkDynHapiAddCallback(stream, &cb)`` delivers an empty message to ``cb``,
from the calling PE's scheduler, when ``stream`` reaches this point. Quiescence
is not detected while such a callback is pending. A binding uses it to resume
a coroutine that waits for a kernel or a copy.

.. _ckdyn-devpersist:

Persistent device buffers
-------------------------

.. code-block:: c

   typedef struct CkDynDevPersist CkDynDevPersist;   /* opaque, valid on the PE that opened it */
   CkDynDevPersist *CkDynDevPersistOpen(const CkDynDevDesc *d, const CkDynCallback *done);
   void   CkDynDevPersistClose(CkDynDevPersist *p);
   size_t CkDynDevPersistHandleSize(void);
   void   CkDynDevPersistHandle(const CkDynDevPersist *p, void *out);
   void   CkDynDevPersistGet(CkDynDevPersist *dst, const void *src_handle, uint32_t refnum);
   void   CkDynDevPersistPut(const void *dst_handle, CkDynDevPersist *src, uint32_t refnum);
   int    CkDynDevPersistMode(const CkDynDevPersist *p, const void *handle);

A persistent device buffer is a device buffer opened once on one PE and then
used for any number of one-sided transfers, without a message or a memory
registration per transfer. ``CkDynDevPersistOpen`` opens the buffer ``d``
names (``ptr``, ``len``, ``device`` and ``stream`` are used; the stream
orders each transfer after earlier work on it); the memory is registered with
the network the first time a transfer needs RDMA, so a run in one process, or
one that uses only IPC, registers nothing. ``CkDynDevPersistHandle`` writes the
buffer's handle, ``CkDynDevPersistHandleSize()`` bytes, which the binding
sends to other PEs in an ordinary message (a BYTES slot, for example).

``CkDynDevPersistGet(dst, src_handle, refnum)`` copies the source buffer
named by the handle into ``dst``, which must be open on the calling PE.
``CkDynDevPersistPut(dst_handle, src, refnum)`` copies the local open buffer
``src`` into the buffer named by the handle. The transfer moves the source's
length; the destination must be at least as long. Either side may be on any
PE. ``done``, the callback given to ``CkDynDevPersistOpen`` of the local
buffer, receives an empty message on the issuing PE once per completed
transfer: a get is complete when the bytes are in ``dst``, a put when they
are in the remote buffer. For a FUTURE callback a non-zero ``refnum`` of the
call is the future id; for the other kinds the message carries the
callback's own reference number. The caller keeps both buffers valid and
does not read the destination before ``done``.

The transfer mode is chosen per pair of buffers, and
``CkDynDevPersistMode`` returns it: 0, a device copy when both buffers are in
one process; 1, CUDA or HIP IPC when they are in different processes on one
host and the peer's IPC handle opens in this process (``+gpushm`` is not
needed; it concerns DEVICE slots); 2, RDMA on the registered memory
otherwise. Between hosts a put is carried out as a get issued by the
destination's PE, so that ``done`` means the data has arrived. RDMA needs a
machine layer that can register device memory; the underlying
``CkDevicePersistent`` aborts an RDMA transfer whose buffer cannot be
registered, or whose buffer's owner has closed it. With ``+gpupersistverbose``
the first use of each mode and operation in a process prints one line
(``[pe] CkDevicePersistent> get: ...``).

The local buffer keeps what a transfer with a peer's handle set up: its
reference on this process's IPC mapping of the peer's allocation and the
peer's RDMA registration, so that later transfers with that handle reuse them.
A handle of the same peer buffer after the peer reopened it replaces the
earlier one. ``CkDynDevPersistClose`` releases those, deregisters the local
memory if a transfer registered it, and frees the local record; it aborts
while a transfer of that buffer is outstanding. A handle stays usable only
while its buffer is open, and only in the run that opened it (a handle saved
in a checkpoint does not name a buffer after a restart). Persistent buffers
are not migrated or checkpointed. Every function aborts in a build without
CUDA or HIP.
charm4py binds them as ``charm.DevicePersistent``.

.. _ckdyn-coro:

Coroutines, Tracing and Load Balancing
======================================

Coroutines
----------

A binding that runs entry methods on coroutines (charm4py's greenlets) faces
two attribution problems once a coroutine suspends on a future and is
resumed later from the ``future`` hook: its time is charged to CkDyn's
manager group, which the load balancer cannot move, and tracing shows
nothing for it. The binding therefore brackets every run of a coroutine:

.. code-block:: c

   CkDynCoroResume(&target, ep, coro_id);   /* right after switching into the coroutine */
   /* ... the coroutine runs until it waits again or finishes ... */
   CkDynCoroSuspend(&target);               /* right before switching out */

``CkDynCoroResume`` pushes the target's object on the running-object stack,
which charges the time to it, and, when tracing is on and ``ep`` is traced,
begins an execute event for ``ep``. For an array element the event carries
the element's index; for other kinds the object id is made from
``coro_id``. ``CkDynCoroSuspend`` ends both. Call the pair outside any
traced entry (from the ``future`` hook, which runs in an untraced entry, or
around each element of a ``deliver_many`` call) so that events do not nest.
charm4py's switch function:

.. code-block:: c++

   CkDynCoroResume(&c.t, c.ep, c.id);
   r = PyGreenlet_Switch(w, argt, nullptr);
   CkDynCoroSuspend(&c.t);

``CkDynCallstackPush(&t)`` and ``CkDynCallstackPop(&t)`` do the
running-object half alone: load balancer timing and, in a CUDA build with
load balancing, the attribution of GPU kernels launched in between to that
object. charm4py uses them for a coroutine entry inside ``deliver_many``,
whose trace events come from its own switch. Each call changes the stack by
exactly one entry: Push of an object that is not local pushes nothing that
the load balancer charges, and Pop after the object destroyed itself in
between (``CkDynArrayDestroyElement``) still pops its entry, so the running
object is again the one before Push.

Load balancing
--------------

- ``CkDynAtSync``, ``resume_from_sync``, ``CkDynMigrateMe`` and
  ``CkDynSetMigratable``: :ref:`ckdyn-pup`.
- ``CkDynLBInstrument(on)`` turns load instrumentation on or off.
- ``CkDynSetObjLoad(&t, secs)`` sets the load of a local array element, for
  a binding that measures time itself.
- ``CkDynHasLB()`` returns 1 when the build has load balancing.

The usual ``+balancer`` options select the strategy.

Tracing
-------

Each dynamic class is its own Charm++ chare type and each entry its own
entry index, registered with the binding's names. Projections therefore
shows ``Class.method`` names, and for array elements the element index, as
for a charmxi program. The usual tracing options (``+traceroot``,
``+logsize``, ``+traceoff``) apply to a build with tracing enabled.

- ``CKDYN_EP_NOTRACE`` on an entry leaves its executions out.
  ``CKDYN_SEND_NOTRACE`` on a send does not: it only keeps the message out
  of the load balancer's communication statistics.
- ``CkDynTraceBeginEntry(ep)`` / ``CkDynTraceEndEntry(ep)`` bracket a region
  as an execute event of ``ep`` with no object.
- User events: ``CkDynTraceRegisterUserEvent(desc, id)`` returns the event
  id; ``CkDynTraceBeginUserBracketEvent(id)`` and
  ``CkDynTraceEndUserBracketEvent(id)`` bracket a region.
- Array broadcasts through ``deliver_many`` need the per-element bracket
  above to appear per element.
- Projections reads an array element's index fields from the dimension
  count of its class in the ``.sts`` file. A class registered with
  ``ndims`` 0 (any dimension count) takes the count of the first element
  constructed in each process, and only PE 0's process writes the ``.sts``:
  if the class's arrays differ in dimension count, or PE 0's process
  constructed none of its elements before the file was written, Projections
  shows the indices of some elements with the wrong number of fields. The
  count is recorded by every PE thread of a process without
  synchronization; the threads write the same value for one array. Register
  a class with a fixed ``ndims`` when its indices matter in Projections.

Other runtime services
----------------------

``ckdyn.h`` wraps the runtime calls a binding needs besides CkDyn's own, so
that it includes no Charm++ header: PE and node numbers
(``CkDynMyPe``, ``CkDynNumPes``, ``CkDynMyNode``, ``CkDynNumNodes``,
``CkDynMyRank``, ``CkDynNodeSize``, ``CkDynNodeFirst``, ``CkDynNodeOf``),
physical node topology, spanning tree edges (``CkDynTopoTreeEdges``),
``CkDynExit``, ``CkDynAbort``, ``CkDynPrintf``, ``CkDynWallTimer``, timed
callbacks (``CkDynCallAfter``), and CCS handlers (``CkDynCcsRegister`` and
the reply functions). They are listed in :ref:`ckdyn-api`.

.. _ckdyn-walkthrough:

Writing a Binding
=================

This chapter builds a minimal binding from the C++ test binding,
``tests/charm++/ckdyn/cbinding.C``: one array class with two methods, an
array, a send, a broadcast and a reduction.

A class as a table
------------------

The binding's own class record holds a method table whose element 0 is the
constructor, and receives the indices from registration:

.. code-block:: c++

   typedef int64_t (*CbMethodFn)(Obj *o, const CkDynMsgView *v);
   struct CbMethod { const char *name; uint32_t flags; CbMethodFn fn; };
   struct CbClass {
     const char *name; CkDynKind kind; int ndims;
     std::vector<CbMethod> methods;     /* methods[0]: constructor */
     int chareIdx, firstEp;
     int ep(int method) const { return firstEp + method; }
   };

   static int64_t hiCtor(Obj *, const CkDynMsgView *) { return 0; }
   static int64_t hiAdd(Obj *o, const CkDynMsgView *v) {     /* add(value, cb) */
     int r = (int)v->slots[0].v.i64;
     CkDynCallback cb;
     CkDynViewCallback(v, 1, &cb);
     CkDynRedData d = { &r, sizeof r, redType("sum_int") };
     CkDynContribute(&o->t, &d, &cb);
     return 0;
   }
   static int64_t hiPrint(Obj *o, const CkDynMsgView *v) {   /* print(x) */
     CkDynPrintf("got a value\n");
     return 0;
   }

   CbClass Hi = { "Hi", CKDYN_ARRAY, 1, {
     { "Hi.Hi",    0, hiCtor },
     { "Hi.add",   CKDYN_EP_REDNTARGET, hiAdd },
     { "Hi.print", 0, hiPrint },
   } };

Registration
------------

Registration turns the table into descriptors. ``user_id`` is the method's
position, so delivery can index the table; ``binding_data`` is the class
record:

.. code-block:: c++

   static void registerClass(CbClass *c) {
     int n = (int)c->methods.size();
     CkDynEntryDesc *e = new CkDynEntryDesc[n];
     for (int i = 0; i < n; i++)
       e[i] = { c->methods[i].name, c->methods[i].flags, (uint32_t)i };
     CkDynClassDesc d;
     memset(&d, 0, sizeof d);
     d.name = c->name; d.kind = c->kind; d.ndims = c->ndims;
     d.num_entries = n; d.entries = e;
     d.binding_data = c;
     c->chareIdx = CkDynRegisterClass(&d, &c->firstEp);
   }

   /* an initnode in the program's .ci file */
   void registerAll(void) {
     CkDynInit(cbHooks());
     registerClass(&Hi);
   }

The hooks
---------

An object is a struct holding its class and its target. ``construct``
makes one and runs the constructor method; ``deliver`` checks that the entry
belongs to the class and runs the method; ``destruct`` frees it:

.. code-block:: c++

   struct Obj { const CbClass *cls; CkDynTarget t; /* state */ };

   static void *hConstruct(void *bd, const CkDynTarget *t, const CkDynMsgView *v) {
     const CbClass *c = (const CbClass *)bd;
     Obj *o = new Obj;
     o->cls = c;
     o->t = *t;
     c->methods[0].fn(o, v);
     return o;
   }

   static void hDeliver(void *bd, void *h, int ep, uint32_t user_id, uint32_t,
                        const CkDynMsgView *v) {
     const CbClass *c = (const CbClass *)bd;
     if (user_id >= c->methods.size() || ep != c->ep((int)user_id))
       CkDynAbort("delivery does not match the class table");
     c->methods[user_id].fn((Obj *)h, v);
   }

   static void hDestruct(void *, void *h) { delete (Obj *)h; }

   const CkDynHooks *cbHooks(void) {
     static CkDynHooks g;
     memset(&g, 0, sizeof g);
     g.version = CKDYN_API_VERSION;
     g.size = sizeof(CkDynHooks);
     g.construct = hConstruct;
     g.destruct = hDestruct;
     g.deliver = hDeliver;
     g.future = hFuture;          /* required: CkDynCreateManagers aborts without it */
     g.pack_size = hPackSize;     /* needed once elements migrate */
     g.pack = hPack;
     g.unpack = hUnpack;
     g.error = hError;
     return &g;
   }

The ``future`` hook looks the id up in a per-PE table of callbacks and runs
the one it finds; ``cbNewFuture(fn)`` allocates an id and stores ``fn``:

.. code-block:: c++

   static void hFuture(void *, int pe, uint32_t id, const CkDynMsgView *v) {
     PeState *s = CkpvAccess(cbPe);
     auto it = s->futures.find(id);
     CbFutureFn fn = std::move(it->second);
     s->futures.erase(it);
     fn(v);
   }

Create, send, broadcast, reduce
-------------------------------

The program's main chare calls ``CkDynCreateManagers()`` and then uses the
class. ``MsgB`` is a small wrapper over the builder functions:

.. code-block:: c++

   CkDynCreateManagers();
   int dims[1] = { 8 };              /* aid below is a global (a readonly in the .ci file) */
   CkDynArrayOpts ao = { 1, dims, -1, 0, 0, -1 };
   uint32_t created = cbNewFuture([](const CkDynMsgView *) {
     /* every element exists: send to element 3 */
     int i = 3;
     CkDynSendArray(aid, 1, &i, Hi.ep(2), MsgB().i64(42).finish(0), nullptr);
     /* broadcast add(1, cb); the sum arrives at a future on PE 0 */
     uint32_t f = cbNewFuture([](const CkDynMsgView *v) {
       const int *sum = (const int *)CkDynViewBufferPtr(v, 0, nullptr);
       CkDynPrintf(*sum == 8 ? "sum ok\n" : "wrong sum\n");
       CkDynExit(0);
     });
     CkDynSendArray(aid, 1, nullptr, Hi.ep(1), MsgB().i64(1).cb(cbFutureCb(0, f)).finish(0), nullptr);
   });
   aid = CkDynCreateArray(Hi.chareIdx,
       MsgB().flags(CKDYN_MSG_CREATION).fut(0, created).finish(Hi.ep(0)), &ao);

Run it as any Reconverse program, for example ``./hi +pe 4``.

A shared-library binding does the same from ``register_classes`` and from its
``mainchare`` hook, with ``CkDynInit`` and ``CkDynStart`` in its start-up
function.

Checklist
---------

The rules a binding must follow, collected from the chapters above:

1. Check ``CkDynApiVersion()`` and set ``version`` and ``size`` in every hook
   table. Check ``CkDynRefnumBytes()`` if future ids may exceed 65534.
2. Register the same classes in the same order on every process, from rank 0.
3. Keep the hook table and what ``binding_data`` points to valid for the
   program's lifetime (the descriptor, the entry table and the name strings
   are copied at registration).
4. Never return NULL from ``construct``, ``mainchare`` or ``unpack``.
5. Abandon a builder on every error path between ``CkDynMsgBegin`` and
   ``CkDynMsgFinish``; free a finished message that will not be sent with
   ``CkDynMsgFree``. Never touch a message after passing it to a send,
   creation, ``CkDynFutureSet`` or ``CkDynCallbackSend``.
6. Treat every view and every pointer read from it as valid only during the
   hook. Call ``CkDynMsgRetain`` before keeping one, and
   ``CkDynMsgRelease(msg)`` once per retain.
7. Keep NCPY and DEVICE source buffers valid until their ``done`` function
   runs; expect ``done`` exactly once, from the sending PE's scheduler, also
   for abandoned or freed messages.
8. Return the same size from both ``pack_size`` passes and write exactly that
   many bytes in ``pack``; return ``CKDYN_PACK_REFUSED`` for an object that
   cannot be packed.
9. Keep per-PE state per thread: hooks of different PEs of one process may
   run at the same time, and ``reduce`` runs on any PE.
10. Expect hooks to nest: an inline send, or a send to a local object with an
    inline entry, runs ``deliver`` inside the caller's hook; ``reduce`` can
    run while the binding is building a message.
11. Catch the binding language's exceptions inside each hook. Report errors
    with ``CkDynAbort``; implement ``error`` to print CkDyn's errors in the
    binding's terms (CkDyn aborts after it returns).
12. Wait for a section's ``done`` callback before sending to it or
    contributing to it; recreate sections after a restart.
13. Bracket every coroutine run with ``CkDynCoroResume`` /
    ``CkDynCoroSuspend``, and every element of a ``deliver_many`` call if
    per-element load and tracing matter.
14. A binding for an interpreted language releases its interpreter lock
    around ``CkDynStart`` and takes it in every hook.

.. _ckdyn-api:

API Reference
=============

Every function of ``ckdyn.h``, grouped as in the header. Types and constants
are described in the chapters linked from each group. Unless stated
otherwise, a function that names a local object by target aborts when the
object is not on the calling PE.

Version and start-up
--------------------

``int CkDynApiVersion(void)``
   The ``CKDYN_API_VERSION`` the library was built with.

``int CkDynRefnumBytes(void)``
   The width of the envelope reference number in bytes (2 by default, 4 with
   ``--with-refnum-type="unsigned int"``); bounds future ids.

``int CkDynHasDeviceSupport(void)``
   1 in a CUDA or HIP build, else 0.

``void CkDynInit(const CkDynHooks *hooks)``
   Copies and checks the hook table; registers the readonly hooks. Before
   ``CkDynStart`` (shared library) or from an ``initnode`` (C++ program).

``void CkDynStart(int argc, char **argv)``
   Starts the runtime; never returns. Shared-library bindings only, after
   ``CkDynInit``.

``void CkDynCreateManagers(void)``
   Creates CkDyn's manager group and multicast manager. Called by a dynamic
   main chare itself; an ordinary C++ main chare calls it from its
   constructor. Later calls do nothing.

Registration
------------

See :ref:`ckdyn-registration`.

``int CkDynRegisterClass(const CkDynClassDesc *desc, int *first_ep)``
   Registers a class on rank 0 before start-up completes; returns the chare
   index and stores the constructor's entry index in ``*first_ep``. Aborts
   after start-up (use ``CkDynRegisterClassLate``), on a NULL name, and when
   an entry index would exceed 65535 (the message header keeps 16 bits).

``void CkDynAllowLateRegistration(void)``
   Enables ``CkDynRegisterClassLate``.

``int CkDynRegisterClassLate(const CkDynClassDesc *desc, int *first_ep)``
   Registers a class after start-up; one PE per process only.

``const CkDynClassDesc *CkDynClassOf(int chare_idx)``
   The descriptor of a dynamic class, or NULL.

Message building
----------------

See :ref:`ckdyn-messages`. Each ``Add`` function appends one slot.

``CkDynMsgBuilder *CkDynMsgBegin(int nslots_hint, size_t data_hint, size_t tail_hint, int prio_bits, int group_dep)``
   Begins a message with ``prio_bits`` priority bits and an optional group
   dependency (-1: none).

``void CkDynMsgSetFlags(CkDynMsgBuilder *b, uint16_t flags)``
   Sets the header flags (``CKDYN_MSG_*``).

``void CkDynMsgSetUserId(CkDynMsgBuilder *b, uint32_t user_id)``
   Sets the header's ``user_id``: the reducer id of a foreign-reducer
   contribution.

``void CkDynMsgAddNone(CkDynMsgBuilder *b)``
   A NONE slot.

``void CkDynMsgAddBool(CkDynMsgBuilder *b, int v)``, ``void CkDynMsgAddI64(CkDynMsgBuilder *b, int64_t v)``, ``void CkDynMsgAddU64(CkDynMsgBuilder *b, uint64_t v)``, ``void CkDynMsgAddF64(CkDynMsgBuilder *b, double v)``
   Scalar slots, stored inline in the slot.

``void CkDynMsgAddBytes(CkDynMsgBuilder *b, const void *p, uint32_t len, uint16_t tag)``
   A BYTES slot; the bytes are copied now.

``void CkDynMsgAddBuffer(CkDynMsgBuilder *b, const CkDynBufDesc *d)``
   A BUFFER slot; the shape is copied now, the contents at ``Finish``, so
   ``d->ptr`` stays valid until ``Finish`` or ``Abandon``. In the ``reduce``
   hook's result builder the contents are copied now.

``void CkDynMsgAddFuture(CkDynMsgBuilder *b, int pe, uint32_t id)``
   A FUTURE slot.

``void CkDynMsgAddCallback(CkDynMsgBuilder *b, const CkDynCallback *cb)``
   A CALLBACK slot.

``void CkDynMsgAddSection(CkDynMsgBuilder *b, const CkDynSectionID *sid)``
   A SECTION slot.

``void CkDynMsgAddProxy(CkDynMsgBuilder *b, const CkDynTarget *t)``
   A PROXY slot.

``void *CkDynMsgReserveTail(CkDynMsgBuilder *b, uint32_t len)``
   Reserves the tail; the binding writes ``len`` bytes at the returned
   pointer.

``CkDynMsg *CkDynMsgFinish(CkDynMsgBuilder *b, int ep)``
   Allocates the message and ends the builder. ``ep`` goes in the header;
   constructor messages need the class's ``first_ep``.

``CkDynMcastMsg *CkDynMsgFinishMcast(CkDynMsgBuilder *b, int ep)``
   The same payload as a section message, for ``CkDynSendSection``.

``void CkDynMsgAbandon(CkDynMsgBuilder *b)``
   Drops a builder; runs the ``done`` functions of its zero-copy slots.

``void *CkDynMsgPriorityPtr(CkDynMsg *m)``
   The reserved priority bits of a finished message, to fill in.

``void CkDynMsgSetQueueing(CkDynMsg *m, int queueing)``
   Sets the queueing strategy (a ``CKDYN_QUEUEING_*`` value).

``void CkDynMsgFree(CkDynMsg *m)``
   Frees a finished message that will not be sent.

``const void *CkDynMsgPayload(CkDynMsg *m, uint32_t *len)``
   The payload bytes (header to tail), for a foreign-reducer contribution.

Reading messages
----------------

``const void *CkDynViewBufferPtr(const CkDynMsgView *v, int slot, const int64_t **shape_out)``
   The data of a BUFFER, NCPY or DEVICE slot (device memory for DEVICE);
   ``*shape_out``, if not NULL, gets the ``ndim`` shape values.

``const void *CkDynViewBytesPtr(const CkDynMsgView *v, int slot)``
   The bytes of a BYTES slot.

``void CkDynViewCallback(const CkDynMsgView *v, int slot, CkDynCallback *out)``, ``void CkDynViewSection(const CkDynMsgView *v, int slot, CkDynSectionID *out)``, ``void CkDynViewProxy(const CkDynMsgView *v, int slot, CkDynTarget *out)``
   Copy a CALLBACK, SECTION or PROXY slot out.

``void CkDynMsgRetain(const CkDynMsgView *v)``
   Keeps the delivered message alive after the hook.

``void CkDynMsgRelease(void *msg)``
   Drops one retain of ``view->msg``.

Sends
-----

See :ref:`ckdyn-messages`. Each takes ownership of the message; ``o`` may be
NULL.

``void CkDynSendChare(const CkDynTarget *t, int ep, CkDynMsg *m, const CkDynSendOpts *o)``
   To a singleton chare or main chare.

``void CkDynSendGroup(int gid, int pe, int ep, CkDynMsg *m, const CkDynSendOpts *o)``
   To a group or array map branch; ``pe`` -1 broadcasts.

``void CkDynSendGroupMulti(int gid, int npes, const int *pes, int ep, CkDynMsg *m, const CkDynSendOpts *o)``
   To the branches on a PE list.

``void CkDynSendNodeGroup(int gid, int node, int ep, CkDynMsg *m, const CkDynSendOpts *o)``
   To a nodegroup branch; ``node`` -1 broadcasts.

``void CkDynSendArray(int aid, int ndims, const int *index, int ep, CkDynMsg *m, const CkDynSendOpts *o)``
   To an array element; ``index`` NULL broadcasts.

``void CkDynSendSection(const CkDynSectionID *sid, int ep, CkDynMcastMsg *m, const CkDynSendOpts *o)``
   Multicast to a ready section; ``o`` is ignored.

``void CkDynSendTarget(const CkDynTarget *t, int ep, CkDynMsg *m, const CkDynSendOpts *o)``
   To whatever object the target names.

Creation
--------

See :ref:`ckdyn-objects`. Each takes ownership of the constructor message.

``void CkDynCreateChare(int chare_idx, int on_pe, CkDynMsg *ctor, CkDynTarget *out)``
   Creates a singleton chare on ``on_pe`` or ``CKDYN_PE_ANY``; ``*out`` is
   usable at once.

``int CkDynCreateGroup(int chare_idx, CkDynMsg *ctor)``
   Creates a group or array map; returns its id. Constrained with
   ``CKDYN_MSG_CONSTRAINED``.

``int CkDynCreateNodeGroup(int chare_idx, CkDynMsg *ctor)``
   Creates a nodegroup; returns its id.

``int CkDynCreateArray(int chare_idx, CkDynMsg *ctor, const CkDynArrayOpts *o)``
   Creates an array, full or empty; returns its id.

``void CkDynArrayInsert(int aid, int ndims, const int *index, int on_pe, CkDynMsg *ctor)``
   Inserts one element; ``ctor`` finished with the array class's
   ``first_ep``.

``void CkDynArrayDoneInserting(int aid)``
   Ends insertion.

``void CkDynArrayDestroyElement(int aid, int ndims, const int *index)``
   Destroys a local element.

Futures and callbacks
---------------------

See :ref:`ckdyn-futures`.

``void CkDynFutureSet(int pe, uint32_t id, CkDynMsg *value)``
   Sends ``value`` to the binding future ``(pe, id)``; the ``future`` hook
   runs on ``pe``.

``void CkDynCallbackSend(const CkDynCallback *cb, CkDynMsg *m)``
   Sends a message through a callback.

``int CkDynGetReductionNumber(const CkDynTarget *t)``
   The reduction number of a local group or nodegroup branch, or array
   element. An array map has none: the call aborts.

Reductions
----------

See :ref:`ckdyn-reductions`.

``int CkDynReducers(const CkDynReducerInfo **out)``
   The built-in reducer table; returns its length.

``int CkDynForeignReducerType(void)``
   The reducer type for binding-defined reducers.

``void CkDynContribute(const CkDynTarget *me, const CkDynRedData *d, const CkDynCallback *cb)``
   Contributes for a local group, nodegroup or array element.

Sections
--------

See :ref:`ckdyn-sections`.

``void CkDynSectionCreateArray(int aid, int nelems, int ndims, const int *indices, int bfactor, const CkDynCallback *done, CkDynSectionID *out)``
   Creates an array section rooted on the calling PE.

``void CkDynSectionCreateGroup(int gid, int npes, const int *pes, int bfactor, const CkDynCallback *done, CkDynSectionID *out)``
   Creates a section of a dynamic group.

``void CkDynSectionCombine(int nparts, const CkDynSectionID *parts, const CkDynCallback *done, CkDynSectionID *out)``
   The union of array and combined sections.

``void CkDynSectionDelete(const CkDynSectionID *sid)``
   Deletes a section.

``int CkDynSectionRedNo(const CkDynTarget *me, const CkDynSectionID *sid)``
   The member's section reduction count.

``void CkDynContributeSection(const CkDynTarget *me, const CkDynSectionID *sid, const CkDynRedData *d, const CkDynCallback *cb)``
   Contributes to a section reduction.

Checkpoint, quiescence, load balancing, tracing
-----------------------------------------------

See :ref:`ckdyn-pup` and :ref:`ckdyn-coro`.

``void CkDynCheckpoint(const char *dirname, const CkDynCallback *done)``
   Writes a checkpoint; ``done`` fires when written and after a restart.

``void CkDynStartQD(const CkDynCallback *cb)``
   Starts quiescence detection.

``void CkDynAtSync(const CkDynTarget *t)``
   ``AtSync`` for a local array element created with
   ``CKDYN_MSG_USESATSYNC``.

``void CkDynSetMigratable(const CkDynTarget *t, int migratable)``
   Allows or forbids migration of a local array element.

``void CkDynMigrateMe(const CkDynTarget *t, int to_pe)``
   Migrates a local array element.

``void CkDynLBInstrument(int on)``
   Turns load balancer instrumentation on or off.

``void CkDynSetObjLoad(const CkDynTarget *t, double secs)``
   Sets a local array element's load.

``void CkDynCallstackPush(const CkDynTarget *t)``, ``void CkDynCallstackPop(const CkDynTarget *t)``
   Push or pop a local object on the running-object stack. Always paired:
   Pop pops one entry even when the object was destroyed after Push.

``void CkDynCoroResume(const CkDynTarget *t, int ep, uint64_t coro_id)``, ``void CkDynCoroSuspend(const CkDynTarget *t)``
   Bracket one run of a coroutine: running object and trace event.

``int CkDynTraceRegisterUserEvent(const char *desc, int id)``
   Registers a user event; returns its id.

``void CkDynTraceBeginUserBracketEvent(int id)``, ``void CkDynTraceEndUserBracketEvent(int id)``
   Bracket a user event.

``void CkDynTraceBeginEntry(int ep)``, ``void CkDynTraceEndEntry(int ep)``
   Bracket a region as an execute event of ``ep``.

Zero-copy and devices
---------------------

See :ref:`ckdyn-zerocopy`.

``void CkDynMsgAddNcpy(CkDynMsgBuilder *b, const CkDynBufDesc *d, CkDynBufDone done, void *arg)``
   A host buffer transferred without a copy into the message.

``void CkDynMsgAddDevice(CkDynMsgBuilder *b, const CkDynDevDesc *d, CkDynBufDone done, void *arg)``
   A device buffer transferred device to device.

``void CkDynPostBuffer(const CkDynTarget *t, int ep, int slot, const CkDynBufDesc *d)``
   Posts a host destination for a slot of the next message to a local
   object's entry.

``void CkDynPostDeviceBuffer(const CkDynTarget *t, int ep, int slot, const CkDynDevDesc *d)``
   Posts a device destination.

``int CkDynUnpostBuffer(const CkDynTarget *t, int ep, int slot)``
   Withdraws an outstanding post; returns 1 if there was one. ``slot`` of a
   post or unpost may carry ``CKDYN_POST_ARG`` to count arguments.

``CkDynDevPersist *CkDynDevPersistOpen(const CkDynDevDesc *d, const CkDynCallback *done)``, ``CkDynDevPersistClose``, ``CkDynDevPersistHandleSize``, ``CkDynDevPersistHandle``, ``CkDynDevPersistGet``, ``CkDynDevPersistPut``, ``CkDynDevPersistMode``
   Persistent device buffers: :ref:`ckdyn-devpersist`.

``void CkDynHapiAddCallback(uint64_t stream, const CkDynCallback *cb)``
   Delivers an empty message to ``cb`` when the stream reaches this point.

Runtime wrappers
----------------

``int CkDynMyPe(void)``, ``int CkDynNumPes(void)``, ``int CkDynMyNode(void)``, ``int CkDynNumNodes(void)``, ``int CkDynMyRank(void)``
   This PE, the PE count, this process's node, the node count, this PE's
   rank in its process.

``int CkDynNodeSize(int node)``, ``int CkDynNodeFirst(int node)``, ``int CkDynNodeOf(int pe)``
   PEs of a node, its first PE, the node of a PE.

``void CkDynExit(int code)``
   Ends the program with an exit code.

``void CkDynAbort(const char *msg)``
   Aborts with a message.

``void CkDynPrintf(const char *text)``
   Prints ``text`` through the runtime (no format processing).

``double CkDynWallTimer(void)``
   Wall-clock seconds.

``int CkDynNumPhysicalNodes(void)``, ``int CkDynPhysicalNodeID(int pe)``, ``int CkDynPhysicalRank(int pe)``, ``int CkDynNumPesOnPhysicalNode(int node)``
   Physical node topology.

``int CkDynGetPesOnPhysicalNode(int node, int *pes, int max)``
   Fills up to ``max`` PEs of a physical node; returns their count.

``int CkDynTopoTreeEdges(int pe, int root, const int *pes, int npes, int bfactor, int *parent, int *children, int max)``
   ``pe``'s parent and up to ``max`` children in the topology-aware spanning
   tree over ``pes`` (NULL: all PEs) rooted at ``root``; returns the child
   count.

``void CkDynCallAfter(void (*fn)(void *arg, double now), void *arg, double msecs)``
   Calls ``fn`` from the scheduler after ``msecs`` milliseconds.

``int CkDynIsRestarting(void)``
   1 in a program started with ``+restart``, from the start of the restart to
   the end of the run (so a ``done`` callback delivered by the restart can
   tell itself from one delivered by a later checkpoint).

``int CkDynHasLB(void)``
   1 when the build has load balancing.

CCS
---

``void CkDynCcsRegister(const char *name, CkDynCcsHandler fn)``
   Registers a CCS handler; ``fn(global, name, data, len)`` runs for each
   request.

``int CkDynCcsIsRemoteRequest(void)``
   1 inside a handler serving a remote request.

``void CkDynCcsSendReply(int len, const void *data)``
   Replies to the current request.

``void *CkDynCcsDelayReply(void)``, ``void CkDynCcsSendDelayedReply(void *token, int len, const void *data)``
   Defer the reply and send it later with the token.

Further Reading
===============

- ``doc/internals/ckdyn.md`` in the Charm++ repository: how CkDyn implements
  each feature (registration tables, the payload layout, delivery paths,
  the section protocol, the zero-copy prefix, pup), with the source files.
- ``src/ck-core/ckdyn.h``: the API, with comments; the authority where this
  manual and the header differ.
- ``tests/charm++/ckdyn/``: the C++ test binding (``cbinding.h``,
  ``cbinding.C``) and the acceptance test (``ckdyntest.C``), which uses
  every feature.
- The charm4py repository: ``src/ckdyn_py.cpp``, the Python binding, and the
  design documents under ``docs/design/``: ``DESIGN.md`` (section 3 gives
  the reasons for CkDyn's choices), ``SPEC-3.md`` (zero-copy and devices),
  ``SPEC-4b.md`` (sections and constrained groups in the binding) and
  ``SPEC-7-device.md`` (persistent device buffers, planned).
