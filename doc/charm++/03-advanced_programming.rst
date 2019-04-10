Advanced Programming Techniques
===============================

Optimizing Entry Method Invocation
----------------------------------

.. _messages:

Messages
~~~~~~~~

Although Charm++ supports automated parameter marshalling for entry
methods, you can also manually handle the process of packing and
unpacking parameters by using messages. A message encapsulates all the
parameters sent to an entry method. Since the parameters are already
encapsulated, sending messages is often more efficient than parameter
marshalling, and can help to avoid unnecessary copying. Moreover, assume
that the receiver is unable to process the contents of the message at
the time that it receives it. For example, consider a tiled matrix
multiplication program, wherein each chare receives an :math:`A`-tile
and a :math:`B`-tile before computing a partial result for
:math:`C = A \times B`. If we were using parameter marshalled entry
methods, a chare would have to copy the first tile it received, in order
to save it for when it has both the tiles it needs. Then, upon receiving
the second tile, the chare would use the second tile and the first
(saved) tile to compute a partial result. However, using messages, we
would just save a *pointer* to the message encapsulating the tile
received first, instead of the tile data itself.

**Managing the memory buffer associated with a message.** As suggested
in the example above, the biggest difference between marshalled
parameters and messages is that an entry method invocation is assumed to
*keep* the message that it is passed. That is, the Charm++ runtime
system assumes that code in the body of the invoked entry method will
explicitly manage the memory associated with the message that it is
passed. Therefore, in order to avoid leaking memory, the body of an
entry method must either delete the message that it is receives, or save
a pointer to it, and delete it a later point in the execution of the
code.

Moreover, in the Charm++ execution model, once you pass a message buffer
to the runtime system (via an asynchronous entry method invocation), you
should *not* reuse the buffer. That is, after you have passed a message
buffer into an asynchronous entry method invocation, you shouldn’t
access its fields, or pass that same buffer into a second entry method
invocation. Note that this rule doesn’t preclude the *single reuse* of
an input message - consider an entry method invocation :math:`i_1`,
which receives as input the message buffer :math:`m_1`. Then,
:math:`m_1` may be passed to an asynchronous entry method invocation
:math:`i_2`. However, once :math:`i_2` has been issued with :math:`m_1`
as its input parameter, :math:`m_1` cannot be used in any further entry
method invocations.

Several kinds of message are available. Regular Charm++ messages are
objects of *fixed size*. One can have messages that contain pointers or
variable length arrays (arrays with sizes specified at runtime) and
still have these pointers as valid when messages are sent across
processors, with some additional coding. Also available is a mechanism
for assigning *priorities* to a message regardless of its type. A
detailed discussion of priorities appears later in this section.

Message Types
^^^^^^^^^^^^^

**Fixed-Size Messages.** The simplest type of message is a *fixed-size*
message. The size of each data member of such a message should be known
at compile time. Therefore, such a message may encapsulate primitive
data types, user-defined data types that *don’t* maintain pointers to
memory locations, and *static* arrays of the aforementioned types.

**Variable-Size Messages.** Very often, the size of the data contained
in a message is not known until runtime. For such scenarios, you can use
variable-size (*varsize*) messages. A *varsize* message can encapsulate
several arrays, each of whose size is determined at run time. The space
required for these encapsulated, variable length arrays is allocated
with the entire message comprises a contiguous buffer of memory.

**Packed Messages.** A *packed* message is used to communicate
non-linear data structures via messages. However, we defer a more
detailed description of their use to
§ :numref:`sec:messages/packed_msgs`.

.. _memory allocation:

Using Messages In Your Program
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

There are five steps to incorporating a (fixed or varsize) message type
in your Charm++ program: (1) Declare message type in .ci file; (2)
Define message type in .h file; (3) Allocate message; (4) Pass message
to asynchronous entry method invocation and (5) Deallocate message to
free associated memory resources.

**Declaring Your Message Type.** Like all other entities involved in
asynchronous entry method invocation, messages must be declared in the
``.ci`` file. This allows the Charm++ translator to generate support
code for messages. Message declaration is straightforward for fixed-size
messages. Given a message of type ``MyFixedSizeMsg``, simply include the
following in the .ci file:

.. code-block:: c++

    message MyFixedSizeMsg;

For varsize messages, the .ci declaration must also include the names
and types of the variable-length arrays that the message will
encapsulate. The following example illustrates this requirement. In it,
a message of type ``MyVarsizeMsg``, which encapsulates three
variable-length arrays of different types, is declared:

.. code-block:: c++

    message MyVarsizeMsg {
      int arr1[];
      double arr2[];
      MyPointerlessStruct arr3[];
    };

**Defining Your Message Type.** Once a message type has been declared to
the Charm++ translator, its type definition must be provided. Your
message type must inherit from a specific generated base class. If the
type of your message is ``T``, then ``class T`` must inherit from
``CMessage_T``. This is true for both fixed and varsize messages. As an
example, for our fixed size message type ``MyFixedSizeMsg`` above, we
might write the following in the .h file:

.. code-block:: c++

   class MyFixedSizeMsg : public CMessage_MyFixedSizeMsg {
     int var1;
     MyPointerlessStruct var2;
     double arr3[10];

     // Normal C++ methods, constructors, etc. go here
   };

In particular, note the inclusion of the static array of ``double``\ s,
``arr3``, whose size is known at compile time to be that of ten
``double``\ s. Similarly, for our example varsize message of type
``MyVarsizeMsg``, we would write something like:

.. code-block:: c++

   class MyVarsizeMsg : public CMessage_MyVarsizeMsg {
     // variable-length arrays
     int *arr1;
     double *arr2;
     MyPointerlessStruct *arr3;

     // members that are not variable-length arrays
     int x,y;
     double z;

     // Normal C++ methods, constructors, etc. go here
   };

Note that the .h definition of the class type must contain data members
whose names and types match those specified in the .ci declaration. In
addition, if any of data members are private or protected, it should
declare class CMessage_MyVarsizeMsg to be a friend class. Finally, there
are no limitations on the member methods of message classes, except that
the message class may not redefine operators ``new`` or ``delete``.

Thus the mtype class declaration should be similar to:

**Creating a Message.** With the .ci declaration and .h definition in
place, messages can be allocated and used in the program. Messages are
allocated using the C++ new operator:

.. code-block:: c++

    MessageType *msgptr =
     new [(int sz1, int sz2, ... , int priobits=0)] MessageType[(constructor arguments)];

The arguments enclosed within the square brackets are optional, and are
used only when allocating messages with variable length arrays or
prioritized messages. These arguments are not specified for fixed size
messages. For instance, to allocate a message of our example message
``MyFixedSizeMsg``, we write:

.. code-block:: c++

   MyFixedSizeMsg *msg = new MyFixedSizeMsg(<constructor args>);

In order to allocate a varsize message, we must pass appropriate values
to the arguments of the overloaded new operator presented previously.
Arguments sz1, sz2, ... denote the size (in number of elements) of the
memory blocks that need to be allocated and assigned to the pointers
(variable-length arrays) that the message contains. The priobits
argument denotes the size of a bitvector (number of bits) that will be
used to store the message priority. So, if we wanted to create
``MyVarsizeMsg`` whose ``arr1``, ``arr2`` and ``arr3`` arrays contain
10, 20 and 7 elements of their respective types, we would write:

.. code-block:: c++

   MyVarsizeMsg *msg = new (10, 20, 7) MyVarsizeMsg(<constructor args>);

Further, to add a 32-bit priority bitvector to this message, we would
write:

.. code-block:: c++

   MyVarsizeMsg *msg = new (10, 20, 7, sizeof(uint32_t)*8) VarsizeMessage;

Notice the last argument to the overloaded new operator, which specifies
the number of bits used to store message priority. The section on
prioritized execution (§ :numref:`prioritized message passing`)
describes how priorities can be employed in your program.

Another version of the overloaded new operator allows you to pass in an
array containing the size of each variable-length array, rather than
specifying individual sizes as separate arguments. For example, we could
create a message of type ``MyVarsizeMsg`` in the following manner:

.. code-block:: c++

   int sizes[3];
   sizes[0] = 10;               // arr1 will have 10 elements
   sizes[1] = 20;               // arr2 will have 20 elements
   sizes[2] = 7;                // arr3 will have 7 elements

   MyVarsizeMsg *msg = new(sizes, 0) MyVarsizeMsg(<constructor args>); // 0 priority bits

**Sending a Message.** Once we have a properly allocated message, we can
set the various elements of the encapsulated arrays in the following
manner:

.. code-block:: c++

     msg->arr1[13] = 1;
     msg->arr2[5] = 32.82;
     msg->arr3[2] = MyPointerlessStruct();
     // etc.

And pass it to an asynchronous entry method invocation, thereby sending
it to the corresponding chare:

.. code-block:: c++

   myChareArray[someIndex].foo(msg);

When a message is *sent*, i.e. passed to an asynchronous entry method
invocation, the programmer relinquishes control of it; the space
allocated for the message is freed by the runtime system. However, when
a message is *received* at an entry point, it is *not* freed by the
runtime system. As mentioned at the start of this section, received
messages may be reused or deleted by the programmer. Finally, messages
are deleted using the standard C++ delete operator.

.. _message packing:

Message Packing
^^^^^^^^^^^^^^^

The Charm++ interface translator generates implementation for three
static methods for the message class CMessage_mtype. These methods have
the prototypes:

.. code-block:: c++

       static void* alloc(int msgnum, size_t size, int* array, int priobits);
       static void* pack(mtype*);
       static mtype* unpack(void*);

One may choose not to use the translator-generated methods and may
override these implementations with their own alloc, pack and unpack
static methods of the mtype class. The alloc method will be called when
the message is allocated using the C++ new operator. The programmer never
needs to explicitly call it. Note that all elements of the message are
allocated when the message is created with new. There is no need to call
new to allocate any of the fields of the message. This differs from a
packed message where each field requires individual allocation. The
alloc method should actually allocate the message using CkAllocMsg,
whose signature is given below:

.. code-block:: c++

   void *CkAllocMsg(int msgnum, int size, int priobits);

For varsize messages, these static methods ``alloc``, ``pack``, and
``unpack`` are generated by the interface translator. For example, these
methods for the VarsizeMessage class above would be similar to:

.. code-block:: c++

   // allocate memory for varmessage so charm can keep track of memory
   static void* alloc(int msgnum, size_t size, int* array, int priobits)
   {
     int totalsize, first_start, second_start;
     // array is passed in when the message is allocated using new (see below).
     // size is the amount of space needed for the part of the message known
     // about at compile time.  Depending on their values, sometimes a segfault
     // will occur if memory addressing is not on 8-byte boundary, so altered
     // with ALIGN8
     first_start = ALIGN8(size);  // 8-byte align with this macro
     second_start = ALIGN8(first_start + array[0]*sizeof(int));
     totalsize = second_start + array[1]*sizeof(double);
     VarsizeMessage* newMsg =
       (VarsizeMessage*) CkAllocMsg(msgnum, totalsize, priobits);
     // make firstArray point to end of newMsg in memory
     newMsg->firstArray = (int*) ((char*)newMsg + first_start);
     // make secondArray point to after end of firstArray in memory
     newMsg->secondArray = (double*) ((char*)newMsg + second_start);

     return (void*) newMsg;
   }

   // returns pointer to memory containing packed message
   static void* pack(VarsizeMessage* in)
   {
     // set firstArray an offset from the start of in
     in->firstArray = (int*) ((char*)in->firstArray - (char*)in);
     // set secondArray to the appropriate offset
     in->secondArray = (double*) ((char*)in->secondArray - (char*)in);
     return in;
   }

   // returns new message from raw memory
   static VarsizeMessage* VarsizeMessage::unpack(void* inbuf)
   {
     VarsizeMessage* me = (VarsizeMessage*)inbuf;
     // return first array to absolute address in memory
     me->firstArray = (int*) ((size_t)me->firstArray + (char*)me);
     // likewise for secondArray
     me->secondArray = (double*) ((size_t)me->secondArray + (char*)me);
     return me;
   }

The pointers in a varsize message can exist in two states. At creation,
they are valid C++ pointers to the start of the arrays. After packing,
they become offsets from the address of the pointer variable to the
start of the pointed-to data. Unpacking restores them to pointers.

.. _sec:messages/packed_msgs:

Custom Packed Messages
''''''''''''''''''''''

In many cases, a message must store a *non-linear* data structure using
pointers. Examples of these are binary trees, hash tables etc. Thus, the
message itself contains only a pointer to the actual data. When the
message is sent to the same processor, these pointers point to the
original locations, which are within the address space of the same
processor. However, when such a message is sent to other processors,
these pointers will point to invalid locations.

Thus, the programmer needs a way to “serialize” these messages *only if*
the message crosses the address-space boundary. Charm++ provides a way
to do this serialization by allowing the developer to override the
default serialization methods generated by the Charm++ interface
translator. Note that this low-level serialization has nothing to do
with parameter marshalling or the PUP framework described later.

Packed messages are declared in the ``.ci`` file the same way as
ordinary messages:

.. code-block:: c++

   message PMessage;

Like all messages, the class PMessage needs to inherit from
CMessage_PMessage and should provide two *static* methods: pack and
unpack. These methods are called by the Charm++ runtime system, when the
message is determined to be crossing address-space boundary. The
prototypes for these methods are as follows:

.. code-block:: c++

   static void *PMessage::pack(PMessage *in);
   static PMessage *PMessage::unpack(void *in);

Typically, the following tasks are done in pack method:

-  Determine size of the buffer needed to serialize message data.

-  Allocate buffer using the CkAllocBuffer function. This function takes
   in two parameters: input message, and size of the buffer needed, and
   returns the buffer.

-  Serialize message data into buffer (along with any control
   information needed to de-serialize it on the receiving side.

-  Free resources occupied by message (including message itself.)

On the receiving processor, the unpack method is called. Typically, the
following tasks are done in the unpack method:

-  Allocate message using CkAllocBuffer function. *Do not use new to
   allocate message here. If the message constructor has to be called,
   it can be done using the in-place new operator.*

-  De-serialize message data from input buffer into the allocated
   message.

-  Free the input buffer using CkFreeMsg.

Here is an example of a packed-message implementation:

.. code-block:: c++

   // File: pgm.ci
   mainmodule PackExample {
     ...
     message PackedMessage;
     ...
   };

   // File: pgm.h
   ...
   class PackedMessage : public CMessage_PackedMessage
   {
     public:
       BinaryTree<char> btree; // A non-linear data structure
       static void* pack(PackedMessage*);
       static PackedMessage* unpack(void*);
       ...
   };
   ...

   // File: pgm.C
   ...
   void*
   PackedMessage::pack(PackedMessage* inmsg)
   {
     int treesize = inmsg->btree.getFlattenedSize();
     int totalsize = treesize + sizeof(int);
     char *buf = (char*)CkAllocBuffer(inmsg, totalsize);
     // buf is now just raw memory to store the data structure
     int num_nodes = inmsg->btree.getNumNodes();
     memcpy(buf, &num_nodes, sizeof(int));  // copy numnodes into buffer
     buf = buf + sizeof(int);               // don't overwrite numnodes
     // copies into buffer, give size of buffer minus header
     inmsg->btree.Flatten((void*)buf, treesize);
     buf = buf - sizeof(int);              // don't lose numnodes
     delete inmsg;
     return (void*) buf;
   }

   PackedMessage*
   PackedMessage::unpack(void* inbuf)
   {
     // inbuf is the raw memory allocated and assigned in pack
     char* buf = (char*) inbuf;
     int num_nodes;
     memcpy(&num_nodes, buf, sizeof(int));
     buf = buf + sizeof(int);
     // allocate the message through Charm RTS
     PackedMessage* pmsg =
       (PackedMessage*)CkAllocBuffer(inbuf, sizeof(PackedMessage));
     // call "inplace" constructor of PackedMessage that calls constructor
     // of PackedMessage using the memory allocated by CkAllocBuffer,
     // takes a raw buffer inbuf, the number of nodes, and constructs the btree
     pmsg = new ((void*)pmsg) PackedMessage(buf, num_nodes);
     CkFreeMsg(inbuf);
     return pmsg;
   }
   ...
   PackedMessage* pm = new PackedMessage();  // just like always
   pm->btree.Insert('A');
   ...

While serializing an arbitrary data structure into a flat buffer, one
must be very wary of any possible alignment problems. Thus, if possible,
the buffer itself should be declared to be a flat struct. This will
allow the C++ compiler to ensure proper alignment of all its member
fields.

.. _attributes:

Entry Method Attributes
~~~~~~~~~~~~~~~~~~~~~~~

Charm++ provides a handful of special attributes that entry methods may
have. In order to give a particular entry method an attribute, you must
specify the keyword for the desired attribute in the attribute list of
that entry method’s ``.ci`` file declaration. The syntax for this is as
follows:

.. code-block:: c++

   entry [attribute1, ..., attributeN] void EntryMethod(parameters);

Charm++ currently offers the following attributes that one may assign to
an entry method: threaded, sync, exclusive, nokeep, notrace, appwork,
immediate, expedited, inline, local, python, reductiontarget, aggregate.

threaded
   entry methods run in their own non-preemptible threads. These entry
   methods may perform blocking operations, such as calls to a sync
   entry method, or explicitly suspending themselves. For more details,
   refer to section :numref:`threaded`.

sync
   entry methods are special in that calls to them are blocking-they do
   not return control to the caller until the method finishes execution
   completely. Sync methods may have return values; however, they may
   only return messages or data types that have the PUP method
   implemented. Callers must run in a thread separate from the runtime
   scheduler, e.g. a threaded entry methods. Calls expecting a return
   value will receive it as the return from the proxy invocation:

   .. code-block:: c++

       ReturnMsg* m;
       m = A[i].foo(a, b, c);

   For more details, refer to section :numref:`sync`.

exclusive
   entry methods should only exist on NodeGroup objects. One such entry
   method will not execute while some other exclusive entry methods
   belonging to the same NodeGroup object are executing on the same
   node. In other words, if one exclusive method of a NodeGroup object
   is executing on node N, and another one is scheduled to run on the
   same node, the second exclusive method will wait to execute until the
   first one finishes. An example can be found in
   ``tests/charm++/pingpong``.

nokeep
   entry methods take only a message as their lone argument, and the
   memory buffer for this message is managed by the Charm++ runtime
   system rather than by the user. This means that the user has to
   guarantee that the message will not be buffered for later usage or be
   freed in the user code. Additionally, users are not allowed to modify
   the contents of a nokeep message, since for a broadcast the same
   message can be reused for all entry method invocations on each PE. If
   a user frees the message or modifies its contents, a runtime error
   may result. An example can be found in
   ``examples/charm++/histogram_group``.

notrace
   entry methods will not be traced during execution. As a result, they
   will not be considered and displayed in Projections for performance
   analysis. Additionally, ``immediate`` entry methods are by default
   ``notrace`` and will not be traced during execution.

appwork
   this entry method will be marked as executing application work. It
   will be used for performance analysis.

immediate
   entry methods are executed in an “immediate” fashion as they skip the
   message scheduling while other normal entry methods do not. Immediate
   entry methods can only be associated with NodeGroup objects,
   otherwise a compilation error will result. If the destination of such
   entry method is on the local node, then the method will be executed
   in the context of the regular PE regardless the execution mode of
   Charm++ runtime. However, in the SMP mode, if the destination of the
   method is on the remote node, then the method will be executed in the
   context of the communication thread. For that reason, immediate entry
   methods should be used for code that is performance critical and does
   not take too long in terms of execution time because long running
   entry methods can delay communication by occupying the communication
   thread for entry method execution rather than remote communication.

   Such entry methods can be useful for implementing
   multicasts/reductions as well as data lookup when such operations are
   on the performance critical path. On a certain Charm++ PE, skipping
   the normal message scheduling prevents the execution of immediate
   entry methods from being delayed by entry functions that could take a
   long time to finish. Immediate entry methods are implicitly
   “exclusive” on each node, meaning that one execution of immediate
   message will not be interrupted by another. Function
   CmiProbeImmediateMsg() can be called in user codes to probe and
   process immediate messages periodically. Also note that ``immediate``
   entry methods are by default ``notrace`` and are not traced during
   execution. An example of ``immediate`` entry method can be found in
   ``examples/charm++/immediateEntryMethod``.

expedited
   entry methods skip the priority-based message queue in Charm++
   runtime. It is useful for messages that require prompt processing
   when adding the immediate attribute to the message does not apply.
   Compared with the immediate attribute, the expedited attribute
   provides a more general solution that works for all types of Charm++
   objects, i.e. Chare, Group, NodeGroup and Chare Array. However,
   expedited entry methods will still be scheduled in the lower-level
   Converse message queue, and be processed in the order of message
   arrival. Therefore, they may still suffer from delays caused by long
   running entry methods. An example can be found in
   ``examples/charm++/satisfiability``.

inline
   entry methods will be called as a normal C++ member function if the
   message recipient happens to be on the same PE. The call to the
   function will happen inline, and control will return to the calling
   function after the inline method completes. Because of this, these
   entry methods need to be re-entrant as they could be called multiple
   times recursively. Parameters to the inline method will be passed by
   reference to avoid any copying, packing, or unpacking of the
   parameters. This makes inline calls effective when large amounts of
   data are being passed, and copying or packing the data would be an
   expensive operation. Perfect forwarding has been implemented to allow
   for seamless passing of both lvalue and rvalue references. Note that
   calls with rvalue references must take place in the same translation
   unit as the .decl.h file to allow for the appropriate template
   instantiations. Alternatively, the method can be made templated and
   referenced from multiple translation units via ``CK_TEMPLATES_ONLY``.
   An explicit instantiation of all lvalue references is provided for
   compatibility with existing code. If the recipient resides on a
   different PE, a regular message is sent with the message arguments
   packed up using PUP, and inline has no effect. An example “inlineem”
   can be found in ``tests/charm++/megatest``.

local
   entry methods are equivalent to normal function calls: the entry
   method is always executed immediately. This feature is available only
   for Group objects and Chare Array objects. The user has to guarantee
   that the recipient chare element resides on the same PE. Otherwise,
   the application will abort with a failure. If the local entry method
   uses parameter marshalling, instead of marshalling input parameters
   into a message, it will pass them directly to the callee. This
   implies that the callee can modify the caller data if method
   parameters are passed by pointer or reference. The above description
   of perfect forwarding for inline entry methods also applies to local
   entry methods. Furthermore, input parameters are not required to be
   PUPable. Considering that these entry methods always execute
   immediately, they are allowed to have a non-void return value. An
   example can be found in ``examples/charm++/hello/local``.

python
   entry methods are enabled to be called from python scripts as
   explained in chapter :numref:`python`. Note that the object owning
   the method must also be declared with the keyword python. Refer to
   chapter :numref:`python` for more details.

reductiontarget
   entry methods can be used as the target of reductions while taking
   arguments of the same type specified in the contribute call rather
   than a message of type CkReductionMsg. See
   section :numref:`reductions` for more information.

aggregate
   data sent to this entry method will be aggregated into larger
   messages before being sent, to reduce fine-grained overhead. The
   aggregation is handled by the Topological Routing and Aggregation
   Module (TRAM). The argument to this entry method must be a single
   fixed-size object. More details on TRAM are given in the `TRAM
   section <http://charm.cs.illinois.edu/manuals/html/libraries/manual-1p.html#TRAM>`__
   of the libraries manual.

Controlling Delivery Order
~~~~~~~~~~~~~~~~~~~~~~~~~~

By default, Charm++ processes the messages sent in roughly FIFO order
when they arrive at a PE. For most programs, this behavior is fine.
However, for optimal performance, some programs need more explicit
control over the order in which messages are processed. Charm++ allows
you to adjust delivery order on a per-message basis.

An example program demonstrating how to modify delivery order for
messages and parameter marshaling can be found in
``examples/charm++/prio``.

.. _queueing strategies:

Queueing Strategies
^^^^^^^^^^^^^^^^^^^

The order in which messages are processed in the recipient’s queue can
be set by explicitly setting the queuing strategy using one the
following constants. These constants can be applied when sending a
message or invoking an entry method using parameter marshaling:

-  ``CK_QUEUEING_FIFO``: FIFO ordering

-  ``CK_QUEUEING_LIFO``: LIFO ordering

-  ``CK_QUEUEING_IFIFO``: FIFO ordering with *integer* priority

-  ``CK_QUEUEING_ILIFO``: LIFO ordering with *integer* priority

-  ``CK_QUEUEING_BFIFO``: FIFO ordering with *bitvector* priority

-  ``CK_QUEUEING_BLIFO``: LIFO ordering with *bitvector* priority

-  ``CK_QUEUEING_LFIFO``: FIFO ordering with *long integer* priority

-  ``CK_QUEUEING_LLIFO``: FIFO ordering with *long integer* priority

Parameter Marshaling
^^^^^^^^^^^^^^^^^^^^

For parameter marshaling, the queueingtype can be set for
CkEntryOptions, which is passed to an entry method invocation as the
optional last parameter.

.. code-block:: c++

     CkEntryOptions opts1, opts2;
     opts1.setQueueing(CK_QUEUEING_FIFO);
     opts2.setQueueing(CK_QUEUEING_LIFO);

     chare.entry_name(arg1, arg2, opts1);
     chare.entry_name(arg1, arg2, opts2);

When the message with opts1 arrives at its destination, it will be
pushed onto the end of the message queue as usual. However, when the
message with opts2 arrives, it will be pushed onto the *front* of the
message queue.

.. _messages-1:

Messages
^^^^^^^^

| For messages, the CkSetQueueing function can be used to change the
  order in which messages are processed, where queueingtype is one of
  the above constants.
| void CkSetQueueing(MsgType message, int queueingtype)

The first two options, CK_QUEUEING_FIFO and CK_QUEUEING_LIFO, are used
as follows:

.. code-block:: c++

     MsgType *msg1 = new MsgType ;
     CkSetQueueing(msg1, CK_QUEUEING_FIFO);

     MsgType *msg2 = new MsgType ;
     CkSetQueueing(msg2, CK_QUEUEING_LIFO);

Similar to the parameter marshalled case described above, msg1 will be
pushed onto the end of the message queue, while msg2 will be pushed onto
the *front* of the message queue.

.. _prioritized message passing:

Prioritized Execution
^^^^^^^^^^^^^^^^^^^^^

The basic FIFO and LIFO strategies are sufficient to approximate
parallel breadth-first and depth-first explorations of a problem space,
but they do not allow more fine-grained control. To provide that degree
of control, Charm++ also allows explicit prioritization of messages.

The other six queueing strategies involve the use of priorities . There
are two kinds of priorities which can be attached to a message: *integer
priorities* and *bitvector priorities* . These correspond to the *I* and
*B* queueing strategies, respectively. In both cases, numerically lower
priorities will be dequeued and delivered before numerically greater
priorities. The FIFO and LIFO queueing strategies then control the
relative order in which messages of the same priority will be delivered.

To attach a priority field to a message, one needs to set aside space
in the message’s buffer while allocating the message . To achieve
this, the size of the priority field in bits should be specified as a
placement argument to the new operator, as described in
section :numref:`memory allocation`. Although the
size of the priority field is specified in bits, it is always padded
to an integral number of ``int``\ s. A pointer to the priority part of
the message buffer can be obtained with this call:
``void *CkPriorityPtr(MsgType msg)``.

Integer priorities are quite straightforward. One allocates a message
with an extra integer parameter to “new” (see the first line of the
example below), which sets aside enough space (in bits) in the message
to hold the priority. One then stores the priority in the message.
Finally, one informs the system that the message contains an integer
priority using CkSetQueueing:

.. code-block:: c++

     MsgType *msg = new (8*sizeof(int)) MsgType;
     *(int*)CkPriorityPtr(msg) = prio;
     CkSetQueueing(msg, CK_QUEUEING_IFIFO);

Bitvector Prioritization
^^^^^^^^^^^^^^^^^^^^^^^^

Bitvector priorities are arbitrary-length bit-strings representing
fixed-point numbers in the range 0 to 1. For example, the bit-string
“001001” represents the number .001001. As with integer priorities,
higher numbers represent lower priorities. However, bitvectors can be of
arbitrary length, and hence the priority numbers they represent can be
of arbitrary precision.

Arbitrary-precision priorities are often useful in AI search-tree
applications. Suppose we have a heuristic suggesting that tree node
:math:`N_1` should be searched before tree node :math:`N_2`. We
therefore designate that node :math:`N_1` and its descendants will use
high priorities, and that node :math:`N_2` and its descendants will use
lower priorities. We have effectively split the range of possible
priorities in two. If several such heuristics fire in sequence, we can
easily split the priority range in two enough times that no significant
bits remain, and the search begins to fail for lack of meaningful
priorities to assign. The solution is to use arbitrary-precision
priorities, i.e. bitvector priorities.

To assign a bitvector priority, two methods are available. The first is
to obtain a pointer to the priority field using CkPriorityPtr, and then
manually set the bits using the bit-setting operations inherent to C. To
achieve this, one must know the format of the bitvector, which is as
follows: the bitvector is represented as an array of unsigned integers.
The most significant bit of the first integer contains the first bit of
the bitvector. The remaining bits of the first integer contain the next
31 bits of the bitvector. Subsequent integers contain 32 bits each. If
the size of the bitvector is not a multiple of 32, then the last integer
contains 0 bits for padding in the least-significant bits of the
integer.

The second way to assign priorities is only useful for those who are
using the priority range-splitting described above. The root of the tree
is assigned the null priority-string. Each child is assigned its
parent’s priority with some number of bits concatenated. The net effect
is that the entire priority of a branch is within a small epsilon of the
priority of its root.

It is possible to utilize unprioritized messages, integer priorities,
and bitvector priorities in the same program. The messages will be
processed in roughly the following order :

-  Among messages enqueued with bitvector priorities, the messages are
   dequeued according to their priority. The priority “0000...” is
   dequeued first, and “1111...” is dequeued last.

-  Unprioritized messages are treated as if they had the priority
   “1000...” (which is the “middle” priority, it lies exactly halfway
   between “0000...” and “1111...”).

-  Integer priorities are converted to bitvector priorities. They are
   normalized so that the integer priority of zero is converted to
   “1000...” (the “middle” priority). To be more specific, the
   conversion is performed by adding 0x80000000 to the integer, and then
   treating the resulting 32-bit quantity as a 32-bit bitvector
   priority.

-  Among messages with the same priority, messages are dequeued in FIFO
   order or LIFO order, depending upon which queuing strategy was used.

Additionally, long integer priorities can be specified by the *L*
strategy.

A final reminder about prioritized execution: Charm++ processes messages
in *roughly* the order you specify; it never guarantees that it will
deliver the messages in *precisely* the order you specify. Thus, the
correctness of your program should never depend on the order in which
the runtime delivers messages. However, it makes a serious attempt to be
“close”, so priorities can strongly affect the efficiency of your
program.

Skipping the Queue
^^^^^^^^^^^^^^^^^^

Some operations that one might want to perform are sufficiently
latency-sensitive that they should never wait in line behind other
messages. The Charm++ runtime offers two attributes for entry methods,
expedited and immediate, to serve these needs. For more information on
these attributes, see Section :numref:`attributes` and the example in
``tests/charm++/megatest/immediatering.ci``.

.. _nocopyapi:

Zero Copy Messaging API
~~~~~~~~~~~~~~~~~~~~~~~

Apart from using messages, Charm++ also provides APIs to avoid sender
and receiver side copies. On RDMA enabled networks like GNI, Verbs, PAMI
and OFI, these internally make use of one-sided communication by using
the underlying Remote Direct Memory Access (RDMA) enabled network. For
large arrays (few 100 KBs or more), the cost of copying during
marshalling the message can be quite high. Using these APIs can help not
only save the expensive copy operation but also reduce the application’s
memory footprint by avoiding data duplication. Saving these costs for
large arrays proves to be a significant optimization in achieving faster
message send and receive times in addition to overall improvement in
performance because of lower memory consumption. On the other hand,
using these APIs for small arrays can lead to a drop in performance due
to the overhead associated with one-sided communication. The overhead is
mainly due to additional small messages required for sending the
metadata message and the acknowledgment message on completion.

For within process data-transfers, this API uses regular memcpy to
achieve zerocopy semantics. Similarly, on CMA-enabled machines, in a few
cases, this API takes advantage of CMA to perform inter-process
intra-physical host data transfers. This API is also functional on
non-RDMA enabled networks like regular ethernet, except that it does not
avoid copies and behaves like a regular Charm++ entry method invocation.

There are two APIs that provide Zero copy semantics in Charm++:

-  Zero Copy Direct API

-  Zero Copy Entry Method Send API

Zero Copy Direct API
^^^^^^^^^^^^^^^^^^^^

The Zero copy Direct API allows users to explicitly invoke a standard
set of methods on predefined buffer information objects to avoid copies.
Unlike the Entry Method API which calls the zero copy operation for
every zero copy entry method invocation, the direct API provides a more
flexible interface by allowing the user to exploit the persistent nature
of iterative applications to perform zero copy operations using the same
buffer information objects across iteration boundaries. It is also more
beneficial than the Zero copy entry method API because unlike the entry
method API, which avoids just the sender side copy, the Zero copy Direct
API avoids both sender and receiver side copies.

To send an array using the zero copy Direct API, define a CkNcpyBuffer
object on the sender chare specifying the pointer, size, a CkCallback
object and an optional mode parameter.

.. code-block:: c++

   CkCallback srcCb(CkIndex_Ping1::sourceDone, thisProxy[thisIndex]);
   // CkNcpyBuffer object representing the source buffer
   CkNcpyBuffer source(arr1, arr1Size * sizeof(int), srcCb, CK_BUFFER_REG);

When used inside a CkNcpyBuffer object that represents the source buffer
information, the callback is specified to notify about the safety of
reusing the buffer and indicates that the get or put call has been
completed. In those cases where the application can determine safety of
reusing the buffer through other synchronization mechanisms, the
callback is not entirely useful and in such cases,
``CkCallback::ignore`` can be passed as the callback parameter. The
optional mode operator is used to determine the network registration
mode for the buffer. It is only relevant on networks requiring explicit
memory registration for performing RDMA operations. These networks
include GNI, OFI and Verbs. When the mode is not specified by the user,
the default mode is considered to be ``CK_BUFFER_REG``

Similarly, to receive an array using the Zero copy Direct API, define
another CkNcpyBuffer object on the receiver chare object specifying the
pointer, the size, a CkCallback object and an optional mode parameter.
When used inside a CkNcpyBuffer object that represents the destination
buffer, the callback is specified to notify the completion of data
transfer into the CkNcpyBuffer buffer. In those cases where the
application can determine data transfer completion through other
synchronization mechanisms, the callback is not entirely useful and in
such cases, ``CkCallback::ignore`` can be passed as the callback
parameter.

.. code-block:: c++

   CkCallback destCb(CkIndex_Ping1::destinationDone, thisProxy[thisIndex]);
   // CkNcpyBuffer object representing the destination buffer
   CkNcpyBuffer dest(arr2, arr2Size * sizeof(int), destCb, CK_BUFFER_REG);

Once the source CkNcpyBuffer and destination CkNcpyBuffer objects have
been defined on the sender and receiver chares, to perform a get
operation, send the source CkNcpyBuffer object to the receiver chare.
This can be done using a regular entry method invocation as shown in the
following code snippet, where the sender, arrProxy[0] sends its source
object to the receiver chare, arrProxy[1].

.. code-block:: c++

   // On Index 0 of arrProxy chare array
   arrProxy[1].recvNcpySrcObj(source);

After receiving the sender’s source CkNcpyBuffer object, the receiver
can perform a get operation on its destination CkNcpyBuffer object by
passing the source object as an argument to the runtime defined get
method as shown in the following code snippet.

.. code-block:: c++

   // On Index 1 of arrProxy chare array
   // Call get on the local destination object passing the source object
   dest.get(source);

This call performs a get operation, reading the remote source buffer
into the local destination buffer.

Similarly, a receiver’s destination CkNcpyBuffer object can be sent to
the sender for the sender to perform a put on its source object by
passing the source CkNcpyBuffer object as an argument to the runtime
defined put method as shown in in the code snippet.

.. code-block:: c++

   // On Index 1 of arrProxy chare array
   arrProxy[0].recvNcpyDestObj(dest);

.. code-block:: c++

   // On Index 0 of arrProxy chare array
   // Call put on the local source object passing the destination object
   source.put(dest);

After the completion of either a get or a put, the callbacks specified
in both the objects are invoked. Within the CkNcpyBuffer source
callback, ``sourceDone()``, the buffer can be safely modified or freed
as shown in the following code snippet.

.. code-block:: c++

   // Invoked by the runtime on source (Index 0)
   void sourceDone() {
       // update the buffer to the next pointer
       updateBuffer();
   }

Similarly, inside the CkNcpyBuffer destination callback,
``destinationDone()``, the user is guaranteed that the data transfer is
complete into the destination buffer and the user can begin operating on
the newly available data as shown in the following code snippet.

.. code-block:: c++

   // Invoked by the runtime on destination (Index 1)
   void destinationDone() {
       // received data, begin computing
       computeValues();
   }

The callback methods can also take a pointer to a ``CkDataMsg`` message.
This message can be used to access the original buffer information
object i.e. the ``CkNcpyBuffer`` objects used for the zero copy
transfer. The buffer information object available in the callback allows
access to all its information including the buffer pointer and the
arbitrary reference pointer set using the method ``setRef``. It is
important to note that only the source ``CkNcpyBuffer`` object is
accessible using the ``CkDataMsg`` in the source callback and similarly,
the destination ``CkNcpyBuffer`` object is accessible using the
``CkDataMsg`` in the destination callback. The following code snippet
illustrates the accessing of the original buffer pointer in the callback
method by casting the ``data`` field of the ``CkDataMsg`` object into a
``CkNcpyBuffer`` object.

.. code-block:: c++

   // Invoked by the runtime on source (Index 0)
   void sourceDone(CkDataMsg *msg) {
       // Cast msg->data to a CkNcpyBuffer to get the source buffer information object
       CkNcpyBuffer *source = (CkNcpyBuffer *)(msg->data);

       // access buffer pointer and free it
       free(source->ptr);
   }

The following code snippet illustrates the usage of the ``setRef``
method.

.. code-block:: c++

   const void *refPtr = &index;
   CkNcpyBuffer source(arr1, arr1Size * sizeof(int), srcCb, CK_BUFFER_REG);
   source.setRef(refPtr);

Similar to the buffer pointer, the user set arbitrary reference pointer
can be also accessed in the callback method. This is shown in the next
code snippet.

.. code-block:: c++

   // Invoked by the runtime on source (Index 0)
   void sourceDone(CkDataMsg *msg) {
       // update the buffer to the next pointer
       updateBuffer();

       // Cast msg->data to a CkNcpyBuffer
       CkNcpyBuffer *src = (CkNcpyBuffer *)(msg->data);

       // access buffer pointer and free it
       free(src->ptr);

       // get reference pointer
       const void *refPtr = src->ref;
   }

The usage of ``CkDataMsg`` and ``setRef`` in order to access the
original pointer and the arbitrary reference pointer is illustrated in
``examples/charm++/zerocopy/direct_api/unreg/simple_get``

Both the source and destination buffers are of the same type i.e.
CkNcpyBuffer. What distinguishes a source buffer from a destination
buffer is the way the get or put call is made. A valid get call using
two CkNcpyBuffer objects ``obj1`` and ``obj2`` is performed as
``obj1.get(obj2)``, where ``obj1`` is the local destination buffer
object and ``obj2`` is the remote source buffer object that was passed
to this PE. Similarly, a valid put call using two CkNcpyBuffer objects
``obj1`` and ``obj2`` is performed as ``obj1.put(obj2)``, where ``obj1``
is the local source buffer object and ``obj2`` is the remote destination
buffer object that was passed to this PE.

In addition to the callbacks, the return values of get and put also
indicate the completion of data transfer between the buffers. When the
source and destination buffers are within the same process or on
different processes within the same CMA-enabled physical node, the
zerocopy data transfer happens immediately without an asynchronous RDMA
call. In such cases, both the methods, get and put return an enum value
of CkNcpyStatus::complete. This value of the API indicates the
completion of the zerocopy data transfer. On the other hand, in the case
of an asynchronous RDMA call, the data transfer is not immediate and the
return enum value of the get and put methods is
CkNcpyStatus::incomplete. This indicates that the data transfer is
in-progress and not necessarily complete. Use of CkNcpyStatus in an
application is illustrated in
``examples/charm++/zerocopy/direct_api/reg/simple_get``.

Since callbacks in Charm++ allow to store a reference number, these
callbacks passed into CkNcpyBuffer can be set with a reference number
using the method ``cb.setRefNum(num)``. Upon callback invocation, these
reference numbers can be used to identify the buffers that were passed
into the CkNcpyBuffer objects. Upon callback invocation, the reference
number of the callback can be accessed using the ``CkDataMsg`` argument
of the callback function. For a callback using a ``CkDataMsg *msg``, the
reference number is obtained by using the method ``CkGetRefNum(msg);``.
This is illustrated in
``examples/charm++/zerocopy/direct_api/unreg/simple_get``. specifically
useful where there is an indexed collection of buffers, where the
reference number can be used to index the collection.

Note that the CkNcpyBuffer objects can be either statically declared or
be dynamically allocated. Additionally, the objects are also reusable
across iteration boundaries i.e. after sending the CkNcpyBuffer object,
the remote PE can use the same object to perform get or put. This
pattern of using the same objects across iterations is demonstrated in
``examples/charm++/zerocopy/direct_api/reg/pingpong``.

This API is demonstrated in ``examples/charm++/zerocopy/direct_api``

Memory Registration and Modes of Operation
''''''''''''''''''''''''''''''''''''''''''

There are four modes of operation for the Zero Copy Direct API. These
modes act as control switches on networks that require memory
registration like GNI, OFI and Verbs, in order to perform RDMA
operations . They dictate the functioning of the API providing flexible
options based on user requirement. On other networks, where network
memory management is not necessary (Netlrts) or is internally handled by
the lower layer networking API (PAMI, MPI), these switches are still
supported to maintain API consistency by all behaving in the similar
default mode of operation.

``CK_BUFFER_REG``:


``CK_BUFFER_REG`` is the default mode that is used when no mode is
passed. This mode doesn’t distinguish between non-network and network
data transfers. When this mode is passed, the buffer is registered
immediately and this can be used for both non-network sends (memcpy) and
network sends without requiring an extra message being sent by the
runtime system for the latter case. This mode is demonstrated in
``examples/charm++/zerocopy/direct_api/reg``

``CK_BUFFER_UNREG``:


When this mode is passed, the buffer is initially unregistered and it
is registered only for network transfers where registration is
absolutely required. For example, if the target buffer is on the same PE
or same logical node (or process), since the get internally performs a
memcpy, registration is avoided for non-network transfers. On the other
hand, if the target buffer resides on a remote PE on a different logical
node, the get is implemented through an RDMA call requiring
registration. In such a case, there is a small message sent by the RTS
to register and perform the RDMA operation. This mode is demonstrated in
``examples/charm++/zerocopy/direct_api/unreg``

``CK_BUFFER_PREREG``:


This mode is only beneficial by implementations that use
pre-registered memory pools. In Charm++, GNI and Verbs machine layers
use pre-registered memory pools for avoiding registration costs. On
other machine layers, this mode is supported, but it behaves similar to
``CK_BUFFER_REG``. A custom allocator, ``CkRdmaAlloc`` can be used to
allocate a buffer from a pool of pre-registered memory to avoid the
expensive malloc and memory registration costs. For a buffer allocated
through ``CkRdmaAlloc``, the mode ``CK_BUFFER_PREREG`` should be passed
to indicate the use of a mempooled buffer to the RTS. A buffer allocated
with ``CkRdmaAlloc`` can be deallocated by calling a custom deallocator,
``CkRdmaFree``. Although the allocator ``CkRdmaAlloc`` and deallocator,
``CkRdmaFree`` are beneficial on GNI and Verbs, the allocators are
functional on other networks and allocate regular memory similar to a
``malloc`` call. Importantly, it should be noted that with the
``CK_BUFFER_PREREG`` mode, the allocated buffer’s pointer should be used
without any offsets. Using a buffer pointer with an offset will cause a
segmentation fault. This mode is demonstrated in
``examples/charm++/zerocopy/direct_api/prereg``

``CK_BUFFER_NOREG``:


This mode is used for just storing pointer information without any
actual networking or registration information. It cannot be used for
performing any zerocopy operations. This mode was added as it was useful
for implementing a runtime system feature.

Memory De-registration
''''''''''''''''''''''

Similar to memory registration, there is a method available to
de-register memory after the completion of the operations. This allows
for other buffers to use the registered memory as machines/networks are
limited by the maximum amount of registered or pinned memory. Registered
memory can be de-registered by calling the ``deregisterMem()`` method on
the CkNcpyBuffer object.

Other Methods
'''''''''''''

In addition to ``deregisterMem()``, there are other methods in a
CkNcpyBuffer object that offer other utilities. The
``init(const void *ptr, size_t cnt, CkCallback &cb, unsigned short int mode=CK_BUFFER_UNREG)``
method can be used to re-initialize the CkNcpyBuffer object to new
values similar to the ones that were passed in the constructor. For
example, after using a CkNcpyBuffer object called ``srcInfo``, the user
can re-initialize the same object with other values. This is shown in
the following code snippet.

.. code-block:: c++

   // initialize src with new values
   src.init(ptr, 200, newCb, CK_BUFFER_REG);

Additionally, the user can use another method ``registerMem`` in order
to register a buffer that has been de-registered. Note that it is not
required to call ``registerMem`` after a new initialization using
``init`` as ``registerMem`` is internally called on every new
initialization. The usage of ``registerMem`` is illustrated in the
following code snippet. Additionally, also note that following
de-registration, if intended to register again, it is required to call
``registerMem`` even in the ``CK_BUFFER_PREREG`` mode when the buffer is
allocated from a preregistered mempool. This is required to set the
registration memory handles and will not incur any registration costs.

.. code-block:: c++

   // register previously de-registered buffer
   src.registerMem();

Zero Copy Entry Method Send API
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The Zero copy Entry Method Send API only allows the user to only avoid
the sender side copy without avoiding the receiver side copy. It
offloads the user from the responsibility of making additional calls to
support zero copy semantics. It extends the capability of the existing
entry methods with slight modifications in order to send large buffers
without a copy.

To send an array using the zero copy message send API, specify the array
parameter in the .ci file with the nocopy specifier.

.. code-block:: c++

   entry void foo (int size, nocopy int arr[size]);

While calling the entry method from the .C file, wrap the array i.e the
pointer in a CkSendBuffer wrapper.

.. code-block:: c++

   arrayProxy[0].foo(500000, CkSendBuffer(arrPtr));

Until the RDMA operation is completed, it is not safe to modify the
buffer. To be notified on completion of the RDMA operation, pass an
optional callback object in the CkSendBuffer wrapper associated with the
specific nocopy array.

.. code-block:: c++

   CkCallback cb(CkIndex_Foo::zerocopySent(NULL), thisProxy[thisIndex]);
   arrayProxy[0].foo(500000, CkSendBuffer(arrPtr, cb));

The callback will be invoked on completion of the RDMA operation
associated with the corresponding array. Inside the callback, it is safe
to overwrite the buffer sent via the zero copy entry method send API and
this buffer can be accessed by dereferencing the CkDataMsg received in
the callback.

.. code-block:: c++

   //called when RDMA operation is completed
   void zerocopySent(CkDataMsg *m)
   {
     //get access to the pointer and free the allocated buffer
     void *ptr = *((void **)(m->data));
     free(ptr);
     delete m;
   }

The RDMA call is associated with a nocopy array rather than the entry
method. In the case of sending multiple nocopy arrays, each RDMA call is
independent of the other. Hence, the callback applies to only the array
it is attached to and not to all the nocopy arrays passed in an entry
method invocation. On completion of the RDMA call for each array, the
corresponding callback is separately invoked.

As an example, for an entry method with two nocopy array parameters,
each called with the same callback, the callback will be invoked twice:
on completing the transfer of each of the two nocopy parameters.

For multiple arrays to be sent via RDMA, declare the entry method in the
.ci file as:

.. code-block:: c++

   entry void foo (int size1, nocopy int arr1[size1], int size2, nocopy double arr2[size2]);

In the .C file, it is also possible to have different callbacks
associated with each nocopy array.

.. code-block:: c++

   CkCallback cb1(CkIndex_Foo::zerocopySent1(NULL), thisProxy[thisIndex]);
   CkCallback cb2(CkIndex_Foo::zerocopySent2(NULL), thisProxy[thisIndex]);
   arrayProxy[0].foo(500000, CkSendBuffer(arrPtr1, cb1), 1024000, CkSendBuffer(arrPtr2, cb2));

This API is demonstrated in
``examples/charm++/zerocopy/entry_method_api`` and
``tests/charm++/pingpong``

It should be noted that calls to entry methods with nocopy specified
parameters are currently only supported for point to point operations
and not for collective operations. Additionally, there is also no
support for migration of chares that have pending RDMA transfer
requests.

It should also be noted that the benefit of this API can be seen for
large arrays on only RDMA enabled networks. On networks which do not
support RDMA and for within process sends (which uses shared memory),
the API is functional but doesn’t show any performance benefit as it
behaves like a regular entry method that copies its arguments.

Table :numref:`tab:rdmathreshold` displays the
message size thresholds for the zero copy entry method send API on
popular systems and build architectures. These results were obtained by
running ``examples/charm++/zerocopy/entry_method_api/pingpong`` in
non-SMP mode on production builds. For message sizes greater than or
equal to the displayed thresholds, the zero copy API is found to perform
better than the regular message send API. For network layers that are
not pamilrts, gni, verbs, ofi or mpi, the generic implementation is
used.

.. table:: Message Size Thresholds for which Zero Copy Entry API performs better than Regular API
   :name: tab:rdmathreshold

   ============================= =============== ====================== =============== ========== ==========
   Machine                       Network         Build Architecture     Intra Processor Intra Host Inter Host
   ============================= =============== ====================== =============== ========== ==========
   Blue Gene/Q (Vesta)           PAMI            ``pamilrts-bluegeneq`` 4 MB            32 KB      256 KB
   Cray XC30 (Edison)            Aries           ``gni-crayxc``         1 MB            2 MB       2 MB
   Cray XC30 (Edison)            Aries           ``mpi-crayxc``         256 KB          8 KB       32 KB
   Dell Cluster (Golub)          Infiniband      ``verbs-linux-x86_64`` 128 KB          2 MB       1 MB
   Dell Cluster (Golub)          Infiniband      ``mpi-linux-x86_64``   128 KB          32 KB      64 KB
   Intel Cluster (Bridges)       Intel Omni-Path ``ofi-linux-x86_64``   64 KB           32 KB      32 KB
   Intel KNL Cluster (Stampede2) Intel Omni-Path ``ofi-linux-x86_64``   1 MB            64 KB      64 KB
   ============================= =============== ====================== =============== ========== ==========

.. _callbacks:

Callbacks
---------

Callbacks provide a generic way to store the information required to
invoke a communication target, such as a chare’s entry method, at a
future time. Callbacks are often encountered when writing library code,
where they provide a simple way to transfer control back to a client
after the library has finished. For example, after finishing a
reduction, you may want the results passed to some chare’s entry method.
To do this, you would create an object of type CkCallback with the
chare’s CkChareID and entry method index, and pass this callback object
to the reduction library.

.. _sec:callbacks/creating:

Creating a CkCallback Object
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

There are several different types of CkCallback objects; the type of the
callback specifies the intended behavior upon invocation of the
callback. Callbacks must be invoked with the Charm++ message of the type
specified when creating the callback. If the callback is being passed
into a library which will return its result through the callback, it is
the user’s responsibility to ensure that the type of the message
delivered by the library is the same as that specified in the callback.
Messages delivered through a callback are not automatically freed by the
Charm RTS. They should be freed or stored for future use by the user.

Callbacks that target chares require an “entry method index”, an integer
that identifies which entry method will be called. An entry method index
is the Charm++ version of a function pointer. The entry method index can
be obtained using the syntax:

.. code-block:: c++

   int myIdx = CkIndex_ChareName::EntryMethod(parameters);

Here, ChareName is the name of the chare (group, or array) containing
the desired entry method, EntryMethod is the name of that entry method,
and parameters are the parameters taken by the method. These parameters
are only used to resolve the proper EntryMethod; they are otherwise
ignored.

Under most circumstances, entry methods to be invoked through a
CkCallback must take a single message pointer as argument. As such, if
the entry method specified in the callback is not overloaded, using NULL
in place of parameters will suffice in fully specifying the intended
target. If the entry method is overloaded, a message pointer of the
appropriate type should be defined and passed in as a parameter when
specifying the entry method. The pointer does not need to be initialized
as the argument is only used to resolve the target entry method.

The intended behavior upon a callback’s invocation is specified through
the choice of callback constructor used when creating the callback.
Possible constructors are:

#. CkCallback(int ep, const CkChareID &id) - When invoked, the callback
   will send its message to the given entry method (specified by the
   entry point index - ep) of the given Chare (specified by the chare
   id). Note that a chare proxy will also work in place of a chare id:

   .. code-block:: c++

      CkCallback(CkIndex_Foo::bar(NULL), thisProxy[thisIndex])

#. CkCallback(void (\*CallbackFn)(void \*, void \*), void \*param) - When
   invoked, the callback will pass param and the result message to the
   given C function, which should have a prototype like:

   .. code-block:: c++

      void myCallbackFn(void *param, void *message)

   This function will be called on the processor where the callback was
   created, so param is allowed to point to heap-allocated data. Hence,
   this constructor should be used only when it is known that the
   callback target (which by definition here is just a C-like function)
   will be on the same processor as from where the constructor was
   called. Of course, you are required to free any storage referenced by
   param.

#. CkCallback(CkCallback::ignore) - When invoked, the callback will do
   nothing. This can be useful if a Charm++ library requires a callback,
   but you don’t care when it finishes, or will find out some other way.

#. CkCallback(CkCallback::ckExit) - When invoked, the callback will call
   CkExit(), ending the Charm++ program.

#. CkCallback(int ep, const CkArrayID &id) - When invoked, the callback
   will broadcast its message to the given entry method of the given
   array. An array proxy will work in the place of an array id.

#. CkCallback(int ep, const CkArrayIndex &idx, const CkArrayID &id) -
   When invoked, the callback will send its message to the given entry
   method of the given array element.

#. CkCallback(int ep, const CkGroupID &id) - When invoked, the callback
   will broadcast its message to the given entry method of the given
   group.

#. CkCallback(int ep, int onPE, const CkGroupID &id) - When invoked, the
   callback will send its message to the given entry method of the given
   group member.

One final type of callback, CkCallbackResumeThread(), can only be used
from within threaded entry methods. This callback type is discussed in
section :numref:`sec:ckcallbackresumethread`.

.. _libraryInterface:

CkCallback Invocation
~~~~~~~~~~~~~~~~~~~~~

A properly initialized CkCallback object stores a global destination
identifier, and as such can be freely copied, marshalled, and sent in
messages. Invocation of a CkCallback is done by calling the function
send on the callback with the result message as an argument. As an
example, a library which accepts a CkCallback object from the user and
then invokes it to return a result may have the following interface:

.. code-block:: c++

   //Main library entry point, called by asynchronous users:
   void myLibrary(...library parameters...,const CkCallback &cb)
   {
     ..start some parallel computation, store cb to be passed to myLibraryDone later...
   }

   //Internal library routine, called when computation is done
   void myLibraryDone(...parameters...,const CkCallback &cb)
   {
     ...prepare a return message...
     cb.send(msg);
   }

A CkCallback will accept any message type, or even NULL. The message is
immediately sent to the user’s client function or entry point. A library
which returns its result through a callback should have a clearly
documented return message type. The type of the message returned by the
library must be the same as the type accepted by the entry method
specified in the callback.

As an alternative to “send”, the callback can be used in a *contribute*
collective operation. This will internally invoke the “send” method on
the callback when the contribute operation has finished.

For examples of how to use the various callback types, please see
``tests/charm++/megatest/callback.C``

.. _sec:ckcallbackresumethread:

Synchronous Execution with CkCallbackResumeThread
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Threaded entry methods can be suspended and resumed through the
*CkCallbackResumeThread* class. *CkCallbackResumeThread* is derived from
*CkCallback* and has specific functionality for threads. This class
automatically suspends the thread when the destructor of the callback is
called. A suspended threaded client will resume when the “send” method
is invoked on the associated callback. It can be used in situations when
the return value is not needed, and only the synchronization is
important. For example:

.. code-block:: c++

   // Call the "doWork" method and wait until it has completed
   void mainControlFlow() {
     ...perform some work...
     // call a library
     doWork(...,CkCallbackResumeThread());
     // or send a broadcast to a chare collection
     myProxy.doWork(...,CkCallbackResumeThread());
     // callback goes out of scope; the thread is suspended until doWork calls 'send' on the callback

     ...some more work...
   }

Alternatively, if doWork returns a value of interest, this can be
retrieved by passing a pointer to *CkCallbackResumeThread*. This pointer
will be modified by *CkCallbackResumeThread* to point to the incoming
message. Notice that the input pointer has to be cast to *(void*&)*:

.. code-block:: c++

   // Call the "doWork" method and wait until it has completed
   void mainControlFlow() {
     ...perform some work...
     MyMessage *mymsg;
     myProxy.doWork(...,CkCallbackResumeThread((void*&)mymsg));
     // The thread is suspended until doWork calls send on the callback

     ...some more work using "mymsg"...
   }

Notice that the instance of *CkCallbackResumeThread* is constructed as
an anonymous parameter to the “doWork” call. This insures that the
callback is destroyed as soon as the function returns, thereby
suspending the thread.

It is also possible to allocate a *CkCallbackResumeThread* on the heap
or on the stack. We suggest that programmers avoid such usage, and favor
the anonymous instance construction shown above. For completeness, we
still present the code for heap and stack allocation of
CkCallbackResumeThread callbacks below.

For heap allocation, the user must explicitly “delete” the callback in
order to suspend the thread.

.. code-block:: c++

   // Call the "doWork" method and wait until it has completed
   void mainControlFlow() {
     ...perform some work...
     CkCallbackResumeThread cb = new CkCallbackResumeThread();
     myProxy.doWork(...,cb);
     ...do not suspend yet, continue some more work...
     delete cb;
     // The thread suspends now

     ...some more work after the thread resumes...
   }

For a callback that is allocated on the stack, its destructor will be
called only when the callback variable goes out of scope. In this
situation, the function “thread_delay” can be invoked on the callback to
force the thread to suspend. This also works for heap allocated
callbacks.

.. code-block:: c++

   // Call the "doWork" method and wait until it has completed
   void mainControlFlow() {
     ...perform some work...
     CkCallbackResumeThread cb;
     myProxy.doWork(...,cb);
     ...do not suspend yet, continue some more work...
     cb.thread_delay();
     // The thread suspends now

     ...some more work after the thread is resumed...
   }

In all cases a *CkCallbackResumeThread* can be used to suspend a
thread only once.
(See Main.cpp of `Barnes-Hut
MiniApp <http://charmplusplus.org/miniApps/#barnes>`__ for a complete
example).
*Deprecated usage*: in the past, “thread_delay” was used to retrieve
the incoming message from the callback. While that is still allowed
for backward compatibility, its usage is deprecated. The old usage is
subject to memory leaks and dangling pointers.

Callbacks can also be tagged with reference numbers which can be
matched inside SDAG code. When the callback is created, the creator
can set the refnum and the runtime system will ensure that the message
invoked on the callback’s destination will have that refnum. This
allows the receiver of the final callback to match the messages based
on the refnum value. (See
``examples/charm++/examples/charm++/ckcallback`` for a complete
example).

Waiting for Completion
----------------------

.. _threaded:

Threaded Entry Methods
~~~~~~~~~~~~~~~~~~~~~~

Typically, entry methods run in the same thread of execution as the
Charm++ scheduler. This prevents them from undertaking any actions that
would cause their thread to block, as blocking would prevent the
receiving and processing of incoming messages.

However, entry methods with the threaded attribute run in their own
user-level nonpreemptible thread, and are therefore able to block
without interrupting the runtime system. This allows them to undertake
blocking operations or explicitly suspend themselves, which is necessary
to use some Charm++ features, such as sync entry methods and futures.

For details on the threads API available to threaded entry methods, see
chapter 3 of the Converse programming manual. The use of threaded entry
methods is demonstrated in an example program located in
``examples/charm++/threaded_ring``.

.. _sync:

Sync Entry Methods
~~~~~~~~~~~~~~~~~~

Generally, entry methods are invoked asynchronously and return ``void``.
Therefore, while an entry method may send data back to its invoker, it
can only do so by invoking another asynchronous entry method on the
chare object that invoked it.

However, it is possible to use sync entry methods, which have blocking
semantics. The data returned by the invocation of such an entry method
is available at the call site when it returns from blocking. This
returned data can either be in the form of a Charm++ message or any type
that has the PUP method implemented. Because the caller of a sync entry
method will block, it must execute in a thread separate from the
scheduler; that is, it must be a threaded entry method (*cf.*
§ :numref:`threaded`, above). If a sync entry method returns a value,
it is provided as the return value from the invocation on the proxy
object:

.. code-block:: c++

    ReturnMsg* m;
    m = A[i].foo(a, b, c);

An example of the use of sync entry methods is given in
``tests/charm++/sync_square``.

Futures
~~~~~~~

Similar to Multilisp and other functional programming languages,
Charm++ provides the abstraction of *futures*. In simple terms, a
*future* is a contract with the runtime system to evaluate an expression
asynchronously with the calling program. This mechanism promotes the
evaluation of expressions in parallel as several threads concurrently
evaluate the futures created by a program.

In some ways, a future resembles lazy evaluation. Each future is
assigned to a particular thread (or to a chare, in Charm++) and,
eventually, its value is delivered to the calling program. Once a future
is created, a *reference* is returned immediately. However, if the
*value* calculated by the future is needed, the calling program blocks
until the value is available.

Charm++ provides all the necessary infrastructure to use futures by
means of the following functions:

.. code-block:: c++

    CkFuture CkCreateFuture(void)
    void CkReleaseFuture(CkFuture fut)
    int CkProbeFuture(CkFuture fut)
    void *CkWaitFuture(CkFuture fut)
    void CkSendToFuture(CkFuture fut, void *msg)

To illustrate the use of all these functions, a Fibonacci example in
Charm++ using futures in presented below:

.. code-block:: c++

   chare fib {
     entry fib(bool amIroot, int n, CkFuture f);
     entry [threaded] void run(bool amIroot, int n, CkFuture f);
   };

.. code-block:: c++

   void  fib::run(bool amIRoot, int n, CkFuture f) {
      if (n < THRESHOLD)
       result = seqFib(n);
     else {
       CkFuture f1 = CkCreateFuture();
       CkFuture f2 = CkCreateFuture();
       CProxy_fib::ckNew(0, n-1, f1);
       CProxy_fib::ckNew(0, n-2, f2);
       ValueMsg * m1 = (ValueMsg *) CkWaitFuture(f1);
       ValueMsg * m2 = (ValueMsg *) CkWaitFuture(f2);
       result = m1->value + m2->value;
       delete m1; delete m2;
     }
     if (amIRoot) {
       CkPrintf("The requested Fibonacci number is : %d\n", result);
       CkExit();
     } else {
       ValueMsg *m = new ValueMsg();
       m->value = result;
       CkSendToFuture(f, m);
     }
   }

The constant *THRESHOLD* sets a limit value for computing the Fibonacci
number with futures or just with the sequential procedure. Given value
*n*, the program creates two futures using *CkCreateFuture*. Those
futures are used to create two new chares that will carry out the
computation. Next, the program blocks until the two component values of
the recurrence have been evaluated. Function *CkWaitFuture* is used for
that purpose. Finally, the program checks whether or not it is the root
of the recursive evaluation. The very first chare created with a future
is the root. If a chare is not the root, it must indicate that its
future has finished computing the value. *CkSendToFuture* is meant to
return the value for the current future.

Other functions complete the API for futures. *CkReleaseFuture* destroys
a future. *CkProbeFuture* tests whether the future has already finished
computing the value of the expression.

The Converse version of future functions can be found in the :ref:`conv-futures`
section.

.. _sec-completion:

Completion Detection
~~~~~~~~~~~~~~~~~~~~

Completion detection is a method for automatically detecting completion
of a distributed process within an application. This functionality is
helpful when the exact number of messages expected by individual objects
is not known. In such cases, the process must achieve global consensus
as to the number of messages produced and the number of messages
consumed. Completion is reached within a distributed process when the
participating objects have produced and consumed an equal number of
events globally. The number of global events that will be produced and
consumed does not need to be known, just the number of producers is
required.

The completion detection feature is implemented in Charm++ as a module,
and therefore is only included when ``-module completion`` is
specified when linking your application.

First, the detector should be constructed. This call would typically
belong in application startup code (it initializes the group that keeps
track of completion):

.. code-block:: c++

   CProxy_CompletionDetector detector = CProxy_CompletionDetector::ckNew();

When it is time to start completion detection, invoke the following
method of the library on *all* branches of the completion detection
group:

.. code-block:: c++

   void start_detection(int num_producers,
                        CkCallback start,
                        CkCallback all_produced,
                        CkCallback finish,
                        int prio);

The ``num_producers`` parameter is the number of objects (chares) that
will produce elements. So if every chare array element will produce one
event, then it would be the size of the array.

The ``start`` callback notifies your program that it is safe to begin
producing and consuming (this state is reached when the module has
finished its internal initialization).

The ``all_produced`` callback notifies your program when the client has
called ``done`` with arguments summing to ``num_producers``.

The ``finish`` callback is invoked when completion has been detected
(all objects participating have produced and consumed an equal number of
elements globally).

The ``prio`` parameter is the priority with which the completion
detector will run. This feature is still under development, but it
should be set below the application’s priority if possible.

For example, the call

.. code-block:: c++

   detector.start_detection(10,
                            CkCallback(CkIndex_chare1::start_test(), thisProxy),
                            CkCallback(CkIndex_chare1::produced_test(), thisProxy),
                            CkCallback(CkIndex_chare1::finish_test(), thisProxy),
                            0);

sets up completion detection for 10 producers. Once initialization is
done, the callback associated with the ``start_test`` method will be
invoked. Once all 10 producers have called ``done`` on the completion
detector, the ``produced_test`` method will be invoked. Furthermore,
when the system detects completion, the callback associated with
``finish_test`` will be invoked. Finally, the priority given to the
completion detection library is set to 0 in this case.

Once initialization is complete (the “start” callback is triggered),
make the following call to the library:

.. code-block:: c++

   void CompletionDetector::produce(int events_produced)
   void CompletionDetector::produce() // 1 by default

For example, within the code for a chare array object, you might make
the following call:

.. code-block:: c++

   detector.ckLocalBranch()->produce(4);

Once all the “events” that this chare is going to produce have been sent
out, make the following call:

.. code-block:: c++

   void CompletionDetector::done(int producers_done)
   void CompletionDetector::done() // 1 by default

.. code-block:: c++

   detector.ckLocalBranch()->done();

At the same time, objects can also consume produced elements, using the
following calls:

.. code-block:: c++

   void CompletionDetector::consume(int events_consumed)
   void CompletionDetector::consume() // 1 by default

.. code-block:: c++

   detector.ckLocalBranch()->consume();

Note that an object may interleave calls to ``produce()`` and
``consume()``, i.e. it could produce a few elements, consume a few, etc.
When it is done producing its elements, it should call ``done()``, after
which cannot ``produce()`` any more elements. However, it can continue
to ``consume()`` elements even after calling ``done()``. When the
library detects that, globally, the number of produced elements equals
the number of consumed elements, and all producers have finished
producing (i.e. called ``done()``), it will invoke the ``finish``
callback. Thereafter, ``start_detection`` can be called again to restart
the process.

.. _sec:qd:

Quiescence Detection
~~~~~~~~~~~~~~~~~~~~

In Charm++, quiescence is defined as the state in which no processor is
executing an entry point, no messages are awaiting processing, and there
are no messages in-flight. Charm++ provides two facilities for detecting
quiescence: CkStartQD and CkWaitQD. CkStartQD registers with the system
a callback that is to be invoked the next time quiescence is detected.
Note that if immediate messages are used, QD cannot be used. CkStartQD
has two variants which expect the following arguments:

#. A CkCallback object. The syntax of this call looks like:

   .. code-block:: c++

        CkStartQD(const CkCallback& cb);

   Upon quiescence detection, the specified callback is called with no
   parameters. Note that using this variant, you could have your program
   terminate after quiescence is detected, by supplying the above method
   with a CkExit callback (§ :numref:`sec:callbacks/creating`).

#. An index corresponding to the entry function that is to be called,
   and a handle to the chare on which that entry function should be
   called. The syntax of this call looks like this:

   .. code-block:: c++

       CkStartQD(int Index,const CkChareID* chareID);

   To retrieve the corresponding index of a particular entry method, you
   must use a static method contained within the (charmc-generated)
   CkIndex object corresponding to the chare containing that entry
   method. The syntax of this call is as follows:

   .. code-block:: c++

      myIdx=CkIndex_ChareClass::entryMethod(parameters);

   where ChareClass is the C++ class of the chare containing the desired
   entry method, entryMethod is the name of that entry method, and
   parameters are the parameters taken by the method. These parameters
   are only used to resolve the proper entryMethod; they are otherwise
   ignored.

CkWaitQD, by contrast, does not register a callback. Rather, CkWaitQD
*blocks* and does not return until quiescence is detected. It takes no
parameters and returns no value. A call to CkWaitQD simply looks like
this:

.. code-block:: c++

     CkWaitQD();

Note that CkWaitQD should only be called from a threaded entry method
because a call to CkWaitQD suspends the current thread of execution
(*cf.* § :numref:`threaded`).

.. _advanced arrays:

More Chare Array Features
-------------------------

The basic array features described previously (creation, messaging,
broadcasts, and reductions) are needed in almost every Charm++ program.
The more advanced techniques that follow are not universally needed, but
represent many useful optimizations.

.. _ckLocal for arrays:

Local Access
~~~~~~~~~~~~

It is possible to get direct access to a local array element using the
proxy’s ckLocal method, which returns an ordinary C++ pointer to the
element if it exists on the local processor, and NULL if the element
does not exist or is on another processor.

.. code-block:: c++

   A1 *a=a1[i].ckLocal();
   if (a==NULL) // ...is remote -- send message
   else // ...is local -- directly use members and methods of a

Note that if the element migrates or is deleted, any pointers obtained
with ckLocal are no longer valid. It is best, then, to either avoid
ckLocal or else call ckLocal each time the element may have migrated;
e.g., at the start of each entry method.

An example of this usage is available in
``examples/charm++/topology/matmul3d``.

.. _advanced array create:

Advanced Array Creation
~~~~~~~~~~~~~~~~~~~~~~~

There are several ways to control the array creation process. You can
adjust the map and bindings before creation, change the way the initial
array elements are created, create elements explicitly during the
computation, and create elements implicitly, “on demand”.

You can create all of an arrays elements using any one of these methods,
or create different elements using different methods. An array element
has the same syntax and semantics no matter how it was created.

.. _CkArrayOptions:

Configuring Array Characteristics Using CkArrayOptions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The array creation method ckNew actually takes a parameter of type
CkArrayOptions. This object describes several optional attributes of the
new array.

The most common form of CkArrayOptions is to set the number of initial
array elements. A CkArrayOptions object will be constructed
automatically in this special common case. Thus the following code
segments all do exactly the same thing:

.. code-block:: c++

   // Implicit CkArrayOptions
   a1=CProxy_A1::ckNew(parameters,nElements);

   // Explicit CkArrayOptions
   a1=CProxy_A1::ckNew(parameters,CkArrayOptions(nElements));

   // Separate CkArrayOptions
   CkArrayOptions opts(nElements);
   a1=CProxy_A1::ckNew(parameters,opts);

Note that the “numElements” in an array element is simply the
numElements passed in when the array was created. The true number of
array elements may grow or shrink during the course of the computation,
so numElements can become out of date. This “bulk” constructor approach
should be preferred where possible, especially for large arrays. Bulk
construction is handled via a broadcast which will be significantly more
efficient in the number of messages required than inserting each element
individually, which will require one message send per element.

Examples of bulk construction are commonplace, see
``examples/charm++/jacobi3d-sdag`` for a demonstration of the slightly
more complicated case of multidimensional chare array bulk construction.

CkArrayOptions can also be used for bulk creation of sparse arrays when
the sparsity of the array can be described in terms of a start index, an
end index, and a step index. The start, end, and step can either be
passed into the CkArrayOptions constructor, or set one at a time. The
following shows two different ways to create CkArrayOptions for a 2D
array with only the odd indices from (1,1) to (10,10) being populated:

.. code-block:: c++

   // Set at construction
   CkArrayOptions options(CkArrayIndex2D(1,1),
                          CkArrayIndex2D(10,10),
                          CkArrayIndex(2,2));

   // Set one at a time
   CkArrayOptions options;
   options.setStart(CkArrayIndex2D(1,1))
          .setEnd(CkArrayIndex2D(10,10))
          .setStep(CkArrayIndex2D(2,2));

The default for start is :math:`0^d` and the default for step is
:math:`1^d` (where :math:`d` is the dimension of the array), so the
following are equivalent:

.. code-block:: c++

   // Specify just the number of elements
   CkArrayOptions options(nElements);

   // Specify just the end index
   CkArrayOptions options;
   options.setEnd(CkArrayIndex1D(nElements));

   // Specify all three indices
   CkArrayOptions options;
   options.setStart(CkArrayIndex1D(0))
          .setEnd(CkArrayIndex1D(nElements))
          .setStep(CkArrayIndex1D(1));

In addition to controlling how many elements and at which indices to
create them, CkArrayOptions contains a few flags that the runtime can
use to optimize handling of a given array. If the array elements will
only migrate at controlled points (such as periodic load balancing with
``AtASync()``), this is signaled to the runtime by calling
``opts.setAnytimeMigration(false)``\  [11]_. If all array elements will
be inserted by bulk creation or by ``fooArray[x].insert()`` calls,
signal this by calling ``opts.setStaticInsertion(true)``  [12]_.

.. _array map:

Initial Placement Using Map Objects
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can use CkArrayOptions to specify a “map object” for an array. The
map object is used by the array manager to determine the “home” PE of
each element. The home PE is the PE upon which it is initially placed,
which will retain responsibility for maintaining the location of the
element.

There is a default map object, which maps 1D array indices in a block
fashion to processors, and maps other array indices based on a hash
function. Some other mappings such as round-robin (RRMap) also exist,
which can be used similar to custom ones described below.

A custom map object is implemented as a group which inherits from
CkArrayMap and defines these virtual methods:

.. code-block:: c++

   class CkArrayMap : public Group
   {
   public:
     // ...

     // Return an ``arrayHdl'', given some information about the array
     virtual int registerArray(CkArrayIndex& numElements,CkArrayID aid);
     // Return the home processor number for this element of this array
     virtual int procNum(int arrayHdl,const CkArrayIndex &element);
   }

For example, a simple 1D blockmapping scheme. Actual mapping is handled
in the procNum function.

.. code-block:: c++

   class BlockMap : public CkArrayMap
   {
    public:
     BlockMap(void) {}
     BlockMap(CkMigrateMessage *m){}
     int registerArray(CkArrayIndex& numElements,CkArrayID aid) {
       return 0;
     }
     int procNum(int /*arrayHdl*/,const CkArrayIndex &idx) {
       int elem=*(int *)idx.data();
       int penum =  (elem/(32/CkNumPes()));
       return penum;
     }
   };

Note that the first argument to the procNum method exists for reasons
internal to the runtime system and is not used in the calculation of
processor numbers.

Once you’ve instantiated a custom map object, you can use it to control
the location of a new array’s elements using the setMap method of the
CkArrayOptions object described above. For example, if you’ve declared a
map object named “BlockMap”:

.. code-block:: c++

   // Create the map group
   CProxy_BlockMap myMap=CProxy_BlockMap::ckNew();

   // Make a new array using that map
   CkArrayOptions opts(nElements);
   opts.setMap(myMap);
   a1=CProxy_A1::ckNew(parameters,opts);

An example which constructs one element per physical node may be found
in ``examples/charm++/PUP/pupDisk``.

Other 3D Torus network oriented map examples are in
``examples/charm++/topology``.

.. _array initial:

Initial Elements
^^^^^^^^^^^^^^^^

The map object described above can also be used to create the initial
set of array elements in a distributed fashion. An array’s initial
elements are created by its map object, by making a call to
populateInitial on each processor.

You can create your own set of elements by creating your own map object
and overriding this virtual function of CkArrayMap:

.. code-block:: c++

   virtual void populateInitial(int arrayHdl,int numInitial,
           void *msg,CkArray *mgr)

In this call, arrayHdl is the value returned by registerArray,
numInitial is the number of elements passed to CkArrayOptions, msg is
the constructor message to pass, and mgr is the array to create.

populateInitial creates new array elements using the method ``void
CkArray::insertInitial(CkArrayIndex idx,void \*ctorMsg)``. For example, to
create one row of 2D array elements on each processor, you would write:

.. code-block:: c++

   void xyElementMap::populateInitial(int arrayHdl,int numInitial,
    void *msg,CkArray *mgr)
   {
     if (numInitial==0) return; //No initial elements requested

     //Create each local element
     int y=CkMyPe();
     for (int x=0;x<numInitial;x++) {
       mgr->insertInitial(CkArrayIndex2D(x,y),CkCopyMsg(&msg));
     }
     mgr->doneInserting();
     CkFreeMsg(msg);
   }

Thus calling ckNew(10) on a 3-processor machine would result in 30
elements being created.

Bound Arrays
^^^^^^^^^^^^

You can “bind” a new array to an existing array using the bindTo method
of CkArrayOptions. Bound arrays act like separate arrays in all ways
except for migration- corresponding elements of bound arrays always
migrate together. For example, this code creates two arrays A and B
which are bound together- A[i] and B[i] will always be on the same
processor.

.. code-block:: c++

   // Create the first array normally
   aProxy=CProxy_A::ckNew(parameters,nElements);
   // Create the second array bound to the first
   CkArrayOptions opts(nElements);
   opts.bindTo(aProxy);
   bProxy=CProxy_B::ckNew(parameters,opts);

An arbitrary number of arrays can be bound together- in the example
above, we could create yet another array C and bind it to A or B. The
result would be the same in either case- A[i], B[i], and C[i] will
always be on the same processor.

There is no relationship between the types of bound arrays- it is
permissible to bind arrays of different types or of the same type. It is
also permissible to have different numbers of elements in the arrays,
although elements of A which have no corresponding element in B obey no
special semantics. Any method may be used to create the elements of any
bound array.

Bound arrays are often useful if A[i] and B[i] perform different aspects
of the same computation, and thus will run most efficiently if they lie
on the same processor. Bound array elements are guaranteed to always be
able to interact using ckLocal (see
section :numref:`ckLocal for arrays`), although the local pointer
must be refreshed after any migration. This should be done during the
pup routine. When migrated, all elements that are bound together will be
created at the new processor before pup is called on any of them,
ensuring that a valid local pointer to any of the bound objects can be
obtained during the pup routine of any of the others.

For example, an array *Alibrary* is implemented as a library module. It
implements a certain functionality by operating on a data array *dest*
which is just a pointer to some user provided data. A user defined array
*UserArray* is created and bound to the array *Alibrary* to take
advantage of the functionality provided by the library. When bound array
element migrated, the *data* pointer in *UserArray* is re-allocated in
*pup()*, thus *UserArray* is responsible to refresh the pointer *dest*
stored in *Alibrary*.

.. code-block:: c++

   class Alibrary: public CProxy_Alibrary {
   public:
     ...
     void set_ptr(double *ptr) { dest = ptr; }
     virtual void pup(PUP::er &p);
   private:
     double *dest; // point to user data in user defined bound array
   };

   class UserArray: public CProxy_UserArray {
   public:
     virtual void pup(PUP::er &p) {
       p|len;
       if(p.isUnpacking()) {
         data = new double[len];
         Alibrary *myfellow = AlibraryProxy(thisIndex).ckLocal();
         myfellow->set_ptr(data); // refresh data in bound array
       }
       p(data, len);
     }
   private:
     CProxy_Alibrary AlibraryProxy; // proxy to my bound array
     double *data; // user allocated data pointer
     int len;
   };

A demonstration of bound arrays can be found in
``tests/charm++/startupTest``

Note that if any bound array element sets *usesAtSync=true* in its
constructor, then users must ensure that *AtSync()* is called on all of
those array elements. If a bound array element does not have the
usesAtSync flag set to true, then it will migrate along with any
elements it is bound to when they migrate. In this case, such an array
element does not need to call *AtSync()* itself.

.. _dynamic_insertion:

Dynamic Insertion
^^^^^^^^^^^^^^^^^

In addition to creating initial array elements using ckNew, you can also
create array elements during the computation.

You insert elements into the array by indexing the proxy and calling
insert. The insert call optionally takes parameters, which are passed to
the constructor; and a processor number, where the element will be
created. Array elements can be inserted in any order from any processor
at any time. Array elements need not be contiguous.

If using insert to create all the elements of the array, you must call
CProxy_Array::doneInserting before using the array.

.. code-block:: c++

   // In the .C file:
   int x,y,z;
   CProxy_A1 a1=CProxy_A1::ckNew(); // Creates a new, empty 1D array
   for (x=...) {
      a1[x  ].insert(parameters); // Bracket syntax
      a1(x+1).insert(parameters); // or equivalent parenthesis syntax
   }
   a1.doneInserting();

   CProxy_A2 a2=CProxy_A2::ckNew(); // Creates 2D array
   for (x=...) for (y=...)
      a2(x,y).insert(parameters); // Can't use brackets!
   a2.doneInserting();

   CProxy_A3 a3=CProxy_A3::ckNew(); // Creates 3D array
   for (x=...) for (y=...) for (z=...)
      a3(x,y,z).insert(parameters);
   a3.doneInserting();

   CProxy_AF aF=CProxy_AF::ckNew(); // Creates user-defined index array
   for (...) {
      aF[CkArrayIndexFoo(...)].insert(parameters); // Use brackets...
      aF(CkArrayIndexFoo(...)).insert(parameters); // ...or parenthesis
   }
   aF.doneInserting();

The doneInserting call starts the reduction manager (see “Array
Reductions”) and load balancer (see  :numref:`lbFramework`)- since
these objects need to know about all the array’s elements, they must be
started after the initial elements are inserted. You may call
doneInserting multiple times, but only the first call actually does
anything. You may even insert or destroy elements after a call to
doneInserting, with different semantics- see the reduction manager and
load balancer sections for details.

If you do not specify one, the system will choose a processor to create
an array element on based on the current map object.

A demonstration of dynamic insertion is available:
``examples/charm++/hello/fancyarray``

Demand Creation
^^^^^^^^^^^^^^^

Demand Creation is a specialized form of dynamic insertion. Normally,
invoking an entry method on a nonexistent array element is an error. But
if you add the attribute [createhere] or [createhome] to an entry
method, the array manager will “demand create” a new element to handle
the message.

With [createhome], the new element will be created on the home
processor, which is most efficient when messages for the element may
arrive from anywhere in the machine. With [createhere], the new element
is created on the sending processor, which is most efficient if when
messages will often be sent from that same processor.

The new element is created by calling its default (taking no parameters)
constructor, which must exist and be listed in the .ci file. A single
array can have a mix of demand-creation and classic entry methods; and
demand-created and normally created elements.

A simple example of demand creation ``tests/charm++/demand_creation``.

.. _asynchronous_array_creation:

Asynchronous Array Creation
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Normally, CProxy_Array::ckNew call must always be made from PE 0.
However, asynchronous array creation can be used to lift this
restriction and let the array creation be made from any PE. To do this,
CkCallback must be given as an argument for ckNew to provide the created
chare array’s CkArrayID to the callback function.

.. code-block:: c++

   CProxy_SomeProxy::ckNew(parameters, nElements, CkCallback(CkIndex_MyClass::someFunction(NULL), thisProxy));

   void someFunction(CkArrayCreatedMsg *m) {
       CProxy_AnotherProxy(m->aid)[index].myFunction(); // m->aid is CkArrayID
       delete m;
   }

Similar to the standard array creation method, arguments to the array
element constructor calls are taken first, followed by the dimensions of
the array. Note that the parameters field can be optional, if the
default constructor is expected to be used.

Alternatively, CkArrayOptions can be used in place of nElements to
further configure array characteristics.

.. code-block:: c++

   // Creating a 3-dimensional chare array with 2 parameters
   CkArrayOptions options(dimX, dimY, dimZ);
   CProxy_AnotherProxy::ckNew(param1, param2, options, CkCallback(CkIndex_SomeClass::anotherFunction(NULL), thisProxy));

A demonstration of asynchronous array creation can be found in
``examples/charm++/hello/darray``.

.. _user-defined array index type:

User-defined Array Indices
~~~~~~~~~~~~~~~~~~~~~~~~~~

Charm++ array indices are arbitrary collections of integers. To define a
new array index type, you create an ordinary C++ class which inherits from
CkArrayIndex, allocates custom data in the space it has set aside for index
data, and sets the “nInts” member to the length, in integers, of the custom
index data.

For example, if you have a structure or class named “Foo”, you can use a
Foo object as an array index by defining the class:

.. code-block:: c++

    // Include to inherit from CkArrayIndex
    #include <charm++.h>

    class CkArrayIndexFoo : public CkArrayIndex {
    private:
      Foo* f;
    public:
      CkArrayIndexFoo(const Foo &in) {
        f = new (index) Foo(in);
        nInts = sizeof(Foo)/sizeof(int);
      }
    };

Note that Foo must be allocated using placement new pointing to the "index"
member of CkArrayIndex. Furthermore, its size must be an integral number of
integers- you must pad it with zero bytes if this is not the case. Also, Foo
must be a simple class- it cannot contain pointers, have virtual functions, or
require a destructor. Finally, there is a Charm++ configuration-time option
called CK_ARRAYINDEX_MAXLEN which is the largest allowable number of integers
in an array index. The default is 3; but you may override this to any value by
passing “-DCK_ARRAYINDEX_MAXLEN=n” to the Charm++ build script as well as all
user code. Larger values will increase the size of each message.

You can then declare an array indexed by Foo objects with

.. code-block:: c++

    // in the .ci file:
    array [Foo] AF { entry AF(); ... }

    // in the .h file:
    class AF : public CBase_AF
    { public: AF() {} ... }

    // in the .C file:
    Foo f;
    CProxy_AF a=CProxy_AF::ckNew();
    a[CkArrayIndexFoo(f)].insert();
    ...

Note that since our CkArrayIndexFoo constructor is not declared with the
explicit keyword, we can equivalently write the last line as:

.. code-block:: c++

    a[f].insert();

The array index (an object of type Foo) is then accessible as “thisIndex”. For
example:

.. code-block:: c++

    // in the .C file:
    AF::AF() {
      Foo myF=thisIndex;
      functionTakingFoo(myF);
    }

A demonstration of user defined indices can be seen in
``examples/charm++/hello/fancyarray``.

.. _array section:

Sections: Subsets of a Chare Array/Group
----------------------------------------

Charm++ supports defining and communicating with subsets of a chare
array or group. This entity is called a chare array section or a group
section (*section*). Section elements are addressed via a section proxy.
Charm++ also supports sections which are a subset of elements of
multiple chare arrays/groups of the same type (see
:numref:`cross array section`).

Multicast operations, a broadcast to all members of a section, are
directly supported by the section proxy. For array sections, multicast
operations by default use optimized spanning trees via the CkMulticast
library in Charm++. For group sections, multicast operations by default
use an unoptimized direct-sending implementation. To optimize messaging,
group sections need to be manually delegated to CkMulticast (see
:numref:`Manual Delegation`). Reductions are also supported for both
arrays and group sections via the CkMulticast library.

Array and group sections work in mostly the same way. Check
``examples/charm++/groupsection`` for a group section example and
``examples/charm++/arraysection`` for an array section example.

.. _section creation:

Section Creation
~~~~~~~~~~~~~~~~

Array sections
^^^^^^^^^^^^^^

For each chare array “A” declared in a ci file, a section proxy of type
“CProxySection_A” is automatically generated in the decl and def header
files. You can create an array section in your application by invoking
ckNew() function of the CProxySection. The user will need to provide
array indexes of all the array section members through either explicit
enumeration, or an index range expression. For example, for a 3D array:

.. code-block:: c++

     std::vector<CkArrayIndex3D> elems;  // add array indices
     for (int i=0; i<10; i++)
       for (int j=0; j<20; j+=2)
         for (int k=0; k<30; k+=2)
            elems.emplace_back(i, j, k);
     CProxySection_Hello proxy = CProxySection_Hello::ckNew(helloArrayID, elems);

Alternatively, one can do the same thing by providing the index range
[lbound:ubound:stride] for each dimension:

.. code-block:: c++

     CProxySection_Hello proxy = CProxySection_Hello::ckNew(helloArrayID, 0, 9, 1, 0, 19, 2, 0, 29, 2);

The above code creates a section proxy that contains array elements
[0:9, 0:19:2, 0:29:2].

For user-defined array index other than CkArrayIndex1D to
CkArrayIndex6D, one needs to use the generic array index type:
CkArrayIndex.

.. code-block:: c++

     std::vector<CkArrayIndex> elems;  // add array indices
     CProxySection_Hello proxy = CProxySection_Hello::ckNew(helloArrayID, elems);

Group sections
^^^^^^^^^^^^^^

Group sections are created in the same way as array sections. A group
“A” will have an associated “CProxySection_A” type which is used to
create a section and obtain a proxy. In this case, ckNew() will receive
the list of PE IDs which will form the section. See
examples/charm++/groupsection for an example.

It is important to note that Charm++ does not automatically delegate
group sections to the internal CkMulticast library, and instead defaults
to a point-to-point implementation of multicasts. To use CkMulticast
with group sections, the user must manually delegate after invoking
group creation. See :numref:`Manual Delegation` for information on how
to do this.

Creation order restrictions
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Important: Array sections should be created in post-constructor entry
methods to avoid race conditions.

If the user wants to invoke section creation from a group, special care
must be taken that the collection for which we are creating a section
(array or group) already exists.

For example, suppose a user wants to create a section of array “A” from
an entry method in group “G”. Because groups are created before arrays
in Charm++, and there is no guarantee of creation order of groups, there
is a risk that array A’s internal structures have not been initialized
yet on every PE, causing section creation to fail. As such, the
application must ensure that A has been created before attempting to
create a section.

If the section is created from inside an array element there is no such
risk.

.. _array_section_multicast:

Section Multicasts
~~~~~~~~~~~~~~~~~~

Once the proxy is obtained at section creation time, the user can
broadcast to all the section members, like this:

.. code-block:: c++

     CProxySection_Hello proxy;
     proxy.someEntry(...); // section broadcast

See examples/charm++/arraysection for examples on how sections are used.

You can send the section proxy in a message to another processor, and
still safely invoke the entry functions on the section proxy.

Optimized multicast via CkMulticast
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Charm++ has a built-in CkMulticast library that optimizes section
communications. By default, the Charm++ runtime system will use this
library for array and cross-array sections. For group sections, the user
must manually delegate the section proxy to CkMulticast (see
:numref:`Manual Delegation`).

By default, CkMulticast builds a spanning tree for multicast/reduction
with a factor of 2 (binary tree). One can specify a different branching
factor when creating the section.

.. code-block:: c++

     CProxySection_Hello sectProxy = CProxySection_Hello::ckNew(..., 3); // factor is 3

Note, to use CkMulticast library, all multicast messages must inherit
from CkMcastBaseMsg, as the following example shows. Note that
CkMcastBaseMsg must come first, this is IMPORTANT for CkMulticast
library to retrieve section information out of the message.

.. code-block:: c++

   class HiMsg : public CkMcastBaseMsg, public CMessage_HiMsg
   {
   public:
     int *data;
   };

Due to this restriction, when using CkMulticast you must define messages
explicitly for multicast entry functions and no parameter marshalling
can be used.

Section Reductions
~~~~~~~~~~~~~~~~~~

Reductions over the elements of a section are supported through the
CkMulticast library. As such, to perform reductions, the section must
have been delegated to CkMulticast, either automatically (which is the
default case for array sections), or manually for group sections.

Since an array element can be a member of multiple array sections, it is
necessary to disambiguate between which array section reduction it is
participating in each time it contributes to one. For this purpose, a
data structure called “CkSectionInfo” is created by CkMulticast library
for each array section that the array element belongs to. During a
section reduction, the array element must pass the CkSectionInfo as a
parameter in the contribute(). The CkSectionInfo for a section can be
retrieved from a message in a multicast entry point using function call
CkGetSectionInfo:

.. code-block:: c++

     CkSectionInfo cookie;

     void SayHi(HiMsg *msg)
     {
       CkGetSectionInfo(cookie, msg); // update section cookie every time
       int data = thisIndex;
       CProxySection_Hello::contribute(sizeof(int), &data, CkReduction::sum_int, cookie, cb);
     }

Note that the cookie cannot be used as a one-time local variable in the
function, the same cookie is needed for the next contribute. This is
because the cookie includes some context-sensitive information (e.g.,
the reduction counter). Subsequent invocations of CkGetSectionInfo()
only updates part of the data in the cookie, rather than creating a
brand new one.

Similar to array reductions, to use section-based reductions, a
reduction client CkCallback object must be created. You may pass the
client callback as an additional parameter to contribute. If different
contribute calls to the same reduction operation pass different
callbacks, some (unspecified, unreliable) callback will be chosen for
use.

See the following example:

.. code-block:: c++

       CkCallback cb(CkIndex_myArrayType::myReductionEntry(NULL),thisProxy);
       CProxySection_Hello::contribute(sizeof(int), &data, CkReduction::sum_int, cookie, cb);

As in an array reduction, users can use built-in reduction types
(Section :numref:`builtin_reduction`) or define his/her own reducer
functions (Section :numref:`new_type_reduction`).

Section Operations and Migration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When using a section reduction, you don’t need to worry about migrations
of array elements. When migration happens, an array element in the array
section can still use the CkSectionInfo it stored previously for doing a
reduction. Reduction messages will be correctly delivered but may not be
as efficient until a new multicast spanning tree is rebuilt internally
in the CkMulticastMgr library. When a new spanning tree is rebuilt, an
updated CkSectionInfo is passed along with a multicast message, so it is
recommended that CkGetSectionInfo() function is always called when a
multicast message arrives (as shown in the above SayHi example).

In the case where a multicast root migrates, the library must
reconstruct the spanning tree to get optimal performance. One will get
the following warning message if this is not done: “Warning: Multicast
not optimized after multicast root migrated.” In the current
implementation, the user needs to initiate the rebuilding process using
resetSection.

.. code-block:: c++

   void Foo::pup(PUP::er & p) {
       // if I am multicast root and it is unpacking
      if (ismcastroot && p.isUnpacking()) {
         CProxySection_Foo fooProxy; // proxy for the section
         fooProxy.resetSection(fooProxy);

         // you may want to reset reduction client to root
         CkCallback *cb = new CkCallback(...);
      }
   }

.. _cross array section:

Cross Array Sections
~~~~~~~~~~~~~~~~~~~~

Cross array sections contain elements from multiple arrays. Construction
and use of cross array sections is similar to normal array sections with
the following restrictions.

-  Arrays in a section must all be of the same type.

-  Each array must be enumerated by array ID.

-  The elements within each array must be enumerated explicitly.

Note: cross section logic also works for groups with analogous
characteristics.

Given three arrays declared thusly:

.. code-block:: c++

  std::vector<CkArrayID> aidArr(3);
  for (int i=0; i<3; i++) {
    CProxy_multisectiontest_array1d Aproxy = CProxy_multisectiontest_array1d::ckNew(masterproxy.ckGetGroupID(), ArraySize);
    aidArr[i] = Aproxy.ckGetArrayID();
  }

One can make a section including the lower half elements of all three
arrays as follows:

.. code-block:: c++

  int aboundary = ArraySize/2;
  int afloor = aboundary;
  int aceiling = ArraySize-1;
  int asectionSize = aceiling-afloor+1;
  // cross section lower half of each array
  std::vector<std::vector<CkArrayIndex> > aelems(3);
  for (int k=0; k<3; k++) {
    aelems[k].resize(asectionSize);
    for (int i=afloor,j=0; i<=aceiling; i++,j++)
      aelems[k][j] = CkArrayIndex1D(i);
  }
  CProxySection_multisectiontest_array1d arrayLowProxy(aidArr, aelems);

The resulting cross section proxy, as in the example arrayLowProxy, can
then be used for multicasts in the same way as a normal array section.

Note: For simplicity the above example has all arrays and sections of
uniform size. The size of each array and the number of elements in each
array within a section can all be set independently. For a more concrete
example on how to use cross array section reduction, please refer to:
``examples/charm++/hello/xarraySection``.

.. _Manual Delegation:

Manual Delegation
~~~~~~~~~~~~~~~~~

By default Charm++ uses the CkMulticast library for optimized broadcasts
and reductions on array sections, but advanced Charm++ users can choose
to delegate [13]_ sections to custom libraries (called delegation
managers). Note that group sections are not automatically delegated to
CkMulticast and hence must be manually delegated to this library to
benefit from the optimized multicast tree implementation. This is
explained in this section, and see examples/charm++/groupsection for an
example.

While creating a chare array one can set the auto delegation flag to
false in CkArrayOptions and the runtime system will not use the default
CkMulticast library. A CkMulticastMgr (or any other delegation manager)
group can then be created by the user, and any section delegated to it.

One only needs to create one delegation manager group, and it can serve
all multicast/reduction delegations for different array/group sections
in an application. In the following we show a manual delegation example
using CkMulticast (the same can be applied to custom delegation
managers):

.. code-block:: c++

     CkArrayOptions opts(...);
     opts.setSectionAutoDelegate(false); // manual delegation
     CProxy_Hello arrayProxy = CProxy_Hello::ckNew(opts,...);
     CProxySection_Hello sectProxy = CProxySection_Hello::ckNew(...);
     CkGroupID mCastGrpId = CProxy_CkMulticastMgr::ckNew();
     CkMulticastMgr *mCastGrp = CProxy_CkMulticastMgr(mCastGrpId).ckLocalBranch();

     sectProxy.ckSectionDelegate(mCastGrp); // initialize section proxy

     sectProxy.someEntry(...); // multicast via delegation library as before

One can also set the default branching factor when creating a
CkMulticastMgr group. Sections created via this manager will use the
specified branching factor for their multicast tree. For example,

.. code-block:: c++

     CkGroupID mCastGrpId = CProxy_CkMulticastMgr::ckNew(3); // factor is 3

Contributing using a custom CkMulticastMgr group:

.. code-block:: c++

     CkSectionInfo cookie;

     void SayHi(HiMsg *msg)
     {
       CkGetSectionInfo(cookie, msg); // update section cookie every time
       int data = thisIndex;
       CkCallback cb(CkIndex_myArrayType::myReductionEntry(NULL),thisProxy);
       mcastGrp->contribute(sizeof(int), &data, CkReduction::sum_int, cookie, cb);
     }

Setting default reduction client for a section when using manual
delegation:

.. code-block:: c++

     CProxySection_Hello sectProxy;
     CkMulticastMgr *mcastGrp = CProxy_CkMulticastMgr(mCastGrpId).ckLocalBranch();
     mcastGrp->setReductionClient(sectProxy, new CkCallback(...));

Writing the pup method:

.. code-block:: c++

    void Foo::pup(PUP::er & p) {
      // if I am multicast root and it is unpacking
      if (ismcastroot && p.isUnpacking()) {
        CProxySection_Foo fooProxy; // proxy for the section
        CkMulticastMgr *mg = CProxy_CkMulticastMgr(mCastGrpId).ckLocalBranch();
        mg->resetSection(fooProxy);

        // you may want to reset reduction client to root
        CkCallback *cb = new CkCallback(...);
        mg->setReductionClient(mcp, cb);
      }
    }

.. _inheritance:

Chare and Message Inheritance
-----------------------------

Charm++ supports C++ like inheritance among Charm++ objects such as
chares, groups, and messages, making it easier to keep applications
modular and allowing reuse of code.

Chare Inheritance
~~~~~~~~~~~~~~~~~

Chare inheritance makes it possible to remotely invoke methods of a base
chare from a proxy of a derived chare. Suppose a base chare is of type
BaseChare, then the derived chare of type DerivedChare needs to be
declared in the Charm++ interface file to be explicitly derived from
BaseChare. Thus, the constructs in the ``.ci`` file should look like:

.. code-block:: c++

     chare BaseChare {
       entry BaseChare(someMessage *);
       entry void baseMethod(void);
       ...
     }
     chare DerivedChare : BaseChare {
       entry DerivedChare(otherMessage *);
       entry void derivedMethod(void);
       ...
     }

Note that the access specifier public is omitted, because
Charm++ interface translator only needs to know about the public
inheritance, and thus public is implicit. A Chare can inherit privately
from other classes too, but the Charm++ interface translator does not
need to know about it, because it generates support classes (*proxies*)
to remotely invoke only public methods.

The class definitions of these chares should look like:

.. code-block:: c++

     class BaseChare : public CBase_BaseChare {
       // private or protected data
       public:
         BaseChare(someMessage *);
         void baseMethod(void);
     };
     class DerivedChare : public CBase_DerivedChare {
       // private or protected data
       public:
         DerivedChare(otherMessage *);
         void derivedMethod(void);
     };

It is possible to create a derived chare, and invoke methods of base
chare from it, or to assign a derived chare proxy to a base chare proxy
as shown below:

.. code-block:: c++

     ...
     otherMessage *msg = new otherMessage();
     CProxy_DerivedChare pd = CProxy_DerivedChare::ckNew(msg);
     pd.baseMethod();     // OK
     pd.derivedMethod();  // OK
     ...
     CProxy_BaseChare pb = pd;
     pb.baseMethod();    // OK
     pb.derivedMethod(); // COMPILE ERROR

To pass constructor arguments from
DerivedChare::DerivedChare(someMessage*) to
BaseChare::BaseChare(someMessage*), they can be forwarded through the
CBase type constructor as follows:

.. code-block:: c++

   DerivedChare::DerivedChare(someMessage *msg)
   : CBase_DerivedChare(msg) // Will forward all arguments to BaseChare::BaseChare
   { }

If no arguments are provided, the generated C++ code for the
CBase_DerivedChare constructor calls the default constructor of the base
class BaseChare.

Entry methods are inherited in the same manner as methods of sequential
C++ objects. To make an entry method virtual, just add the keyword virtual
to the corresponding chare method declaration in the class header- no
change is needed in the interface file. Pure virtual entry methods also
require no special description in the interface file.

Inheritance for Messages
~~~~~~~~~~~~~~~~~~~~~~~~

Messages cannot inherit from other messages. A message can, however,
inherit from a regular C++ class. For example:

.. code-block:: c++

    // In the .ci file:
    message BaseMessage1;
    message BaseMessage2;

    // In the .h file:
    class Base {
      // ...
    };
    class BaseMessage1 : public Base, public CMessage_BaseMessage1 {
      // ...
    };
    class BaseMessage2 : public Base, public CMessage_BaseMessage2 {
      // ...
    };

Messages cannot contain virtual methods or virtual base classes unless
you use a packed message. Parameter marshalling has complete support for
inheritance, virtual methods, and virtual base classes via the PUP::able
framework.

.. _templates:

Generic and Meta Programming with Templates
-------------------------------------------

Templates are a mechanism provided by the C++ language to parametrize code
over various types and constants with compile-time code specialization
for each instance. Charm++ allows developers to implement various
entities using C++ templates to gain their advantages in abstraction,
flexibility, and performance. Because the Charm++ runtime system
requires some generated code for each entity type that is used in a
program, template entities must each have a declaration in a .ci file, a
definition in a C++ header, and declarations of their instantiations in
one or more .ci files.

The first step to implementing a templated Charm++ entity is declaring
it as such in a .ci file. This declaration takes the same form as any C++
template: the ``template`` keyword, a list of template parameters
surrounded by angle brackets, and the normal declaration of the entity
with possible reference to the template parameters. The Charm++
interface translator will generate corresponding templated code for the
entity, similar to what it would generate for a non-templated entity of
the same kind. Differences in how one uses this generated code are
described below.

A message template might be declared as follows:

.. code-block:: c++

   module A {
     template <class DType, int N=3>
     message TMessage;
   };

Note that default template parameters are supported.

If one wished to include variable-length arrays in a message template,
those can be accomodated as well:

.. code-block:: c++

   module B {
     template <class DType>
     message TVarMessage {
       DType payload[];
     };
   };

Similarly, chare class templates (for various kinds of chares) would be
written:

.. code-block:: c++

   module C {
     template <typename T>
     chare TChare {
       entry TChare();
       entry void doStuff(T t);
     };

     template <typename U>
     group TGroup {
       entry TGroup();
       entry void doSomethingElse(U u, int n);
     };

     template <typename V, int s>
     array [2D] TArray {
       entry TArray(V v);
     };

     template <typename W>
     nodegroup TNodeGroup {
       entry TNodeGroup();
       entry void doAnotherThing(W w);
     };
   };

Entry method templates are declared like so:

.. code-block:: c++

   module D {
       array [1D] libArray {
           entry libArray(int _dataSize);
           template <typename T>
           entry void doSomething(T t, CkCallback redCB);
       };
   };

The definition of templated Charm++ entities works almost identically to
the definition of non-template entities, with the addition of the
expected template signature:

.. code-block:: c++

   // A.h
   #include "A.decl.h"

   template <class DType, int N=3>
   struct TMessage : public CMessage_TMessage<DType, N> {
     DType d[N];
   };

   #define CK_TEMPLATES_ONLY
   #include "A.def.h"
   #undef CK_TEMPLATES_ONLY

The distinguishing change is the additional requirement to include parts
of the generated .def.h file that relate to the templates being defined.
This exposes the generated code that provides registration and other
supporting routines to client code that will need to instantiate it. As
with C++ template code in general, the entire definition of the templated
entity must be visible to the code that eventually references it to
allow instantiation. In circumstances where ``module A`` contains only
template code, some source file including ``A.def.h`` without the
template macro will still have to be compiled and linked to incorporate
module-level generated code.

Code that references particular templated entities needs to ask the
interface translator to instantiate registration and delivery code for
those entities. This is accomplished by a declaration in a ``.ci`` file
that names the entity and the actual template arguments for which an
instantiation is desired.

For the message and chare templates described above, a few
instantiations might look like

.. code-block:: c++

   module D {
     extern module A;
     message TMessage<float, 7>;
     message TMessage<double>;
     message TMessage<int, 1>;

     extern module C;
     array [2D] TArray<std::string, 4>;
     group TGroup<char>;
   };

Instantiations of entry method templates are slightly more complex,
because they must specify the chare class containing them. The template
arguments are also specified directly in the method’s parameters, rather
than as distinct template arguments.

.. code-block:: c++

   module E {
     extern module D;

     // syntax: extern entry void chareClassName templateEntryMethodName(list, of, actual, arguments);
     extern entry void libArray doSomething(int&, CkCallback redCB);
   };

To enable generic programming using Charm++ entities, we define a number
of type trait utilities. These can be used to determine at compile-time
if a type is a certain kind of Charm++ type:

.. code-block:: c++

   #include "charm++_type_traits.h"

   // Is T a chare array proxy?
   using result = charmxx::is_array_proxy<T>;

   // Is T a group proxy?
   using result = charmxx::is_group_proxy<T>;

   // Is T a node group proxy?
   using result = charmxx::is_node_group_proxy<T>;

   // Is T a chare proxy?
   using result = charmxx::is_chare_proxy<T>;

   // Is T a bound array?
   using result = charmxx::is_bound_array<T>;

   // Does T have a PUP routine defined for it?
   using result = charmxx::is_pupable<T>;

Collectives
-----------

.. _reductionClients:

Reduction Clients
~~~~~~~~~~~~~~~~~

After the data is reduced, it is passed to you via a callback object, as
described in section :numref:`callbacks`. The message passed
to the callback is of type CkReductionMsg. Unlike typed reductions
briefed in Section :numref:`reductions`, here we discuss callbacks
that take CkReductionMsg\* argument. The important members of
CkReductionMsg are getSize(), which returns the number of bytes of
reduction data; and getData(), which returns a “void \*” to the actual
reduced data.

The callback to be invoked when the reduction is complete is specified
as an additional parameter to contribute. It is an error for chare array
elements to specify different callbacks to the same reduction
contribution.

.. code-block:: c++

       double forces[2]=get_my_forces();
       // When done, broadcast the CkReductionMsg to "myReductionEntry"
       CkCallback cb(CkIndex_myArrayType::myReductionEntry(NULL), thisProxy);
       contribute(2*sizeof(double), forces,CkReduction::sum_double, cb);

In the case of the reduced version used for synchronization purposes,
the callback parameter will be the only input parameter:

.. code-block:: c++

       CkCallback cb(CkIndex_myArrayType::myReductionEntry(NULL), thisProxy);
       contribute(cb);

and the corresponding callback function:

.. code-block:: c++

   void myReductionEntry(CkReductionMsg *msg)
   {
     int reducedArrSize=msg->getSize() / sizeof(double);
     double *output=(double *) msg->getData();
     for(int i=0 ; i<reducedArrSize ; i++)
     {
      // Do something with the reduction results in each output[i] array element
      .
      .
      .
     }
     delete msg;
   }

(See ``examples/charm++/reductions/simple_reduction`` for a complete
example).

If the target of a reduction is an entry method defined by a *when*
clause in SDAG (Section :numref:`sec:sdag`), one may wish to set a
reference number (or tag) that SDAG can use to match the resulting
reduction message. To set the tag on a reduction message, call the
``CkCallback::setRefNum(CMK_REFNUM_TYPE refnum)`` method on the callback
passed to the ``contribute()`` call.

.. _new_type_reduction:

Defining a New Reduction Type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

It is possible to define a new type of reduction, performing a
user-defined operation on user-defined data. This is done by creating a
*reduction function*, which combines separate contributions into a
single combined value.

The input to a reduction function is a list of CkReductionMsgs. A
CkReductionMsg is a thin wrapper around a buffer of untyped data to be
reduced. The output of a reduction function is a single CkReductionMsg
containing the reduced data, which you should create using the
CkReductionMsg::buildNew(int nBytes,const void \*data) method.

Thus every reduction function has the prototype:

.. code-block:: c++

   CkReductionMsg *reductionFn(int nMsg,CkReductionMsg **msgs);

For example, a reduction function to add up contributions consisting of
two machine ``short int``\ s would be:

.. code-block:: c++

   CkReductionMsg *sumTwoShorts(int nMsg,CkReductionMsg **msgs)
   {
     // Sum starts off at zero
     short ret[2]={0,0};
     for (int i=0;i<nMsg;i++) {
       // Sanity check:
       CkAssert(msgs[i]->getSize()==2*sizeof(short));
       // Extract this message's data
       short *m=(short *)msgs[i]->getData();
       ret[0]+=m[0];
       ret[1]+=m[1];
     }
     return CkReductionMsg::buildNew(2*sizeof(short),ret);
   }

The reduction function must be registered with Charm++ using
CkReduction::addReducer(reducerFn fn=NULL, bool streamable=false, const
char\* name=NULL) from an initnode routine (see
section :numref:`initnode` for details on the initnode
mechanism). It takes a required parameter, reducerFn fn, a function
pointer to the reduction function, and an optional parameter bool
streamable, which indicates if the function is streamable or not (see
section :numref:`streamable_reductions` for more information).
CkReduction::addReducer returns a CkReduction::reducerType which you can
later pass to contribute. Since initnode routines are executed once on
every node, you can safely store the CkReduction::reducerType in a
global or class-static variable. For the example above, the reduction
function is registered and used in the following manner:

.. code-block:: c++

   // In the .ci file:
   initnode void registerSumTwoShorts(void);

   // In some .C file:
   /*global*/ CkReduction::reducerType sumTwoShortsType;
   /*initnode*/ void registerSumTwoShorts(void)
   {
     sumTwoShortsType=CkReduction::addReducer(sumTwoShorts);
   }

   // In some member function, contribute data to the customized reduction:
   short data[2]=...;
   contribute(2*sizeof(short),data,sumTwoShortsType);

| Note that typed reductions briefed in Section :numref:`reductions`
  can also be used for custom reductions. The target reduction client
  can be declared as in Section :numref:`reductions` but the
  reduction functions will be defined as explained above.
| Note that you cannot call CkReduction::addReducer from anywhere but an
  initnode routine.
| (See Reduction.cpp of `Barnes-Hut
  MiniApp <http://charmplusplus.org/miniApps/#barnes>`__ for a complete
  example).

.. _streamable_reductions:

Streamable Reductions
^^^^^^^^^^^^^^^^^^^^^

For custom reductions over fixed sized messages, it is often desirable
that the runtime process each contribution in a streaming fashion, i.e.
as soon as a contribution is received from a chare array element, that
data should be combined with the current aggregation of other
contributions on that PE. This results in a smaller memory footprint
because contributions are immediately combined as they come in rather
than waiting for all contributions to be received. Users can write their
own custom streamable reducers by reusing the message memory of the
zeroth message in their reducer function by passing it as the last
argument to CkReduction::buildNew:

.. code-block:: c++

   CkReductionMsg *sumTwoShorts(int nMsg,CkReductionMsg **msgs)
   {
     // reuse msgs[0]'s memory:
     short *retData = (short*)msgs[0]->getData();
     for (int i=1;i<nMsg;i++) {
       // Sanity check:
       CkAssert(msgs[i]->getSize()==2*sizeof(short));
       // Extract this message's data
       short *m=(short *)msgs[i]->getData();
       retData[0]+=m[0];
       retData[1]+=m[1];
     }
     return CkReductionMsg::buildNew(2*sizeof(short), retData, sumTwoShortsReducer, msgs[0]);
   }

Note that *only message zero* is allowed to be reused. For reducer
functions that do not operate on fixed sized messages, such as set and
concat, streaming would result in quadratic memory allocation and so is
not desirable. Users can specify that a custom reducer is streamable
when calling CkReduction::addReducer by specifying an optional boolean
parameter (default is false). They can also provide a name string for
their reducer to aid in debugging (default is NULL).

.. code-block:: c++

   static void initNodeFn(void) {
       sumTwoShorts = CkReduction::addReducer(sumTwoShorts, /* streamable = */ true, /* name = */ "sumTwoShorts");
   }

Serializing Complex Types
-------------------------

This section describes advanced functionality in the PUP framework. The
first subsections describe features supporting complex objects, with
multiple levels of inheritance, or with dynamic changes in heap usage.
The latter subsections describe additional language bindings, and
features supporting PUP modes which can be used to copy object state
from and to long-term storage for checkpointing, or other application
level purposes.

.. _sec:pupdynalloc:

Dynamic Allocation
~~~~~~~~~~~~~~~~~~

If your class has fields that are dynamically allocated, when unpacking,
these need to be allocated in the usual way before you pup them.
Deallocation should be left to the class destructor as usual.

No allocation
^^^^^^^^^^^^^

The simplest case is when there is no dynamic allocation. Example:

.. code-block:: c++

   class keepsFoo : public mySuperclass {
   private:
       foo f; /* simple foo object */
   public:
       keepsFoo(void) { }
       void pup(PUP::er &p) {
         mySuperclass::pup(p);
         p|f; // pup f's fields (calls f.pup(p);)
       }
       ~keepsFoo() { }
   };

Allocation outside pup
^^^^^^^^^^^^^^^^^^^^^^

The next simplest case is when we contain a class that is always
allocated during our constructor, and deallocated during our destructor.
Then no allocation is needed within the pup routine.

.. code-block:: c++

   class keepsHeapFoo : public mySuperclass {
   private:
       foo *f; /* Heap-allocated foo object */
   public:
       keepsHeapFoo(void) {
         f=new foo;
       }
       void pup(PUP::er &p) {
         mySuperclass::pup(p);
         p|*f; // pup f's fields (calls f->pup(p))
       }
       ~keepsHeapFoo() { delete f; }
   };

Allocation during pup
^^^^^^^^^^^^^^^^^^^^^

If we need values obtained during the pup routine before we can allocate
the class, we must allocate the class inside the pup routine. Be sure to
protect the allocation with ``if (p.isUnpacking())``.

.. code-block:: c++

   class keepsOneFoo : public mySuperclass {
   private:
       foo *f; /* Heap-allocated foo object */
   public:
       keepsOneFoo(...) { f=new foo(...); }
       keepsOneFoo() { f=NULL; } /* pup constructor */
       void pup(PUP::er &p) {
         mySuperclass::pup(p);
         // ...
         if (p.isUnpacking()) /* must allocate foo now */
            f=new foo(...);
         p|*f; // pup f's fields
       }
       ~keepsOneFoo() { delete f; }
   };

Allocatable array
^^^^^^^^^^^^^^^^^

For example, if we keep an array of doubles, we need to know how many
doubles there are before we can allocate the array. Hence we must first
pup the array length, do our allocation, and then pup the array data. We
could allocate memory using malloc/free or other allocators in exactly
the same way.

.. code-block:: c++

   class keepsDoubles : public mySuperclass {
   private:
       int n;
       double *arr; /* new'd array of n doubles */
   public:
       keepsDoubles(int n_) {
         n=n_;
         arr=new double[n];
       }
       keepsDoubles() { }

       void pup(PUP::er &p) {
         mySuperclass::pup(p);
         p|n; // pup the array length n
         if (p.isUnpacking()) arr=new double[n];
         PUParray(p,arr,n); // pup data in the array
       }

       ~keepsDoubles() { delete[] arr; }
   };

NULL object pointer
^^^^^^^^^^^^^^^^^^^

If our allocated object may be ``NULL``, our allocation becomes much more
complicated. We must first check and pup a flag to indicate whether the
object exists, then depending on the flag, pup the object.

.. code-block:: c++

   class keepsNullFoo : public mySuperclass {
   private:
       foo *f; /*Heap-allocated foo object, or NULL*/
   public:
       keepsNullFoo(...) { if (...) f=new foo(...); }
       keepsNullFoo() { f=NULL; }
       void pup(PUP::er &p) {
         mySuperclass::pup(p);
         int has_f = (f!=NULL);
         p|has_f;
         if (has_f) {
           if (p.isUnpacking()) f=new foo;
           p|*f;
         } else {
           f=NULL;
         }
       }
       ~keepsNullFoo() { delete f; }
   };

This sort of code is normally much longer and more error-prone if split
into the various packing/unpacking cases.

Array of classes
^^^^^^^^^^^^^^^^

An array of actual classes can be treated exactly the same way as an
array of basic types. ``PUParray`` will pup each element of the array
properly, calling the appropriate ``operator|``.

.. code-block:: c++

   class keepsFoos : public mySuperclass {
   private:
       int n;
       foo *arr; /* new'd array of n foos */
   public:
       keepsFoos(int n_) {
         n=n_;
         arr=new foo[n];
       }
       keepsFoos() { arr=NULL; }

       void pup(PUP::er &p) {
         mySuperclass::pup(p);
         p|n; // pup the array length n
         if (p.isUnpacking())  arr=new foo[n];
         PUParray(p,arr,n); // pup each foo in the array
       }

       ~keepsFoos() { delete[] arr; }
   };

Array of pointers to classes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

An array of pointers to classes must handle each element separately,
since the PUParray routine does not work with pointers. An "allocate"
routine to set up the array could simplify this code. More ambitious is
to construct a "smart pointer" class that includes a pup routine.

.. code-block:: c++

   class keepsFooPtrs : public mySuperclass {
   private:
       int n;
       foo **arr; /* new'd array of n pointer-to-foos */
   public:
       keepsFooPtrs(int n_) {
         n=n_;
         arr=new foo*[n]; // allocate array
         for (int i=0; i<n; i++) arr[i]=new foo(...); // allocate i'th foo
       }
       keepsFooPtrs() { arr=NULL; }

       void pup(PUP::er &p) {
         mySuperclass::pup(p);
         p|n; // pup the array length n
         if (p.isUnpacking()) arr=new foo*[n]; // allocate array
         for (int i=0; i<n; i++) {
           if (p.isUnpacking()) arr[i]=new foo(...); // allocate i'th foo
           p|*arr[i];  // pup the i'th foo
         }
       }

       ~keepsFooPtrs() {
          for (int i=0; i<n; i++) delete arr[i];
          delete[] arr;
        }
   };

Note that this will not properly handle the case where some elements of
the array are actually subclasses of foo, with virtual methods. The
``PUP::able`` framework described in the next section can be helpful in this
case.

.. _sec:pup::able:

Subclass allocation via PUP::able
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If the class *foo* above might have been a subclass, instead of simply
using ``new foo`` above we would have had to allocate an object of the
appropriate subclass. Since determining the proper subclass and calling
the appropriate constructor yourself can be difficult, the PUP framework
provides a scheme for automatically determining and dynamically
allocating subobjects of the appropriate type.

Your superclass must inherit from ``PUP::able``, which provides the basic
machinery used to move the class. A concrete superclass and all its
concrete subclasses require these four features:

-  A line declaring ``PUPable className;`` in the .ci file. This registers
   the class' constructor. If ``className`` is a templated class, each concrete
   instantiation should have its own fully specified ``PUPable`` declaration in
   the .ci file.

-  A call to the macro ``PUPable_decl(className)`` in the class'
   declaration, in the header file. This adds a virtual method to your
   class to allow PUP::able to determine your class' type. If ``className`` is a
   templated class, instead use ``PUPable_decl_base_template(baseClassName,
   className)``, where ``baseClassName`` is the name of the base class. Both
   class names should include template specifications if necessary.

-  A migration constructor — a constructor that takes ``CkMigrateMessage *``.
   This is used to create the new object on the receive side,
   immediately before calling the new object's pup routine.

-  A working, virtual ``pup`` method. You can omit this if your class has no
   data that needs to be packed.

As an added note for ``PUP::able`` classes which are templated: just as with
templated chares, you will also need to include the .def.h surrounded by
``CK_TEMPLATES_ONLY`` preprocessor guards in an appropriate location, as
described in Section :numref:`templates`.

An abstract superclass — a superclass that will never actually be
packed — only needs to inherit from ``PUP::able`` and include a
``PUPable_abstract(className)`` macro in their body. For these abstract
classes, the .ci file, ``PUPable_decl`` macro, and constructor are not
needed.

For example, if *parent* is a concrete superclass, and *child* and *tchild* are
its subclasses:

.. code-block:: c++

   // --------- In the .ci file ---------
   PUPable parent;
   PUPable child; // Could also have said `PUPable parent, child;`
   // One PUPable declaration per concrete intantiantion of tchild
   PUPable tchild<int, double>;
   PUPable tchild<char, Foo>;

   // --------- In the .h file ---------
   class parent : public PUP::able {
       // ... data members ...
   public:
       // ... other methods ...
       parent() {...}

       // PUP::able support: decl, migration constructor, and pup
       PUPable_decl(parent);
       parent(CkMigrateMessage *m) : PUP::able(m) {}
       virtual void pup(PUP::er &p) {
           PUP::able::pup(p); // Call base class
           // ... pup data members as usual ...
       }
   };

   class child : public parent {
       // ... more data members ...
   public:
       // ... more methods, possibly virtual ...
       child() {...}

       // PUP::able support: decl, migration constructor, and pup
       PUPable_decl(child);
       child(CkMigrateMessage *m) : parent(m) {}
       virtual void pup(PUP::er &p) {
           parent::pup(p); // Call base class
           // ... pup child's data members as usual ...
       }
   };

   template <typename T1, typename T2>
   class tchild : public parent {
       // ... more data members ...
   public:
       // ... more methods, possibly virtual ...
       tchild() { ... }

       // PUP::able support for templated classes
       // If parent were templated, we'd also include template args for parent
       PUPable_decl_base_template(parent, tchild<T1,T2>);
       tchild(CkMigrateMessage* m) : parent(m) {}
       virtual void pup(PUP::er &p) {
           parent::pup(p); // Call base class
           // ... pup tchild's data memebers as usual ...
       }
   };

   // Because tchild is a templated class with PUPable decls in the .ci file ...
   #define CK_TEMPLATES_ONLY
   #include "module_name.def.h"
   #undef CK_TEMPLATES_ONLY

With these declarations, we can automatically allocate and pup a
pointer to a parent or child using the vertical bar ``PUP::er`` syntax,
which on the receive side will create a new object of the appropriate
type:

.. code-block:: c++

   class keepsParent {
       parent *obj; // May actually point to a child class (or be NULL)
   public:
       // ...
       ~keepsParent() {
           delete obj;
       }
       void pup(PUP::er &p)
       {
           p|obj;
       }
   };

This will properly pack, allocate, and unpack ``obj`` whether it is actually
a parent or child object. The child class can use all the usual
C++ features, such as virtual functions and extra private data.

If ``obj`` is ``NULL`` when packed, it will be restored to ``NULL`` when unpacked.
For example, if the nodes of a binary tree are PUP::able, one may write
a recursive pup routine for the tree quite easily:

.. code-block:: c++

   // --------- In the .ci file ---------
   PUPable treeNode;

   // --------- In the .h file ---------
   class treeNode : public PUP::able {
       treeNode *left; // Left subtree
       treeNode *right; // Right subtree
       // ... other fields ...
   public:
       treeNode(treeNode *l=NULL, treeNode *r=NULL);
       ~treeNode() { delete left; delete right; }

       // The usual PUP::able support:
       PUPable_decl(treeNode);
       treeNode(CkMigrateMessage *m) : PUP::able(m) { left = right = NULL; }
       void pup(PUP::er &p) {
           PUP::able::pup(p); // Call base class
           p|left;
           p|right;
           // ... pup other fields as usual ...
       }
   };

This same implementation will also work properly even if the tree's
internal nodes are actually subclasses of ``treeNode``.

You may prefer to use the macros ``PUPable_def(className)`` and
``PUPable_reg(className)`` rather than using ``PUPable`` in the .ci file.
``PUPable_def`` provides routine definitions used by the PUP::able
machinery, and should be included in exactly one source file at file
scope. ``PUPable_reg`` registers this class with the runtime system, and
should be executed exactly once per node during program startup.

Finally, a ``PUP::able`` superclass like *parent* above must normally be
passed around via a pointer or reference, because the object might
actually be some subclass like *child*. Because pointers and references
cannot be passed across processors, for parameter marshalling you must
use the special templated smart pointer classes ``CkPointer`` and
``CkReference``, which only need to be listed in the .ci file.

A ``CkReference`` is a read-only reference to a ``PUP::able`` object — it is only
valid for the duration of the method call. A ``CkPointer`` transfers
ownership of the unmarshalled ``PUP::able`` to the method, so the pointer
can be kept and the object used indefinitely.

For example, if the entry method bar needs a ``PUP::able`` parent object for
in-call processing, you would use a ``CkReference`` like this:

.. code-block:: c++

   // --------- In the .ci file ---------
   entry void barRef(int x, CkReference<parent> p);

   // --------- In the .h file ---------
   void barRef(int x, parent &p) {
     // can use p here, but only during this method invocation
   }

If the entry method needs to keep its parameter, use a ``CkPointer`` like
this:

.. code-block:: c++

   // --------- In the .ci file ---------
   entry void barPtr(int x, CkPointer<parent> p);

   // --------- In the .h file ---------
   void barPtr(int x, parent *p) {
     // can keep this pointer indefinitely, but must eventually delete it
   }

Both ``CkReference`` and ``CkPointer`` are read-only from the send side — unlike
messages, which are consumed when sent, the same object can be passed to
several parameter marshalled entry methods. In the example above, we
could do:

.. code-block:: c++

      parent *p = new child;
      someProxy.barRef(x, *p);
      someProxy.barPtr(x, p); // Makes a copy of p
      delete p; // We allocated p, so we destroy it.

C and Fortran bindings
~~~~~~~~~~~~~~~~~~~~~~

C and Fortran programmers can use a limited subset of the ``PUP::er``
capability. The routines all take a handle named ``pup_er``. The routines
have the prototype:

.. code-block:: c++

   void pup_type(pup_er p, type *val);
   void pup_types(pup_er p, type *vals, int nVals);

The first call is for use with a single element; the second call is for
use with an array. The supported types are ``char``, ``short``, ``int``, ``long``,
``uchar``, ``ushort``, ``uint``, ``ulong``, ``float``, and ``double``, which all have the usual
C meanings.

A byte-packing routine

.. code-block:: c++

   void pup_bytes(pup_er p, void *data, int nBytes);

is also provided, but its use is discouraged for cross-platform puping.

``pup_isSizing``, ``pup_isPacking``, ``pup_isUnpacking``, and ``pup_isDeleting`` calls
are also available. Since C and Fortran have no destructors, you should
actually deallocate all data when passed a deleting ``pup_er``.

C and Fortran users cannot use ``PUP::able`` objects, seeking, or write
custom ``PUP::ers``. Using the C++ interface is recommended.

.. _sec:PUP:CommonPUPers:

Common PUP::ers
~~~~~~~~~~~~~~~

The most common *PUP::ers* used are ``PUP::sizer``, ``PUP::toMem``, and
``PUP::fromMem``. These are sizing, packing, and unpacking PUP::ers,
respectively.

``PUP::sizer`` simply sums up the sizes of the native binary representation
of the objects it is passed. ``PUP::toMem`` copies the binary representation
of the objects passed into a preallocated contiguous memory buffer.
``PUP::fromMem`` copies binary data from a contiguous memory buffer into the
objects passed. All three support the size method, which returns the
number of bytes used by the objects seen so far.

Other common PUP::ers are ``PUP::toDisk``, ``PUP::fromDisk``, and ``PUP::xlater``.
The first two are simple filesystem variants of the ``PUP::toMem`` and
``PUP::fromMem`` classes; ``PUP::xlater`` translates binary data from an
unpacking PUP::er into the machine's native binary format, based on a
``machineInfo`` structure that describes the format used by the source
machine.

An example of ``PUP::toDisk`` is available in ``examples/charm++/PUP/pupDisk``.

PUP::seekBlock
~~~~~~~~~~~~~~

It may rarely occur that you require items to be unpacked in a different
order than they are packed. That is, you want a seek capability.
*PUP::ers* support a limited form of seeking.

To begin a seek block, create a ``PUP::seekBlock`` object with your current
``PUP::er`` and the number of "sections" to create. Seek to a (0-based)
section number with the ``seek`` method, and end the seeking with the
``endBlock`` method. For example, if we have two objects A and B, where A's
pup depends on and affects some object B, we can pup the two with:

.. code-block:: c++

   void pupAB(PUP::er &p)
   {
     // ... other fields ...
     PUP::seekBlock s(p,2); // 2 seek sections
     if (p.isUnpacking())
     { // In this case, pup B first
       s.seek(1);
       B.pup(p);
     }
     s.seek(0);
     A.pup(p,B);

     if (!p.isUnpacking())
     { // In this case, pup B last
       s.seek(1);
       B.pup(p);
     }
     s.endBlock(); // End of seeking block
     // ... other fields ...
   };

Note that without the seek block, A’s fields would be unpacked over B’s
memory, with *disastrous* consequences. The packing or sizing path must
traverse the seek sections in numerical order; the unpack path may
traverse them in any order. There is currently a small fixed limit of **3**
on the maximum number of seek sections.

Writing a PUP::er
~~~~~~~~~~~~~~~~~

System-level programmers may occasionally find it useful to define their
own ``PUP::er`` objects. The system ``PUP::er`` class is an abstract base class
that funnels all incoming pup requests to a single subroutine:

.. code-block:: c++

       virtual void bytes(void *p, int n, size_t itemSize, dataType t);

The parameters are, in order, the field address, the number of items,
the size of each item, and the type of the items. The PUP::er is allowed
to use these fields in any way. However, an ``isSizing`` or ``isPacking``
PUP::er may not modify the referenced user data; while an ``isUnpacking``
PUP::er may not read the original values of the user data. If your
PUP::er is not clearly packing (saving values to some format) or
unpacking (restoring values), declare it as sizing PUP::er.

.. _topo:

Querying Network Topology
-------------------------

The following calls provide information about the machine upon which the
parallel program is executed. A processing element (PE) is a unit of
mapping and scheduling, which takes the form of an OS thread in SMP mode
and an OS process in non-SMP mode. A node (specifically, a logical node)
refers to an OS process: a set of one or more PEs that share memory
(i.e. an address space). PEs and nodes are ranked separately starting
from zero: PEs are ranked from ``0`` to ``CmiNumPes()``, and nodes are ranked
from ``0`` to ``CmiNumNodes()``.

Charm++ provides a unified abstraction for querying topology information of
IBM's BG/Q and Cray's XE6. The ``TopoManager`` singleton object, which can be
used by including ``TopoManager.h``, contains the following methods:

getDimNX(), getDimNY(), getDimNZ():
   Returns the length of X, Y and Z dimensions (except BG/Q).

getDimNA(), getDimNB(), getDimNC(), getDimND(), getDimNE():
   Returns the length of A, B, C, D and E dimensions on BG/Q.

getDimNT():
   Returns the length of T dimension. TopoManager uses the T dimension to
   represent different cores that reside within a physical node.

rankToCoordinates(int pe, int &x, int &y, int &z, int &t):
   Get the coordinates of PE with rank *pe* (except BG/Q).

rankToCoordinates(int pe, int &a, int &b, int &c, int &d, int &e, int &t):
   Get the coordinates of PE with rank *pe* on BG/Q.

coordinatesToRank(int x, int y, int z, int t):
   Returns the rank of PE with given coordinates (except BG/Q).

coordinatesToRank(int a, int b, int c, int d, int e, int t):
   Returns the rank of PE with given coordinates on BG/Q.

getHopsBetweenRanks(int pe1, int pe2):
   Returns the distance between the given PEs in terms of the hops count
   on the network between the two PEs.

printAllocation(FILE \*fp):
   Outputs the allocation for a particular execution to the given file.

For example, one can obtain the rank of a processor, whose coordinates are
known, on Cray XE6 using the following code:

.. code-block:: c++

   TopoManager *tmgr = TopoManager::getTopoManager();
   int rank, x, y, z, t;
   x = y = z = t = 2;
   rank = tmgr->coordinatesToRank(x, y, z, t);

For more examples, please refer to ``examples/charm++/topology``.

.. _physical:

Physical Node API
-----------------

The following calls provide information about the division and mapping
of physical hardware in Charm++. A processing element (PE) is a unit of
mapping and scheduling, which takes the form of an OS thread in SMP mode
and an OS process in non-SMP mode. A logical node (often shortened to
*node*) refers to an OS process: a set of one or more PEs that share
memory (i.e. an address space). A physical node refers to an individual
hardware machine (or, more precisely, an operating system instance on
which Charm++ processes execute, or, in networking terminology, a *host*).

Communication between PEs on the same logical node is faster than
communication between different logical nodes because OS threads share
the same address space and can directly interact through shared memory.
Communication between PEs on the same *physical* node may also be faster
than between different physical nodes depending on the availability of
OS features such as POSIX shared memory and Cross Memory Attach, the
abilities of the network interconnect in use, and the speed of network
loopback.

PEs are ranked in the range ``0`` to ``CmiNumPes()``. Likewise, logical nodes
are ranked from ``0`` to ``CmiNumNodes()``, and physical nodes are ranked from
``0`` to ``CmiNumPhysicalNodes()``.

Charm++ provides a set of functions for querying information about the
mapping of PE's to physical nodes. The ``cputopology.C`` module, contains the
following globally accessible functions:

int CmiPeOnSamePhysicalNode(int pe1, int pe2)
   Returns 1 if PEs ``pe1`` and ``pe2`` are on the same physical node and 0
   otherwise.

int CmiNumPhysicalNodes()
   Returns the number of physical nodes that the program is running on.

int CmiNumPesOnPhysicalNode(int node)
   Returns the number of PEs that reside within a physical node.

void CmiGetPesOnPhysicalNode(int node, int \**pelist, int \*num)
   After execution ``pelist`` will point to a list of all PEs that reside
   within a physical ``node`` and ``num`` will point to the length of the list.
   One should be careful to not free or alter ``pelist`` since it points to
   reserved memory.

int CmiPhysicalRank(int pe)
   Returns the rank of a PE among all PEs running on the same physical
   node.

int CmiPhysicalNodeID(int pe)
   Returns the node ID of the physical node in which a PE resides.

int CmiGetFirstPeOnPhysicalNode(int node)
   Returns the lowest numbered processor on a physical node.

.. _sec:checkpoint:

Checkpoint/Restart-Based Fault Tolerance
----------------------------------------

Charm++ offers two checkpoint/restart mechanisms. Each of these
targets a specific need in parallel programming. However, both of them
are based on the same infrastructure.

Traditional chare-array-based Charm++ applications, including AMPI
applications, can be checkpointed to storage buffers (either files or
memory regions) and be restarted later from those buffers. The basic
idea behind this is straightforward: checkpointing an application is
like migrating its parallel objects from the processors onto buffers,
and restarting is the reverse. Thanks to the migration utilities like
PUP methods (Section :numref:`sec:pup`), users can decide what
data to save in checkpoints and how to save them. However, unlike
migration (where certain objects do not need a PUP method), checkpoint
requires all the objects to implement the PUP method.

The two checkpoint/restart schemes implemented are:

-  Shared filesystem: provides support for *split execution*, where the
   execution of an application is interrupted and later resumed.

-  Double local-storage: offers an online *fault tolerance* mechanism
   for applications running on unreliable machines.

Split Execution
~~~~~~~~~~~~~~~

There are several reasons for having to split the execution of an
application. These include protection against job failure, a single
execution needing to run beyond a machine's job time limit, and resuming
execution from an intermediate point with different parameters. All of
these scenarios are supported by a mechanism to record execution state,
and resume execution from it later.

Parallel machines are assembled from many complicated components, each
of which can potentially fail and interrupt execution unexpectedly.
Thus, parallel applications that take long enough to run from start to
completion need to protect themselves from losing work and having to
start over. They can achieve this by periodically taking a checkpoint of
their execution state from which they can later resume.

Another use of checkpoint/restart is where the total execution time of
the application exceeds the maximum allocation time for a job in a
supercomputer. For that case, an application may checkpoint before the
allocation time expires and then restart from the checkpoint in a
subsequent allocation.

A third reason for having a split execution is when an application
consists of *phases* and each phase may be run a different number of
times with varying parameters. Consider, for instance, an application
with two phases where the first phase only has a possible configuration
(it is run only once). The second phase may have several configuration
(for testing various algorithms). In that case, once the first phase is
complete, the application checkpoints the result. Further executions of
the second phase may just resume from that checkpoint.

An example of Charm++'s support for split execution can be seen in
``tests/charm++/chkpt/hello``.

.. _sec:diskcheckpoint:

Checkpointing
^^^^^^^^^^^^^

The API to checkpoint the application is:

.. code-block:: c++

     void CkStartCheckpoint(char* dirname, const CkCallback& cb);

The string ``dirname`` is the destination directory where the checkpoint
files will be stored, and ``cb`` is the callback function which will be
invoked after the checkpoint is done, as well as when the restart is
complete. Here is an example of a typical use:

.. code-block:: c++

     /* ... */ CkCallback cb(CkIndex_Hello::SayHi(), helloProxy);
     CkStartCheckpoint("log", cb);

A chare array usually has a PUP routine for the sake of migration. The
PUP routine is also used in the checkpointing and restarting process.
Therefore, it is up to the programmer what to save and restore for the
application. One illustration of this flexibility is a complicated
scientific computation application with 9 matrices, 8 of which hold
intermediate results and 1 that holds the final results of each
timestep. To save resources, the PUP routine can well omit the 8
intermediate matrices and checkpoint the matrix with the final results
of each timestep.

Group, nodegroup (Section :numref:`sec:group`) and singleton chare
objects are normally not meant to be migrated. In order to checkpoint
them, however, the user has to write PUP routines for the groups and
chare and declare them as ``[migratable]`` in the .ci file. Some
programs use *mainchares* to hold key control data like global object
counts, and thus mainchares need to be checkpointed too. To do this, the
programmer should write a PUP routine for the mainchare and declare them
as ``[migratable]`` in the .ci file, just as in the case of Group and
NodeGroup.

The checkpoint must be recorded at a synchronization point in the
application, to ensure a consistent state upon restart. One easy way to
achieve this is to synchronize through a reduction to a single chare
(such as the mainchare used at startup) and have that chare make the
call to initiate the checkpoint.

After ``CkStartCheckpoint`` is executed, a directory of the designated
name is created and a collection of checkpoint files are written into
it.

Restarting
^^^^^^^^^^

The user can choose to run the Charm++ application in restart mode,
i.e., restarting execution from a previously-created checkpoint. The
command line option ``+restart DIRNAME`` is required to invoke this
mode. For example:

.. code-block:: bash

     $ ./charmrun hello +p4 +restart log

Restarting is the reverse process of checkpointing. Charm++ allows
restarting the old checkpoint on a different number of physical
processors. This provides the flexibility to expand or shrink your
application when the availability of computing resources changes.

Note that on restart, if an array or group reduction client was set to a
static function, the function pointer might be lost and the user needs
to register it again. A better alternative is to always use an entry
method of a chare object. Since all the entry methods are registered
inside Charm++ system, in the restart phase, the reduction client will
be automatically restored.

After a failure, the system may contain fewer or more processors. Once
the failed components have been repaired, some processors may become
available again. Therefore, the user may need the flexibility to restart
on a different number of processors than in the checkpointing phase.
This is allowable by giving a different ``+pN`` option at runtime. One
thing to note is that the new load distribution might differ from the
previous one at checkpoint time, so running a load balancer (see
Section :numref:`loadbalancing`) after restart is suggested.

If restart is not done on the same number of processors, the
processor-specific data in a group/nodegroup branch cannot (and usually
should not) be restored individually. A copy from processor 0 will be
propagated to all the processors.

Choosing What to Save
^^^^^^^^^^^^^^^^^^^^^

In your programs, you may use chare groups for different types of
purposes. For example, groups holding read-only data can avoid excessive
data copying, while groups maintaining processor-specific information
are used as a local manager of the processor. In the latter situation,
the data is sometimes too complicated to save and restore but easy to
re-compute. For the read-only data, you want to save and restore it in
the PUP’er routine and leave empty the migration constructor, via which
the new object is created during restart. For the easy-to-recompute type
of data, we just omit the PUP’er routine and do the data reconstruction
in the group’s migration constructor.

A similar example is the program mentioned above, where there are two
types of chare arrays, one maintaining intermediate results while the
other type holds the final result for each timestep. The programmer can
take advantage of the flexibility by leaving PUP’er routine empty for
intermediate objects, and do save/restore only for the important
objects.

.. _sec:MemCheckpointing:

Online Fault Tolerance
~~~~~~~~~~~~~~~~~~~~~~

As supercomputers grow in size, their reliability decreases
correspondingly. This is due to the fact that the ability to assemble
components in a machine surpasses the increase in reliability per
component. What we can expect in the future is that applications will
run on unreliable hardware.

The previous disk-based checkpoint/restart can be used as a fault
tolerance scheme. However, it would be a very basic scheme in that when
a failure occurs, the whole program gets killed and the user has to
manually restart the application from the checkpoint files. The double
local-storage checkpoint/restart protocol described in this subsection
provides an automatic fault tolerance solution. When a failure occurs,
the program can automatically detect the failure and restart from the
checkpoint. Further, this fault-tolerance protocol does not rely on any
reliable external storage (as needed in the previous method). Instead,
it stores two copies of checkpoint data to two different locations (can
be memory or local disk). This double checkpointing ensures the
availability of one checkpoint in case the other is lost. The double
in-memory checkpoint/restart scheme is useful and efficient for
applications with small memory footprint at the checkpoint state. The
double in-disk variant stores checkpoints into local disk, thus can be
useful for applications with large memory footprint.

Checkpointing
^^^^^^^^^^^^^

The function that application developers can call to record a checkpoint
in a chare-array-based application is:

.. code-block:: c++

         void CkStartMemCheckpoint(CkCallback &cb)

where ``cb`` has the same meaning as in
section :numref:`sec:diskcheckpoint`. Just like the above disk
checkpoint described, it is up to the programmer to decide what to save.
The programmer is responsible for choosing when to activate
checkpointing so that the size of a global checkpoint state, and
consequently the time to record it, is minimized.

In AMPI applications, the user just needs to create an ``MPI_Info`` object
with the key ``"ampi_checkpoint"`` and a value of either ``"in_memory"`` (for a
double in-memory checkpoint) or ``"to_file=file_name"`` (to checkpoint to
disk), and pass that object to the function ``AMPI_Migrate()`` as in the
following:

.. code-block:: c++

   // Setup
   MPI_Info in_memory, to_file;

   MPI_Info_create(&in_memory);
   MPI_Info_set(in_memory, "ampi_checkpoint", "in_memory");

   MPI_Info_create(&to_file);
   MPI_Info_set(to_file, "ampi_checkpoint", "to_file=chkpt_dir");

   ...

   // Main time-stepping loop
   for (int iter=0; iter < max_iters; iter++) {

     // Time step work ...

     if (iter % chkpt_freq == 0)
       AMPI_Migrate(in_memory);
   }

.. _restarting-1:

Restarting
^^^^^^^^^^

When a processor crashes, the restart protocol will be automatically
invoked to recover all objects using the last checkpoints. The program
will continue to run on the surviving processors. This is based on the
assumption that there are no extra processors to replace the crashed
ones.

However, if there are a pool of extra processors to replace the crashed
ones, the fault-tolerance protocol can also take advantage of this to
grab one free processor and let the program run on the same number of
processors as before the crash. In order to achieve this, Charm++ needs
to be compiled with the macro option ``CK_NO_PROC_POOL`` turned on.

Double in-disk checkpoint/restart
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A variant of double memory checkpoint/restart, *double in-disk
checkpoint/restart*, can be applied to applications with large memory
footprint. In this scheme, instead of storing checkpoints in the memory,
it stores them in the local disk. The checkpoint files are named
``ckpt[CkMyPe]-[idx]-XXXXX`` and are stored under the ``/tmp`` directory.

Users can pass the runtime option ``+ftc_disk`` to activate this mode. For
example:

.. code-block:: c++

      ./charmrun hello +p8 +ftc_disk

Building Instructions
^^^^^^^^^^^^^^^^^^^^^

In order to have the double local-storage checkpoint/restart
functionality available, the parameter ``syncft`` must be provided at
build time:

.. code-block:: c++

      ./build charm++ netlrts-linux-x86_64 syncft

At present, only a few of the machine layers underlying the Charm++
runtime system support resilient execution. These include the TCP-based
``net`` builds on Linux and Mac OS X. For clusters overbearing
job-schedulers that kill a job if a node goes down, the way to
demonstrate the killing of a process is show in
Section :numref:`ft:inject` . Charm++ runtime system can
automatically detect failures and restart from checkpoint.

Failure Injection
^^^^^^^^^^^^^^^^^

To test that your application is able to successfully recover from
failures using the double local-storage mechanism, we provide a failure
injection mechanism that lets you specify which PEs will fail at what
point in time. You must create a text file with two columns. The first
colum will store the PEs that will fail. The second column will store
the time at which the corresponding PE will fail. Make sure all the
failures occur after the first checkpoint. The runtime parameter
``kill_file`` has to be added to the command line along with the file
name:

.. code-block:: bash

   $ ./charmrun hello +p8 +kill_file <file>

An example of this usage can be found in the ``syncfttest`` targets in
``tests/charm++/jacobi3d``.

.. _ft:inject:

Failure Demonstration
^^^^^^^^^^^^^^^^^^^^^

For HPC clusters, the job-schedulers usually kills a job if a node goes
down. To demonstrate restarting after failures on such clusters,
``CkDieNow()`` function can be used. You just need to place it at any
place in the code. When it is called by a processor, the processor will
hang and stop responding to any communication. A spare processor will
replace the crashed processor and continue execution after getting the
checkpoint of the crashed processor. To make it work, you need to add
the command line option ``+wp``, the number following that option is the
working processors and the remaining are the spare processors in the
system.

.. _sec:looplevel:

Support for Loop-level Parallelism
----------------------------------

To better utilize the multicore chip, it has become increasingly popular
to adopt shared-memory multithreading programming methods to exploit
parallelism on a node. For example, in hybrid MPI programs, OpenMP is
the most popular choice. When launching such hybrid programs, users have
to make sure there are spare physical cores allocated to the
shared-memory multithreading runtime. Otherwise, the runtime that
handles distributed-memory programming may interfere with resource
contention because the two independent runtime systems are not
coordinated. If spare cores are allocated, in the same way of launching
a MPI+OpenMP hybrid program, Charm++ will work perfectly with any
shared-memory parallel programming languages (e.g. OpenMP). As with
ordinary OpenMP applications, the number of threads used in the OpenMP
parts of the program can be controlled with the ``OMP_NUM_THREADS``
environment variable. See Sec. :numref:`charmrun` for details on how
to propagate such environment variables.

If there are no spare cores allocated, to avoid resource contention, a
*unified runtime* is needed to support both intra-node shared-memory
multithreading parallelism and inter-node distributed-memory
message-passing parallelism. Additionally, considering that a parallel
application may have only a small fraction of its critical computation
be suitable for porting to shared-memory parallelism (the savings on
critical computation may also reduce the communication cost, thus
leading to more performance improvement), dedicating physical cores on
every node to the shared-memory multithreading runtime will waste
computational power because those dedicated cores are not utilized at
all during most of the application's execution time. This case indicates
the necessity of a unified runtime supporting both types of parallelism.

CkLoop library
~~~~~~~~~~~~~~

The *CkLoop* library is an add-on to the Charm++ runtime to achieve such
a unified runtime. The library implements a simple OpenMP-like
shared-memory multithreading runtime that reuses Charm++ PEs to perform
tasks spawned by the multithreading runtime. This library targets the
SMP mode of Charm++.

The *CkLoop* library is built in
``$CHARM_DIR/$MACH_LAYER/tmp/libs/ck-libs/ckloop`` by executing ``make``. To
use it for user applications, one has to include ``CkLoopAPI.h`` in the
source code. The interface functions of this library are as follows:

-  CProxy_FuncCkLoop **CkLoop_Init**\ (int numThreads=0): This function
   initializes the CkLoop library, and it only needs to be called once
   on a single PE during the initialization phase of the application.
   The argument ``numThreads`` is only used in non-SMP mode, specifying
   the number of threads to be created for the single-node shared-memory
   parallelism. It will be ignored in SMP mode.

-  void **CkLoop_SetSchedPolicy**\ (CkLoop_sched
   schedPolicy=CKLOOP_NODE_QUEUE) : This function sets the scheduling
   policy of CkLoop work, three options available: ``CKLOOP_NODE_QUEUE``,
   ``CKLOOP_TREE`` and ``CKLOOP_LIST``. The default policy, ``CKLOOP_NODE_QUEUE`` on
   supported environments is to use node_queue message so that master or
   another idle PE delievers the CkLoop work to all other PEs.
   ``CKLOOP_TREE`` policy is set by default for builds not supporting a node
   queue. This policy delivers CkLoop messages on the implicit tree.
   ``CKLOOP_LIST`` uses list to deliver the messages.

-  void **CkLoop_Exit**\ (CProxy_FuncCkLoop ckLoop): This function is
   intended to be used in non-SMP mode, as it frees the resources (e.g.
   terminating the spawned threads) used by the CkLoop library. It
   should be called on just one PE.

-  | void **CkLoop_Parallelize**\ (
   | HelperFn func, /\* the function that finishes partial work on
     another thread \*/
   | int paramNum, /\* the number of parameters for func \*/
   | void \* param, /\* the input parameters for the above func \*/
   | int numChunks, /\* number of chunks to be partitioned \*/
   | int lowerRange, /\* lower range of the loop-like parallelization
     [lowerRange, upperRange] \*/
   | int upperRange, /\* upper range of the loop-like parallelization
     [lowerRange, upperRange] \*/
   | int sync=1, /\* toggle implicit barrier after each parallelized
     loop \*/
   | void \*redResult=NULL, /\* the reduction result, ONLY SUPPORT
     SINGLE VAR of TYPE int/float/double \*/
   | REDUCTION_TYPE type=CKLOOP_NONE /\* type of the reduction result
     \*/
   | CallerFn cfunc=NULL, /\* caller PE will call this function before
     ckloop is done and before starting to work on its chunks \*/
   | int cparamNum=0, void \*cparam=NULL /\* the input parameters to the
     above function \*/
   | )
   | The "HelperFn" is defined as "typedef void (\*HelperFn)(int
     first,int last, void \*result, int paramNum, void \*param);" and
     the "result" is the buffer for reduction result on a single
     simple-type variable. The "CallerFn" is defined as "typedef void
     (\*CallerFn)(int paramNum, void \*param);"

Lambda syntax for *CkLoop* is also supported. The interface for using
lambda syntax is as follows:

.. code-block:: c++

      void CkLoop_Parallelize(
      int numChunks, int lowerRange, int upperRange,
          [=](int first, int last, void *result) {
          for (int i = first; i <= last; ++i ) {
            // work to parallelize goes here
          }
        }, void *redResult=NULL, REDUCTION_TYPE type=CKLOOP_NONE,
        std::function<void()> cfunc=NULL
        }
      );

Examples using this library can be found in ``examples/charm++/ckloop``
and the widely used molecular dynamics simulation application
NAMD [14]_.

The CkLoop Hybrid library
^^^^^^^^^^^^^^^^^^^^^^^^^

The CkLoop_Hybrid library is a mode of CkLoop that incorporates specific
adaptive scheduling strategies aimed at providing a tradeoff between
dynamic load balance and spatial locality. It is used in a build of
Charm++ where all chares are placed on core 0 of each node (called the
drone-mode, or all-drones-mode). It incorporates a strategy called
staggered static-dynamic scheduling (from dissertation work of Vivek
Kale). The iteration space is first tentatively divided approximately
equally to all available PEs. Each PE's share of the iteration space is
divided into a static portion, specified by the staticFraction parameter
below, and the remaining dynamic portion. The dynamic portion of a PE is
divided into chunks of specified chunksize, and enqueued in the
task-queue associated with that PE. Each PE works on its static portion,
and then on its own task queue (thus preserving spatial locality, as
well as persistence of allocations across outer iterations), and after
finishing that, steals work from other PE's task queues.

CkLoopHybrid support requires the SMP mode of Charm++ and the additional
flags ``-enable-drone-mode`` and ``-enable-task-queue`` to be passed as build
options when Charm++ is built.

The changes to the CkLoop API call are the following:

-  **CkLoop_Init** does not need to be called

-  **CkLoop_SetSchedPolicy** is not supported

-  **CkLoop_Exit** does not need to be called

-  **CkLoop_Parallelize** call is similar to CkLoop but has an
   additional variable that provides the fraction of iterations that are
   statically scheduled:

   .. code-block:: c++

      void CkLoop_ParallelizeHybrid(
      float staticFraction,
      HelperFn func, /* the function that finishes partial work on another thread */
      int paramNum, /* the number of parameters for func */
      void * param, /* the input parameters for the above func */
      int numChunks, /* number of chunks to be partitioned */
      int lowerRange, /* lower range of the loop-like parallelization [lowerRange, upperRange] */
      int upperRange, /* upper range of the loop-like parallelization [lowerRange, upperRange] */
      int sync=1, /* toggle implicit barrier after each parallelized loop */
      void *redResult=NULL, /* the reduction result, ONLY SUPPORT SINGLE VAR of TYPE int/float/double */
      REDUCTION_TYPE type=CKLOOP_NONE /* type of the reduction result */
      CallerFn cfunc=NULL, /* caller PE will call this function before ckloop is done and before starting to work on its chunks */
      int cparamNum=0, void *cparam=NULL /* the input parameters to the above function */
      )

Reduction is supported for type ``CKLOOP_INT_SUM``, ``CKLOOP_FLOAT_SUM``,
``CKLOOP_DOUBLE_SUM``. It is recommended to use this mode without reduction.

Charm++/Converse Runtime Scheduler Integrated OpenMP
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The compiler-provided OpenMP runtime library can work with Charm++ but
it creates its own thread pool so that Charm++ and OpenMP can have
oversubscription problem. The integrated OpenMP runtime library
parallelizes OpenMP regions in each chare and runs on the Charm++
runtime without oversubscription. The integrated runtime creates OpenMP
user-level threads, which can migrate among PEs within a node. This
fine-grained parallelism by the integrated runtime helps resolve load
imbalance within a node easily. When PEs become idle, they help other
busy PEs within a node via work-stealing.

Instructions to build and use the integrated OpenMP library
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Instructions to build
'''''''''''''''''''''

The OpenMP library can be built with ``omp`` keyword on any smp version
of Charm++ including multicore build when you build Charm++ or AMPI,
for example:

.. code-block:: bash

   $ $CHARM_DIR/build charm++ multicore-linux64 omp
   $ $CHARM_DIR/build charm++ netlrts-linux-x86_64 smp omp

This library is based on the LLVM OpenMP runtime library. So it supports
the ABI used by clang, intel and gcc compilers.
The following is the list of compilers which are verified to support
this integrated library on Linux.

-  GCC: 4.8 or newer

-  ICC: 15.0 or newer

-  Clang: 3.7 or newer

You can use this integrated OpenMP with *clang* on IBM Blue Gene machines
without special compilation flags (don't need to add -fopenmp or
-openmp on Blue Gene clang).

On Linux, the OpenMP supported version of clang has been installed in
default recently. For example, Ubuntu has been released with clang
higher than 3.7 since 15.10. Depending on which version of clang is
installed in your working environments, you should follow additional
instructions to use this integrated OpenMP with Clang. The following is
the instruction to use clang on Ubuntu where the default clang is older
than 3.7. If you want to use clang on other Linux distributions, you can
use package managers on those Linux distributions to install clang and
OpenMP library. This installation of clang will add headers for OpenMP
environmental routines and allow you to parse the OpenMP directives.
However, on Ubuntu, the installation of clang doesn't come with its
OpenMP runtime library so it results in an error message saying that it
fails to link the compiler provided OpenMP library. This library is not
needed to use the integrated OpenMP runtime but you need to avoid this
error to succeed compiling your codes. The following are the instructions
to avoid the error:

.. code-block:: bash

   # When you want to compile Integrated OpenMP on Ubuntu where the pre-installed clang
   # is older than 3.7, you can use integrated openmp with the following instructions.
   # e.g.) Ubuntu 14.04, the version of default clang is 3.4.
   $ sudo apt-get install clang-3.8 //you can use any version of clang higher than 3.8
   $ sudo ln -svT /usr/bin/clang-3.8 /usr/bin/clang
   $ sudo ln -svT /usr/bin/clang++-3.8 /usr/bin/clang

   $ $CHARM_DIR/build charm++ multicore-linux64 clang omp --with-production -j8
   $ echo '!<arch>' > $(CHARM_DIR)/lib/libomp.a  # Dummy library. This will make you avoid the error message.

On Mac, the Apple-provided clang installed in default doesn’t have
OpenMP feature. We're working on the support of this library on Mac
with OpenMP enabled clang which can be downloaded and installed
through Homebrew or MacPorts. Currently, this integrated library is
built and compiled on Mac with the normal GCC which can be downloaded
and installed via Homebrew and MacPorts. If installed globally, GCC
will be accessible by appending the major version number and adding it
to the invocation of the Charm++ build script. For example:

.. code-block:: bash

   $ $CHARM_DIR/build charm++ multicore-linux64 omp gcc-7
   $ $CHARM_DIR/build charm++ netlrts-linux-x86_64 smp omp gcc-7

If this does not work, you should set environment variables so that the
Charm++ build script uses the normal gcc installed from Homebrew or
MacPorts. The following is an example using Homebrew on Mac OS X
10.12.5:

.. code-block:: bash

   # Install Homebrew from https://brew.sh
   # Install gcc using 'brew' */
   $ brew install gcc

   # gcc, g++ and other binaries are installed at /usr/local/Cellar/gcc/<version>/bin
   # You need to make symbolic links to the gcc binaries at /usr/local/bin
   # In this example, gcc 7.1.0 is installed at the directory.
   $ cd /usr/local/bin
   $ ln -sv /usr/local/Cellar/gcc/7.1.0/bin/gcc-7 gcc
   $ ln -sv /usr/local/Cellar/gcc/7.1.0/bin/g++-7 g++
   $ ln -sv /usr/local/Cellar/gcc/7.1.0/bin/gcc-nm-7 gcc-nm
   $ ln -sv /usr/local/Cellar/gcc/7.1.0/bin/gcc-ranlib-7 gcc-ranlib
   $ ln -sv /usr/local/Cellar/gcc/7.1.0/bin/gcc-ar-7 gcc-ar

   # Finally, you should set PATH variable so that these binaries are accessed first in the build script.
   $ export PATH=/usr/local/bin:$PATH

In addition, this library will be supported on Windows in the next
release of Charm++.

How to use the integrated OpenMP on Charm++
'''''''''''''''''''''''''''''''''''''''''''

To use this library on your applications, you have to add ``-module OmpCharm``
in compile flags to link this library instead of the
compiler-provided library in compilers. Without ``-module OmpCharm``, your
application will use the compiler-provided OpenMP library which running
on its own separate runtime (you don't need to add ``-fopenmp`` or ``-openmp``
with gcc and icc. These flags are included in the predefined
compile options when you build Charm++ with ``omp``).

This integrated OpenMP adjusts the number of OpenMP instances on each
chare so the number of OpenMP instances can be changed for each OpenMP
region over execution. If your code shares some data structures among
OpenMP instances in a parallel region, you can set the size of the data
structures before the start of the OpenMP region with
``omp_get_max_threads()`` and use the data structure within each OpenMP
instance with ``omp_get_thread_num()``. After the OpenMP region, you can
iterate over the data structure to combine partial results with
``CmiGetCurKnownOmpThreads()``. ``CmiGetCurKnownOmpThreads()`` returns the
number of OpenMP threads for the latest OpenMP region on the PE where a
chare is running. The following is an example to describe how you can
use shared data structures for OpenMP regions on the integrated OpenMP
with Charm++:

.. code-block:: c++

   /* Maximum possible number of OpenMP threads in the upcoming OpenMP region.
      Users can restrict this number with 'omp_set_num_threads()' for each chare
      and the environmental variable, 'OMP_NUM_THREADS' for all chares.
      By default, omp_get_max_threads() returns the number of PEs for each logical node.
   */
   int maxAvailableThreads = omp_get_max_threads();
   int *partialResult = new int[maxAvailableThreads]{0};

   /* Partial sum for subsets of iterations assigned to each OpenMP thread.
      The integrated OpenMP runtime decides how many OpenMP threads to create
      with some heuristics internally.
   */
   #pragma omp parallel for
   for (int i = 0; i < 128; i++) {
     partialResult[omp_get_thread_num()] +=i;
   }
   /* We can know how many OpenMP threads are created in the latest OpenMP region
      by CmiCurKnownOmpthreads().
      You can get partial results each OpenMP thread generated */
   for (int j = 0; j < CmiCurKnownOmpThreads(); j++)
     CkPrintf("partial sum of thread %d: %d \n", j, partialResult[j]);

The list of supported pragmas
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This library is forked from LLVM OpenMP Library supporting OpenMP 4.0.
Among many number of directives specified in OpenMP 4.0, limited set of
directives are supported. The following list of supported pragmas is
verified from the OpenMP conformance test suite which forked from LLVM
OpenMP library and ported to Charm++ program running multiple OpenMP
instances on chares. The test suite can be found in
``tests/converse/openmp_test``.

.. code-block:: c++

   /* omp_<directive>_<clauses> */
   omp_atomic
   omp_barrier
   omp_critical
   omp_flush

   /* the following directives means they work within omp parallel region */
   omp_for_firstprivate
   omp_for_lastprivate
   omp_for_nowait
   omp_for_private
   omp_for_reduction
   omp_for_schedule_dynamic
   omp_for_schedule_guided
   omp_for_schedule_static
   omp_section_firstprivate
   omp_section_lastprivate
   omp_section_private
   omp_sections_nowait
   omp_sections_reduction

   omp_get_num_threads
   omp_get_wtick
   omp_get_wtime
   omp_in_parallel
   omp_master
   omp_master_3
   omp_parallel_default
   omp_parallel_firstprivate

   /* the following directives means the combination of 'omp parallel and omp for/section' works */
   omp_parallel_for_firstprivate
   omp_parallel_for_if
   omp_parallel_for_lastprivate
   omp_parallel_for_private
   omp_parallel_for_reduction
   omp_parallel_sections_firstprivate
   omp_parallel_sections_lastprivate
   omp_parallel_sections_private
   omp_parallel_sections_reduction

   omp_parallel_if
   omp_parallel_private
   omp_parallel_reduction
   omp_parallel_shared
   omp_single
   omp_single_nowait
   omp_single_private

The other directives in OpenMP standard will be supported in the next
version.

A simple example using this library can be found in
``examples/charm++/openmp``. You can compare CkLoop and the integrated
OpenMP with this example. You can see that the total execution time of
this example with enough big size of problem is faster with OpenMP than
CkLoop thanks to load balancing through work-stealing between threads
within a node while the execution time of each chare can be slower on
OpenMP because idle PEs helping busy PEs.

API to control which PEs participating in CkLoop/OpenMP work
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

User may want certain PE not to be involved in other PE’s loop-level
parallelization for some cases because it may add latency to works in
the PE by helping other PEs. User can enable or disable each PE to
participate in the loop-level parallelization through the following API:

void **CkSetPeHelpsOtherThreads** (int value)

value can be 0 or 1, 0 means this API disable the current PE to help
other PEs. value 1 or others can enable the current PE for loop-level
parallelization. By default, all the PEs are enabled for the loop-level
parallelization by CkLoop and OpenMP. User should explicitly enable the
PE again by calling this API with value 1 after they disable it during
certain procedure so that the PE can help others after that. The
following example shows how this API can be used.

.. code-block:: c++

   CkSetPeHelpsOtherThreads(0);

   /* codes which can run without the current PE
   interuppted by loop-level work from other PEs */

   CkSetPeHelpsOtherThreads(1);

.. _sec:interop:

Charm-MPI Interoperation
------------------------

Codes and libraries written in Charm++ and MPI can also be used in an
interoperable manner. Currently, this functionality is supported only if
Charm++ is built using MPI, PAMILRTS, or GNI as the network layer (e.g.
mpi-linux-x86_64 build). An example program to demonstrate the
interoperation is available in ``examples/charm++/mpi-coexist``. In the
following text, we will refer to this example program for the ease of
understanding.

Control Flow and Memory Structure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The control flow and memory structure of a Charm++-MPI interoperable
program is similar to that of a pure MPI program that uses external
MPI libraries. The execution of program begins with pure MPI code’s
*main*. At some point after MPI_Init() has been invoked, the following
function call should be made to initialize Charm++:

**void CharmLibInit(MPI_Comm newComm, int argc, char \**argv)**

If Charm++ is build on top of MPI, ``newComm`` is the MPI communicator
that Charm++ will use for the setup and communication. All the MPI
ranks that belong to ``newComm`` should call this function collectively.
A collection of MPI ranks that make the ``CharmLibInit`` call defines a
new Charm++ instance. Different MPI ranks that belong to different
communicators can make this call independently, and separate Charm++
instances that are not aware of each other will be created. This
results in a space division. As of now, a particular MPI rank can only
be part of one unique Charm++ instance. For PAMILRTS and GNI, the
``newComm`` argument is ignored. These layers do not support the space
division of given processors, but require all processors to make the
``CharmLibInit`` call. The mode of interoperating here is called time
division, and can be used with MPI-based Charm++ also if the size of
``newComm`` is same as ``MPI_COMM_WORLD``. Arguments ``argc`` and ``argv``
should contain the information required by Charm++ such as the load balancing
strategy, etc.

During initialization, control is transferred from MPI to the
Charm++ RTS on the MPI ranks that made the call. Along with basic setup,
the Charm++ RTS also invokes the constructors of all mainchares during
initialization. Once the initial set up is done, control is transferred
back to MPI as if returning from a function call. Since Charm++
initialization is made via a function call from the pure MPI program,
Charm++ resides in the same memory space as the pure MPI program. This
makes transfer of data to/from MPI from/to Charm++ convenient (using
pointers).

Writing Interoperable Charm++ Libraries
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Minor modifications are required to make a Charm++ program interoperable
with a pure MPI program:

-  If the interoperable Charm++ library does not contain a main module,
   the Charm++ RTS provides a main module and the control is returned
   back to MPI after the initialization is complete. In the other case,
   the library should explicitly call ``CkExit`` to mark the end of
   initialization and the user should provide ``-nomain-module`` link time
   flag when building the final executable.

-  ``CkExit`` should be used the same way a *return* statement is used for
   returning back from a function call. ``CkExit`` should be called only
   once from one of the processors. This unique call marks the transfer
   of control from Charm++ RTS to MPI.

-  Include ``mpi-interoperate.h`` - if not included in the files that call
   ``CkExit``, invoking ``CkExit`` will result in unspecified behavior.

-  Since the ``CharmLibInit`` call invokes the constructors of mainchares,
   the constructors of mainchares should only perform basic set up such
   as creation of chare arrays etc, i.e. the set up should not result in
   invocation of actual work, which should be done using interface
   functions (when desired from the pure MPI program). However, if the
   main module is provided by the library, ``CharmLibInit`` behaves like a
   regular Charm++ startup and execution which is stopped when ``CkExit`` is
   explicitly called by the library. One can also avoid use of
   mainchares, and perform the necessary initializations in an interface
   function as demonstrated in the interoperable library
   ``examples/charm++/mpi-coexist/libs/hello``.

-  *Interface functions* - Every library needs to define interface
   function(s) that can be invoked from pure MPI programs, and transfers
   the control to the Charm++ RTS. The interface functions are simple
   functions whose task is to start work for the Charm++ libraries. Here
   is an example interface function for the *hello* library.

   .. code-block:: c++

      void HelloStart(int elems)
      {
        if(CkMyPe() == 0) {
          CkPrintf("HelloStart - Starting lib by calling constructor of MainHello\n");
          CProxy_MainHello mainhello = CProxy_MainHello::ckNew(elems);
        }
        StartCharmScheduler(-1);
      }

   This function creates a new chare (``mainHello``) defined in the *hello*
   library which subsequently results in work being done in *hello*
   library. More examples of such interface functions can be found in hi
   (HiStart) and kNeighbor (kNeighbor) directories in
   ``examples/charm++/mpi-coexist/libs``. Note that a scheduler call
   ``StartCharmScheduler()`` should be made from the interface functions
   to start the message reception by Charm++ RTS.

Writing Interoperable MPI Programs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

An MPI program that invokes Charm++ libraries should include
``mpi-interoperate.h``. As mentioned earlier, an initialization call,
``CharmLibInit`` is required after invoking ``MPI_Init`` to perform the
initial set up of Charm++. It is advisable to call an ``MPI_Barrier`` after
a control transfer between Charm++ and MPI to avoid any side effects.
Thereafter, a Charm++ library can be invoked at any point using the
interface functions. One may look at
``examples/charm++/mpi-coexist/multirun.cpp`` for a working example. Based
on the way interfaces are defined, a library can be invoked multiple
times. In the end, one should call ``CharmLibExit`` to free resources
reserved by Charm++.

Compilation and Execution
~~~~~~~~~~~~~~~~~~~~~~~~~

An interoperable Charm++ library can be compiled as usual using
*charmc*. Instead of producing an executable in the end, one should
create a library (\*.a) as shown in
``examples/charm++/mpi-coexist/libs/hi/Makefile``. The compilation process
of the MPI program, however, needs modification. One has to include the
charm directory (``-I$(CHARMDIR)/include``) to help the compiler find the
location of included ``mpi-interoperate.h``. The linking step to create
the executable should be done using ``charmc``, which in turn uses the
compiler used to build charm. In the linking step, it is required to
pass ``-mpi`` as an argument because of which *charmc* performs the
linking for interoperation. The charm libraries, which one wants to be
linked, should be passed using ``-module`` option. Refer to
``examples/charm++/mpi-coexist/Makefile`` to view a working example. For
execution on BG/Q systems, the following additional argument should be
added to the launch command: ``-envs PAMI_CLIENTS=MPI,Converse``.

User Driven Mode
~~~~~~~~~~~~~~~~

In addition to the above technique for interoperation, one can also
interoperate with Charm++ in user driven mode. User driven mode is
intended for cases where the developer has direct control over the both
the Charm++ code and the non-Charm++ code, and would like a more tightly
coupled relation between the two. When executing in user driven mode,
*main* is called on every rank as in the above example. To initialize
the Charm++ runtime, a call to ``CharmInit`` should be called on every
rank:

**void CharmInit(int argc, char **argv)**

``CharmInit`` starts the Charm++ runtime in user driven mode, and
executes the constructor of the main chare. Control returns to user
code when a call to ``CkExit`` is made. Once control is returned, user
code can do other work as needed, including creating chares, and
invoking entry methods on proxies. Any messages created by the user
code will be sent/received the next time the user calls
``StartCharmScheduler``. Calls to ``StartCharmScheduler`` allow the
Charm++ runtime to resume sending and processing messages, and control
returns to user code when ``CkExit`` is called. The Charm++ scheduler
can be started and stopped in this fashion as many times as necessary.
``CharmLibExit`` should be called by the user code at the end of
execution.

A small example of user driven interoperation can be found in
``examples/charm++/user-driven-interop``.

.. _sec:partition:

Partitioning in Charm++
-----------------------

Starting with the 6.5.0 release, Charm++ was augmented with support
for partitioning. The key idea is to divide the allocated set of nodes
into subsets that run independent Charm++ instances. These Charm++
instances (called *partitions* from now on) have a unique identifier, can
be programmed to do different tasks, and can interact with each other.
Addition of the partitioning scheme does not affect the existing code
base or codes that do not want to use partitioning. Some of the use
cases of partitioning are replicated NAMD, replica-based fault
tolerance, studying mapping performance etc. In some aspects,
partitioning is similar to disjoint communicator creation in MPI.

Overview
~~~~~~~~

The Charm++ stack has three components - Charm++, Converse and a machine
layer. In general, machine layer handles the exchange of messages among
nodes, and interacts with the next layer in the stack - Converse.
Converse is responsible for scheduling of tasks (including user code)
and is used by Charm++ to execute the user application. Charm++ is the
top-most level in which the applications are written. During
partitioning, Charm++ and machine layers are unaware of the
partitioning. Charm++ assumes its partition to be the entire world,
whereas machine layer considers the whole set of allocated nodes as one
partition. During start up, converse divides the allocated set of nodes
into partitions, in each of which Charm++ instances are run. It performs
the necessary translations as interactions happen with Charm++ and the
machine layer. The partitions can communicate with each other using the
Converse API described later.

Ranking
~~~~~~~

The Charm++ stack assigns a rank to every processing element (PE). In the
non-partitioned version, a rank assigned to a PE is the same at all three
layers of the Charm++ stack. This rank also generally coincides with the
rank provided to processors/cores by the underlying job scheduler. The
importance of these ranks derive from the fact that they are used for
multiple purposes. Partitioning leads to segregation of the notion of
ranks at different levels of Charm++ stack. What used to be the PE is
now a local rank within a partition running a Charm++ instance. Existing
methods such as ``CkMyPe()``, ``CkMyNode()``, ``CmiMyPe()``, etc.
continue to provide these local ranks. Hence, existing codes do not
require any change as long as inter-partition interaction is not
required.

On the other hand, machine layer is provided with the target ranks that
are globally unique. These ranks can be obtained using functions with the
*Global* suffix such as ``CmiNumNodesGlobal()``, ``CmiMyNodeGlobal()``,
``CmiMyPeGlobal()`` etc.

Converse, which operates at a layer between Charm++ and machine layer,
performs the required transitions. It maintains relevant information for
any conversion. Information related to partitions can be obtained using
Converse level functions such as ``CmiMyPartition()``,
``CmiNumPartitions()``, etc. If required, one can also obtain the
mapping of a local rank to a global rank using functions such as
``CmiGetPeGlobal(int perank, int partition)`` and
``CmiGetNodeGlobal(int noderank, int partition)``. These functions take
two arguments - the local rank and the partition number. For example,
CmiGetNodeGlobal(5, 2) will return the global rank of the node that
belongs to partition 2 and has a local rank of 5 in partition 2. The
inverse translation, from global rank to local rank, is not supported.

Startup and Partitioning
~~~~~~~~~~~~~~~~~~~~~~~~

A number of compile time and runtime parameters are available for users
who want to run multiple partitions in one single job.

-  Runtime parameter: ``+partitions <part_number>`` or
   ``+replicas <replica_number>`` - number of partitions to be created.
   If no further options are provided, allocated cores/nodes are divided
   equally among partitions. Only this option is supported from the
   6.5.0 release; remaining options are supported starting 6.6.0.

-  Runtime parameter: ``+master_partition`` - assign one core/node as
   the master partition (partition 0), and divide the remaining
   cores/nodes equally among remaining partitions.

-  Runtime parameter: ``+partition_sizes L[-U[:S[.R]]]#W[,...]`` -
   defines the size of partitions. A single number identifies a
   particular partition. Two numbers separated by a dash identify an
   inclusive range (*lower bound* and *upper bound*). If they are
   followed by a colon and another number (a *stride*), that range will
   be stepped through in increments of the additional number. Within
   each stride, a dot followed by a *run* will indicate how many
   partitions to use from that starting point. Finally, a compulsory
   number sign (#) followed by a *width* defines the size of each of the
   partitions identified so far. For example, the sequence
   ``0-4:2#10,1#5,3#15`` states that partitions 0, 2, 4 should be of
   size 10, partition 1 of size 5 and partition 3 of size 15. In SMP
   mode, these sizes are in terms of nodes. All workers threads
   associated with a node are assigned to the partition of the node.
   This option conflicts with ``+assign_master``.

-  Runtime parameter: ``+partition_topology`` - use a default topology
   aware scheme to partition the allocated nodes.

-  Runtime parameter: ``+partition_topology_scheme <scheme>`` - use the
   given scheme to partition the allocated nodes. Currently, two
   generalized schemes are supported that should be useful on torus
   networks. If scheme is set to 1, allocated nodes are traversed plane
   by plane during partitioning. A hilbert curve based traversal is used
   with scheme 2.

-  Compilation parameter: ``-custom-part``, runtime parameter:
   ``+use_custom_partition`` - enables use of user defined
   partitioning. In order to implement a new partitioning scheme, a
   user must link an object exporting a C function with following
   prototype:

   | ``extern "C" void createCustomPartitions(int numparts, int *partitionSize, int *nodeMap);``
   | ``numparts`` (input) - number of partitions to be created.
   | ``partitionSize`` (input) - an array that contains the size of each partition.
   | ``nodeMap`` (output, preallocated) - a preallocated array of length ``CmiNumNodesGlobal()``.
     Entry *i* in this array specifies the new
     global node rank of a node with default node rank *i*. The entries
     in this array are block-wise divided to create partitions, i.e.
     entries 0 to partitionSize[0]-1 belong to partition 1,
     partitionSize[0] to partitionSize[0]+partitionSize[1]-1 to
     partition 2 and so on.
   | When this function is invoked to create partitions, TopoManager is
     configured to view all the allocated nodes as one partition.
     Partition based API is yet to be initialized, and should not be
     used. A link time parameter ``-custom-part`` is required to be
     passed to ``charmc`` for successful compilation.

Redirecting output from individual partitions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Output to standard output (stdout) from various partitions can be
directed to separate files by passing the target path as a command line
option. The run time parameter ``+stdout <path>`` is to be used for this
purpose. The ``<path>`` may contain the C format specifier ``%d``, which
will be replaced by the partition number. In case, ``%d`` is specified
multiple times, only the first three instances from the left will be
replaced by the partition number (other or additional format specifiers
will result in undefined behavior). If a format specifier is not
specified, the partition number will be appended as a suffix to the
specified path. Example usage:

-  ``+stdout out/%d/log`` will write to *out/0/log, out/1/log,
   out/2/log,* :math:`\cdots`.

-  ``+stdout log`` will write to *log.0, log.1, log.2,* :math:`\cdots`.

-  ``+stdout out/%d/log%d`` will write to *out/0/log0, out/1/log1,
   out/2/log2,* :math:`\cdots`.

Inter-partition Communication
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A new API was added to Converse to enable sending messages from one
replica to another. Currently, the following functions are available:

-  **CmiInterSyncSend(local_rank, partition, size, message)**

-  **CmiInterSyncSendAndFree(local_rank, partition, size, message)**

-  **CmiInterSyncNodeSend(local_node, partition, size, message)**

-  **CmiInterSyncNodeSendAndFree(local_node, partition, size, message)**

Users who have coded in Converse will find these functions to be very
similar to basic Converse functions for send - ``CmiSyncSend`` and
``CmiSyncSendAndFree``. Given the local rank of a PE and the partition it
belongs to, these two functions will pass the message to the machine
layer. ``CmiInterSyncSend`` does not return until ``message`` is ready for
reuse. ``CmiInterSyncSendAndFree`` passes the ownership of ``message`` to
the Charm++ RTS, which will free the message when the send is complete. Each
converse message contains a message header, which makes those messages
active - they contain information about their handlers. These handlers
can be registered using existing API in Charm++ - ``CmiRegisterHandler``.
``CmiInterNodeSend`` and ``CmiInterNodeSendAndFree`` are counterparts to these
functions that allow sending of a message to a node (in SMP mode).
