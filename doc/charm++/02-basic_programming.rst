Basic Charm++ Programming
=========================

Program Structure, Compilation and Utilities
--------------------------------------------

A Charm++ program is essentially a C++ program where some components
describe its parallel structure. Sequential code can be written using
any programming technologies that cooperate with the C++ toolchain. This
includes C and Fortran. Parallel entities in the user’s code are written
in C++. These entities interact with the Charm++ framework via inherited
classes and function calls.

Charm++ Interface (.ci) Files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All user program components that comprise its parallel interface (such
as messages, chares, entry methods, etc.) are granted this elevated
status by declaring them in separate *charm++ interface* description
files. These files have a *.ci* suffix and adopt a C++-like declaration
syntax with several additional keywords. In some declaration contexts,
they may also contain some sequential C++ source code. Charm++ parses
these interface descriptions and generates C++ code (base classes, utility
classes, wrapper functions etc.) that facilitates the interaction of the
user program’s entities with the framework. A program may have several
interface description files.

Syntax Highlighting of .ci Files
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Vim
'''

To enable syntax highlighting of .ci files in Vim, do the following:

.. code-block:: bash

   $ cp charm/contrib/ci.vim ~/.vim/syntax/.
   $ vim ~/.vim/filetype.vim

And paste the following line in that file:

.. code-block:: vim

   au! BufRead,BufNewFile *.ci set filetype=ci

Sublime Text
''''''''''''

Syntax highlighting in Sublime Text (version 3 or newer) can be enabled
by installing the *Charmci* package through Package Control.

Emacs
'''''

Syntax highlighting in Emacs can be enabled by triggering C++ handling on the .ci file extension by adding the following line to your .emacs file.

.. code-block:: emacs

   (add-to-list 'auto-mode-alist '("\\.ci\\'" . c++-mode))

Modules
~~~~~~~

The top-level construct in a *ci* file is a named container for
interface declarations called a *module*. Modules allow related
declarations to be grouped together, and cause generated code for these
declarations to be grouped into files named after the module. Modules
cannot be nested, but each *ci* file can have several modules. Modules
are specified using the keyword *module*. A module name must be a valid
C++ identifier.

.. code-block:: c++

   module myFirstModule {
       // Parallel interface declarations go here
       ...
   };

Generated Files
~~~~~~~~~~~~~~~

Each module present in a *ci* file is parsed to generate two files. The
basename of these files is the same as the name of the module and their
suffixes are *.decl.h* and *.def.h*. For e.g., the module defined
earlier will produce the files “myFirstModule.decl.h” and
“myFirstModule.def.h”. As the suffixes indicate, they contain the
declarations and definitions respectively, of all the classes and
functions that are generated based on the parallel interface
description.

We recommend that the header file containing the declarations (decl.h)
be included at the top of the files that contain the declarations or
definitions of the user program entities mentioned in the corresponding
module. The def.h is not actually a header file because it contains
definitions for the generated entities. To avoid multiple definition
errors, it should be compiled into just one object file. A convention we
find useful is to place the def.h file at the bottom of the source file
(.C, .cpp, .cc etc.) which includes the definitions of the corresponding
user program entities.

It should be noted that the generated files have no dependence on the
name of the *ci* file, but only on the names of the modules. This can
make automated dependency-based build systems slightly more complicated.

Module Dependencies
~~~~~~~~~~~~~~~~~~~

A module may depend on the parallel entities declared in another module.
It can express this dependency using the *extern* keyword. *extern* ed
modules do not have to be present in the same *ci* file.

.. code-block:: c++

   module mySecondModule {

       // Entities in this module depend on those declared in another module
       extern module myFirstModule;

       // More parallel interface declarations
       ...
   };

The *extern* keyword places an include statement for the decl.h file of
the *extern* ed module in the generated code of the current module. Hence,
decl.h files generated from *extern* ed modules are required during the
compilation of the source code for the current module. This is usually
required anyway because of the dependencies between user program
entities across the two modules.

The Main Module and Reachable Modules
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Charm++ software can contain several module definitions from several
independently developed libraries / components. However, the user
program must specify exactly one module as containing the starting point
of the program’s execution. This module is called the *mainmodule*. Every
Charm++ program has to contain precisely one *mainmodule*.

All modules that are “reachable” from the *mainmodule* via a chain of
*extern* ed module dependencies are included in a Charm++ program. More
precisely, during program execution, the Charm++ runtime system will
recognize only the user program entities that are declared in reachable
modules. The decl.h and def.h files may be generated for other modules,
but the runtime system is not aware of entities declared in such
unreachable modules.

.. code-block:: c++

   module A {
       ...
   };

   module B {
       extern module A;
       ...
   };

   module C {
       extern module A;
       ...
   };

   module D {
       extern module B;
       ...
   };

   module E {
       ...
   };

   mainmodule M {
       extern module C;
       extern module D;
       // Only modules A, B, C and D are reachable and known to the runtime system
       // Module E is unreachable via any chain of externed modules
       ...
   };

Including other headers
~~~~~~~~~~~~~~~~~~~~~~~

There can be occasions where code generated from the module definitions
requires other declarations / definitions in the user program’s
sequential code. Usually, this can be achieved by placing such user code
before the point of inclusion of the decl.h file. However, this can
become laborious if the decl.h file has to included in several places.
Charm++ supports the keyword *include* in *ci* files to permit the
inclusion of any header directly into the generated decl.h files.

.. code-block:: c++

   module A {
       include "myUtilityClass.h"; //< Note the semicolon
       // Interface declarations that depend on myUtilityClass
       ...
   };

   module B {
       include "someUserTypedefs.h";
       // Interface declarations that require user typedefs
       ...
   };

   module C {
       extern module A;
       extern module B;
       // The user includes will be indirectly visible here too
       ...
   };

The main() function
~~~~~~~~~~~~~~~~~~~

The Charm++ framework implements its own main function and
retains control until the parallel execution environment is initialized
and ready for executing user code. Hence, the user program must not
define a *main()* function. Control enters the user code via the
*mainchare* of the *mainmodule*. This will be discussed in further detail
in :numref:`mainchare`.

Using the facilities described thus far, the parallel interface
declarations for a Charm++ program can be spread across multiple ci
files and multiple modules, permitting good control over the grouping
and export of parallel API. This aids the encapsulation of parallel
software.

Compiling Charm++ Programs
~~~~~~~~~~~~~~~~~~~~~~~~~~

Charm++ provides a compiler-wrapper called *charmc* that handles all *ci*,
C, C++ and Fortran source files that are part of a user program. Users can
invoke charmc to parse their interface descriptions, compile source code
and link objects into binaries. It also links against the appropriate
set of charm framework objects and libraries while producing a binary.
*charmc* and its functionality is described in :numref:`sec:compile`.

.. _basic utility fns:

Utility Functions
~~~~~~~~~~~~~~~~~

The following calls provide basic rank information and utilities useful
when running a Charm++ program.

``void CkAssert(int expression)``
Aborts the program if expression is 0.

``void CkAbort(const char \*message)``
Causes the program to abort, printing
the given error message. This function never returns.

``void CkExit()``
This call informs the Charm RTS that computation on all
processors should terminate. This routine never returns, so any code
after the call to CkExit() inside the function that calls it will not
execute. Other processors will continue executing until they receive
notification to stop, so it is a good idea to ensure through
synchronization that all useful work has finished before calling
CkExit().

``double CkWallTimer()``
Returns the elapsed wall time since the start of execution.

Information about Logical Machine Entities
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

As described in section :numref:`machineModel`, Charm++ recognizes
two logical machine entities: “node” and PE (processing element). The
following functions provide basic information about such logical machine
that a Charm++ program runs on. PE and “node” are numbered starting from
zero.

``int CkNumPes()``
Returns the total number of PEs across all nodes.

``int CkMyPe()``
Returns the index of the PE on which the call was made.

``int CkNumNodes()``
Returns the total number of logical Charm++ nodes.

``int CkMyNode()``
Returns the index of the “node” on which the call was
made.

``int CkMyRank()``
Returns the rank number of the PE on a “node” on which
the call was made. PEs within a “node” are also ranked starting from
zero.

``int CkNodeFirst(int nd)``
Returns the index of the first PE on the logical
node :math:`nd`.

``int CkNodeSize(int nd)``
Returns the number of PEs on the logical node
:math:`nd` on which the call was made.

``int CkNodeOf(int pe)``
Returns the “node” number that PE :math:`pe`
belongs to.

``int CkRankOf(int pe)``
Returns the rank of the given PE within its node.

Terminal I/O
^^^^^^^^^^^^

Charm++ provides both C and C++ style methods of doing terminal I/O.

In place of C-style printf and scanf, Charm++ provides CkPrintf and
CkScanf. These functions have interfaces that are identical to their C
counterparts, but there are some differences in their behavior that
should be mentioned.

Charm++ also supports all forms of printf, cout, etc. in addition to the
special forms shown below. The special forms below are still useful,
however, since they obey well-defined (but still lax) ordering
requirements.

``int CkPrintf(format [, arg]*)``
This call is used for atomic terminal
output. Its usage is similar to ``printf`` in C. However, CkPrintf has
some special properties that make it more suited for parallel
programming. CkPrintf routes all terminal output to a single end point
which prints the output. This guarantees that the output for a single
call to CkPrintf will be printed completely without being interleaved
with other calls to CkPrintf. Note that CkPrintf is implemented using an
asynchronous send, meaning that the call to CkPrintf returns immediately
after the message has been sent, and most likely before the message has
actually been received, processed, and displayed. As such, there is no
guarantee of order in which the output for concurrent calls to CkPrintf
is printed. Imposing such an order requires proper synchronization
between the calls to CkPrintf in the parallel application.

``void CkError(format [, arg]*))``
Like CkPrintf, but used to print error messages on stderr.

``int CkScanf(format [, arg]*)``
This call is used for atomic terminal input. Its usage is similar to scanf in C. A call to CkScanf, unlike CkPrintf, blocks all execution on the processor it is called from, and returns only after all input has been retrieved.

For C++ style stream-based I/O, Charm++ offers ``ckout`` and ``ckerr`` in place of
``cout`` and ``cerr``. The C++ streams and their Charm++ equivalents are related
in the same manner as printf and scanf are to ``CkPrintf`` and ``CkScanf``. The
Charm++ streams are all used through the same interface as the
C++ streams, and all behave in a slightly different way, just like C-style
I/O.

Basic Syntax
------------

.. _entry:

Entry Methods
~~~~~~~~~~~~~

Member functions in the user program which function as entry methods
have to be defined in public scope within the class definition. Entry
methods typically do not return data and have a “void” return type. An
entry method with the same name as its enclosing class is a constructor
entry method and is used to create or spawn chare objects during
execution. Class member functions are annotated as entry methods by
declaring them in the interface file as:

.. code-block:: c++

   entry void Entry1(parameters);

Parameters is either a list of serializable parameters, (e.g., “int i,
double x”), or a message type (e.g., “MyMessage \*msg”). Since
parameters get marshalled into a message before being sent across the
network, in this manual we use “message” to mean either a message type
or a set of marshalled parameters.

Messages are lower level, more efficient, more flexible to use than
parameter marshalling.

For example, a chare could have this entry method declaration in the
interface (``.ci``) file:

.. code-block:: c++

     entry void foo(int i,int k);

Then invoking foo(2,3) on the chare proxy will eventually invoke
foo(2,3) on the chare object.

Since Charm++ runs on distributed memory machines, we cannot pass an
array via a pointer in the usual C++ way. Instead, we must specify the
length of the array in the interface file, as:

.. code-block:: c++

     entry void bar(int n,double arr[n]);

Since C++ does not recognize this syntax, the array data must be passed to
the chare proxy as a simple pointer. The array data will be copied and
sent to the destination processor, where the chare will receive the copy
via a simple pointer again. The remote copy of the data will be kept
until the remote method returns, when it will be freed. This means any
modifications made locally after the call will not be seen by the remote
chare; and the remote chare’s modifications will be lost after the
remote method returns- Charm++ always uses call-by-value, even for
arrays and structures.

This also means the data must be copied on the sending side, and to be
kept must be copied again at the receive side. Especially for large
arrays, this is less efficient than messages, as described in the next
section.

Array parameters and other parameters can be combined in arbitrary ways,
as:

.. code-block:: c++

     entry void doLine(float data[n],int n);
     entry void doPlane(float data[n*n],int n);
     entry void doSpace(int n,int m,int o,float data[n*m*o]);
     entry void doGeneral(int nd,int dims[nd],float data[product(dims,nd)]);

The array length expression between the square brackets can be any valid
C++ expression, including a fixed constant, and may depend in any manner
on any of the passed parameters or even on global functions or global
data. The array length expression is evaluated exactly once per
invocation, on the sending side only. Thus executing the doGeneral
method above will invoke the (user-defined) product function exactly
once on the sending processor.

Marshalling User-Defined Structures and Classes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The marshalling system uses the pup framework to copy data, meaning
every user class that is marshalled needs either a pup routine, a
“PUPbytes” declaration, or a working operator|. See the PUP description
in Section :numref:`sec:pup` for more details on these
routines.

Any user-defined types in the argument list must be declared before
including the “.decl.h” file. Any user-defined types must be fully
defined before the entry method declaration that consumes it. This is
typically done by including the header defining the type in the ``.ci``
file. Alternatively, one may define it before including the ``.decl.h``
file. As usual in C, it is often dramatically more efficient to pass a
large structure by reference than by value.

As an example, refer to the following code from
``examples/charm++/PUP/HeapPUP``:

.. code-block:: c++

   // In HeapObject.h:

   class HeapObject {
    public:
     int publicInt;

     // ... other methods ...

     void pup(PUP::er &p) {
       // remember to pup your superclass if there is one
       p|publicInt;
       p|privateBool;
       if (p.isUnpacking())
         data = new float[publicInt];
       PUParray(p, data, publicInt);
     }

    private:
     bool privateBool;
     float *data;
   };

   // In SimplePup.ci:

   mainmodule SimplePUP {
     include "HeapObject.h";

     // ... other Chare declarations ...

     array [1D] SimpleArray{
       entry SimpleArray();
       entry void acceptData(HeapObject &inData);
     };
   };

   // In SimplePup.h:

   #include "SimplePUP.decl.h"

   // ... other definitions ...

   class SimpleArray : public CBase_SimpleArray {
    public:
     void acceptData(HeapObject &inData) {
       // ... code using marshalled parameter ...
     }
   };

   // In SimplePup.C:

   #include "SimplePUP.h"

   main::main(CkArgMsg *m)
   {
     // normal object construction
     HeapObject exampleObject(... parameters ...);

     // normal chare array construction
     CProxy_SimpleArray simpleProxy = CProxy_SimpleArray::ckNew(30);

     // pass object to remote method invocation on the chare array
     simpleProxy[29].acceptData(exampleObject);
   }

   #include "SimplePUP.def.h"

Chare Objects
~~~~~~~~~~~~~

Chares are concurrent objects with methods that can be invoked remotely.
These methods are known as entry methods. All chares must have a
constructor that is an entry method, and may have any number of other
entry methods. All chare classes and their entry methods are declared in
the interface (``.ci``) file:

.. code-block:: c++

       chare ChareType
       {
           entry ChareType(parameters1);
           entry void EntryMethodName(parameters2);
       };

Although it is *declared* in an interface file, a chare is a C++ object
and must have a normal C++ *implementation* (definition) in addition. A
chare class ``ChareType`` must inherit from the class
``CBase_ChareType``, which is a special class that is generated by the
Charm++ translator from the interface file. Note that C++ namespace
constructs can be used in the interface file, as demonstrated in
``examples/charm++/namespace``.

To be concrete, the C++ definition of the chare above might have the
following definition in a ``.h`` file:

.. code-block:: c++

      class ChareType : public CBase_ChareType {
          // Data and member functions as in C++
          public:
              ChareType(parameters1);
              void EntryMethodName2(parameters2);
      };

Each chare encapsulates data associated with medium-grained units of
work in a parallel application. Chares can be dynamically created on any
processor; there may be thousands of chares on a processor. The location
of a chare is usually determined by the dynamic load balancing strategy.
However, once a chare commences execution on a processor, it does not
migrate to other processors [3]_. Chares do not have a default “thread
of control”: the entry methods in a chare execute in a message driven
fashion upon the arrival of a message [4]_.

The entry method definition specifies a function that is executed
*without interruption* when a message is received and scheduled for
processing. Only one message per chare is processed at a time. Entry
methods are defined exactly as normal C++ function members, except that
they must have the return value void (except for the constructor entry
method which may not have a return value, and for a *synchronous* entry
method, which is invoked by a *threaded* method in a remote chare). Each
entry method can either take no arguments, take a list of arguments that
the runtime system can automatically pack into a message and send (see
section :numref:`entry`), or take a single argument that is a pointer
to a Charm++ message (see section :numref:`messages`).

A chare’s entry methods can be invoked via *proxies* (see
section :numref:`proxies`). Proxies to a chare of type ``chareType``
have type ``CProxy_chareType``. By inheriting from the CBase parent
class, each chare gets a ``thisProxy`` member variable, which holds a
proxy to itself. This proxy can be sent to other chares, allowing them
to invoke entry methods on this chare.

.. _chare creation:

Chare Creation
^^^^^^^^^^^^^^

Once you have declared and defined a chare class, you will want to
create some chare objects to use. Chares are created by the ``ckNew``
method, which is a static method of the chare’s proxy class:

.. code-block:: c++

      CProxy_chareType::ckNew(parameters, int destPE);

The ``parameters`` correspond to the parameters of the chare’s
constructor. Even if the constructor takes several arguments, all of the
arguments should be passed in order to ``ckNew``. If the constructor
takes no arguments, the parameters are omitted. By default, the new
chare’s location is determined by the runtime system. However, this can
be overridden by passing a value for ``destPE``, which specifies the PE
where the chare will be created.

The chare creation method deposits the *seed* for a chare in a pool of
seeds and returns immediately. The chare will be created later on some
processor, as determined by the dynamic load balancing strategy (or by
``destPE``). When a chare is created, it is initialized by calling its
constructor entry method with the parameters specified by ``ckNew``.

Suppose we have declared a chare class ``C`` with a constructor that
takes two arguments, an ``int`` and a ``double``.

#. This will create a new chare of type C on any processor and return a
   proxy to that chare:

   .. code-block:: c++

         CProxy_C chareProxy = CProxy_C::ckNew(1, 10.0);

#. This will create a new chare of type C on processor destPE and return
   a proxy to that chare:

   .. code-block:: c++

         CProxy_C chareProxy = CProxy_C::ckNew(1, 10.0, destPE);

For an example of chare creation in a full application, see
``examples/charm++/fib`` in the Charm++ software distribution, which
calculates Fibonacci numbers in parallel.

Method Invocation on Chares
^^^^^^^^^^^^^^^^^^^^^^^^^^^

A message may be sent to a chare through a proxy object using the
notation:

.. code-block:: c++

       chareProxy.EntryMethod(parameters)

This invokes the entry method EntryMethod on the chare referred to by
the proxy chareProxy. This call is asynchronous and non-blocking; it
returns immediately after sending the message.

Local Access
^^^^^^^^^^^^

You can get direct access to a local chare using the proxy’s ckLocal
method, which returns an ordinary C++ pointer to the chare if it exists on
the local processor, and NULL otherwise.

.. code-block:: c++

       C *c=chareProxy.ckLocal();
       if (c==NULL) {
           // object is remote; send message
       } else {
           // object is local; directly use members and methods of c
       }

.. _readonly:

Read-only Data
~~~~~~~~~~~~~~

Since Charm++ does not allow global variables, it provides a special
mechanism for sharing data amongst all objects. *Read-only* variables of
simple data types or compound data types including messages and arrays
are used to share information that is obtained only after the program
begins execution and does not change after they are initialized in the
dynamic scope of the ``main`` function of the mainchare. They are
broadcast to every Charm++ Node (process) by the Charm++ runtime, and
can be accessed in the same way as C++ “global” variables on any process.
Compound data structures containing pointers can be made available as
read-only variables using read-only messages(see
section :numref:`messages`) or read-only arrays(see
section :numref:`basic arrays`). Note that memory has to be
allocated for read-only messages by using new to create the message in
the ``main`` function of the mainchare.

Read-only variables are declared by using the type modifier readonly,
which is similar to const in C++. Read-only data is specified in the
``.ci`` file (the interface file) as:

.. code-block:: c++

    readonly Type ReadonlyVarName;

The variable ReadonlyVarName is declared to be a read-only variable of
type Type. Type must be a single token and not a type expression.

.. code-block:: c++

    readonly message MessageType *ReadonlyMsgName;

The variable ReadonlyMsgName is declared to be a read-only message of
type MessageType. Pointers are not allowed to be readonly variables
unless they are pointers to message types. In this case, the message
will be initialized on every PE.

.. code-block:: c++

    readonly Type ReadonlyArrayName [arraysize];

The variable ReadonlyArrayName is declared to be a read-only array of
type Type with arraysize elements. Type must be a single token and not a
type expression. The value of arraysize must be known at compile time.

Read-only variables must be declared either as global or as public class
static data in the C/C++ implementation files, and these declarations have
the usual form:

.. code-block:: c++

    Type ReadonlyVarName;
    MessageType *ReadonlyMsgName;
    Type ReadonlyArrayName [arraysize];

Similar declarations preceded by extern would appear in the ``.h`` file.

*Note:* The current Charm++ translator cannot prevent assignments to
read-only variables. The user must make sure that no assignments occur
in the program outside of the mainchare constructor.

For concrete examples for using read-only variables, please refer to
examples such as ``examples/charm++/array`` and
``examples/charm++/gaussSeidel3D``.

Users can get the same functionality of readonly variables by making
such variables members of Charm++ Node Group objects and constructing
the Node Group in the mainchare’s main routine.

.. _basic arrays:

Chare Arrays
------------

Chare arrays are arbitrarily-sized, possibly-sparse collections of
chares that are distributed across the processors. The entire array has
a globally unique identifier of type CkArrayID, and each element has a
unique index of type CkArrayIndex. A CkArrayIndex can be a single
integer (i.e. a one-dimensional array), several integers (i.e. a
multi-dimensional array), or an arbitrary string of bytes (e.g. a binary
tree index).

Array elements can be dynamically created and destroyed on any PE,
migrated between PEs, and messages for the elements will still arrive
properly. Array elements can be migrated at any time, allowing arrays to
be efficiently load balanced. A chare array (or a subset of array
elements) can receive a broadcast/multicast or contribute to a
reduction.

An example program can be found here: ``examples/charm++/array``.

Declaring a One-dimensional Array
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can declare a one-dimensional (1D) chare array as:

.. code-block:: c++

   //In the .ci file:
   array [1D] A {
     entry A(parameters1);
     entry void someEntry(parameters2);
   };

Array elements extend the system class CBase_ClassName, inheriting
several fields:

-  thisProxy: the proxy to the entire chare array that can be indexed to
   obtain a proxy to a specific array element (e.g. for a 1D chare array
   thisProxy[10]; for a 2D chare array thisProxy(10, 20))

-  thisArrayID: the array’s globally unique identifier

-  thisIndex: the element’s array index (an array element can obtain a
   proxy to itself like this thisProxy[thisIndex])

.. code-block:: c++

   class A : public CBase_A {
     public:
       A(parameters1);

       void someEntry(parameters2);
   };

Note that A must have a *migration constructor*, which is typically
empty:

.. code-block:: c++

   //In the .C file:
   A::A(void)
   {
     //... constructor code ...
   }

   A::someEntry(parameters2)
   {
     // ... code for someEntry ...
   }

See the section :numref:`arraymigratable` on migratable array
elements for more information on the migration constructor that takes
CkMigrateMessage \* as the argument.

Declaring Multi-dimensional Arrays
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Charm++ supports multi-dimensional or user-defined indices. These array
types can be declared as:

.. code-block:: c++

   //In the .ci file:
   array [1D]  ArrayA { entry ArrayA(); entry void e(parameters);}
   array [2D]  ArrayB { entry ArrayB(); entry void e(parameters);}
   array [3D]  ArrayC { entry ArrayC(); entry void e(parameters);}
   array [4D]  ArrayD { entry ArrayD(); entry void e(parameters);}
   array [5D]  ArrayE { entry ArrayE(); entry void e(parameters);}
   array [6D]  ArrayF { entry ArrayF(); entry void e(parameters);}
   array [Foo] ArrayG { entry ArrayG(); entry void e(parameters);}
   array [Bar<3>] ArrayH { entry ArrayH(); entry void e(parameters);}

The declaration of ArrayG expects an array index of type
CkArrayIndexFoo, which must be defined before including the ``.decl.h``
file (see section :numref:`user-defined array index type` on
user-defined array indices for more information).

.. code-block:: c++

   //In the .h file:
   class ArrayA : public CBase_ArrayA { public: ArrayA(){} ...};
   class ArrayB : public CBase_ArrayB { public: ArrayB(){} ...};
   class ArrayC : public CBase_ArrayC { public: ArrayC(){} ...};
   class ArrayD : public CBase_ArrayD { public: ArrayD(){} ...};
   class ArrayE : public CBase_ArrayE { public: ArrayE(){} ...};
   class ArrayF : public CBase_ArrayF { public: ArrayF(){} ...};
   class ArrayG : public CBase_ArrayG { public: ArrayG(){} ...};
   class ArrayH : public CBase_ArrayH { public: ArrayH(){} ...};

The fields in thisIndex are different depending on the dimensionality of
the chare array:

-  1D array: thisIndex

-  2D array (:math:`x`,\ :math:`y`): thisIndex.x, thisIndex.y

-  3D array (:math:`x`,\ :math:`y`,\ :math:`z`): thisIndex.x,
   thisIndex.y, thisIndex.z

-  4D array (:math:`w`,\ :math:`x`,\ :math:`y`,\ :math:`z`):
   thisIndex.w, thisIndex.x, thisIndex.y, thisIndex.z

-  5D array (:math:`v`,\ :math:`w`,\ :math:`x`,\ :math:`y`,\ :math:`z`):
   thisIndex.v, thisIndex.w, thisIndex.x, thisIndex.y, thisIndex.z

-  6D array
   (:math:`x_1`,\ :math:`y_1`,\ :math:`z_1`,\ :math:`x_2`,\ :math:`y_2`,\ :math:`z_2`):
   thisIndex.x1, thisIndex.y1, thisIndex.z1, thisIndex.x2, thisIndex.y2,
   thisIndex.z2

-  Foo array: thisIndex

-  Bar<3> array: thisIndex

.. _basic array creation:

Creating an Array
~~~~~~~~~~~~~~~~~

An array is created using the CProxy_Array::ckNew routine, which must be
called from PE 0. To create an array from any PE, asynchronous array
creation using a callback can be used. See
section :numref:`asynchronous_array_creation` for asynchronous
array creation. CProxy_Array::ckNew returns a proxy object, which can be
kept, copied, or sent in messages. The following creates a 1D array
containing elements indexed (0, 1, …, dimX-1):

.. code-block:: c++

   CProxy_ArrayA a1 = CProxy_ArrayA::ckNew(params, dimX);

Likewise, a dense multidimensional array can be created by passing the
extents at creation time to ckNew.

.. code-block:: c++

   CProxy_ArrayB a2 = CProxy_ArrayB::ckNew(params, dimX, dimY);
   CProxy_ArrayC a3 = CProxy_ArrayC::ckNew(params, dimX, dimY, dimZ);
   CProxy_ArrayD a4 = CProxy_ArrayC::ckNew(params, dimW, dimX, dimY, dimZ);
   CProxy_ArrayE a5 = CProxy_ArrayC::ckNew(params, dimV, dimW, dimX, dimY, dimZ);
   CProxy_ArrayF a6 = CProxy_ArrayC::ckNew(params, dimX1, dimY1, dimZ1, dimX2, dimY2, dimZ2);

For user-defined arrays, this functionality cannot be used. The array
elements must be inserted individually as described in
section :numref:`dynamic_insertion`.

During creation, the constructor is invoked on each array element. For
more options when creating the array, see
section :numref:`advanced array create`.

Entry Method Invocation
~~~~~~~~~~~~~~~~~~~~~~~

To obtain a proxy to a specific element in chare array, the chare array
proxy (e.g. thisProxy) must be indexed by the appropriate index call
depending on the dimensionality of the array:

-  1D array, to obtain a proxy to element :math:`i`:
   thisProxy[:math:`i`] or thisProxy(\ :math:`i`)

-  2D array, to obtain a proxy to element :math:`(i,j)`:
   thisProxy(\ :math:`i`,\ :math:`j`)

-  3D array, to obtain a proxy to element :math:`(i,j,k)`:
   thisProxy(\ :math:`i`,\ :math:`j`,\ :math:`k`)

-  4D array, to obtain a proxy to element :math:`(i,j,k,l)`:
   thisProxy(\ :math:`i`,\ :math:`j`,\ :math:`k`,\ :math:`l`)

-  5D array, to obtain a proxy to element :math:`(i,j,k,l,m)`:
   thisProxy(\ :math:`i`,\ :math:`j`,\ :math:`k`,\ :math:`l`,\ :math:`m`)

-  6D array, to obtain a proxy to element :math:`(i,j,k,l,m,n)`:
   thisProxy(\ :math:`i`,\ :math:`j`,\ :math:`k`,\ :math:`l`,\ :math:`m`,\ :math:`n`)

-  User-defined array, to obtain a proxy to element :math:`i`:
   thisProxy[:math:`i`] or thisProxy(\ :math:`i`)

To send a message to an array element, index the proxy and call the
method name:

.. code-block:: c++

   a1[i].doSomething(parameters);
   a3(x,y,z).doAnother(parameters);
   aF[CkArrayIndexFoo(...)].doAgain(parameters);

You may invoke methods on array elements that have not yet been created.
The Charm++ runtime system will buffer the message until the element is
created.  [5]_

Messages are not guaranteed to be delivered in order. For instance, if a
method is invoked on method A and then method B; it is possible that B
is executed before A.

.. code-block:: c++

   a1[i].A();
   a1[i].B();

Messages sent to migrating elements will be delivered after the
migrating element arrives on the destination PE. It is an error to send
a message to a deleted array element.

Broadcasts on Chare Arrays
~~~~~~~~~~~~~~~~~~~~~~~~~~

To broadcast a message to all the current elements of an array, simply
omit the index (invoke an entry method on the chare array proxy):

.. code-block:: c++

   a1.doIt(parameters); //<- invokes doIt on each array element

The broadcast message will be delivered to every existing array element
exactly once. Broadcasts work properly even with ongoing migrations,
insertions, and deletions.

.. _reductions:

Reductions on Chare Arrays
~~~~~~~~~~~~~~~~~~~~~~~~~~

A reduction applies a single operation (e.g. add, max, min, ...) to data
items scattered across many processors and collects the result in one
place. Charm++ supports reductions over the members of an array or
group.

The data to be reduced comes from a call to the member contribute
method:

.. code-block:: c++

   void contribute(int nBytes, const void *data, CkReduction::reducerType type);

This call contributes nBytes bytes starting at data to the reduction
type (see Section :numref:`builtin_reduction`). Unlike sending a
message, you may use data after the call to contribute. All members of
the chare array or group must call contribute, and all of them must use
the same reduction type.

For example, if we want to sum each array/group member’s single integer
myInt, we would use:

.. code-block:: c++

       // Inside any member method
       int myInt=get_myInt();
       contribute(sizeof(int),&myInt,CkReduction::sum_int);

The built-in reduction types (see below) can also handle arrays of
numbers. For example, if each element of a chare array has a pair of
doubles forces[2], the corresponding elements of which are to be added
across all elements, from each element call:

.. code-block:: c++

       double forces[2]=get_my_forces();
       contribute(2*sizeof(double),forces,CkReduction::sum_double);

Note that since C++ arrays (like forces[2]) are already pointers, we
don’t use &forces.

A slightly simpler interface is available for ``std::vector<T>``, since
the class determines the size and count of the underlying type:

.. code-block:: c++

       CkCallback cb(...);
       vector<double> forces(2);
       get_my_forces(forces);
       contribute(forces, CkReduction::sum_double, cb);

Either of these will result in a ``double`` array of 2 elements, the
first of which contains the sum of all forces[0] values, with the second
element holding the sum of all forces[1] values of the chare array
elements.

Typically the client entry method of a reduction takes a single argument
of type CkReductionMsg (see Section :numref:`reductionClients`).
However, by giving an entry method the reductiontarget attribute in the
``.ci`` file, you can instead use entry methods that take arguments of
the same type as specified by the *contribute* call. When creating a
callback to the reduction target, the entry method index is generated by
``CkReductionTarget(ChareClass, method_name)`` instead of
``CkIndex_ChareClass::method_name(...)``. For example, the code for a
typed reduction that yields an ``int``, would look like this:

.. code-block:: c++

     // In the .ci file...
     entry [reductiontarget] void done(int result);

     // In some .C file:
     // Create a callback that invokes the typed reduction client
     // driverProxy is a proxy to the chare object on which
     // the reduction target method "done" is called upon completion
     // of the reduction
     CkCallback cb(CkReductionTarget(Driver, done), driverProxy);

     // Contribution to the reduction...
     contribute(sizeof(int), &intData, CkReduction::sum_int, cb);

     // Definition of the reduction client...
     void Driver::done(int result)
     {
       CkPrintf("Reduction value: %d", result);
     }

This will also work for arrays of data
elements(\ ``entry [reductiontarget] void done(int n, int result[n])``),
and for any user-defined type with a PUP method (see
:numref:`sec:pup`). If you know that the reduction will yield a
particular number of elements, say 3 ``int``\ s, you can also specify a
reduction target which takes 3 ``int``\ s and it will be invoked
correctly.

Reductions do not have to specify commutative-associative operations on
data; they can also be used to signal the fact that all array/group
members have reached a certain synchronization point. In this case, a
simpler version of contribute may be used:

.. code-block:: c++

       contribute();

In all cases, the result of the reduction operation is passed to the
*reduction client*. Many different kinds of reduction clients can be
used, as explained in Section :numref:`reductionClients`.

Please refer to ``examples/charm++/reductions/typed_reduction`` for a
working example of reductions in Charm++.

Note that the reduction will complete properly even if chare array
elements are *migrated* or *deleted* during the reduction. Additionally,
when you create a new chare array element, it is expected to contribute
to the next reduction not already in progress on that processor.

.. _builtin_reduction:

Built-in Reduction Types
^^^^^^^^^^^^^^^^^^^^^^^^

Charm++ includes several built-in reduction types, used to combine
individual contributions. Any of them may be passed as an argument of
type CkReduction::reducerType to contribute.

The first four operations (``sum``, ``product``, ``max``, and ``min``)
work on ``char``, ``short``, ``int``, ``long``, ``long long``,
``float``, or ``double`` data as indicated by the suffix. The logical
reductions (``and``, ``or``) only work on bool and integer data. All the
built-in reductions work on either single numbers (pass a pointer) or
arrays- just pass the correct number of bytes to contribute.

#. CkReduction::nop : no operation performed.

#. CkReduction::sum_char, sum_short, sum_int, sum_long, sum_long_long,
   sum_uchar, sum_ushort, sum_uint, sum_ulong, sum_ulong_long,
   sum_float, sum_double : the result will be the sum of the given
   numbers.

#. CkReduction::product_char, product_short, product_int, product_long,
   product_long_long, product_uchar, product_ushort, product_uint,
   product_ulong, product_ulong_long, product_float, product_double :
   the result will be the product of the given numbers.

#. CkReduction::max_char, max_short, max_int, max_long, max_long_long,
   max_uchar, max_ushort, max_uint, max_ulong, max_ulong_long,
   max_float, max_double : the result will be the largest of the given
   numbers.

#. CkReduction::min_char, min_short, min_int, min_long, min_long_long,
   min_uchar, min_ushort, min_uint, min_ulong, min_ulong_long,
   min_float, min_double : the result will be the smallest of the given
   numbers.

#. CkReduction::logical_and_bool, logical_and_int : the result will be
   the logical AND of the given values.

#. CkReduction::logical_or_bool, logical_or_int : the result will be the
   logical OR of the given values.

#. CkReduction::logical_xor_bool, logical_xor_int : the result will be
   the logical XOR of the given values.

#. CkReduction::bitvec_and_bool, bitvec_and_int : the result will be the
   bitvector AND of the given values.

#. CkReduction::bitvec_or_bool, bitvec_or_int : the result will be the
   bitvector OR of the given values.

#. CkReduction::bitvec_xor_bool, bitvec_xor_int : the result will be the
   bitvector XOR of the given values.

#. CkReduction::set : the result will be a verbatim concatenation of all
   the contributed data, separated into CkReduction::setElement records.
   The data contributed can be of any length, and can vary across array
   elements or reductions. To extract the data from each element, see
   the description below.

#. CkReduction::concat : the result will be a byte-by-byte concatenation
   of all the contributed data. The contributed elements are not
   delimiter-separated.

#. CkReduction::random : the result will be a single randomly selected
   value of all of the contributed values.

#. CkReduction::statistics : returns a CkReduction::statisticsElement
   struct, containing summary statistics of the contributed data.
   Specifically, the struct contains the following fields: int count,
   double mean, and double m2, and the following functions: double
   variance() and double stddev().

CkReduction::set returns a collection of CkReduction::setElement
objects, one per contribution. This class has the definition:

.. code-block:: c++

   class CkReduction::setElement
   {
   public:
     int dataSize; //The length of the `data' array in bytes.
     char data[1]; //A place holder that marks the start of the data array.
     CkReduction::setElement *next(void);
   };

Example: Suppose you would like to contribute 3 integers from each array
element. In the reduction method you would do the following:

.. code-block:: c++

   void ArrayClass::methodName (CkCallback &cb)
   {
     int result[3];
     result[0] = 1;            // Copy the desired values into the result.
     result[1] = 2;
     result[2] = 3;
     // Contribute the result to the reductiontarget cb.
     contribute(3*sizeof(int), result, CkReduction::set, cb);
   }

Inside the reduction’s target method, the contributions can be accessed
by using the ``CkReduction::setElement->next()`` iterator.

.. code-block:: c++

   void SomeClass::reductionTargetMethod (CkReductionMsg *msg)
   {
     // Get the initial element in the set.
     CkReduction::setElement *current = (CkReduction::setElement*) msg->getData();
     while(current != NULL) // Loop over elements in set.
     {
       // Get the pointer to the packed int's.
       int *result = (int*) &current->data;
       // Do something with result.
       current = current->next(); // Iterate.
     }
   }

The reduction set order is undefined. You should add a source field to
the contributed elements if you need to know which array element gave a
particular contribution. Additionally, if the contributed elements are
of a complex data type, you will likely have to supply code for
serializing/deserializing them. Consider using the PUP interface
(§ :numref:`sec:pup`) to simplify your object serialization
needs.

If the outcome of your reduction is dependent on the order in which data
elements are processed, or if your data is just too heterogeneous to be
handled elegantly by the predefined types and you don’t want to
undertake multiple reductions, you can use a tuple reduction or define
your own custom reduction type.

Tuple reductions allow performing multiple different reductions in the
same message. The reductions can be on the same or different data, and
the reducer type for each reduction can be set independently as well.
The contributions that make up a single tuple reduction message are all
reduced in the same order as each other. As an example, a chare array
element can contribute to a gatherv-like operation using a tuple
reduction that consists of two set reductions.

.. code-block:: c++

   int tupleSize = 2;
   CkReduction::tupleElement tupleRedn[] = {
     CkReduction::tupleElement(sizeof(int), &thisIndex, CkReduction::set),
     CkReduction::tupleElement(sizeData, data, CkReduction::set)
   };
   CkReductionMsg* msg = CkReductionMsg::buildFromTuple(tupleRedn, tupleSize);
   CkCallback allgathervCB(CkIndex_Foo::allgathervResult(0), thisProxy);
   msg->setCallback(allgathervCB);
   contribute(msg);

Note that ``CkReduction::tupleElement`` only holds pointers to the data that
will make up the reduction message, therefore any local variables used must
remain in scope until ``CkReductionMsg::buildFromTuple`` completes.

The result of this reduction is a single CkReductionMsg that can be
processed as multiple reductions:

.. code-block:: c++

   void Foo::allgathervResult (CkReductionMsg* msg)
   {
     int numReductions;
     CkReduction::tupleElement* results;

     msg->toTuple(&results, &numReductions);
     CkReduction::setElement* currSrc  = (CkReduction::setElement*)results[0].data;
     CkReduction::setElement* currData = (CkReduction::setElement*)results[1].data;

     // ... process currSrc and currData

     delete [] results;
   }

See the next section (Section :numref:`new_type_reduction`) for details
on custom reduction types.

Destroying Array Elements
~~~~~~~~~~~~~~~~~~~~~~~~~

To destroy an array element - detach it from the array, call its
destructor, and release its memory-invoke its Array destroy method, as:

.. code-block:: c++

   a1[i].ckDestroy();

Note that this method can also be invoked remotely i.e. from a process
different from the one on which the array element resides. You must
ensure that no messages are sent to a deleted element. After destroying
an element, you may insert a new element at its index.

.. _sec:sdag:

Structured Control Flow: Structured Dagger
------------------------------------------

Charm++ is based on the message-driven parallel programming paradigm. In
contrast to many other approaches, Charm++ programmers encode entry
points to their parallel objects, but do not explicitly wait (i.e.
block) on the runtime to indicate completion of posted ‘receive’
requests. Thus, a Charm++ object’s overall flow of control can end up
fragmented across a number of separate methods, obscuring the sequence
in which code is expected to execute. Furthermore, there are often
constraints on when different pieces of code should execute relative to
one another, related to data and synchronization dependencies.

Consider one way of expressing these constraints using flags, buffers,
and counters, as in the following example:

.. code-block:: c++

   // in .ci file
   chare ComputeObject {
     entry void ComputeObject();
     entry void startStep();
     entry void firstInput(Input i);
     entry void secondInput(Input j);
   };

   // in C++ file
   class ComputeObject : public CBase_ComputeObject {
     int   expectedMessageCount;
     Input first, second;

   public:
     ComputeObject() {
       startStep();
     }
     void startStep() {
       expectedMessageCount = 2;
     }

     void firstInput(Input i) {
       first = i;
       if (--expectedMessageCount == 0)
         computeInteractions(first, second);
       }
     void recv_second(Input j) {
       second = j;
       if (--expectedMessageCount == 0)
         computeInteractions(first, second);
     }

     void computeInteractions(Input a, Input b) {
       // do computations using a and b
       ...
       // send off results
       ...
       // reset for next step
       startStep();
     }
   };

In each step, this object expects pairs of messages, and waits to
process the incoming data until it has both of them. This sequencing is
encoded across 4 different functions, which in real code could be much
larger and more numerous, resulting in a spaghetti-code mess.

Instead, it would be preferable to express this flow of control using
structured constructs, such as loops. Charm++ provides such constructs
for structured control flow across an object’s entry methods in a
notation called Structured Dagger. The basic constructs of Structured
Dagger (SDAG) provide for *program-order execution* of the entry methods
and code blocks that they define. These definitions appear in the
``.ci`` file definition of the enclosing chare class as a ‘body’ of an
entry method following its signature.

The most basic construct in SDAG is the ``serial`` (aka the ``atomic``)
block. Serial blocks contain sequential C++ code. They’re also called
atomic because the code within them executes without returning control
to the Charm++ runtime scheduler, and thus avoiding interruption from
incoming messages. The keywords atomic and serial are synonymous, and
you can find example programs that use atomic. However, we recommend the
use of serial and are considering the deprecation of the atomic keyword.
Typically serial blocks hold the code that actually deals with incoming
messages in a ``when`` statement, or to do local operations before a
message is sent or after it’s received. The earlier example can be
adapted to use serial blocks as follows:

.. code-block:: c++

   // in .ci file
   chare ComputeObject {
     entry void ComputeObject();
     entry void startStep();
     entry void firstInput(Input i) {
       serial {
         first = i;
         if (--expectedMessageCount == 0)
           computeInteractions(first, second);
       }
     };
     entry void secondInput(Input j) {
       serial {
         second = j;
         if (--expectedMessageCount == 0)
           computeInteractions(first, second);
       }
     };
   };

   // in C++ file
   class ComputeObject : public CBase_ComputeObject {
     ComputeObject_SDAG_CODE
     int   expectedMessageCount;
     Input first, second;

   public:
     ComputeObject() {
       startStep();
     }
     void startStep() {
       expectedMessageCount = 2;
     }

     void computeInteractions(Input a, Input b) {
       // do computations using a and b
       . . .
       // send off results
       . . .
       // reset for next step
       startStep();
     }
   };

Note that chare classes containing SDAG code must include a few
additional declarations in addition to inheriting from their
``CBase_Foo`` class, by incorporating the ``Foo_SDAG_CODE``
generated-code macro in the class.

Serial blocks can also specify a textual ‘label’ that will appear in
traces, as follows:

.. code-block:: c++

     entry void firstInput(Input i) {
       serial "process first" {
         first = i;
         if (--expectedMessageCount == 0)
           computeInteractions(first, second);
       }
     };

In order to control the sequence in which entry methods are processed,
SDAG provides the ``when`` construct. These statements, also called
triggers, indicate that we expect an incoming message of a particular
type, and provide code to handle that message when it arrives. From the
perspective of a chare object reaching a ``when`` statement, it is
effectively a ‘blocking receive.’

Entry methods defined by a ``when`` are not executed immediately when a
message targeting them is delivered, but instead are held until control
flow in the chare reaches a corresponding ``when`` clause. Conversely,
when control flow reaches a ``when`` clause, the generated code checks
whether a corresponding message has arrived: if one has arrived, it is
processed; otherwise, control is returned to the Charm++ scheduler.

The use of ``when`` substantially simplifies the example from above:

.. code-block:: c++

   // in .ci file
   chare ComputeObject {
     entry void ComputeObject();
     entry void startStep() {
       when firstInput(Input first)
         when secondInput(Input second)
           serial {
             computeInteractions(first, second);
           }
     };
     entry void firstInput(Input i);
     entry void secondInput(Input j);
   };

   // in C++ file
   class ComputeObject : public CBase_ComputeObject {
     ComputeObject_SDAG_CODE

   public:
     ComputeObject() {
       startStep();
     }

     void computeInteractions(Input a, Input b) {
       // do computations using a and b
       . . .
       // send off results
       . . .
       // reset for next step
       startStep();
     }
   };

Like an ``if`` or ``while`` in C code, each ``when`` clause has a body
made up of the statement or block following it. The variables declared
as arguments to the entry method triggering the when are available in
the scope of the body. By using the sequenced execution of SDAG code and
the availability of parameters to when-defined entry methods in their
bodies, the counter ``expectedMessageCount`` and the intermediate copies
of the received input are eliminated. Note that the entry methods
``firstInput`` and ``secondInput`` are still declared in the ``.ci``
file, but their definition is in the SDAG code. The interface translator
generates code to handle buffering and triggering them appropriately.

For simplicity, ``when`` constructs can also specify multiple expected
entry methods that all feed into a single body, by separating their
prototypes with commas:

.. code-block:: c++

   entry void startStep() {
     when firstInput(Input first),
          secondInput(Input second)
       serial {
         computeInteractions(first, second);
       }
   };

A single entry method is allowed to appear in more than one ``when``
statement. If only one of those ``when`` statements has been triggered
when the runtime delivers a message to that entry method, that ``when``
statement is guaranteed to process it. If there is no trigger waiting
for that entry method, then the next corresponding ``when`` to be
reached will receive that message. If there is more than one ``when``
waiting on that method, which one will receive it is not specified, and
should not be relied upon. For an example of multiple ``when``
statements handling the same entry method without reaching the
unspecified case, see the CharmLU benchmark.

To more finely control the correspondence between incoming messages and
``when`` clauses associated with the target entry method, SDAG supports
*matching* on reference numbers. Matching is typically used to denote an
iteration of a program that executes asynchronously (without any sort of
barrier or other synchronization between steps) or a particular piece of
the problem being solved. Matching is requested by placing an expression
denoting the desired reference number in square brackets between the
entry method name and its parameter list. For parameter marshalled entry
methods, the reference number expression will be compared for equality
with the entry method’s first argument. For entry methods that accept an
explicit message (§ :numref:`messages`), the reference number on the
message can be set by calling the function
``CkSetRefNum(void *msg, CMK_REFNUM_TYPE ref)``. Matching is used in the
loop example below, and in
``examples/charm++/jacobi2d-sdag/jacobi2d.ci``. Multiple ``when``
triggers for an entry method with different matching reference numbers
will not conflict - each will receive only corresponding messages.

SDAG supports the ``for`` and ``while`` loop constructs mostly as if
they appeared in plain C or C++ code. In the running example,
``computeInteractions()`` calls ``startStep()`` when it is finished to
start the next step. Instead of this arrangement, the loop structure can
be made explicit:

.. code-block:: c++

   // in .ci file
   chare ComputeObject {
     entry void ComputeObject();
     entry void runForever() {
       while(true) {
         when firstInput(Input first),
              secondInput(Input second) serial {
             computeInteractions(first, second);
         }
       }
     };
     entry void firstInput(Input i);
     entry void secondInput(Input j);
   };

   // in C++ file
   class ComputeObject : public CBase_ComputeObject {
     ComputeObject_SDAG_CODE

   public:
     ComputeObject() {
       runForever();
     }

     void computeInteractions(Input a, Input b) {
       // do computations using a and b
       . . .
       // send off results
       . . .
     }
   };

If this code should instead run for a fixed number of iterations, we can
instead use a for loop:

.. code-block:: c++

   // in .ci file
   chare ComputeObject {
     entry void ComputeObject();
     entry void runForever() {
       for(iter = 0; iter < n; ++iter) {
         // Match to only accept inputs for the current iteration
         when firstInput[iter](int a, Input first),
              secondInput[iter](int b, Input second) serial {
           computeInteractions(first, second);
         }
       }
     };
     entry void firstInput(int a, Input i);
     entry void secondInput(int b, Input j);
   };

   // in C++ file
   class ComputeObject : public CBase_ComputeObject {
     ComputeObject_SDAG_CODE
     int n, iter;

   public:
     ComputeObject() {
       n = 10;
       runForever();
     }

     void computeInteractions(Input a, Input b) {
       // do computations using a and b
       . . .
       // send off results
       . . .
     }
   };

Note that ``int iter;`` is declared in the chare’s class definition and
not in the ``.ci`` file. This is necessary because the Charm++ interface
translator does not fully parse the declarations in the ``for`` loop
header, because of the inherent complexities of C++.

SDAG also supports conditional execution of statements and blocks with
``if`` statements. The syntax of SDAG ``if`` statements matches that of
C and C++. However, if one encounters a syntax error on correct-looking
code in a loop or conditional statement, try assigning the condition
expression to a boolean variable in a serial block preceding the
statement and then testing that boolean’s value. This can be necessary
because of the complexity of parsing C++ code.

In cases where multiple tasks must be processed before execution
continues, but with no dependencies or interactions among them, SDAG
provides the ``overlap`` construct. Overlap blocks contain a series of
SDAG statements within them which can occur in any order. Commonly these
blocks are used to hold a series of ``when`` triggers which can be
received and processed in any order. Flow of control doesn’t leave the
overlap block until all the statements within it have been processed.

In the running example, suppose each input needs to be preprocessed
independently before the call to ``computeInteractions``. Since we don’t
care which order they get processed in, and want it to happen as soon as
possible, we can apply ``overlap``:

.. code-block:: c++

   // in .ci file
   chare ComputeObject {
     entry void ComputeObject();
     entry void startStep() {
       overlap {
         when firstInput(Input i)
           serial { first = preprocess(i); }
         when secondInput(Input j)
           serial { second = preprocess(j); }
        }
        serial {
          computeInteractions(first, second);
        }
     };
     entry void firstInput(Input i);
     entry void secondInput(Input j);
   };

   // in C++ file
   class ComputeObject : public CBase_ComputeObject {
     ComputeObject_SDAG_CODE

   public:
     ComputeObject() {
       startStep();
     }

     void computeInteractions(Input a, Input b) {
       // do computations using a and b
       . . .
       // send off results
       . . .
       // reset for next step
       startStep();
     }
   };

Another construct offered by SDAG is the ``forall`` loop. These loops
are used when the iterations of a loop can be performed independently
and in any order. This is in contrast to a regular ``for`` loop, in
which each iteration is executed sequentially. The loop iterations are
executed entirely on the calling PE, so they do not run in parallel.
However, they are executed concurrently, in that execution of different
iterations can interleave at ``when`` statements, like any other SDAG
code. SDAG statements following the ``forall`` loop will not execute
until all iterations have completed. The ``forall`` loop can be seen as
an ``overlap`` with an indexed set of otherwise identical statements in
the body.

The syntax of ``forall`` is

.. code-block:: c++

   forall [IDENT] (MIN:MAX,STRIDE) BODY

The range from MIN to MAX is inclusive. In each iteration instance of
``BODY``, the ``IDENT`` variable will take on one of the values in the
specified range. The ``IDENT`` variable must be declared in the
application C++ code as a member of the enclosing chare class.

Use of ``forall`` is demonstrated through distributed parallel
matrix-matrix multiply shown in ``examples/charm++/matmul/matmul.ci``

The ``case`` Statement
~~~~~~~~~~~~~~~~~~~~~~

The ``case`` statement in SDAG expresses a disjunction over a set of
``when`` clauses. In other words, if it is known that one dependency out
of a set will be satisfied, but which one is not known, this statement
allows the set to be specified and will execute the corresponding block
based on which dependency ends up being fulfilled.

The following is a basic example of the ``case`` statement. Note that
the trigger ``b(), d()`` will only be fulfilled if both ``b()`` and
``d()`` arrive. If only one arrives, then it will partially match, and
the runtime will not “commit” to this branch until the second arrives.
If another dependency fully matches, the partial match will be ignored
and can be used to trigger another ``when`` later in the execution.

.. code-block:: c++

   case {
     when a() { }
     when b(), d() { }
     when c() { }
   }

A full example of the ``case`` statement can be found
``tests/charm++/sdag/case/caseTest.ci``.

Usage Notes
~~~~~~~~~~~

SDAG Code Declaration
^^^^^^^^^^^^^^^^^^^^^

If you’ve added *Structured Dagger* code to your class, you must link in
the code by adding “*className*\ \_SDAG_CODE” inside the class
declaration in the .h file. This macro defines the entry points and
support code used by *Structured Dagger*. Forgetting this results in a
compile error (undefined SDAG entry methods referenced from the .def.h
file).

For example, an array named “Foo” that uses sdag code might contain:

.. code-block:: c++

   class Foo : public CBase_Foo {
   public:
       Foo_SDAG_CODE
       Foo(...) {
          ...
       }
       Foo(CkMigrateMessage *m) { }

       void pup(PUP::er &p) {
          ...
       }
       . . .
   };

Direct Calls to SDAG Entry Methods
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

An SDAG entry method that contains one or more when clause(s) cannot be
directly called and will result in a runtime error with an error
message. It has to be only called through a proxy. This is a runtime
requirement that is enforced in order to prevent accidental calls to
SDAG entry methods that are asynchronous in nature. Additionally, since
they are called using a proxy, it enhances understandability and clarity
as to not be confused for a regular function call that returns
immediately.

For example, in the first example discussed, it is invalid to call the
SDAG entry method ``startStep`` directly as ``startStep();`` because it
contains when clauses. It has to be only called using the proxy i.e.
``computeObj.startStep();`` , where ``computeObj`` is the proxy to
``ComputeObject``.

.. _sec:pup:

Serialization Using the PUP Framework
-------------------------------------

The PUP (Pack/Unpack) framework is a generic way to describe the data in
an object and to use that description for serialization. The
Charm++ system can use this description to pack the object into a
message and unpack the message into a new object on another processor,
to pack and unpack migratable objects for load balancing or
checkpoint/restart-based fault tolerance. The PUP framework also
includes support special for STL containers to ease development in C++.

Like many C++ concepts, the PUP framework is easier to use than describe:

.. code-block:: c++

   class foo : public mySuperclass {
    private:
       double a;
       int x;
       char y;
       unsigned long z;
       float arr[3];
    public:
       ...other methods...

       //pack/unpack method: describe my fields to charm++
       void pup(PUP::er &p) {
         mySuperclass::pup(p);
         p|a;
         p|x; p|y; p|z;
         PUParray(p,arr,3);
       }
   };

This class’s pup method describes the fields of the class to Charm++.
This allows Charm++ to marshall parameters of type foo across
processors, translate foo objects across processor architectures, read
and write foo objects to files on disk, inspect and modify foo objects
in the debugger, and checkpoint and restart programs involving foo
objects.

.. _sec:pupcontract:

PUP contract
~~~~~~~~~~~~

Your object’s pup method must save and restore all your object’s data.
As shown, you save and restore a class’s contents by writing a method
called “pup” which passes all the parts of the class to an object of
type PUP::er, which does the saving or restoring. This manual will often
use “pup” as a verb, meaning “to save/restore the value of” or
equivalently, “to call the pup method of”.

Pup methods for complicated objects normally call the pup methods for
their simpler parts. Since all objects depend on their immediate
superclass, the first line of every pup method is a call to the
superclass’s pup method—the only time you shouldn’t call your
superclass’s pup method is when you don’t have a superclass. If your
superclass has no pup method, you must pup the values in the superclass
yourself.

.. _sec:pupoperator:

PUP operator
^^^^^^^^^^^^

The recommended way to pup any object ``a`` is to use ``p|a;``. This
syntax is an operator ``|`` applied to the PUP::er ``p`` and the user
variable ``a``.

The ``p|a;`` syntax works wherever ``a`` is:

-  A simple type, including char, short, int, long, float, or double. In
   this case, ``p|a;`` copies the data in-place. This is equivalent to
   passing the type directly to the PUP::er using ``p(a)``.

-  Any object with a pup method. In this case, ``p|a;`` calls the
   object’s pup method. This is equivalent to the statement a.pup(p);.

-  A pointer to a PUP::able object, as described in
   Section :numref:`sec:pup::able`. In this case, ``p|a;`` allocates
   and copies the appropriate subclass.

-  An object with a PUPbytes(myClass) declaration in the header. In this
   case, ``p|a;`` copies the object as plain bytes, like memcpy.

-  An object with a custom ``operator |`` defined. In this case,
   ``p|a;`` calls the custom ``operator |``.

See ``examples/charm++/PUP``

For container types, you must simply pup each element of the container.
For arrays, you can use the utility method PUParray, which takes the
PUP::er, the array base pointer, and the array length. This utility
method is defined for user-defined types T as:

.. code-block:: c++

       template<class T>
       inline void PUParray(PUP::er &p,T *array,int length) {
          for (int i=0;i<length;i++) p|array[i];
       }

.. _sec:pupstl:

PUP STL Container Objects
^^^^^^^^^^^^^^^^^^^^^^^^^

If the variable is from the C++ Standard Template Library, you can
include operator\ ``|``\ ’s for STL containers such as vector, map, set,
list, pair, and string, templated on anything, by including the header
“pup_stl.h”.

See ``examples/charm++/PUP/STLPUP``

PUP Dynamic Data
^^^^^^^^^^^^^^^^

As usual in C++, pointers and allocatable objects usually require special
handling. Typically this only requires a p.isUnpacking() conditional
block, where you perform the appropriate allocation. See
Section :numref:`sec:pupdynalloc` for more information and examples.

If the object does not have a pup method, and you cannot add one or use
PUPbytes, you can define an operator\ ``|`` to pup the object. For
example, if myClass contains two fields a and b, the operator\ ``|``
might look like:

.. code-block:: c++

     inline void operator|(PUP::er &p,myClass &c) {
       p|c.a;
       p|c.b;
     }

See ``examples/charm++/PUP/HeapPUP``

.. _sec:pupbytes:

PUP as bytes
^^^^^^^^^^^^

For classes and structs with many fields, it can be tedious and
error-prone to list all the fields in the pup method. You can avoid this
listing in two ways, as long as the object can be safely copied as raw
bytes—this is normally the case for simple structs and classes without
pointers.

-  Use the ``PUPbytes(myClass)`` macro in your header file. This lets
   you use the ``p|*myPtr;`` syntax to pup the entire class as
   sizeof(myClass) raw bytes.

-  Use ``p((void *)myPtr,sizeof(myClass));`` in the pup method. This is
   a direct call to pup a set of bytes.

-  Use ``p((char *)myCharArray,arraySize);`` in the pup method. This is
   a direct call to pup a set of bytes. Other primitive types may also
   be used.

Note that pupping as bytes is just like using ‘memcpy’: it does nothing
to the data other than copy it whole. For example, if the class contains
any pointers, you must make sure to do any allocation needed, and pup
the referenced data yourself.

Pupping as bytes may prevent your pup method from ever being able to
work across different machine architectures. This is currently an
uncommon scenario, but heterogeneous architectures may become more
common, so pupping as bytes is discouraged.

.. _sec:pupoverhead:

PUP overhead
^^^^^^^^^^^^

The PUP::er overhead is very small—one virtual function call for each
item or array to be packed/unpacked. The actual packing/unpacking is
normally a simple memory-to-memory binary copy.

For arrays and vectors of builtin arithmetic types like “int" and
“double", or of types declared as “PUPbytes”, PUParray uses an even
faster block transfer, with one virtual function call per array or
vector.

Thus, if an object does not contain pointers, you should prefer
declaring it as PUPbytes.

For types of objects whose default constructors do more than necessary
when an object will be unpacked from PUP, it is possible to tell the
runtime system to call a more minimalistic alternative. This can apply
to types used as both member variables of chares and as marshalled
arguments to entry methods. A non-chare class can define a constructor
that takes an argument of type ``PUP::reconstruct`` for this purpose.
The runtime system code will call a ``PUP::reconstruct`` constructor in
preference to a default constructor when it’s available. Where
necessary, constructors taking ``PUP::reconstruct`` should call the
constructors of members variables with ``PUP::reconstruct`` if
applicable to that member.

.. _sec:pupmodes:

PUP modes
^^^^^^^^^

Charm++ uses your pup method to both pack and unpack, by passing
different types of PUP::ers to it. The method p.isUnpacking() returns
true if your object is being unpacked—that is, your object’s values are
being restored. Your pup method must work properly in sizing, packing,
and unpacking modes; and to save and restore properly, the same fields
must be passed to the PUP::er, in the exact same order, in all modes.
This means most pup methods can ignore the pup mode.

Three modes are used, with three separate types of PUP::er: sizing,
which only computes the size of your data without modifying it; packing,
which reads/saves values out of your data; and unpacking, which
writes/restores values into your data. You can determine exactly which
type of PUP::er was passed to you using the p.isSizing(), p.isPacking(),
and p.isUnpacking() methods. However, sizing and packing should almost
always be handled identically, so most programs should use
p.isUnpacking() and !p.isUnpacking(). Any program that calls
p.isPacking() and does not also call p.isSizing() is probably buggy,
because sizing and packing must see exactly the same data.

The p.isDeleting() flag indicates the object will be deleted after
calling the pup method. This is normally only needed for pup methods
called via the C or f90 interface, as provided by AMPI or the other
frameworks. Other Charm++ array elements, marshalled parameters, and
other C++ interface objects have their destructor called when they are
deleted, so the p.isDeleting() call is not normally required—instead,
memory should be deallocated in the destructor as usual.

More specialized modes and PUP::ers are described in
section :numref:`sec:PUP:CommonPUPers`.

.. _sec:lifecycle:

PUP Usage Sequence
~~~~~~~~~~~~~~~~~~

.. figure:: fig/pup.png
   :name: fig:pup
   :width: 6in

   Method sequence of an object with a pup method.

Typical method invocation sequence of an object with a pup method is
shown in Figure :numref:`fig:pup`. As usual in C++, objects are
constructed, do some processing, and are then destroyed.

Objects can be created in one of two ways: they can be created using a
normal constructor as usual; or they can be created using their pup
constructor. The pup constructor for Charm++ array elements and
PUP::able objects is a “migration constructor” that takes a single
“CkMigrateMessage \*"; for other objects, such as parameter marshalled
objects, the pup constructor has no parameters. The pup constructor is
always followed by a call to the object’s pup method in ``isUnpacking``
mode.

Once objects are created, they respond to regular user methods and
remote entry methods as usual. At any time, the object pup method can be
called in ``isSizing`` or ``isPacking`` mode. User methods and sizing or
packing pup methods can be called repeatedly over the object lifetime.

Finally, objects are destroyed by calling their destructor as usual.

.. _arraymigratable:

Migratable Array Elements using PUP
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Array objects can migrate from one PE to another. For example, the load
balancer (see section :numref:`lbFramework`) might migrate array
elements to better balance the load between processors. For an array
element to be migratable, it must implement a pup method. The standard
PUP contract (see section :numref:`sec:pupcontract`) and constraints
wrt to serializing data apply. The one exception for chare , group and
node group types is that since the runtime system will be the one to
invoke their PUP routines, the runtime will automatically call PUP on
the generated CBase\_ superclasses so users do not need to call PUP on
generated superclasses.

A simple example for an array follows:

.. code-block:: c++

   //In the .h file:
   class A2 : public CBase_A2 {
   private: //My data members:
       int nt;
       unsigned char chr;
       float flt[7];
       int numDbl;
       double *dbl;
   public:
       //...other declarations

       virtual void pup(PUP::er &p);
   };

   //In the .C file:
   void A2::pup(PUP::er &p)
   {
       // The runtime will automatically call CBase_A2::pup()
       p|nt;
       p|chr;
       p(flt,7);
       p|numDbl;
       if (p.isUnpacking()) dbl=new double[numDbl];
       p(dbl,numDbl);
   }

The default assumption, as used in the example above, for the object
state at PUP time is that a chare, and its member objects, could be
migrated at any time while it is inactive, i.e. not executing an entry
method. Actual migration time can be controlled (see
section :numref:`lbFramework`) to be less frequent. If migration
timing is fully user controlled, e.g., at the end of a synchronized load
balancing step, then PUP implementation can be simplified to only
transport “live” ephemeral data matching the object state which
coincides with migration. More intricate state based PUPing, for objects
whose memory footprint varies substantially with computation phase, can
be handled by explicitly maintaining the object’s phase in a member
variable and implementing phase conditional logic in the PUP method (see
section :numref:`sec:pupdynalloc`).

Marshalling User Defined Data Types via PUP
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Parameter marshalling requires serialization and is therefore
implemented using the PUP framework. User defined data types passed as
parameters must abide by the standard PUP contract (see section
:numref:`sec:pupcontract`).

A simple example of using PUP to marshall user defined data types
follows:

.. code-block:: c++

   class Buffer {
   public:
   //...other declarations
     void pup(PUP::er &p) {
       // remember to pup your superclass if there is one
       p|size;
       if (p.isUnpacking())
         data = new int[size];
       PUParray(p, data, size);
     }

   private:
     int size;
     int *data;
   };


   // In some .ci file
   entry void process(Buffer &buf);

For efficiency, arrays are always copied as blocks of bytes and passed
via pointers. This means classes that need their pup routines to be
called, such as those with dynamically allocated data or virtual methods
cannot be passed as arrays-use STL vectors to pass lists of complicated
user-defined classes. For historical reasons, pointer-accessible
structures cannot appear alone in the parameter list (because they are
confused with messages).

The order of marshalling operations on the send side is:

-  Call “p\ ``|``\ a” on each marshalled parameter with a sizing
   PUP::er.

-  Compute the lengths of each array.

-  Call “p\ ``|``\ a” on each marshalled parameter with a packing
   PUP::er.

-  memcpy each arrays’ data.

The order of marshalling operations on the receive side is:

-  Create an instance of each marshalled parameter using its default
   constructor.

-  Call “p\ ``|``\ a” on each marshalled parameter using an unpacking
   PUP::er.

-  Compute pointers into the message for each array.

Finally, very large structures are most efficiently passed via messages,
because messages are an efficient, low-level construct that minimizes
copying and overhead; but very complicated structures are often most
easily passed via marshalling, because marshalling uses the high-level
pup framework.

See ``examples/charm++/PUP/HeapPUP``

.. _loadbalancing:

Load Balancing
--------------

Load balancing in Charm++ is enabled by its ability to place, or
migrate, chares or chare array elements. Typical application usage to
exploit this feature will construct many more chares than processors,
and enable their runtime migration.

Iterative applications, which are commonplace in physical simulations,
are the most suitable target for Charm++’s measurement based load
balancing techniques. Such applications may contain a series of
time-steps, and/or iterative solvers that run to convergence. For such
computations, typically, the heuristic principle that we call “principle
of persistence” holds: the computational loads and communication
patterns between objects (chares) tend to persist over multiple
iterations, even in dynamic applications. In such cases, the recent past
is a good predictor of the near future. Measurement-based chare
migration strategies are useful in this context. Currently these apply
to chare-array elements, but they may be extended to chares in the
future.

For applications without such iterative structure, or with iterative
structure, but without predictability (i.e. where the principle of
persistence does not apply), Charm++ supports “seed balancers” that move
“seeds” for new chares among processors (possibly repeatedly) to achieve
load balance. These strategies are currently available for both chares
and chare-arrays. Seed balancers were the original load balancers
provided in Charm since the late 80’s. They are extremely useful for
state-space search applications, and are also useful in other
computations, as well as in conjunction with migration strategies.

For iterative computations when there is a correlation between
iterations/steps, but either it is not strong, or the machine
environment is not predictable (due to noise from OS interrupts on small
time steps, or time-shared desktop machines), one can use a combination
of the two kinds of strategies. The baseline load balancing is provided
by migration strategies, but in each iteration one also spawns off work
in the form of chares that can run on any processor. The seed balancer
will handle such work as it arises.

Examples are in ``examples/charm++/load_balancing`` and
``tests/charm++/load_balancing``

.. _lbFramework:

Measurement-based Object Migration Strategies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In Charm++, objects (except groups, nodegroups) can migrate from
processor to processor at runtime. Object migration can potentially
improve the performance of the parallel program by migrating objects
from overloaded processors to underloaded ones.

Charm++ implements a generic, measurement-based load balancing framework
which automatically instruments all Charm++ objects, collects
computation load and communication structure during execution and stores
them into a load balancing database. Charm++ then provides a collection
of load balancing strategies whose job it is to decide on a new mapping
of objects to processors based on the information from the database.
Such measurement based strategies are efficient when we can reasonably
assume that objects in a Charm++ application tend to exhibit temporal
correlation in their computation and communication patterns, i.e. future
can be to some extent predicted using the historical measurement data,
allowing effective measurement-based load balancing without
application-specific knowledge.

Two key terms in the Charm++ load balancing framework are:

-  Load balancing database provides the interface of almost all load
   balancing calls. On each processor, it stores the load balancing
   instrumented data and coordinates the load balancing manager and
   balancer. It is implemented as a Chare Group called LBDatabase.

-  Load balancer or strategy takes the load balancing database and
   produces the new mapping of the objects. In Charm++, it is
   implemented as Chare Group inherited from BaseLB. Three kinds of
   schemes are implemented: (a) centralized load balancers, (b) fully
   distributed load balancers and (c) hierarchical load balancers.

.. _lbStrategy:

Available Load Balancing Strategies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Load balancing can be performed in either a centralized, a fully
distributed, or an hierarchical fashion.

In centralized approaches, the entire machine’s load and communication
structure are accumulated to a single point, typically processor 0,
followed by a decision making process to determine the new distribution
of Charm++ objects. Centralized load balancing requires synchronization
which may incur an overhead and delay. However, due to the fact that the
decision process has a high degree of the knowledge about the entire
platform, it tends to be more accurate.

In distributed approaches, load data is only exchanged among neighboring
processors. There is no global synchronization. However, they will not,
in general, provide an immediate restoration for load balance - the
process is iterated until the load balance can be achieved.

In hierarchical approaches, processors are divided into independent
autonomous sets of processor groups and these groups are organized in
hierarchies, thereby decentralizing the load balancing task. Different
strategies can be used to balance the load on processors inside each
processor group, and processors across groups in a hierarchical fashion.

Listed below are some of the available non-trivial centralized load
balancers and their brief descriptions:

-  **GreedyLB**: Uses a greedy algorithm that always assigns the
   heaviest object to the least loaded processor.

-  **GreedyRefineLB**: Uses a greedy algorithm that assigns the heaviest
   object to the least loaded processor when the benefit outweighs the
   migration cost, otherwise leaves the object on its current processor.
   It takes an optional command-line argument *+LBPercentMoves*,which
   specifies the percentage of migrations that can be tolerated.

-  **TopoCentLB**: Extends the greedy algorithm to take processor
   topology into account.

-  **RefineLB**: Moves objects away from the most overloaded processors
   to reach average, limits the number of objects migrated.

-  **RefineSwapLB**: Moves objects away from the most overloaded
   processors to reach average. In case it cannot migrate an object from
   an overloaded processor to an underloaded processor, it swaps objects
   to reduce the load on the overloaded processor. This strategy limits
   the number of objects migrated.

-  **RefineTopoLB**: Same idea as in RefineLB, but takes processor
   topology into account.

-  **BlockLB**: This strategy does a blocked distribution of objects to
   processors.

-  **ComboCentLB**: A special load balancer that can be used to combine
   any number of centralized load balancers mentioned above.

Listed below are some of the communication-aware load balancers:

-  **MetisLB**: Uses `METIS <http://glaros.dtc.umn.edu/gkhome/metis/metis/overview>`__
   to partition the object communication graph.

-  **ScotchLB**: Uses the `SCOTCH <http://www.labri.fr/perso/pelegrin/scotch/>`__
   library for partitioning the object
   communication graph, while also taking object load imbalance into
   account.

-  **GreedyCommLB**: Extends the greedy algorithm to take the
   communication graph into account.

-  **RefineCommLB**: Same idea as in RefineLB, but takes communication
   into account.

Listed below are the distributed load balancers:

-  **NeighborLB**: A neighborhood load balancer in which each processor
   tries to average out its load only among its neighbors.

-  **WSLB**: A load balancer for workstation clusters, which can detect
   load changes on desktops (and other timeshared processors) and adjust
   load without interfering with other’s use of the desktop.

-  **DistributedLB**: A load balancer which uses partial information
   about underloaded and overloaded processors in the system to do
   probabilistic transfer of load. This is a refinement based strategy.

An example of a hierarchical strategy can be found in:

-  **HybridLB**: This calls GreedyLB at the lower level and RefineLB at
   the root.

Listed below are load balancers for debugging purposes:

-  **RandCentLB**: Randomly assigns objects to processors;

-  **RotateLB**: This strategy moves objects to the next available PE
   every time it is called. It is useful for debugging PUP routines and
   other migration related bugs.

Users can choose any load balancing strategy they think is appropriate
for their application. We recommend using GreedyRefineLB with
applications in general. For applications where the cost of migrating
objects is very high, say, due to frequent load balancing to handle
frequent load changes or due to size of data in the object being large,
a strategy which favors migration minimization at the cost of balance
(eg: RefineLB) is more suitable. DistributedLB and HybridLB are suitable
for large number of nodes. Communication-aware load balancers like
MetisLB and ScotchLB are suitable for communication intensive
applications. RotateLB and RandCentLB are more useful for debugging
object migrations. The compiler and runtime options are described in
section :numref:`lbOption`.

**Metabalancer to automatically schedule load balancing**

Metabalancer can be invoked to automatically decide when to invoke the
load balancer, given the load-balancing strategy. Metabalancer uses a
linear prediction model to set the load balancing period based on
observed load imbalance.

The runtime option *+MetaLB* can be used to invoke this feature to
automatically invoke the load balancing strategy based on the imbalance
observed. This option needs to be specified alongside the *+balancer*
option to specify the load balancing strategy to use. Metabalancer
relies on the AtSync() calls specified in Section :numref:`lbarray`
to collect load statistics.

*+MetaLBModelDir* ``<path-to-model>`` can be used to invoke the
Metabalancer feature to automatically decide which load balancing
strategy to invoke. A model trained on a generic representative load
imbalance benchmark can be found in ``charm/src/ck-ldb/rf_model``.
Metabalancer makes a decision on which load balancing strategy to invoke
out of a subset of strategies, namely GreedyLB, RefineLB, HybridLB,
DistributedLB, MetisLB and ScotchLB. For using the model based
prediction in Metabalancer, Charm++ needs to be built with all the above
load balancing strategies, including ScotchLB that relies on the
external partitioning library SCOTCH specified in the
Section :numref:`lbOption`.

.. _lbarray:

Load Balancing Chare Arrays
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The load balancing framework is well integrated with chare array
implementation - when a chare array is created, it automatically
registers its elements with the load balancing framework. The
instrumentation of compute time (WALL/CPU time) and communication
pattern is done automatically and APIs are provided for users to trigger
the load balancing. To use the load balancer, you must make your array
elements migratable (see migration section above) and choose a load
balancing strategy (see the section :numref:`lbStrategy` for a
description of available load balancing strategies).

There are three different ways to use load balancing for chare arrays to
meet different needs of the applications. These methods are different in
how and when a load balancing phase starts. The three methods are:
**periodic load balancing mode**, **at sync mode** and **manual mode**.

In *periodic load balancing mode*, a user specifies only how often load
balancing is to occur, using +LBPeriod runtime option to specify the
time interval.

In *at sync mode*, the application invokes the load balancer explicitly
at appropriate (generally at a pre-existing synchronization boundary) to
trigger load balancing by inserting a function call (AtSync) in the
application source code.

In the prior two load balancing modes, users do not need to worry about
how to start load balancing. However, in one scenario, those automatic
load balancers will fail to work - when array elements are created by
dynamic insertion. This is because the above two load balancing modes
require an application to have fixed the number of objects at the time
of load balancing. The array manager needs to maintain a head count of
local array elements for the local barrier. In this case, the
application must use the *manual mode* to trigger load balancer.

The detailed APIs of these three methods are described as follows:

#. **Periodical load balancing mode**: In the default setting, load
   balancing happens whenever the array elements are ready, with an
   interval of 1 second. It is desirable for the application to set a
   larger interval using +LBPeriod runtime option. For example
   “+LBPeriod 5.0” can be used to start load balancing roughly every 5
   seconds. By default, array elements may be asked to migrate at any
   time, provided that they are not in the middle of executing an entry
   method. The array element’s variable usesAtSync being false
   attributes to this default behavior.

#. **At sync mode**: Using this method, elements can be migrated only at
   certain points in the execution when the application invokes
   AtSync(). In order to use the at sync mode, one should set usesAtSync
   to true in the array element constructor. When an element is ready to
   migrate, call AtSync()  [6]_. When all local elements call AtSync,
   the load balancer is triggered. Once all migrations are completed,
   the load balancer calls the virtual function
   ArrayElement::ResumeFromSync() on each of the array elements. This
   function can be redefined in the application.

   Note that the minimum time for AtSync() load balancing to occur is
   controlled by the LBPeriod. Unusually high frequency load balancing
   (more frequent than 500ms) will perform better if this value is set
   via +LBPeriod or SetLBPeriod() to a number shorter than your load
   balancing interval.

   Note that *AtSync()* is not a blocking call, it just gives a hint to
   load balancing that it is time for load balancing. During the time
   between *AtSync* and *ResumeFromSync*, the object may be migrated.
   One can choose to let objects continue working with incoming
   messages, however keep in mind the object may suddenly show up in
   another processor and make sure no operations that could possibly
   prevent migration be performed. This is the automatic way of doing
   load balancing where the application does not need to define
   ResumeFromSync().

   The more commonly used approach is to force the object to be idle
   until load balancing finishes. The user places an AtSync call at the
   end of some iteration and when all elements reach that call load
   balancing is triggered. The objects can start executing again when
   ResumeFromSync() is called. In this case, the user redefines
   ResumeFromSync() to trigger the next iteration of the application.
   This manual way of using the at sync mode results in a barrier at
   load balancing (see example here :numref:`lbexample`).

#. **Manual mode**: The load balancer can be programmed to be started
   manually. To switch to the manual mode, the application calls
   *TurnManualLBOn()* on every processor to prevent the load balancer
   from starting automatically. *TurnManualLBOn()* should be called as
   early as possible in the program. It could be called at the
   initialization part of the program, for example from a global
   variable constructor, or in an initproc call
   (Section :numref:`initproc`). It can also be called in the
   constructor of a static array or before the *doneInserting* call for
   a dynamic array. It can be called multiple times on one processor,
   but only the last one takes effect.

   The function call *CkStartLB()* starts load balancing immediately.
   This call should be made at only one place on only one processor.
   This function is not blocking, the object will continue to process
   messages and the load balancing, when triggered, happens in the
   background.

   *TurnManualLBOff()* turns off manual load balancing and switches back
   to the automatic load balancing mode.

.. _lbmigobj:

Migrating objects
~~~~~~~~~~~~~~~~~

Load balancers migrate objects automatically. For an array element to
migrate, user can refer to Section :numref:`arraymigratable` for how
to write a “pup” for an array element.

In general one needs to pack the whole snapshot of the member data in an
array element in the pup subroutine. This is because the migration of
the object may happen at any time. In certain load balancing schemes
where the user explicitly controls when load balancing occurs, the user
may choose to pack only a part of the data and may skip temporary data.

An array element can migrate by calling the migrateMe(destination
processor) member function- this call must be the last action in an
element entry method. The system can also migrate array elements for
load balancing (see the section :numref:`lbarray`).

To migrate your array element to another processor, the Charm++ runtime
will:

-  Call your ckAboutToMigrate method

-  Call your pup method with a sizing PUP::er to determine how big a
   message it needs to hold your element.

-  Call your pup method again with a packing PUP::er to pack your
   element into a message.

-  Call your element’s destructor (deleting the old copy).

-  Send the message (containing your element) across the network.

-  Call your element’s migration constructor on the new processor.

-  Call your pup method on with an unpacking PUP::er to unpack the
   element.

-  Call your ckJustMigrated method

Migration constructors, then, are normally empty- all the unpacking and
allocation of the data items is done in the element’s pup routine.
Deallocation is done in the element destructor as usual.

Other utility functions
~~~~~~~~~~~~~~~~~~~~~~~

There are several utility functions that can be called in applications
to configure the load balancer, etc. These functions are:

-  **LBTurnInstrumentOn()** and **LBTurnInstrumentOff()**: are plain C
   functions to control the load balancing statistics instrumentation on
   or off on the calling processor. No implicit broadcast or
   synchronization exists in these functions. Fortran interface:
   **FLBTURNINSTRUMENTON()** and **FLBTURNINSTRUMENTOFF()**.

-  **setMigratable(bool migratable)**: is a member function of array
   element. This function can be called in an array element constructor
   to tell the load balancer whether this object is migratable or
   not [7]_.

-  **LBSetPeriod(double s)**: this function can be called anywhere (even
   in Charm++ initnodes or initprocs) to specify the load balancing
   period time in seconds. It tells load balancer not to start next load
   balancing in less than :math:`s` seconds. This can be used to prevent
   load balancing from occurring too often in *automatic without sync
   mode*. Here is how to use it:

   .. code-block:: c++

      // if used in an array element
      LBDatabase *lbdb = getLBDB();
      lbdb->SetLBPeriod(5.0);

      // if used outside of an array element
      LBSetPeriod(5.0);

   Alternatively, one can specify +LBPeriod {seconds} at command line.

.. _lbOption:

Compiler and runtime options to use load balancing module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Load balancing strategies are implemented as libraries in Charm++. This
allows programmers to easily experiment with different existing
strategies by simply linking a pool of strategy modules and choosing one
to use at runtime via a command line option.

**Note:** linking a load balancing module is different from activating
it:

-  link an LB module: is to link a Load Balancer module(library) at
   compile time. You can link against multiple LB libraries as
   candidates.

-  activate an LB: is to actually ask the runtime to create an LB
   strategy and start it. You can only activate load balancers that have
   been linked at compile time.

Below are the descriptions about the compiler and runtime options:

#. **compile time options:**

   -  | *-module NeighborLB -module GreedyCommLB ...*
      | links the modules NeighborLB, GreedyCommLB etc into an
        application, but these load balancers will remain inactive at
        execution time unless overridden by other runtime options.

   -  | *-module CommonLBs*
      | links a special module CommonLBs which includes some commonly
        used Charm++ built-in load balancers. The commonly used load
        balancers include: *DummyLB, GreedyLB, GreedyRefineLB, CommLB, RandCentLB, RefineLB, RefineCommLB, RotateLB, DistributedLB, HybridLB, ComboCentLB, RefineSwapLB, NeighborLB, OrbLB, BlockLB, GreedyCommLB*

   -  | *-balancer GreedyCommLB*
      | links the load balancer GreedyCommLB and invokes it at runtime.

   -  | *-balancer GreedyCommLB -balancer RefineLB*
      | invokes GreedyCommLB at the first load balancing step and
        RefineLB in all subsequent load balancing steps.

   -  | *-balancer ComboCentLB:GreedyLB,RefineLB*
      | You can create a new combination load balancer made of multiple
        load balancers. In the above example, GreedyLB and RefineLB
        strategies are applied one after the other in each load
        balancing step.

   The list of existing load balancers is given in Section
   :numref:`lbStrategy`. Note: you can have multiple -module \*LB
   options. LB modules are linked into a program, but they are not
   activated automatically at runtime. Using -balancer A at compile time
   will activate load balancer A automatically at runtime. Having
   -balancer A implies -module A, so you don’t have to write -module A
   again, although that is not invalid. Using CommonLBs is a convenient
   way to link against the commonly used existing load balancers.

   The SCOTCH-based load balancer(s) use an external partitioning
   library requiring 3rd party software:

   SCOTCH can be downloaded from:
   http://www.labri.fr/perso/pelegrin/scotch/

   Use the *-incdir and -libdir* build time option to add your
   installation of any third party libraries you wish to use to the
   Charm++ search paths.

#. **Building individual load balancers**

   Load balancers can be built individually by changing the current
   working directory to the *tmp* subdirectory of your build and making
   them by name.

   .. code-block:: bash

      $ cd netlrts-linux-x86_64/tmp
      $ make PhasebyArrayLB

#. **Write and use your own load balancer**

   Refer Section :numref:`lbWriteNewLB` for writing a new load
   balancer. Compile it in the form of library and name it
   *libmoduleFooLB.a* where *FooLB* is the new load balancer. Add the
   path to the library and link the load balancer into an application
   using *-module FooLB*.

   You can create a library by modifying the Makefile in the following
   way. This will create *libmoduleFooLB.a*.

   .. code-block:: makefile

      libmoduleFooLB.a: FooLB.o
        $(CHARMC) -o libmoduleFooLB.a FooLB.o

   To include this balancer in your application, the Makefile can be
   changed in the following way

   .. code-block:: makefile

      $(TARGET): $(OBJECTS)
        $(CHARMC) -o $(TARGET) -L/path-to-the-lib $(OBJS) -module FooLB

#. **runtime options:**

   Runtime balancer selection options are similar to the compile time
   options as described above, but they can be used to override those
   compile time options.

   -  | *+balancer help*
      | displays all available balancers that have been linked in.

   -  | *+balancer GreedyCommLB*
      | invokes GreedyCommLB

   -  | *+balancer GreedyCommLB +balancer RefineLB*
      | invokes GreedyCommLB at the first load balancing step and
        RefineLB in all subsequent load balancing steps.

   -  | *+balancer ComboCentLB:GreedyLB,RefineLB*
      | same as the example in the -balancer compile time option.

   Note: +balancer option works only if you have already linked the
   corresponding load balancers module at compile time. Giving +balancer
   with a wrong LB name will result in a runtime error. When you have
   used -balancer A as compile time option, you do not need to use
   +balancer A again to activate it at runtime. However, you can use
   +balancer B to override the compile time option and choose to
   activate B instead of A.

#. **Handling the case that no load balancer is activated by users**

   When no balancer is linked by users, but the program counts on a load
   balancer because it used *AtSync()* and expect *ResumeFromSync()* to
   be called to continue, a special load balancer called *NullLB* will
   be automatically created to run the program. This default load
   balancer calls *ResumeFromSync()* after *AtSync()*. It keeps a
   program from hanging after calling *AtSync()*. *NullLB* will be
   suppressed if another load balancer is created.

#. **Other useful runtime options**

   There are a few other runtime options for load balancing that may be
   useful:

   -  | *+LBDebug {verbose level}*
      | {verbose level} can be any positive integer number. 0 is to turn
        off the verbose. This option asks load balancer to output load
        balancing information to stdout. The bigger the verbose level
        is, the more verbose the output is.

   -  | *+LBPeriod {seconds}*
      | {Seconds} can be any float number. This option sets the minimum
        period time in seconds between two consecutive load balancing
        steps. The default value is 1 second. That is to say that a load
        balancing step will not happen until 1 second after the last
        load balancing step.

   -  | *+LBSameCpus*
      | This option simply tells load balancer that all processors are
        of same speed. The load balancer will then skip the measurement
        of CPU speed at runtime. This is the default.

   -  | *+LBTestPESpeed*
      | This option tells the load balancer to test the speed of all
        processors at runtime. The load balancer may use this
        measurement to perform speed-aware load balancing.

   -  | *+LBObjOnly*
      | This tells load balancer to ignore processor background load
        when making migration decisions.

   -  | *+LBSyncResume*
      | After load balancing step, normally a processor can resume
        computation once all objects are received on that processor,
        even when other processors are still working on migrations. If
        this turns out to be a problem, that is when some processors
        start working on computation while the other processors are
        still busy migrating objects, then this option can be used to
        force a global barrier on all processors to make sure that
        processors can only resume computation after migrations are
        completed on all processors.

   -  | *+LBOff*
      | This option turns off load balancing instrumentation of both CPU
        and communication usage at startup time.

   -  | *+LBCommOff*
      | This option turns off load balancing instrumentation of
        communication at startup time. The instrument of CPU usage is
        left on.

.. _seedlb:

Seed load balancers - load balancing Chares at creation time
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Seed load balancing involves the movement of object creation messages,
or "seeds", to create a balance of work across a set of processors. This
seed load balancing scheme is used to balance chares at creation time.
After the chare constructor is executed on a processor, the seed
balancer does not migrate it. Depending on the movement strategy,
several seed load balancers are available now. Examples can be found
``examples/charm++/NQueen``.

#. | *random*
   | A strategy that places seeds randomly when they are created and
     does no movement of seeds thereafter. This is used as the default
     seed load balancer.

#. | *neighbor*
   | A strategy which imposes a virtual topology on the processors, load
     exchange happens among neighbors only. The overloaded processors
     initiate the load balancing and send work to its neighbors when it
     becomes overloaded. The default topology is mesh2D, one can use
     command line option to choose other topology such as ring, mesh3D
     and dense graph.

#. | *spray*
   | A strategy which imposes a spanning tree organization on the
     processors, results in communication via global reduction among all
     processors to compute global average load via periodic reduction.
     It uses averaging of loads to determine how seeds should be
     distributed.

#. | *workstealing*
   | A strategy that the idle processor requests a random processor and
     steal chares.

Other strategies can also be explored by following the simple API of the
seed load balancer.

**Compile and run time options for seed load balancers**

To choose a seed load balancer other than the default *rand* strategy,
use link time command line option **-balance foo**.

When using neighbor seed load balancer, one can also specify the virtual
topology at runtime. Use **+LBTopo topo**, where *topo* can be one of:
(a) ring, (b) mesh2d, (c) mesh3d and (d) graph.

To write a seed load balancer, name your file as *cldb.foo.c*, where
*foo* is the strategy name. Compile it in the form of library under
charm/lib, named as *libcldb-foo.a*, where *foo* is the strategy name
used above. Now one can use **-balance foo** as compile time option to
**charmc** to link with the *foo* seed load balancer.

.. _lbexample:

Simple Load Balancer Usage Example - Automatic with Sync LB
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A simple example of how to use a load balancer in sync mode in one’s
application is presented below.

.. code-block:: c++

   /*** lbexample.ci ***/
   mainmodule lbexample {
     readonly CProxy_Main mainProxy;
     readonly int nElements;

     mainchare Main {
       entry Main(CkArgMsg *m);
       entry void done(void);
     };

     array [1D] LBExample {
       entry LBExample(void);
       entry void doWork();
     };
   };

——————————————————————————-

.. code-block:: c++

   /*** lbexample.C ***/
   #include <stdio.h>
   #include "lbexample.decl.h"

   /*readonly*/ CProxy_Main mainProxy;
   /*readonly*/ int nElements;

   #define MAX_WORK_CNT 50
   #define LB_INTERVAL 5

   /*mainchare*/
   class Main : public CBase_Main
   {
   private:
     int count;
   public:
     Main(CkArgMsg* m)
     {
       /*....Initialization....*/
       mainProxy = thisProxy;
       CProxy_LBExample arr = CProxy_LBExample::ckNew(nElements);
       arr.doWork();
     };

     void done(void)
     {
       count++;
       if(count==nElements){
         CkPrintf("All done");
         CkExit();
       }
     };
   };

   /*array [1D]*/
   class LBExample : public CBase_LBExample
   {
   private:
     int workcnt;
   public:
     LBExample()
     {
       workcnt=0;
       /* May initialize some variables to be used in doWork */
       //Must be set to true to make AtSync work
       usesAtSync = true;
     }

     LBExample(CkMigrateMessage *m) { /* Migration constructor -- invoked when chare migrates */ }

     /* Must be written for migration to succeed */
     void pup(PUP::er &p){
       p|workcnt;
       /* There may be some more variables used in doWork */
     }

     void doWork()
     {
       /* Do work proportional to the chare index to see the effects of LB */

       workcnt++;
       if(workcnt==MAX_WORK_CNT)
         mainProxy.done();

       if(workcnt%LB_INTERVAL==0)
         AtSync();
       else
         doWork();
     }

     void ResumeFromSync(){
       doWork();
     }
   };

   #include "lbexample.def.h"

Processor-Aware Chare Collections
---------------------------------

So far, we have discussed chares separately from the underlying hardware
resources to which they are mapped. However, for writing lower-level
libraries or optimizing application performance it is sometimes useful
to create chare collections where a single chare is mapped per specified
resource used in the run. The group  [8]_ and node group constructs
provide this facility by creating a collection of chares with a single
chare (or branch) on each PE (in the case of groups) or process (for
node groups).

.. _sec:group:

Group Objects
~~~~~~~~~~~~~

Groups have a definition syntax similar to normal chares, and they have
to inherit from the system-defined class CBase_ClassName, where
ClassName is the name of the group’s C++ class  [9]_.

Group Definition
^^^^^^^^^^^^^^^^

In the interface (``.ci``) file, we declare

.. code-block:: c++

   group Foo {
     // Interface specifications as for normal chares

     // For instance, the constructor ...
     entry Foo(parameters1);

     // ... and an entry method
     entry void someEntryMethod(parameters2);
   };

The definition of the ``Foo`` class is given in the ``.h`` file, as
follows:

.. code-block:: c++

   class Foo : public CBase_Foo {
     // Data and member functions as in C++
     // Entry functions as for normal chares

     public:
       Foo(parameters1);
       void someEntryMethod(parameters2);
   };

.. _sec:groups/creation:

Group Creation
^^^^^^^^^^^^^^

Groups are created using ckNew like chares and chare arrays. Given the
declarations and definitions of group ``Foo`` from above, we can create
a group in the following manner:

.. code-block:: c++

   CProxy_Foo fooProxy = CProxy_Foo::ckNew(parameters1);

One can also use ckNew to get a CkGroupID as shown below:

.. code-block:: c++

   CkGroupID fooGroupID = CProxy_Foo::ckNew(parameters1);

A CkGroupID is useful to specify dependence in group creations using
CkEntryOptions. For example, in the following code, the creation of
group ``GroupB`` on each PE depends on the creation of ``GroupA`` on
that PE.

.. code-block:: c++

   // Create GroupA
   CkGroupID groupAID = CProxy_GroupA::ckNew(parameters1);

   // Create GroupB. However, for each PE, do this only
   // after GroupA has been created on it

   // Specify the dependency through a `CkEntryOptions' object
   CkEntryOptions opts;
   opts.setGroupDepID(groupAId);

   // The last argument to `ckNew' is the `CkEntryOptions' object from above
   CkGroupID groupBID = CProxy_GroupB::ckNew(parameters2, opts);

Note that there can be several instances of each group type. In such a
case, each instance has a unique group identifier, and its own set of
branches.

Method Invocation on Groups
^^^^^^^^^^^^^^^^^^^^^^^^^^^

An asynchronous entry method can be invoked on a particular branch of a
group through a proxy of that group. If we have a group with a proxy
``fooProxy`` and we wish to invoke entry method ``someEntryMethod`` on
that branch of the group which resides on PE ``somePE``, we would
accomplish this with the following syntax:

.. code-block:: c++

   fooProxy[somePE].someEntryMethod(parameters);

This call is asynchronous and non-blocking; it returns immediately after
sending the message. A message may be broadcast to all branches of a
group (i.e., to all PEs) using the notation :

.. code-block:: c++

   fooProxy.anotherEntryMethod(parameters);

This invokes entry method anotherEntryMethod with the given parameters
on all branches of the group. This call is also asynchronous and
non-blocking, and it, too, returns immediately after sending the
message.

Recall that each PE hosts a branch of every instantiated group.
Sequential objects, chares and other groups can gain access to this
*PE-local* branch using ckLocalBranch():

.. code-block:: c++

   GroupType *g=groupProxy.ckLocalBranch();

This call returns a regular C++ pointer to the actual object (not a proxy)
referred to by the proxy groupProxy. Once a proxy to the local branch of
a group is obtained, that branch can be accessed as a regular C++ object.
Its public methods can return values, and its public data is readily
accessible.

Let us end with an example use-case for groups. Suppose that we have a
task-parallel program in which we dynamically spawn new chares.
Furthermore, assume that each one of these chares has some data to send
to the mainchare. Instead of creating a separate message for each
chare’s data, we create a group. When a particular chare finishes its
work, it reports its findings to the local branch of the group. When all
the chares on a PE have finished their work, the local branch can send a
single message to the main chare. This reduces the number of messages
sent to the mainchare from the number of chares created to the number of
processors.

For a more concrete example on how to use groups, please refer to
``examples/charm++/histogram_group``. It presents a parallel
histogramming operation in which chare array elements funnel their bin
counts through a group, instead of contributing directly to a reduction
across all chares.

NodeGroup Objects
~~~~~~~~~~~~~~~~~

The *node group* construct is similar to the group construct discussed
above. Rather than having one chare per PE, a node group is a collection
of chares with one chare per *process*, or *logical node*. Therefore,
each logical node hosts a single branch of the node group. As with
groups, node groups can be addressed via globally unique identifiers.
Nonetheless, there are significant differences in the semantics of node
groups as compared to groups and chare arrays. When an entry method of a
node group is executed on one of its branches, it executes on *some* PE
within the logical node. Also, multiple entry method calls can execute
concurrently on a single node group branch. This makes node groups
significantly different from groups and requires some care when using
them.

NodeGroup Declaration
^^^^^^^^^^^^^^^^^^^^^

Node groups are defined in a similar way to groups.  [10]_ For example,
in the interface file, we declare:

.. code-block:: c++

    nodegroup NodeGroupType {
     // Interface specifications as for normal chares
    };

In the ``.h`` file, we define NodeGroupType as follows:

.. code-block:: c++

    class NodeGroupType : public CBase_NodeGroupType {
     // Data and member functions as in C++
     // Entry functions as for normal chares
    };

Like groups, NodeGroups are identified by a globally unique identifier
of type CkGroupID. Just as with groups, this identifier is common to all
branches of the NodeGroup, and can be obtained from the inherited data
member thisgroup. There can be many instances corresponding to a single
NodeGroup type, and each instance has a different identifier, and its
own set of branches.

Method Invocation on NodeGroups
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

As with chares, chare arrays and groups, entry methods are invoked on
NodeGroup branches via proxy objects. An entry method may be invoked on
a *particular* branch of a nodegroup by specifying a *logical node
number* argument to the square bracket operator of the proxy object. A
broadcast is expressed by omitting the square bracket notation. For
completeness, example syntax for these two cases is shown below:

.. code-block:: c++

    // Invoke `someEntryMethod' on the i-th logical node of
    // a NodeGroup whose proxy is `myNodeGroupProxy':
    myNodeGroupProxy[i].someEntryMethod(parameters);

    // Invoke `someEntryMethod' on all logical nodes of
    // a NodeGroup whose proxy is `myNodeGroupProxy':
    myNodeGroupProxy.someEntryMethod(parameters);

It is worth restating that when an entry method is invoked on a
particular branch of a nodegroup, it may be executed by *any* PE in that
logical node. Thus two invocations of a single entry method on a
particular branch of a NodeGroup may be executed *concurrently* by two
different PEs in the logical node. If this could cause data races in
your program, please consult § :numref:`sec:nodegroups/exclusive`
(below).

.. _sec:nodegroups/exclusive:

NodeGroups and exclusive Entry Methods
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Node groups may have exclusive entry methods. The execution of an
exclusive entry method invocation is *mutually exclusive* with those of
all other exclusive entry methods invocations. That is, an exclusive
entry method invocation is not executed on a logical node as long as
another exclusive entry method is executing on it. More explicitly, if a
method M of a nodegroup NG is marked exclusive, it means that while an
instance of M is being executed by a PE within a logical node, no other
PE within that logical node will execute any other *exclusive* methods.
However, PEs in the logical node may still execute *non-exclusive* entry
method invocations. An entry method can be marked exclusive by tagging
it with the exclusive attribute, as explained in
§ :numref:`attributes`.

Accessing the Local Branch of a NodeGroup
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The local branch of a NodeGroup NG, and hence its member fields and
methods, can be accessed through the method NG\*
CProxy_NG::ckLocalBranch() of its proxy. Note that accessing data
members of a NodeGroup branch in this manner is *not* thread-safe by
default, although you may implement your own mutual exclusion schemes to
ensure safety. One way to ensure safety is to use node-level locks,
which are described in the Converse manual.

NodeGroups can be used in a similar way to groups so as to implement
lower-level optimizations such as data sharing and message reduction.

Initializations at Program Startup
----------------------------------

.. _initnode:
.. _initproc:

initnode and initproc Routines
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Some registration routines need be executed
exactly once before the computation begins. You may choose to declare a
regular C++ subroutine initnode in the .ci file to ask Charm++ to execute
the routine exactly once on *every logical node* before the computation
begins, or to declare a regular C++ subroutine initproc to be executed
exactly once on *every PE*.

.. code-block:: c++

   module foo {
       initnode void fooNodeInit(void);
       initproc void fooProcInit(void);
       chare bar {
           ...
           initnode void barNodeInit(void);
           initproc void barProcInit(void);
       };
   };

This code will execute the routines fooNodeInit and static
bar::barNodeInit once on every logical node and fooProcInit and
bar::barProcInit on every PE before the main computation starts.
Initnode calls are always executed before initproc calls. Both init
calls (declared as static member functions) can be used in chares, chare
arrays and groups.

Note that these routines should only implement registration and startup
functionality, and not parallel computation, since the Charm++ run time
system will not have started up fully when they are invoked. In order to
begin the parallel computation, you should use a mainchare instead,
which gets executed on only PE 0.

Event Sequence During Charm++ Startup
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

At startup, every Charm++ program performs the following actions, in
sequence:

#. Module Registration: all modules given in the .ci files are
   registered in the order of their specification in the linking stage
   of program compilation. For example, if you specified
   “``-module A -module B``” while linking your Charm++ program, then
   module ``A`` is registered before module ``B`` at runtime.

#. initnode, initproc Calls: all initnode and initproc functions are
   invoked before the creation of Charm++ data structures, and before
   the invocation of any mainchares’ main() methods.

#. readonly Variables: readonly variables are initialized on PE 0 in the
   mainchare, following program order as given in the ``main()`` method.
   After initialization, they are broadcast to all other PEs making them
   available in the constructors groups, chares, chare arrays, etc. (see
   below.)

#. Group and NodeGroup Creation: on PE 0, constructors of these objects
   are invoked in program order. However, on all other PEs, their
   creation is triggered by messages. Since message order is not
   guaranteed in Charm++ program, constructors of groups and nodegroups
   should **not** depend on other Group or NodeGroup objects on a PE.
   However, if your program structure requires it, you can explicitly
   specify that the creation of certain Groups/NodeGroups depends on the
   creation of others, as described in
   § :numref:`sec:groups/creation`. In addition, since those
   objects are initialized after the initialization of readonly
   variables, readonly variables can be used in the constructors of
   Groups and NodeGroups.

#. Charm++ Array Creation: the order in which array constructors are
   called on PEs is similar to that described for groups and nodegroups,
   above. Therefore, the creation of one array should **not** depend on
   other arrays. As Array objects are initialized last, their
   constructors can use readonly variables and local branches of Group
   or NodeGroup objects.
