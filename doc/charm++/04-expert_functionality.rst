Expert-Level Functionality
==========================

.. _advancedlb:

Tuning and Developing Load Balancers
------------------------------------

Load Balancing Simulation
~~~~~~~~~~~~~~~~~~~~~~~~~

The simulation feature of the load balancing framework allows the users
to collect information about the compute WALL/CPU time and communication
of the chares during a particular run of the program and use this
information later to test the different load balancing strategies to see
which one is suitable for the program behavior. Currently, this feature
is supported only for the centralized load balancing strategies. For
this, the load balancing framework accepts the following command line
options:

#. | *+LBDump StepStart*
   | This will dump the compute and the communication data collected by
     the load balancing framework starting from the load balancing step
     *StepStart* into a file on the disk. The name of the file is given
     by the *+LBDumpFile* option. The load balancing step in the program
     is numbered starting from 0. Negative value for *StepStart* will be
     converted to 0.

#. | *+LBDumpSteps StepsNo*
   | This option specifies the number of load balancing steps for which
     data will be dumped to disk. If omitted, its default value is 1.
     The program will exit after *StepsNo* files are created.

#. | *+LBDumpFile FileName*
   | This option specifies the base name of the file created with the
     load balancing data. If this option is not specified, the framework
     uses the default file ``lbdata.dat``. Since multiple steps are
     allowed, a number corresponding to the step number is appended to
     the filename in the form ``Filename.#``; this applies to both dump
     and simulation.

#. | *+LBSim StepStart*
   | This option instructs the framework to do the simulation starting
     from *StepStart* step. When this option is specified, the load
     balancing data along with the step number will be read from the
     file specified in the *+LBDumpFile* option. The program will print
     the results of the balancing for a number of steps given by the
     *+LBSimSteps* option, and then will exit.

#. | *+LBSimSteps StepsNo*
   | This option is applicable only to the simulation mode. It specifies
     the number of load balancing steps for which the data will be
     dumped. The default value is 1.

#. | *+LBSimProcs*
   | With this option, the user can change the number of processors
     specified to the load balancing strategy. It may be used to test
     the strategy in the cases where some processor crashes or a new
     processor becomes available. If this number is not changed since
     the original run, starting from the second step file, the program
     will print other additional information about how the simulated
     load differs from the real load during the run (considering all
     strategies that were applied while running). This may be used to
     test the validity of a load balancer prediction over the reality.
     If the strategies used during run and simulation differ, the
     additional data printed may not be useful.

Here is an example which collects the data for a 1000 processor run of a
program

.. code-block:: bash

   $ ./charmrun pgm +p1000 +balancer RandCentLB +LBDump 2 +LBDumpSteps 4 +LBDumpFile lbsim.dat

This will collect data on files lbsim.dat.2,3,4,5. We can use this data
to analyze the performance of various centralized strategies using:

.. code-block:: bash

   $ ./charmrun pgm +balancer <Strategy to test> +LBSim 2 +LBSimSteps 4 +LBDumpFile lbsim.dat
   [+LBSimProcs 900]

Please note that this does not invoke the real application. In fact,
"pgm" can be replaced with any generic application which calls
centralized load balancer. An example can be found in
``tests/charm++/load_balancing/lb_test``.

Future load predictor
~~~~~~~~~~~~~~~~~~~~~

When objects do not follow the assumption that the future workload will
be the same as the past, the load balancer might not have the right
information to do a good rebalancing job. To prevent this, the user can
provide a transition function to the load balancer to predict what will
be the future workload, given the past instrumented one. For this, the
user can provide a specific class which inherits from
``LBPredictorFunction`` and implement the appropriate functions. Here is
the abstract class:

.. code-block:: c++

   class LBPredictorFunction {
   public:
     int num_params;

     virtual void initialize_params(double *x);

     virtual double predict(double x, double *params) =0;
     virtual void print(double *params) {PredictorPrintf("LB: unknown model");};
     virtual void function(double x, double *param, double &y, double *dyda) =0;
   };

-  ``initialize_params`` by default initializes the parameters randomly.
   If the user knows how they should be, this function can be
   re-implemented.

-  ``predict`` is the function that predicts the future load based on
   the function parameters. An example for the *predict* function is
   given below.

   .. code-block:: c++

      double predict(double x, double *param) {return (param[0]*x + param[1]);}

-  ``print`` is useful for debugging and it can be re-implemented to
   have a meaningful print of the learned model

-  ``function`` is a function internally needed to learn the parameters,
   ``x`` and ``param`` are input, ``y`` and ``dyda`` are output (the
   computed function and all its derivatives with respect to the
   parameters, respectively). For the function in the example should
   look like:

   .. code-block:: c++

      void function(double x, double *param, double &y, double *dyda) {
        y = predict(x, param);
        dyda[0] = x;
        dyda[1] = 1;
      }

Other than these functions, the user should provide a constructor which
must initialize ``num_params`` to the number of parameters the model has
to learn. This number is the dimension of ``param`` and ``dyda`` in the
previous functions. For the given example, the constructor is
``{num_params = 2;}``.

If the model for computation is not known, the user can leave the system
to use the default function.

As seen, the function can have several parameters which will be learned
during the execution of the program. For this, user can be add the
following command line arguments to specify the learning behavior:

#. | *+LBPredictorWindow size*
   | This parameter specifies the number of statistics steps the load
     balancer will store. The greater this number is, the better the
     approximation of the workload will be, but more memory is required
     to store the intermediate information. The default is 20.

#. | *+LBPredictorDelay steps*
   | This will tell how many load balancer steps to wait before
     considering the function parameters learned and starting to use the
     mode. The load balancer will collect statistics for a
     *+LBPredictorWindow* steps, but it will start using the model as
     soon as *+LBPredictorDelay* information are collected. The default
     is 10.

| Moreover, another flag can be set to enable the predictor from command
  line: *+LBPredictor*.
| Other than the command line options, there are some methods which can
  be called from the user program to modify the predictor. These methods
  are:

-  ``void PredictorOn(LBPredictorFunction *model);``

-  ``void PredictorOn(LBPredictorFunction *model,int window);``

-  ``void PredictorOff();``

-  ``void ChangePredictor(LBPredictorFunction *model);``

An example can be found in
``tests/charm++/load_balancing/lb_test/predictor``.

Control CPU Load Statistics
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Charm++ programmers can modify the CPU load data in the load balancing
database before a load balancing phase starts (which is the time when
load balancing database is collected and used by load balancing
strategies).

In an array element, the following function can be invoked to overwrite
the CPU load that is measured by the load balancing framework.

.. code-block:: c++

      double newTiming;
      setObjTime(newTiming);

*setObjTime()* is defined as a method of class *CkMigratable*, which is
the superclass of all array elements.

The users can also retrieve the current timing that the load balancing
runtime has measured for the current array element using *getObjTime()*.

.. code-block:: c++

      double measuredTiming;
      measuredTiming = getObjTime();

This is useful when the users want to derive a new CPU load based on the
existing one.

Model-based Load Balancing
~~~~~~~~~~~~~~~~~~~~~~~~~~

The user can choose to feed load balancer with their own CPU timing for
each Chare based on certain computational model of the applications.

To do so, in the array element’s constructor, the user first needs to
turn off automatic CPU load measurement completely by setting

.. code-block:: c++

      usesAutoMeasure = false;

The user must also implement the following function to the chare array
classes:

.. code-block:: c++

      virtual void CkMigratable::UserSetLBLoad();      // defined in base class

This function serves as a callback that is called on each chare object
when *AtSync()* is called and ready to do load balancing. The
implementation of *UserSetLBLoad()* is simply to set the current chare
object’s CPU load in load balancing framework. *setObjTime()* described
above can be used for this.

.. _lbWriteNewLB:

Writing a new load balancing strategy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Charm++ programmers can choose an existing load balancing strategy from
Charm++’s built-in strategies(see :numref:`lbStrategy`) for the best
performance based on the characteristics of their applications. However,
they can also choose to write their own load balancing strategies.

The Charm++ load balancing framework provides a simple scheme to
incorporate new load balancing strategies. The programmer needs to write
their strategy for load balancing based on the instrumented ProcArray
and ObjGraph provided by the load balancing framework. This strategy is
implemented within this function:

.. code-block:: c++

   void FooLB::work(LDStats *stats) {
     /** ========================== INITIALIZATION ============================= */
     ProcArray *parr = new ProcArray(stats);
     ObjGraph *ogr = new ObjGraph(stats);

     /** ============================= STRATEGY ================================ */
     /// The strategy goes here
     /// The strategy goes here
     /// The strategy goes here
     /// The strategy goes here
     /// The strategy goes here

     /** ============================== CLEANUP ================================ */
     ogr->convertDecisions(stats);
   }

Figure :numref:`fig:ckgraph` explains the two data structures
available to the strategy: ProcArray and ObjGraph. Using them, the
strategy should assign objects to new processors where it wants to be
migrated through the setNewPe() method. ``src/ck-ldb/GreedyLB.C`` can be
referred.

.. figure:: fig/ckgraph.png
   :name: fig:ckgraph
   :width: 6in

   ProcArray and ObjGraph data structures to be used when writing a load
   balancing strategy

Incorporating this strategy into the Charm++ build framework is
explained in the next section.

Adding a load balancer to Charm++
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Let us assume that we are writing a new centralized load balancer called
FooLB. The next few steps explain the steps of adding the load balancer
to the Charm++ build system:

#. Create files named *FooLB.ci, FooLB.h and FooLB.C* in directory of
   ``src/ck-ldb``. One can choose to copy and rename the files
   GraphPartLB.\* and rename the class name in those files.

#. Implement the strategy in the *FooLB* class method —
   **FooLB::work(LDStats\* stats)** as described in the previous
   section.

#. Build charm for your platform (This will create the required links in
   the tmp directory).

#. To compile the strategy files, first add *FooLB* in the ALL_LDBS list
   in charm/tmp/Makefile_lb.sh. Also comment out the line containing
   UNCOMMON_LDBS in Makefile_lb.sh. If FooLB will require some libraries
   at link time, you also need to create the dependency file called
   libmoduleFooLB.dep. Run the script in charm/tmp, which creates the
   new Makefile named “Make.lb”.

#. Run ``make depends`` to update dependence rule of Charm++ files. And
   run ``make charm++`` to compile Charm++ which includes the new load
   balancing strategy files.

.. _lbdatabase:

Understand Load Balancing Database Data Structure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To write a load balancing strategy, you need to know what information is
measured during the runtime and how it is represented in the load
balancing database data structure.

There are mainly 3 categories of information: a) processor information
including processor speed, background load; b) object information
including per object CPU/WallClock compute time and c) communication
information .

The database data structure named LDStats is defined in *CentralLB.h*:

.. code-block:: c++


     struct ProcStats {  // per processor
       LBRealType total_walltime;
       LBRealType total_cputime;
       LBRealType idletime;
       LBRealType bg_walltime;
       LBRealType bg_cputime;
       int pe_speed;
       double utilization;
       bool available;
       int   n_objs;
     }

     struct LDStats { // load balancing database
       ProcStats  *procs;
       int count;

       int   n_objs;
       int   n_migrateobjs;
       LDObjData* objData;

       int   n_comm;
       LDCommData* commData;

       int  *from_proc, *to_proc;
     }

#. *LBRealType* is the data type for load balancer measured time. It is
   "double" by default. User can specify the type to float at Charm++
   compile time if want. For example, ./build charm++
   netlrts-linux-x86_64 --with-lbtime-type=float;

#. *procs* array defines processor attributes and usage data for each
   processor;

#. *objData* array records per object information, *LDObjData* is
   defined in *lbdb.h*;

#. *commData* array records per communication information. *LDCommData*
   is defined in *lbdb.h*.

.. _python:

Dynamic Code Injection
----------------------

The Python scripting language in Charm++ allows the user to dynamically
execute pieces of code inside a running application, without the need to
recompile. This is performed through the CCS (Converse Client Server)
framework (see :ref:`converse_client_server` for more information about this). The user specifies
which elements of the system will be accessible through the interface,
as we will see later, and then run a client which connects to the
server.

In order to exploit this functionality, Python interpreter needs to be
installed into the system, and Charm++ LIBS need to be built with:
``./build LIBS <arch> <options>``

The interface provides three different types of requests:

Execute
   requests to execute a code, it will contain the code to be executed
   on the server, together with the instructions on how to handle the
   environment;

Print
   asks the server to send back all the strings which have been printed
   by the script until now;

Finished
   asks the server if the current script has finished or it is still
   running.

There are three modes to run code on the server, ordered here by
increase of functionality, and decrease of dynamic flexibility:

-  **simple read/write** By implementing the read and write methods of
   the object exposed to python, in this way single variables may be
   exposed, and the code will have the possibility to modify them
   individually as desired. (see section :numref:`pythonServerRW`)

-  **iteration** By implementing the iterator functions in the server
   (see :numref:`pythonServerIterator`), the user can upload the code
   of a Python function and a user-defined iterator structure, and the
   system will apply the specified function to all the objects reflected
   by the iterator structure.

-  **high level** By implementing python entry methods, the Python code
   uploaded can access them and activate complex, parallel operations
   that will be performed by the Charm++ application. (see
   section :numref:`pythonHighLevel`)

This documentation will describe the client API first, and then the
server API.

.. _pythonClient:

Client API
~~~~~~~~~~

In order to facilitate the interface between the client and the server,
some classes are available to the user to include into the client.
Currently C++ and java interfaces are provided.

C++ programs need to include ``PythonCCS-client.h`` into their code. This
file is among the Charm++ include files. For java, the package
``charm.ccs`` needs to be imported. This is located under the java
directory on the Charm++ distribution, and it provides both the Python
and CCS interface classes.

There are three main classes provided: ``PythonExecute``,
``PythonPrint``, and ``PythonFinished`` which are used for the three
different types of request.

All of them have two common methods to enable communication across
different platforms:

int size();
   Returns the size of the class, as number of bytes that will be
   transmitted through the network (this includes the code and other
   dynamic variables in the case of ``PythonExecute``).

char \*pack();
   Returns a new memory location containing the data to be sent to the
   server, this is the data which has to be passed to the
   ``CcsSendRequest`` function. The original class will be unmodified
   and can be reused in subsequent calls.

A typical invocation to send a request from the client to the server has
the following format:

.. code-block:: c++

   CcsSendRequest (&server, "pyCode", 0, request.size(), request.pack());

.. _pythonExecute:

PythonExecute
~~~~~~~~~~~~~

To execute a Python script on a running server, the client has to create
an instance of ``PythonExecute``, the two constructors have the
following signature (java has a corresponding functionality):

.. code-block:: c++

   PythonExecute(char *code, bool persistent=false, bool highlevel=false, CmiUInt4 interpreter=0);
   PythonExecute(char *code, char *method, PythonIterator *info, bool persistent=false,
                 bool highlevel=false, CmiUInt4 interpreter=0);

The second one is used for iterative requests
(see :numref:`pythonIterator`). The only required argument is the code,
a null terminated string, which will not be modified by the system. All
the other parameters are optional. They refer to the possible variants
for an execution request. In particular, this is a list of all the
options:

iterative
   If the request is a single code (false) or if it represents a
   function over which to iterate (true) (see :numref:`pythonIterator`
   for more details).

persistent
   It is possible to store information on the server which will be
   retained across different client calls (from simple data all the way
   up to complete libraries). True means that the information will be
   retained on the server, false means that the information will be
   deleted when the script terminates. In order to properly release the
   memory, when the last call is made (and the data is no longer
   required), this flag should be set to false. To reuse persistent
   data, the interpreter field of the request should be set to handle
   returned by a previous persistent call (see later in this
   subsection).

high level
   In order to have the ability to call high level Charm++ functions
   (available through the keyword python) this flag must be set to true.
   If it is false, the entire module “charm” will not be present, but
   the startup of the script will be faster.

print retain
   When the requested action triggers printed output for the client,
   this data can be retrieved with a PythonPrint request. If the output
   is not desired, this flag can be set to false, and the output will be
   discarded. If it is set to true the output will be buffered pending
   retrieval by the client. The data will survive also after the
   termination of the Python script, and if not retrieved will bloat
   memory usage on the server.

busy waiting
   Instead of returning a handle immediately to the client, that can be
   used to retrieve prints and check if the script has finished, the
   server will answer to the client only when the script has terminated
   to run (and it will effectively work as a PythonFinished request).

These flags can be set and checked with the following routines (CmiUInt4
represent a 4 byte unsigned integer):

.. code-block:: c++

   void setCode(char *set);
   void setPersistent(bool set);
   void setIterate(bool set);
   void setHighLevel(bool set);
   void setKeepPrint(bool set);
   void setWait(bool set);
   void setInterpreter(CmiUInt4 i);

   bool isPersistent();
   bool isIterate();
   bool isHighLevel();
   bool isKeepPrint();
   bool isWait();
   CmiUInt4 getInterpreter();

From a PythonExecute request, the server will answer with a 4 byte
integer value, which is a handle for the interpreter that is running. It
can be used to request for prints, check if the script has finished, and
for reusing the same interpreter (if it was persistent).

A value of 0 means that there was an error and the script didn’t run.
This is typically due to a request to reuse an existing interpreter
which is not available, either because it was not persistent or because
another script is still running on that interpreter.

.. _pythonModules:

Auto-imported modules
~~~~~~~~~~~~~~~~~~~~~

When a Python script is run inside a Charm++ application, two Python
modules are made available by the system. One is **ck**, the other is
**charm**. The first one is always present and it represent basic
functions, the second is related to high level scripting and it is
present only when this is enabled (see :numref:`pythonExecute` for how
to enable it, and :numref:`pythonHighLevel` for a description on how
to implement charm functions).

The methods present in the ``ck`` module are the following:

printstr
   It accepts a string as parameter. It will write into the server
   stdout that string using the ``CkPrintf`` function call.

printclient
   It accepts a string as parameter. It will forward the string back to
   the client when it issues a PythonPrint request. It will buffer the
   strings until requested by PythonPrint if the ``KeepPrint`` option is
   true, otherwise it will discard them.

mype
   Requires no parameters, and will return an integer representing the
   current processor where the code is executing. It is equivalent to
   the Charm++ function ``CkMyPe()``.

numpes
   Requires no parameters, and will return an integer representing the
   total number of processors that the application is using. It is
   equivalent to the Charm++ function ``CkNumPes()``.

myindex
   Requires no parameters, and will return the index of the current
   element inside the array, if the object under which Python is running
   is an array, or None if it is running under a Chare, a Group or a
   Nodegroup. The index will be a tuple containing as many numbers as
   the dimension of the array.

read
   It accepts one object parameter, and it will perform a read request
   to the Charm++ object connected to the Python script, and return an
   object containing the data read (see :numref:`pythonServerRW` for a
   description of this functionality). An example of a call can be:
   *value = ck.read((number, param, var2, var3))* where the double
   parenthesis are needed to create a single tuple object containing
   four values passed as a single paramter, instead of four different
   parameters.

write
   It accepts two object parameters, and it will perform a write request
   to the Charm++ object connected to the Python script. For a
   description of this method, see :numref:`pythonServerRW`. Again,
   only two objects need to be passed, so extra parenthesis may be
   needed to create tuples from individual values.

.. _pythonIterator:

Iterate mode
~~~~~~~~~~~~

Sometimes some operations need to be iterated over all the elements in
the system. This “iterative” functionality provides a shortcut for the
client user to do this. As an example, suppose we have a system which
contains particles, with their position, velocity and mass. If we
implement ``read`` and ``write`` routines which allow us to access
single particle attributes, we may upload a script which doubles the
mass of the particles with velocity greater than 1:

.. code-block:: python

   size = ck.read((``numparticles'', 0));
   for i in range(0, size):
       vel = ck.read((``velocity'', i));
       mass = ck.read((``mass'', i));
       mass = mass * 2;
       if (vel > 1): ck.write((``mass'', i), mass);

Instead of all these read and writes, it will be better to be able to
write:

.. code-block:: python

   def increase(p):
       if (p.velocity > 1): p.mass = p.mass * 2;

This is what the “iterative” functionality provides. In order for this
to work, the server has to implement two additional functions
(see :numref:`pythonServerIterator`), and the client has to pass some
more information together with the code. This information is the name of
the function that has to be called (which can be defined in the “code”
or was previously uploaded to a persistent interpreter), and a user
defined structure which specifies over what data the function should be
invoked. These values can be specified either while constructing the
PythonExecute variable (see the second constructor in
section :numref:`pythonExecute`), or with the following methods:

.. code-block:: c++

   void setMethodName(char *name);
   void setIterator(PythonIterator *iter);

The ``PythonIterator`` object must be defined by the user, and the user
must insure that the same definition is present inside both the client
and the server. The Charm++ system will simply pass this structure as a
void pointer. This structure must inherit from ``PythonIterator``. In
the simple case (highly recommended), wherein no pointers or dynamic
allocation are used inside this class, nothing else needs to be done
because it is trivial to serialize such objects.

If instead pointers or dynamic memory allocation are used, the following
methods have to be reimplemented to support correct serialization:

.. code-block:: c++

   int size();
   char * pack();
   void unpack();

The first returns the size of the class/structure after being packed.
The second returns a pointer to a newly allocated memory containing all
the packed data, the returned memory must be compatible with the class
itself, since later on this same memory a call to unpack will be
performed. Finally, the third will do the work opposite to pack and fix
all the pointers. This method will not return anything and is supposed
to fix the pointers “inline”.

.. _pythonPrint:

PythonPrint
~~~~~~~~~~~

In order to receive the output printed by the Python script, the client
needs to send a PythonPrint request to the server. The constructor is:

*PythonPrint(CmiUInt4 interpreter, bool Wait=true, bool Kill=false);*

The interpreter for which the request is made is mandatory. The other
parameters are optional. The wait parameter represents whether a reply
will be sent back immediately to the client even if there is no output
(false), or if the answer will be delayed until there is an output
(true). The kill option set to true means that this is not a normal
request, but a signal to unblock the latest print request which was
blocking.

The returned data will be a non null-terminated string if some data is
present (or if the request is blocking), or a 4 byte zero data if
nothing is present. This zero reply can happen in different situations:

-  If the request is non blocking and no data is available on the
   server;

-  If a kill request is sent, the previous blocking request is squashed;

-  If the Python code ends without any output and it is not persistent;

-  If another print request arrives, the previous one is squashed and
   the second one is kept.

As for a print kill request, no data is expected to come back, so it is
safe to call ``CcsNoResponse(server)``.

The two options can also be dynamically set with the following methods:

.. code-block:: c++

   void setWait(bool set);
   bool isWait();

   void setKill(bool set);
   bool isKill();

.. _pythonFinished:

PythonFinished
~~~~~~~~~~~~~~

In order to know when a Python code has finished executing, especially
when using persistent interpreters, and a serialization of the scripts
is needed, a PythonFinished request is available. The constructor is the
following:

*PythonFinished(CmiUInt4 interpreter, bool Wait=true);*

The interpreter corresponds to the handle for which the request was
sent, while the wait option refers to a blocking call (true), or
immediate return (false).

The wait option can be dynamically modified with the two methods:

.. code-block:: c++

   void setWait(bool set);
   bool isWait();

This request will return a 4 byte integer containing the same
interpreter value if the Python script has already finished, or zero if
the script is still running.

.. _pythonServer:

Server API
~~~~~~~~~~

In order for a Charm++ object (chare, array, node, or nodegroup) to
receive python requests, it is necessary to define it as
python-compliant. This is done through the keyword python placed in
square brackets before the object name in the .ci file. Some examples
follow:

.. code-block:: c++

   mainchare [python] main {...}
   array [1D] [python] myArray {...}
   group [python] myGroup {...}

In order to register a newly created object to receive Python scripts,
the method ``registerPython`` of the proxy should be called. As an
example, the following code creates a 10 element array myArray, and then
registers it to receive scripts directed to “pycode”. The argument of
``registerPython`` is the string that CCS will use to address the Python
scripting capability of the object.

.. code-block:: c++

   Cproxy_myArray localVar = CProxy_myArray::ckNew(10);
   localVar.registerPython("pycode");

.. _pythonServerRW:

Server read and write functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

As explained previously in subsection :numref:`pythonModules`, some
functions are automatically made available to the scripting code through
the *ck* module. Two of these, **read** and **write** are only available
if redefined by the object. The signatures of the two methods to
redefine are:

.. code-block:: c++

   PyObject* read(PyObject* where);
   void write(PyObject* where, PyObject* what);

The read function receives as a parameter an object specifying from
where the data will be read, and returns an object with the information
required. The write function will receive two parameters: where the data
will be written and what data, and will perform the update. All these
``PyObject``\ s are generic, and need to be coherent with the protocol
specified by the application. In order to parse the parameters, and
create the value of the read, please refer to the manual `Extending and Embedding the Python Interpreter <http://docs.python.org/>`__, and in
particular to the functions ``PyArg_ParseTuple`` and ``Py_BuildValue``.

.. _pythonServerIterator:

Server iterator functions
~~~~~~~~~~~~~~~~~~~~~~~~~

In order to use the iterative mode as explained in
subsection :numref:`pythonIterator`, it is necessary to implement two
functions which will be called by the system. These two functions have
the following signatures:

.. code-block:: c++

   int buildIterator(PyObject*, void*);
   int nextIteratorUpdate(PyObject*, PyObject*, void*);

The first one is called once before the first execution of the Python
code, and receives two parameters. The first is a pointer to an empty
PyObject to be filled with the data needed by the Python code. In order
to manage this object, some utility functions are provided. They are
explained in subsection :numref:`pythonUtilityFuncs`.

The second is a void pointer containing information of what the
iteration should run over. This parameter may contain any data
structure, and an agreement between the client and the user object is
necessary. The system treats it as a void pointer since it has no
information about what user defined data it contains.

The second function (``nextIteratorUpdate``) has three parameters. The
first parameter contains the object to be filled (similar to
``buildIterator``), but the second object contains the PyObject which
was provided for the last iteration, potentially modified by the Python
function. Its content can be read with the provided routines, used to
retrieve the next logical element in the iterator (with which to update
the parameter itself), and possibly update the content of the data
inside the Charm++ object. The second parameter is the object returned
by the last call to the Python function, and the third parameter is the
same data structure passed to ``buildIterator``.

Both functions return an integer which will be interpreted by the system
as follows:

1
   - a new iterator in the first parameter has been provided, and the
     Python function should be called with it;

0
   - there are no more elements to iterate.

.. _pythonUtilityFuncs:

Server utility functions
~~~~~~~~~~~~~~~~~~~~~~~~

These are inherited when declaring an object as Python-compliant, and
therefore they are available inside the object code. All of them accept
a PyObject pointer where to read/write the data, a string with the name
of a field, and one or two values containing the data to be read/written
(note that to read the data from the PyObject, a pointer needs to be
passed). The strings used to identify the fields will be the same
strings that the Python script will use to access the data inside the
object.

The name of the function identifies the type of Python object stored
inside the PyObject container (i.e String, Int, Long, Float, Complex),
while the parameter of the functions identifies the C++ object type.

.. code-block:: c++

   void pythonSetString(PyObject*, char*, char*);
   void pythonSetString(PyObject*, char*, char*, int);
   void pythonSetInt(PyObject*, char*, long);
   void pythonSetLong(PyObject*, char*, long);
   void pythonSetLong(PyObject*, char*, unsigned long);
   void pythonSetLong(PyObject*, char*, double);
   void pythonSetFloat(PyObject*, char*, double);
   void pythonSetComplex(PyObject*, char*, double, double);

   void pythonGetString(PyObject*, char*, char**);
   void pythonGetInt(PyObject*, char*, long*);
   void pythonGetLong(PyObject*, char*, long*);
   void pythonGetLong(PyObject*, char*, unsigned long*);
   void pythonGetLong(PyObject*, char*, double*);
   void pythonGetFloat(PyObject*, char*, double*);
   void pythonGetComplex(PyObject*, char*, double*, double*);

To handle more complicated structures like Dictionaries, Lists or
Tuples, please refer to `Python/C API Reference Manual <http://docs.python.org/>`__.

.. _pythonHighLevel:

High level scripting
~~~~~~~~~~~~~~~~~~~~

When in addition to the definition of the Charm++ object as python, an
entry method is also defined as python, this entry method can be
accessed directly by a Python script through the *charm* module. For
example, the following definition will be accessible with the python
call: *result = charm.highMethod(var1, var2, var3)* It can accept any
number of parameters (even complex like tuples or dictionaries), and it
can return an object as complex as needed.

The method must have the following signature:

.. code-block:: c++

   entry [python] void highMethod(int handle);

The parameter is a handle that is passed by the system, and can be used
in subsequent calls to return values to the Python code.

The arguments passed by the Python caller can be retrieved using the
function:

*PyObject \*pythonGetArg(int handle);*

which returns a PyObject. This object is a Tuple containing a vector of
all parameters. It can be parsed using ``PyArg_ParseTuple`` to extract
the single parameters.

When the Charm++’s entry method terminates (by means of ``return`` or
termination of the function), control is returned to the waiting Python
script. Since the python entry methods execute within an user-level
thread, it is possible to suspend the entry method while some
computation is carried on in Charm++. To start parallel computation, the
entry method can send regular messages, as every other threaded entry
method (see :numref:`libraryInterface` for more information on how this
can be done using CkCallbackResumeThread callbacks). The only difference
with other threaded entry methods is that here the callback
``CkCallbackPython`` must be used instead of CkCallbackResumeThread. The
more specialized CkCallbackPython callback works exactly like the other
one, except that it correctly handles Python internal locks.

At the end of the computation, the following special function will
return a value to the Python script:

*void pythonReturn(int handle, PyObject\* result);*

where the second parameter is the Python object representing the
returned value. The function ``Py_BuildValue`` can be used to create
this value. This function in itself does not terminate the entry method,
but only sets the returning value for Python to read when the entry
method terminates.

A characteristic of Python is that in a multithreaded environment (like
the one provided in Charm++), the running thread needs to keep a lock to
prevent other threads to access any variable. When using high level
scripting, and the Python script is suspended for long periods of time
while waiting for the Charm++ application to perform the required task,
the Python internal locks are automatically released and re-acquired by
the ``CkCallbackPython`` class when it suspends.

.. _delegation:

Intercepting Messages via Delegation
------------------------------------

*Delegation* is a means by which a library writer can intercept messages
sent via a proxy. This is typically used to construct communication
libraries. A library creates a special kind of Group called a
*DelegationManager*, which receives the messages sent via a delegated
proxy.

There are two parts to the delegation interface- a very small
client-side interface to enable delegation, and a more complex
manager-side interface to handle the resulting redirected messages.

Client Interface
~~~~~~~~~~~~~~~~

All proxies (Chare, Group, Array, ...) in Charm++ support the following
delegation routines.

``void CProxy::ckDelegate(CkGroupID delMgr);`` Begin delegating messages
sent via this proxy to the given delegation manager. This only affects
the proxy it is called on- other proxies for the same object are *not*
changed. If the proxy is already delegated, this call changes the
delegation manager.

``CkGroupID CProxy::ckDelegatedIdx(void) const;`` Get this proxy’s current
delegation manager.

``void CProxy::ckUndelegate(void);`` Stop delegating messages sent via this
proxy. This restores the proxy to normal operation.

One use of these routines might be:

.. code-block:: c++

     CkGroupID mgr=somebodyElsesCommLib(...);
     CProxy_foo p=...;
     p.someEntry1(...); //Sent to foo normally
     p.ckDelegate(mgr);
     p.someEntry2(...); //Handled by mgr, not foo!
     p.someEntry3(...); //Handled by mgr again
     p.ckUndelegate();
     p.someEntry4(...); //Back to foo

The client interface is very simple; but it is often not called by users
directly. Often the delegate manager library needs some other
initialization, so a more typical use would be:

.. code-block:: c++

     CProxy_foo p=...;
     p.someEntry1(...); //Sent to foo normally
     startCommLib(p,...); // Calls ckDelegate on proxy
     p.someEntry2(...); //Handled by library, not foo!
     p.someEntry3(...); //Handled by library again
     finishCommLib(p,...); // Calls ckUndelegate on proxy
     p.someEntry4(...); //Back to foo

Sync entry methods, group and nodegroup multicast messages, and messages
for virtual chares that have not yet been created are never delegated.
Instead, these kinds of entry methods execute as usual, even if the
proxy is delegated.

Manager Interface
~~~~~~~~~~~~~~~~~

A delegation manager is a group which inherits from *CkDelegateMgr* and
overrides certain virtual methods. Since *CkDelegateMgr* does not do any
communication itself, it need not be mentioned in the .ci file; you can
simply declare a group as usual and inherit the C++ implementation from
*CkDelegateMgr*.

Your delegation manager will be called by Charm++ any time a proxy
delegated to it is used. Since any kind of proxy can be delegated, there
are separate virtual methods for delegated Chares, Groups, NodeGroups,
and Arrays.

.. code-block:: c++

   class CkDelegateMgr : public Group {
   public:
     virtual void ChareSend(int ep,void *m,const CkChareID *c,int onPE);

     virtual void GroupSend(int ep,void *m,int onPE,CkGroupID g);
     virtual void GroupBroadcast(int ep,void *m,CkGroupID g);

     virtual void NodeGroupSend(int ep,void *m,int onNode,CkNodeGroupID g);
     virtual void NodeGroupBroadcast(int ep,void *m,CkNodeGroupID g);

     virtual void ArrayCreate(int ep,void *m,const CkArrayIndex &idx,int onPE,CkArrayID a);
     virtual void ArraySend(int ep,void *m,const CkArrayIndex &idx,CkArrayID a);
     virtual void ArrayBroadcast(int ep,void *m,CkArrayID a);
     virtual void ArraySectionSend(int ep,void *m,CkArrayID a,CkSectionID &s);
   };

These routines are called on the send side only. They are called after
parameter marshalling; but before the messages are packed. The
parameters passed in have the following descriptions.

#. **ep** The entry point begin called, passed as an index into the Charm++
   entry table. This information is also stored in the message’s header;
   it is duplicated here for convenience.

#. **m** The Charm++ message. This is a pointer to the start of the user
   data; use the system routine UsrToEnv to get the corresponding
   envelope. The messages are not necessarily packed; be sure to use
   CkPackMessage.

#. **c** The destination CkChareID. This information is already stored in
   the message header.

#. **onPE** The destination processor number. For chare messages, this
   indicates the processor the chare lives on. For group messages, this
   indicates the destination processor. For array create messages, this
   indicates the desired processor.

#. **g** The destination CkGroupID. This is also stored in the message
   header.

#. **onNode** The destination node.

#. **idx** The destination array index. This may be looked up using the
   lastKnown method of the array manager, e.g., using:

   .. code-block:: c++

     int lastPE=CProxy_CkArray(a).ckLocalBranch()->lastKnown(idx);

#. **s** The destination array section.

The *CkDelegateMgr* superclass implements all these methods; so you only
need to implement those you wish to optimize. You can also call the
superclass to do the final delivery after you’ve sent your messages.
