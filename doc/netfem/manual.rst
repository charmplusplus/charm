=====================
Charm++ NetFEM Manual
=====================

.. contents::
   :depth: 3

Introduction
============

NetFEM was built to provide an easy way to visualize the current state
of a finite-element simulation, or any parallel program that computes on
an unstructured mesh. NetFEM is designed to require very little effort
to add to a program, and connects to the running program over the
network via the network protocol CCS (Converse Client/Server).

Compiling and Installing
========================

NetFEM is part of Charm++, so it can be downloaded as part of charm. To
build NetFEM, just build FEM normally, or else do a make in
charm/netlrts-linux/tmp/libs/ck-libs/netfem/.

To link with NetFEM, add -module netfem to your program’s link line.
Note that you do *not* need to use the FEM framework to use NetFEM.

The netfem header file for C is called “netfem.h”, the header for
fortran is called ‘netfemf.h’. A simple example NetFEM program is in
charm/pgms/charm++/fem/simple2D/. A more complicated example is in
charm/pgms/charm++/fem/crack2D/.

Running NetFEM Online
=====================

Once you have a NetFEM program, you can run it and view the results
online by starting the program with CCS enabled:

.. code-block:: bash

      foo.bar.edu>  ./charmrun ./myprogram +p2 ++server ++server-port 1234

“++server-port” controls the TCP port number to use for CCS—here, we use
1234. Currently, NetFEM only works with one chunk per processor—that is,
the -vp option cannot be used.

To view the results online, you then start the NetFEM client, which can
be downloaded for Linux or Windows from http://charm.cs.illinois.edu/research/fem/netfem/.

Enter the name of the machine running charmrun and the TCP port number
into the NetFEM client—for example, you might run:

.. code-block:: bash

     netfem foo.bar.edu:1234

The NetFEM client will then connect to the program, download the most
recent mesh registered with NetFEM_POINTAT, and display it. At any time,
you can press the “update” button to reload the latest mesh.

Running NetFEM Offline
======================

Rather than using CCS as above, you can register your meshes using
NetFEM_WRITE, which makes the server write out binary output dump files.
For example, to view timestep 10, which is written to the “NetFEM/10/“
directory, you’d run the client program as:

.. code-block:: bash

     netfem NetFEM/10

In offline mode, the “update” button fetches the next extant timestep
directory.

NetFEM with other Visualization Tools
=====================================

You can use a provided converter program to convert the offline NetFEM
files into an XML format compatible with the powerful offline
visualization tool ParaView (http://paraview.org). The converter is
located in .../charm/src/libs/ck-libs/netfem/ParaviewConverter/. Build
the converter by simply issuing a “make” command in that
directory(assuming NetFEM already has been built).

Run the converter from the parent directory of the "NetFEM" directory to
be converted. The converter will generate a directory called
“ParaViewData”, which contains subdirectories for each timestep, along
with a “timestep”directory for index files for each timestep. All files
in the ParaViewData directory can be opened by ParaView. To open all
chunks for a given timestep, open the desired timestep file in
“ParaViewData/timesteps”. Also, individual partition files can also be
opened from “ParaViewData / :math:`<`\ timestep\ :math:`>` /
:math:`<`\ partition_num\ :math:`>`”.

Interface Basics
================

You publish your data via NetFEM by making a series of calls to describe
the current state of your data. There are only 6 possible calls you can
make.

NetFEM_Begin is the first routine you call. NetFEM_End is the last
routine to call. These two calls bracket all the other NetFEM calls.

NetFEM_Nodes describes the properties of the nodes, or vertices of the
domain. NetFEM_Elements describes the properties of your elements
(triangles, tetrahedra, etc.). After making one of these calls, you list
the different data arrays associated with your nodes or elements by
making calls to NetFEM_Scalar or NetFEM_Vector.

For example, a typical finite element simulation might have a scalar
mass and vector position, velocity, and net force associated with each
node; and have a scalar stress value associated with each element. The
sequence of NetFEM calls this application would make would be:

.. code-block:: none

     NetFEM_Begin
       NetFEM_Nodes -- lists position of each node
         NetFEM_Vector -- lists velocity of each node
         NetFEM_Vector -- lists net force on each node
         NetFEM_Scalar -- lists mass of each node

       NetFEM_Elements -- lists the nodes of each element
         NetFEM_Scalar -- lists the stress of each element

     NetFEM_End

.. figure:: fig/example.pdf
   :name: fig:example
   :width: 5in

   These arrays, typical of a finite element analysis program, might be
   passed into NetFEM.

Simple Interface
================

The details of how to make each call are:

| NetFEM NetFEM_Begin(int source, int step, int dim, int flavor);
  integer function NetFEM_Begin(source,step,dim,flavor)
| Begins describing a single piece of a mesh. Returns a handle that is
  used for each subsequent call until NetFEM_End. This call, like all
  NetFEM calls, is collective—every processor should make the same calls
  in the same order.

source identifies the piece of the mesh—use FEM_My_partition or CkMyPe.

step identifies which version of the mesh this is—for example, you might
use the timestep number. This is only used to identify the mesh in the
client.

dim is the number of spatial dimensions. For example, in a 2D
computation, you’d pass dim==2; in a 3D computation, dim==3. The client
currently only supports 2D or 3D computations.

flavor specifies what to do with the data. This can take the value
NetFEM_POINTAT, which is used in online visualization, and specifies
that NetFEM should only keep a pointer to your data rather than copy it
out of your arrays. Or it can take the value NetFEM_WRITE, which writes
out the data to files named “NetFEM/step/source.dat” for offline
visualization.

| void NetFEM_End(NetFEM n); subroutine NetFEM_End(n)
| Finishes describing a single piece of a mesh, which then makes the
  mesh available for display.

| void NetFEM_Nodes(NetFEM n,int nNodes,const double \*loc,const char
  \*name); subroutine NetFEM_Nodes(n,nNodes,loc,name)
| Describes the nodes in this piece of the mesh.

n is the NetFEM handle obtained from NetFEM_Begin.

nNodes is the number of nodes listed here.

loc is the location of each node. This must be double-precision array,
laid out with the same number of dimensions as passed to NetFEM_Begin.
For example, in C the location of a 2D node :math:`n` is stored in
loc[2*n+0] (x coordinate) and loc[2*n+1] (y coordinate). In Fortran,
location of a node :math:`n` is stored in loc(:,n).

name is a human-readable name for the node locations to display in the
client. We recommend also including the location units here, for example
"Position (m)".

| void NetFEM_Elements(NetFEM n,int nElements,int nodePerEl,const int
  \*conn,const char \*name); subroutine
  NetFEM_Elements(n,nElements,nodePerEl,conn,name)
| Describes the elements in this piece of the mesh. Unlike NetFEM_Nodes,
  this call can be repeated if there are different types of elements
  (For example, some meshes contain a mix of triangles and
  quadrilaterals).

n is the NetFEM handle obtained from NetFEM_Begin.

nElements is the number of elements listed here.

nodePerEl is the number of nodes for each element. For example, a
triangle has 3 nodes per element; while tetrahedra have 4.

conn gives the index of each element’s nodes. Note that when called from
C, the first node is listed in conn as 0 (0-based node indexing), and
element :math:`e`\ ’s first node is stored in conn[e*nodePerEl+0]. When
called from Fortran, the first node is listed as 1 (1-based node
indexing), and element :math:`e`\ ’s first node is stored in conn(1,e)
or conn((e-1)*nodePerEl+1).

name is a human-readable name for the elements to display in the client.
For example, this might be "Linear-Strain Triangles".

| void NetFEM_Vector(NetFEM n,const double \*data,const char \*name);
  subroutine NetFEM_Vector(n,data,name)
| Describes a spatial vector associated with each node or element in the
  mesh. Attaches the vector to the most recently listed node or element.
  You can repeat this call several times to describe different vectors.

n is the NetFEM handle obtained from NetFEM_Begin.

data is the double-precision array of vector values. The dimensions of
the array have to match up with the node or element the data is
associated with-in C, a 2D element :math:`e`\ ’s vector starts at
data[2*e]; in Fortran, element :math:`e`\ ’s vector is data(:,e).

name is a human-readable name for this vector data. For example, this
might be "Velocity (m/s)".

| void NetFEM_Scalar(NetFEM n,const double \*data,int dataPer,const char
  \*name); subroutine NetFEM_Scalar(n,data,dataPer,name)
| Describes some scalar data associated with each node or element in the
  mesh. Like NetFEM_Vector, this data is attached to the most recently
  listed node or element and this call can be repeated. For a node or
  element, you can make the calls to NetFEM_Vector and NetFEM_Scalar in
  any order.

n is the NetFEM handle obtained from NetFEM_Begin.

data is the double-precision array of values. In C, an element
:math:`e`\ ’s scalar values start at data[dataPer*e]; in Fortran,
element :math:`e`\ ’s values are in data(:,e).

dataPer is the number of values associated with each node or element.
For true scalar data, this is 1; but can be any value. Even if dataPer
happens to equal the number of dimensions, the client knows that this
data does not represent a spatial vector.

name is a human-readable name for this scalar data. For example, this
might be "Mass (Kg)" or "Stresses (pure)".

Advanced “Field” Interface
==========================

This more advanced interface can be used if you store your node or
element data in arrays of C structs or Fortran TYPEs. To use this
interface, you’ll have to provide the name of your struct and field.
Each “field” routine is just an extended version of a regular NetFEM
call described above, and can be used in place of the regular NetFEM
call. In each case, you pass a description of your field in addition to
the usual NetFEM parameters.

In C, use the macro “NetFEM_Field(theStruct,theField)” to describe the
FIELD. For example, to describe the field “loc” of your structure named
“node_t”,

::

      node_t *myNodes=...;
      ..., NetFEM_Field(node_t,loc), ...

In Fortran, you must pass as FIELD the byte offset from the start of the
structure to the start of the field, then the size of the structure. The
FEM "foffsetof" routine, which returns the number of bytes between its
arguments, can be used for this. For example, to describe the field
“loc” of your named type “NODE”,

.. code-block:: fortran

      TYPE(NODE), ALLOCATABLE :: n(:)
      ..., foffsetof(n(1),n(1)%loc),foffsetof(n(1),n(2)), ...

| void NetFEM_Nodes_field(NetFEM n,int nNodes,FIELD,const void \*loc,const
  char \*name);
| subroutine NetFEM_Nodes_field(n,nNodes,FIELD,loc,name)
| A FIELD version of NetFEM_Nodes.

| void NetFEM_Elements_field(NetFEM n,int nElements,int
  nodePerEl,FIELD,int idxBase,const int \*conn,const char \*name);
| subroutine NetFEM_Elements_field(n,nElements,nodePerEl,FIELD,idxBase,conn,name)
| A FIELD version of NetFEM_Elements. This version also allows you to
  control the starting node index of the connectivity array—in C, this is
  normally 0; in Fortran, this is normally 1.

| void NetFEM_Vector_field(NetFEM n,const double \*data,FIELD,const char
  \*name);
| subroutine NetFEM_Vector_field(n,data,FIELD,name)
| A FIELD version of NetFEM_Vector.

| void NetFEM_Scalar_field(NetFEM n,const double \*data,int
  dataPer,FIELD,const char \*name); subroutine
  NetFEM_Scalar(n,data,dataPer,FIELD,name)
| A FIELD version of NetFEM_Scalar.
