=============================================================
Charm++ Iterative Finite Element Matrix (IFEM) Library Manual
=============================================================

.. contents::
   :depth: 3

Introduction
============

This manual presents the Iterative Finite Element Matrix (IFEM) library,
a library for easily solving matrix problems derived from finite-element
formulations. The library is designed to be matrix-free, in that the
only matrix operation required is matrix-vector product, and hence the
entire matrix need never be assembled

IFEM is built on the mesh and communication capabilities of the Charm++
FEM Framework, so for details on the basic runtime, problem setup, and
partitioning see the FEM Framework manual.

Terminology
-----------

A FEM program manipulates elements and nodes. An **element** is a
portion of the problem domain, also known as a cell, and is typically
some simple shape like a triangle, square, or hexagon in 2D; or
tetrahedron or rectangular solid in 3D. A **node** is a point in the
domain, and is often the vertex of several elements. Together, the
elements and nodes form a **mesh**, which is the central data structure
in the FEM framework. See the FEM manual for details.

.. _sec:solver:

Solvers
=======

A IFEM **solver** is a subroutine that controls the search for the
solution.

Solvers often take extra parameters, which are listed in a type called
in C ``ILSI_Param``, which in Fortran is an array of ``ILSI_PARAM`` doubles. You
initialize these solver parameters using the subroutine ``ILSI_Param_new``,
which takes the parameters as its only argument. The input and output
parameters in an ``ILSI_Param`` are listed in
Table :numref:`table:solverIn` and
Table :numref:`table:solverOut`.

.. table:: ``ILSI_Param`` solver input parameters.
   :name: table:solverIn

   ============= ==================== ========================================================
   C Field Name  Fortran Field Offset Use
   ============= ==================== ========================================================
   maxResidual   1                    If nonzero, termination criteria: magnitude of residual.
   maxIterations 2                    If nonzero, termination criteria: number of iterations.
   solverIn[8]   3-10                 Solver-specific input parameters.
   ============= ==================== ========================================================

.. table:: ``ILSI_Param`` solver output parameters.
   :name: table:solverOut

   ============ ==================== ========================================
   C Field Name Fortran Field Offset Use
   ============ ==================== ========================================
   residual     11                   Magnitude of residual of final solution.
   iterations   12                   Number of iterations actually taken.
   solverOut[8] 13-20                Solver-specific output parameters.
   ============ ==================== ========================================

Conjugate Gradient Solver
-------------------------

The only solver currently written using IFEM is the conjugate gradient
solver. This linear solver requires the matrix to be real, symmetric and
positive definite.

Each iteration of the conjugate gradient solver requires one
matrix-vector product and two global dot products. For well-conditioned
problems, the solver typically converges in some small multiple of the
diameter of the mesh-the number of elements along the largest side of
the mesh.

You access the conjugate gradient solver via the subroutine name
``ILSI_CG_Solver``.

Solving Shared-Node Systems
===========================

Many problems encountered in FEM analysis place the entries of the known
and unknown vectors at the nodes-the vertices of the domain. Elements
provide linear relationships between the known and unknown node values,
and the entire matrix expresses the combination of all these element
relations.

For example, in a structural statics problem, we know the net force at
each node, :math:`f`, and seek the displacements of each node,
:math:`u`. Elements provide the relationship, often called a stiffness
matrix :math:`K`, between a nodes’ displacements and its net forces:

.. math:: f=K u

We normally label the known vector :math:`b` (in the example, the
forces), the unknown vector :math:`x` (in the example, the
displacements), and the matrix :math:`A`:

.. math:: b=A x

IFEM provides two routines for solving problems of this type. The first
routine, ``IFEM_Solve_shared``, solves for the entire :math:`x` vector based
on the known values of the :math:`b` vector. The second,
``IFEM_Solve_shared_bc``, allows certain entries in the :math:`x` vector to
be given specific values before the problem is solved, creating values
for the :math:`b` vector.

IFEM_Solve_shared
-----------------

::

  void IFEM_Solve_shared(ILSI_Solver s, ILSI_Param *p, int fem_mesh, int
    fem_entity, int length, int width, IFEM_Matrix_product_c A, void *ptr,
    const double *b, double *x);

.. code-block:: fortran

  subroutine IFEM_Solve_shared(s, p, fem_mesh, fem_entity, length, width, A, ptr, b, x)
  external solver subroutine :: s
  double precision, intent(inout) :: p(ILSI PARAM)
  integer, intent(in) :: fem mesh, fem entity, length, width
  external matrix-vector product subroutine :: A
  TYPE(varies), pointer :: ptr
  double precision, intent(in) :: b(width,length)
  double precision, intent(inout) :: x(width,length)

This routine solves the linear system :math:`A x = b` for the unknown
vector :math:`x`. s and p give the particular linear solver to use,
and are described in more detail in Section :numref:`sec:solver`.
fem_mesh and fem_entity give the FEM framework mesh (often
``FEM_Mesh_default_read()``) and entity (often ``FEM_NODE``) with which the
known and unknown vectors are listed.

width gives the number of degrees of freedom (entries in the vector) per
node. For example, if there is one degree of freedom per node, width is
one. length should always equal the number of FEM nodes.

A is a local matrix-vector product routine you must write. Its interface
is described in Section :numref:`sec:mvp`. ptr is a pointer passed
down to A-it is not otherwise used by the framework.

b is the known vector. x, on input, is the initial guess for the unknown
vector. On output, x is the final value for the unknown vector. b and x
should both have length \* width entries. In C, DOF :math:`i` of node
:math:`n` should be indexed as :math:`x[n*`\ width\ :math:`+i]`. In
Fortran, these arrays should be allocated like x(width,length).

When this routine returns, x is the final value for the unknown vector,
and the output values of the solver parameters p will have been written.

::

   // C++ Example
   int mesh=FEM_Mesh_default_read();
   int nNodes=FEM_Mesh_get_length(mesh,FEM_NODE);
   int width=3; //A 3D problem
   ILSI_Param solverParam;
   struct myProblemData myData;

   double *b=new double[nNodes*width];
   double *x=new double[nNodes*width];
   ... prepare solution target b and guess x ...

   ILSI_Param_new(&solverParam);
   solverParam.maxResidual=1.0e-4;
   solverParam.maxIterations=500;

   IFEM_Solve_shared(IFEM_CG_Solver,&solverParam,
          mesh,FEM_NODE,nNodes,width,
          myMatrixVectorProduct,&myData,b,x);

.. code-block:: fortran

   ! F90 Example
   include 'ifemf.h'
   INTEGER :: mesh, nNodes,width
   DOUBLE PRECISION, ALLOCATABLE :: b(:,:), x(:,:)
   DOUBLE PRECISION :: solverParam(ILSI_PARAM)
   TYPE(myProblemData) :: myData

   mesh=FEM_Mesh_default_read()
   nNodes=FEM_Mesh_get_length(mesh,FEM_NODE)
   width=3   ! A 3D problem

   ALLOCATE(b(width,nNodes), x(width,nNodes))
   ... prepare solution target b and guess x ..

   ILSI_Param_new(&solverParam);
   solverParam(1)=1.0e-4;
   solverParam(2)=500;

   IFEM_Solve_shared(IFEM_CG_Solver,solverParam,
          mesh,FEM_NODE,nNodes,width,
          myMatrixVectorProduct,myData,b,x);

.. _sec:mvp:

Matrix-vector product routine
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

IFEM requires you to write a matrix-vector product routine that will
evaluate :math:`A x` for various vectors :math:`x`. You may use any
subroutine name, but it must take these arguments:

::

  void IFEM_Matrix_product(void *ptr, int length, int width, const double
    *src, double *dest);

.. code-block:: fortran

  subroutine IFEM_Matrix_product(ptr, length, width, src, dest)
  TYPE(varies), pointer :: ptr
  integer, intent(in) :: length, width
  double precision, intent(in) :: src(width, length)
  double precision, intent(out) :: dest(width, length)


The framework calls this user-written routine when it requires a
matrix-vector product. This routine should compute
:math:`dest = A \, src`, interpreting :math:`src` and :math:`dest` as
vectors. length gives the number of nodes and width gives the number
of degrees of freedom per node, as above.

In writing this routine, you are responsible for choosing a
representation for the matrix :math:`A`. For many problems, there is no
need to represent :math:`A` explicitly-instead, you simply evaluate
:math:`dest` by looping over local elements, taking into account the
values of :math:`src`. This example shows how to write the matrix-vector
product routine for simple 1D linear elastic springs, while solving for
displacement given net forces.

After calling this routine, the framework will handle combining the
overlapping portions of these vectors across processors to arrive at a
consistent global matrix-vector product.

::

   // C++ Example
   #include "ifemc.h"

   typedef struct {
     int nElements; //Number of local elements
     int *conn; // Nodes adjacent to each element: 2*nElements entries
     double k; //Uniform spring constant
   } myProblemData;

   void myMatrixVectorProduct(void *ptr,int nNodes,int dofPerNode,
             const double *src,double *dest)
   {
     myProblemData *d=(myProblemData *)ptr;
     int n,e;
     // Zero out output force vector:
     for (n=0;n<nNodes;n++) dest[n]=0;
     // Add in forces from local elements
     for (e=0;e<d->nElements;e++) {
       int n1=d->conn[2*e+0]; // Left node
       int n2=d->conn[2*e+1]; // Right node
       double f=d->k * (src[n2]-src[n1]); //Force
       dest[n1]+=f;
       dest[n2]-=f;
     }
   }

.. code-block:: fortran

   ! F90 Example
   TYPE(myProblemData)
     INTEGER :: nElements
     INTEGER, ALLOCATABLE :: conn(2,:)
     DOUBLE PRECISION :: k
   END TYPE

   SUBROUTINE myMatrixVectorProduct(d,nNodes,dofPerNode,src,dest)
     include 'ifemf.h'
     TYPE(myProblemData), pointer :: d
     INTEGER :: nNodes,dofPerNode
     DOUBLE PRECISION :: src(dofPerNode,nNodes), dest(dofPerNode,nNodes)
     INTEGER :: e,n1,n2
     DOUBLE PRECISION :: f

     dest(:,:)=0.0
     do e=1,d%nElements
       n1=d%conn(1,e)
       n2=d%conn(2,e)
       f=d%k * (src(1,n2)-src(1,n1))
       dest(1,n1)=dest(1,n1)+f
       dest(1,n2)=dest(1,n2)+f
     end do
   END SUBROUTINE

IFEM_Solve_shared_bc
--------------------

::

  void IFEM_Solve_shared_bc(ILSI_Solver s, ILSI_Param *p, int fem_mesh,
  int fem_entity, int length, int width, int bcCount, const int *bcDOF,
  const double *bcValue, IFEM_Matrix_product_c A, void *ptr, const
  double *b, double *x);

.. code-block:: fortran

  subroutine IFEM_Solve_shared_bc(s, p, fem_mesh, fem_entity, length, width,
  bcCount, bcDOF, bcValue, A, ptr, b, x)
  external solver subroutine :: s
  double precision, intent(inout) :: p(ILSI_PARAM)
  integer, intent(in) :: fem_mesh, fem_entity, length,width
  integer, intent(in) :: bcCount
  integer, intent(in) :: bcDOF(bcCount)
  double precision, intent(in) :: bcValue(bcCount)
  external matrix-vector product subroutine :: A
  TYPE(varies), pointer :: ptr
  double precision, intent(in) :: b(width,length)
  double precision, intent(inout) :: x(width,length)

Like ``IFEM_Solve_shared``, this routine solves the linear system
:math:`A x = b` for the unknown vector :math:`x`. This routine,
however, adds support for boundary conditions associated with
:math:`x`. These so-called "essential" boundary conditions restrict
the values of some unknowns. For example, in structural dynamics, a
fixed displacement is such an essential boundary condition.

The only form of boundary condition currently supported is to impose a
fixed value on certain unknowns, listed by their degree of freedom-that
is, their entry in the unknown vector. In general, the :math:`i`\ ’th
DOF of node :math:`n` has DOF number :math:`n*width+i` in C and
:math:`(n-1)*width+i` in Fortran. The framework guarantees that, on
output, for all :math:`bcCount` boundary conditions,
:math:`x(bcDOF(f))=bcValue(f)`.

For example, if :math:`width` is 3 in a 3d problem, we would set node
:math:`ny`\ ’s y coordinate to 4.6 and node :math:`nz`\ ’s z coordinate
to 7.3 like this:

::

   // C++ Example
   int bcCount=2;
   int bcDOF[bcCount];
   double bcValue[bcCount];
   // Fix node ny's y coordinate
   bcDOF[0]=ny*width+1; // y is coordinate 1
   bcValue[0]=4.6;
   // Fix node nz's z coordinate
   bcDOF[1]=nz*width+2; // z is coordinate 2
   bcValue[1]=2.0;

.. code-block:: fortran

   ! F90 Example
   integer :: bcCount=2;
   integer :: bcDOF(bcCount);
   double precision :: bcValue(bcCount);
   // Fix node ny's y coordinate
   bcDOF(1)=(ny-1)*width+2; // y is coordinate 2
   bcValue(1)=4.6;
   // Fix node nz's z coordinate
   bcDOF(2)=(nz-1)*width+3; // z is coordinate 3
   bcValue(2)=2.0;

Mathematically, what is happening is we are splitting the partially
unknown vector :math:`x` into a completely unknown portion :math:`y` and
a known part :math:`f`:

.. math:: A x = b

.. math:: A (y + f) = b

.. math:: A y = b - A f

We can then define a new right hand side vector :math:`c=b-A f` and
solve the new linear system :math:`A y=c` normally. Rather than
renumbering, we do this by zeroing out the known portion of :math:`x` to
make :math:`y`. The creation of the new linear system, and the
substitution back to solve the original system are all done inside this
subroutine.

One important missing feature is the ability to specify general linear
constraints on the unknowns, rather than imposing specific values.

