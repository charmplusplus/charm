C ( C/Fortran data structure declaration )
C ( Orion Sky Lawlor- olawlor@acm.org- 2003/7/23 )

C ( Describes a (piece of a) FEM mesh )
CREATE_TYPE(mesh)
C ( FEM framework ID for this mesh )
  INTEGER fem_mesh;

C ( Entity type FEM_NODE. )
  INTEGER nNodes, nGhostNode;
#define NODE_COORDS 2
  DECLARE_ARRAY2D(DOUBLE PRECISION,coord,NODE_COORDS);

C ( Entity type FEM_ELEM+0: triangles. )
  INTEGER nTri, nGhostTri;
#define TRI_NODES 3
  DECLARE_ARRAY2D(INTEGER,tri,TRI_NODES);
  
C ( Entity type FEM_ELEM+1: quads. )
  INTEGER nQuad, nGhostQuad;
#define QUAD_NODES 4
  DECLARE_ARRAY2D(INTEGER,quad,QUAD_NODES);
END_TYPE

CREATE_TYPE(global)
  INTEGER comm; C( MPI Communicator )
  INTEGER myRank; C( my 0-based rank in the communicator )
  INTEGER commSize; C( the size of the communicator )
  TYPE(mesh) m; C( Testing mesh-- data )
  TYPE(mesh) mg; C( Ghost nodes and elements on testing mesh )
END_TYPE

C( Error return type for MPI routines. )
INTEGER mpierr;
#if TST_F90 /* In f90, MPI routines take an "error" argument. */
#  define MPIERR ,mpierr
#else /* In C, MPI routines return their error. */
#  define MPIERR 
#endif

