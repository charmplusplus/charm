C ( C/Fortran test driver: Overall driver )
C ( Orion Sky Lawlor- olawlor@acm.org- 2003/7/22 )


C ( ------------------ Utility routines ------------------- )
#include "test_assert.tst"

C ( ------------------ Tests ------------------- )
#include "test_idxl_get.tst"

C ( ------------------ Mesh Creation ------------------- )

C ( Allocate (uninitialized) memory for this mesh object )
SUBROUTINE1(TST_mesh_allocate, TYPE_POINTER(mesh),m)
  ALLOCATE_ARRAY2D(m v coord, NODE_COORDS, m v nNodes);
  ALLOCATE_ARRAY2D(m v tri, TRI_NODES, m v nTri);
  ALLOCATE_ARRAY2D(m v quad, QUAD_NODES, m v nQuad);
END

C ( Deallocate storage for this mesh object )
SUBROUTINE1(TST_mesh_deallocate, TYPE_POINTER(mesh),m)
  DEALLOCATE(m v coord);
  DEALLOCATE(m v tri);
  DEALLOCATE(m v quad);
  CALL FEM_Mesh_deallocate(m v fem_mesh);
END

C ( Check (uninitialized) memory for this mesh object )
SUBROUTINE1(TST_mesh_check, TYPE_POINTER(mesh),m)
  INTEGER t,q,n;
  FOR(t,m v nTri)
    FOR(n,TRI_NODES)
      CALL TST_assert_index_range(m v tri(n,t),m v nNodes);
    END_FOR
  END_FOR
  FOR(q,m v nQuad)
    FOR(n,QUAD_NODES)
      CALL TST_assert_index_range(m v quad(n,q),m v nNodes);
    END_FOR
  END_FOR
END

C ( Create a dim x dim grid of 2d nodes connected by triangle-pairs. )
SUBROUTINE2(TST_mesh_create, TYPE_POINTER(mesh),m, INTEGER,dim)
  INTEGER x,y,quadRow,n,t,q;
  m v fem_mesh = FEM_Mesh_allocate();
  m v nNodes = dim*dim;
  quadRow=dim/2;
  m v nTri = 2*quadRow*(dim-1);
  m v nQuad = (dim-1-quadRow)*(dim-1);
  CALL TST_mesh_allocate(m);
  n = IDXBASE;
  t = IDXBASE;
  q = IDXBASE;
  FOR(y,dim) 
    FOR(x,dim)
      C ( Assign coordinates of this node )
      m v coord(IDXBASE+0,n) = x*0.1;
      m v coord(IDXBASE+1,n) = y*0.2;
      IF (x -IDXBASE+1 < dim) THEN   C (skips the rightmost column of nodes)
        IF (y -IDXBASE < quadRow) THEN
          C ( Attach two triangles to this node )
          m v tri(IDXBASE+0,t) = n;
          m v tri(IDXBASE+1,t) = n+1;
          m v tri(IDXBASE+2,t) = n+dim;
          m v tri(IDXBASE+0,t+1) = n+1;
          m v tri(IDXBASE+1,t+1) = n+dim+1;
          m v tri(IDXBASE+2,t+1) = n+dim;
          t = t+2;
        ELSE
          IF (y -IDXBASE+1 < dim) THEN    C (skips the bottom row of nodes)
            C ( Attach a quad to this node )
            m v quad(IDXBASE+0,q) = n;
            m v quad(IDXBASE+1,q) = n+1;
            m v quad(IDXBASE+2,q) = n+dim;
            m v quad(IDXBASE+3,q) = n+dim+1;
            q = q+1;
          END_IF
        END_IF
      END_IF
      n = n+1;
    END_FOR 
  END_FOR
  CALL TST_assert_equal(n,m v nNodes+IDXBASE);
  CALL TST_assert_equal(t,m v nTri+IDXBASE);
  CALL TST_assert_equal(q,m v nQuad+IDXBASE);
  CALL TST_mesh_check(m);
END

C ( Copy this mesh into the FEM framework )
SUBROUTINE2(TST_mesh_set, TYPE_POINTER(mesh),m, INTEGER,ent)
  INTEGER fem_mesh;
  fem_mesh = m v fem_mesh;
  CALL FEM_Mesh_data(fem_mesh,ent+FEM_NODE,FEM_DATA+0,m v coord, IDXBASE,m v nNodes,FEM_DOUBLE,NODE_COORDS);
  CALL FEM_Mesh_data(fem_mesh,ent+FEM_ELEM+0,FEM_CONN,m v tri, IDXBASE,m v nTri,IDXL_INDEX_0+IDXBASE,TRI_NODES);
  CALL FEM_Mesh_data(fem_mesh,ent+FEM_ELEM+1,FEM_CONN,m v quad, IDXBASE,m v nQuad,IDXL_INDEX_0+IDXBASE,QUAD_NODES);
END

C ( Extract this mesh from the FEM framework )
SUBROUTINE3(TST_mesh_get, TYPE_POINTER(mesh),m, INTEGER,fem_mesh, INTEGER,ent)
  m v fem_mesh = fem_mesh;
  m v nNodes = FEM_Mesh_get_length(fem_mesh,ent+FEM_NODE);
  m v nTri = FEM_Mesh_get_length(fem_mesh,ent+FEM_ELEM+0);
  m v nQuad = FEM_Mesh_get_length(fem_mesh,ent+FEM_ELEM+1);
  CALL TST_mesh_allocate(m);
  CALL TST_mesh_set(m,ent); C( <- because "FEM_Mesh_data" works both ways )
END

C ( Prepare the FEM framework to partition these mesh ghosts )
SUBROUTINE1(TST_mesh_ghostprep, TYPE_POINTER(mesh),m)
  INTEGER i;
  DECLARE_ARRAY(INTEGER,t); C( Triangle -> face mapping )
  DECLARE_ARRAY(INTEGER,q); C( Quad -> face mapping )
  ALLOCATE_ARRAY(t,6);
  ALLOCATE_ARRAY(q,8);
  i=IDXBASE;
  t(i+0)=i+0; t(i+1)=i+1; 
  t(i+2)=i+1; t(i+3)=i+2;
  t(i+4)=i+2; t(i+5)=i+0;
  
  q(i+0)=i+0; q(i+1)=i+1; 
  q(i+2)=i+1; q(i+3)=i+3;
  q(i+4)=i+3; q(i+5)=i+2;
  q(i+6)=i+2; q(i+7)=i+0;
  
C( CALL FEM_Mesh_set_default_write(m v fem_mesh); )
  CALL FEM_Add_ghost_layer(2,1);
  CALL FEM_Add_ghost_elem(0,3,t);
  CALL FEM_Add_ghost_elem(1,4,q);
END

C ( ------------------ Driver ------------------- )
C (Call all test routines)
SUBROUTINE0(RUN_Test)
  INTEGER i,j;
  INTEGER serialMesh, fem_mesh;
  INTEGER nodes, elements;
  DECLARE_ARRAY2D(DOUBLE PRECISION,nodeCoord,2);
  TYPE(global) g;
  TYPE(mesh) sm; C( serial mesh )
  g v comm=MPI_COMM_WORLD;
  CALL MPI_Comm_rank(g v comm,ADDR g v myRank MPIERR);
  CALL MPI_Comm_size(g v comm,ADDR g v commSize MPIERR);
  CALL TST_assert_is_chunk(g, g v myRank);
  
  IF (g v myRank == 0) THEN
    CALL FEM_Print("Creating mesh...");
    serialMesh=FEM_Mesh_allocate();
    CALL TST_mesh_create(sm, 4);
    CALL TST_print_int("Serial mesh nodes",sm v nNodes);
    CALL TST_mesh_ghostprep(sm);
    CALL TST_mesh_set(sm,0);
  END_IF
  CALL FEM_Print("Splitting up mesh");
  fem_mesh=FEM_Mesh_broadcast(sm v fem_mesh,0,g v comm);
  IF (g v myRank == 0) THEN
    CALL TST_mesh_deallocate(sm);
  END_IF
  CALL TST_mesh_get(g v m,fem_mesh,0);
  CALL TST_print_int("Split mesh nodes",g v m v nNodes);
  CALL TST_mesh_get(g v mg,fem_mesh,FEM_GHOST);
  CALL TST_print_int("Split mesh ghost nodes",g v mg v nNodes);
  
  CALL FEM_Print("Running IDXL tests");
  CALL TST_test_idxl(g,fem_mesh);
  
  
  CALL TST_mesh_deallocate(g v m);
  
  CALL MPI_Barrier(g v comm MPIERR);
END


