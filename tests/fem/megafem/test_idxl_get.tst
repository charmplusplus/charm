C ( C/Fortran test driver: IDXL routines )
C ( Orion Sky Lawlor- olawlor@acm.org- 2003/7/22 )

C (Test out the IDXL_Get routines on this IDXL_Side_t.)
SUBROUTINE3(TST_test_idxl_side, TYPE_POINTER(global),g, INTEGER,s, INTEGER,n)
  INTEGER nPartners, nShared;
  INTEGER p; C (Partner number)
  INTEGER pRank;
  INTEGER i;
  DECLARE_ARRAY(INTEGER,list);
  CALL TST_status("IDXL_Get_*");
  nPartners=IDXL_Get_partners(s);
  CALL TST_assert_isnt_negative(nPartners);
  C( CALL TST_print_int("IDXL Comm shared partners",nPartners); )
  FOR(p,nPartners)
    pRank=IDXL_Get_partner(s,p);
    CALL TST_assert_is_chunk(g,pRank);
    nShared=IDXL_Get_count(s,p);
    C( CALL TST_print_int2("  IDXL Comm partner/nComm",pRank,nShared); )
    ALLOCATE_ARRAY(list,nShared);
    CALL IDXL_Get_list(s,p,list);
    FOR(i,nShared)
      CALL TST_assert_index_range(list(i),n);
    END_FOR
    DEALLOCATE(list)
  END_FOR
  CALL IDXL_Get_end(s);
END

C ( Test out the ghost communication for this entity type )
SUBROUTINE3(TST_test_idxl_ghost, TYPE_POINTER(global),g, INTEGER,fem_mesh, INTEGER,ent)
  INTEGER n,idxl;
  idxl=FEM_Comm_ghost(fem_mesh,ent);
  n=FEM_Mesh_get_length(fem_mesh,ent);
  CALL TST_test_idxl_side(g,IDXL_Get_send(idxl),n);
  n=FEM_Mesh_get_length(fem_mesh,FEM_GHOST+ent);
  CALL TST_test_idxl_side(g,IDXL_Get_recv(idxl),n);
END

C (Test out the IDXL routines on this Mesh.)
SUBROUTINE2(TST_test_idxl, TYPE_POINTER(global),g, INTEGER,fem_mesh)
  INTEGER n,idxl;
  CALL TST_status("FEM_Comm_*");
C ( Test out shared nodes: )
  idxl=FEM_Comm_shared(fem_mesh,FEM_NODE);
  n=FEM_Mesh_get_length(fem_mesh,FEM_NODE);
  CALL TST_test_idxl_side(g,IDXL_Get_send(idxl),n);

C ( Test out each ghost element type: )
  CALL TST_test_idxl_ghost(g,fem_mesh,FEM_NODE);
  CALL TST_test_idxl_ghost(g,fem_mesh,FEM_ELEM+0);
  CALL TST_test_idxl_ghost(g,fem_mesh,FEM_ELEM+1);
END

