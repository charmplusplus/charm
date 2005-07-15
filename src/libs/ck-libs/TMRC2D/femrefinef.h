interface
   subroutine FEM_REFINE2D_Init()
   end subroutine FEM_REFINE2D_Init
   subroutine FEM_REFINE2D_NewMesh(meshID,nodeID,elemID,nodeBoundary)
     integer, intent(in) :: meshID,nodeID,elemID,nodeBoundary
   end subroutine FEM_REFINE2D_NewMesh
   subroutine FEM_REFINE2D_Split(mID,nID,coord,eID,areas)
     integer, intent(in) :: mID,nID,eID
     double precision, intent(in) :: coord(:)
     double precision, intent(in) :: areas(:)
   end subroutine FEM_REFINE2D_Split
end interface
integer,parameter :: FEM_VALID=(FEM_ATTRIB_TAG_MAX-1)
