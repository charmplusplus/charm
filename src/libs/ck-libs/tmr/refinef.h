       external REFINE2D_Get_Split

       interface
       subroutine REFINE2D_Init()
       end subroutine
       subroutine REFINE2D_NewMesh(nEl,nGhost,conn,gid)
         integer, intent(in) :: nEl,nGhost
         integer, intent(in) :: conn(3,nGhost)
         integer, intent(in) :: gid(2,nGhost)
       end subroutine
       subroutine REFINE2D_Split(nNode,coord,nEl,desiredArea)
         integer, intent(in) :: nNode,nEl
         double precision, intent(in) :: coord(2,nNode)
         double precision, intent(in) :: desiredArea(nEl)
       end subroutine
       function REFINE2D_Get_Split_Length()
          integer  :: REFINE2D_Get_Split_Length
       end function REFINE2D_Get_Split_Length
!       subroutine REFINE2D_Get_Split(splitNo,conn,tri,A,B,C,frac)
!          integer, intent(in) :: splitNo
!          integer, intent(in) :: conn(:,:)
!          integer, intent(out) :: tri
!          integer, intent(out) :: A
!          integer, intent(out) :: B
!          integer, intent(out) :: C
!          double precision, intent(out) :: frac
!          !real, intent(out) :: frac
!       end subroutine 
       subroutine REFINE2D_Check(nEl,conn,nNode)
          integer, intent(in) :: nEl,nNode
          integer, intent(in) :: conn(3,nEl)
       end subroutine 
       end interface
