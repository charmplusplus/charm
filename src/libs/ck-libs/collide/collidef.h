       interface
       INTEGER FUNCTION COLLIDE_Init(mpi_comm,gridStart,gridSize)
          integer, intent(in) :: mpi_comm
          double precision, intent(in) :: gridStart(3), gridSize(3)
       end subroutine

       subroutine COLLIDE_Boxes(c,nObj,boxes)
          integer, intent(in) :: c, nObj
          double precision, intent(in) :: boxes(6,nObj)
       end subroutine
       subroutine COLLIDE_Boxes_prio(c,nObj,boxes,prio)
          integer, intent(in) :: c, nObj
          double precision, intent(in) :: boxes(6,nObj)
          integer, intent(in) :: prio(nObj)
       end subroutine

       integer function COLLIDE_Count(c)
          integer, intent(in) :: c
       end function 
       subroutine COLLIDE_List(c,collisions)
          integer, intent(in) :: c
          integer, intent(out) :: collisions(3,:)
       end subroutine

       subroutine COLLIDE_Destroy(c)
          integer, intent(in) :: c
       end subroutine 

       end interface
