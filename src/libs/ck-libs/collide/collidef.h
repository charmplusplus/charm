       interface
       subroutine CollideInit(gridStart,gridSize)
          double precision, intent(in) :: gridStart(3), gridSize(3)
       end subroutine

       subroutine CollideRegister(chunkNo)
          integer, intent(in) :: chunkNo
       end subroutine
       subroutine CollideUnregister(chunkNo)
          integer, intent(in) :: chunkNo
       end subroutine

       subroutine Collide(chunkNo,nObj,boxes)
          integer, intent(in) :: chunkNo, nObj
          double precision, intent(in) :: boxes(6,nObj)
       end subroutine
       function CollideCount(chunkNo)
          integer, intent(in) :: chunkNo
          integer  :: CollideCount
       end function 
       subroutine CollideList(chunkNo,nColl,collisions)
          integer, intent(in) :: chunkNo, nColl
          integer, intent(out) :: collisions(3,nColl)
       end subroutine

       end interface
