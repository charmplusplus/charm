      MODULE  HelloMod
      TYPE Hello
      integer data
      END TYPE

!    define Object Pointer used to communicate with charm kernel
      TYPE HelloPtr
      TYPE (Hello), POINTER ::  obj
      integer*8 aid
      END TYPE

      END MODULE

!    this subroutine can be generated automatically
      SUBROUTINE Hello_allocate(objPtr, aid, index1, index2)
      USE HelloMod
      TYPE(HelloPtr) objPtr 
      integer*8 aid
      integer :: index1, index2

      allocate(objPtr%obj)
      objPtr%aid = aid
!     set Chare data here in constructor
      objPtr%obj%data = index1
      END SUBROUTINE

!   user MUST write this puper subroutine
      SUBROUTINE hello_pup(p, objPtr, aid)
        USE HelloMod
        IMPLICIT NONE
        INCLUDE 'pupf.h'
        INTEGER :: p
        TYPE(HelloPtr),target :: objPtr
        integer*8 aid

        if (fpup_isUnpacking(p)) then
          allocate(objPtr%obj)
          objPtr%aid = aid;
        endif
        CALL fpup_int(p, objPtr%obj%data)
      END SUBROUTINE

! user MUST write this for load balancing
      SUBROUTINE hello_resumefromsync(objPtr, aid, index1, index2)
        USE HelloMod
        TYPE(HelloPtr) objPtr
        integer*8 aid
        integer index1, index2

        ! empty
      END SUBROUTINE


!    define fortran entry function
      SUBROUTINE SayHi(objPtr, myIndex1, myIndex2, data, data2, len, s)
      USE HelloMod
      IMPLICIT NONE

      TYPE(HelloPtr) objPtr
      integer :: myIndex1, myIndex2;
      integer :: newIndex1, newIndex2;
      integer data
      double precision data2
      integer si, len
      integer  s(len)
      integer chunkSize, myPe

! print parameters
      call CkMyPe(myPe)
      call CkPrintf("[%d] SayHi: myIndex:(%d %d) data:%d data2:%F\n$$", myPe, myIndex1, myIndex2, data, data2)
      call CkPrintf("SayHi: s: %d %d %d %d\n$$",s(1), s(2), s(3), s(4))

!  get readonly variable
      call get_ChunkSize(chunkSize)
      call CkPrintf("chunkSize(readonly):%d \n$$", chunkSize)

!  get Chare data
      call CkPrintf("Chare data: %d\n$$", objPtr%obj%data)

      newIndex1 = myIndex1
      newIndex2 = myIndex2 + 1
      if (newIndex2 .eq. 2) then
        newIndex2 = 0
        newIndex1 = newIndex1 + 1
        if (newIndex1 .eq. 5) newIndex1 = 0
      endif
      if (newIndex1 .ne. 0 .or. newIndex2 .ne. 0) then
          call SendTo_Hello_SayHi(objPtr%aid, newIndex1, newIndex2, 1, data2, len, s);
      else 
	  call CkExit()
      endif

      END SUBROUTINE


      SUBROUTINE f90charmmain()
      USE HelloMod
      integer i
      double precision d
      integer*8 aid
      integer  s(8)

      call Hello_CkNew(5, 2, aid)

      call set_ChunkSize(10);

      do i=1,8
	  s(i) = i;
      enddo
      d = 2.50
      call SendTo_Hello_SayHi(aid, 0, 0, 1, d, 4, s(3:6));

      END

