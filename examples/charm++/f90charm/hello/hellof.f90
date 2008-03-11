      MODULE  HelloMod

      TYPE Hello
        integer data
        integer iter
        double precision  data2
        integer len
        integer s(4)
      END TYPE

!    define Object Pointer used to communicate with charm kernel
      TYPE HelloPtr
        TYPE (Hello), POINTER ::  obj
        integer*8 aid
      END TYPE

      END MODULE

!  user MUST write this subroutine to allocate the object data
      SUBROUTINE Hello_allocate(objPtr, aid, index)
        USE HelloMod
        IMPLICIT NONE
        TYPE(HelloPtr) objPtr 
        integer*8 aid
        integer index

        allocate(objPtr%obj)
        objPtr%aid = aid
        ! initialize Chare data here in constructor
        objPtr%obj%data = index
        objPtr%obj%iter = 0
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
            ! allocate chare and restore aid
          allocate(objPtr%obj)
          objPtr%aid = aid;
        endif
        CALL fpup_int(p, objPtr%obj%data)
        CALL fpup_int(p, objPtr%obj%iter)
        CALL fpup_double(p, objPtr%obj%data2)
        CALL fpup_int(p, objPtr%obj%len)
        CALL fpup_ints(p, objPtr%obj%s, objPtr%obj%len)
      END SUBROUTINE

! user MUST write this for load balancing
      SUBROUTINE hello_resumefromsync(objPtr, aid, index)
        USE HelloMod
        TYPE(HelloPtr) objPtr 
        integer*8 aid
        integer index

        if (index .eq.  4) THEN
            call SendTo_Hello_SayHi(objPtr%aid, 0, 1, objPtr%obj%data2,  &
                                    objPtr%obj%len, objPtr%obj%s);
        endif
      END SUBROUTINE

!    define fortran entry function
      SUBROUTINE SayHi(objPtr, myIndex, data, data2, len, s)
        USE HelloMod
        IMPLICIT NONE

        TYPE(HelloPtr) objPtr
        integer myIndex
        integer data
        double precision data2
        integer si, len
        integer  s(len)
        integer chunkSize, myPe, next

          ! print parameters
        call CkMyPe(myPe)
        call CkPrintf("[%d] SayHi: myIndex:%d data:%d data2:%F\n$$", myPe, myIndex, data, data2)
        call CkPrintf("SayHi: s: %d %d %d %d\n$$",s(1), s(2), s(3), s(4))

          !  get readonly variable
        call get_ChunkSize(chunkSize)
        call CkPrintf("chunkSize(readonly):%d \n$$", chunkSize)

          !  print Chare data
        call CkPrintf("Chare data: %d iteration: %d\n$$", objPtr%obj%data, objPtr%obj%iter)

        objPtr%obj%iter = objPtr%obj%iter + 1

        next = myIndex+1;
        if (next .eq. 5) next = 0;

        if (objPtr%obj%iter == 3) then
          if (next .ne. 0) then
            call SendTo_Hello_SayHi(objPtr%aid, next, 1, data2, len, s);
          else
            objPtr%obj%data2 = data2
            objPtr%obj%len = len
            objPtr%obj%s = s
          endif
          call hello_atsync(objPtr%aid, myIndex);
        else if (objPtr%obj%iter == 5) then
          call CkExit();
        else 
          call SendTo_Hello_SayHi(objPtr%aid, next, 1, data2, len, s);
        endif

      END SUBROUTINE


!   MAIN subroutine, user MUST write
!   called once only on processor 0
      SUBROUTINE f90charmmain()
        USE HelloMod
        IMPLICIT NONE
        integer i
        double precision d
        integer*8 aid
        integer  s(8)

        call Hello_CkNew(5, aid)
 
        call set_ChunkSize(10);

        do i=1,8
	  s(i) = i;
        enddo
        d = 2.50
        call SendTo_Hello_SayHi(aid, 0, 1, d, 4, s(3:6));
      END SUBROUTINE

