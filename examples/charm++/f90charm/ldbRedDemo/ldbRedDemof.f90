      MODULE  LdbDemoMod
        USE charm
        TYPE LdbDemo
          integer iterations
          REAL*8  t0
          integer next, iteration, n, count
          REAL*4, pointer :: myData(:)
        END TYPE

          ! define Object Pointer used to communicate with charm kernel
        TYPE LdbDemoPtr
          TYPE (LdbDemo), POINTER ::  obj
          integer*8 aid
        END TYPE
      END MODULE

!  user MUST write this subroutine to allocate the object data
      SUBROUTINE BalanceMe_allocate(objPtr, aid, index)
        USE LdbDemoMod
        IMPLICIT NONE
        TYPE(LdbDemoPtr) objPtr
        INTEGER*8 aid
        INTEGER index, nElements

        allocate(objPtr%obj)
        objPtr%aid = aid
          ! initialize Chare data here in constructor
        objPtr%obj%iterations = 0
        objPtr%obj%t0 =  CmiWallTimer()

        objPtr%obj%n = 200+MOD(index*31757, 2000)
        call CkPrintf("Constructor of element %d n: %d \n$$", index, objPtr%obj%n)
        allocate(objPtr%obj%myData(objPtr%obj%n))
        call get_nElements(nElements)
        objPtr%obj%next = MOD( index+1, nElements)
        objPtr%obj%iteration = 0
        objPtr%obj%count = 0

        call BalanceMe_Entry_nextStep(objPtr, index)
      END SUBROUTINE

!   user MUST write this puper subroutine
      SUBROUTINE BalanceMe_pup(p, objPtr, aid)
        USE LdbDemoMod
        IMPLICIT NONE
        INCLUDE 'pupf.h'
        INTEGER p
        TYPE(LdbDemoPtr),target :: objPtr
        INTEGER*8 aid

        if (fpup_isUnpacking(p)) then
            ! allocate chare and restore aid
          allocate(objPtr%obj)
          objPtr%aid = aid;
        endif
        CALL fpup_int(p, objPtr%obj%iterations)
        CALL fpup_double(p, objPtr%obj%t0)
        CALL fpup_int(p, objPtr%obj%next)
        CALL fpup_int(p, objPtr%obj%iteration)
        CALL fpup_int(p, objPtr%obj%n)
        CALL fpup_int(p, objPtr%obj%count)
        if (fpup_isUnpacking(p)) then
            ! allocate array
          allocate(objPtr%obj%myData(objPtr%obj%n))
        ENDIF
        CALL fpup_reals(p, objPtr%obj%myData, objPtr%obj%n)
           ! free up memory
        if (fpup_isDeleting(p)) deallocate(objPtr%obj%myData)
      END SUBROUTINE

! user MUST write this for load balancing
      SUBROUTINE BalanceMe_resumefromsync(objPtr, aid, index)
        USE LdbDemoMod
        TYPE(LdbDemoPtr) objPtr
        integer*8 aid
        integer index

          ! load balancing finish, start next step
        call BalanceMe_Entry_nextStep(objPtr, index)
      END SUBROUTINE

      INTEGER FUNCTION doWork(workTime)
        USE charm
        IMPLICIT NONE
        REAL*8 workTime, recvTimeStamp
        INTEGER k

        recvTimeStamp = CmiWallTimer()
        DO WHILE (CmiWallTimer() - recvTimeStamp < workTime )
          k = k+1;
        END DO

        doWork = k;
     END FUNCTION

!    define fortran entry function
      SUBROUTINE BalanceMe_Entry_nbrData(objPtr, myIndex, size, D, k)
        USE LdbDemoMod
        IMPLICIT NONE
        INTEGER doWork

        TYPE(LdbDemoPtr) objPtr
        integer myIndex
        integer size, k, res
        REAL*4  D(size)
        REAL*8  n
        external BalanceMe_ReductionTarget_barrier

        n = objPtr%obj%n
        res = doWork(n * 0.00001)

        objPtr%obj%iteration = objPtr%obj%iteration+1
        IF (MOD(objPtr%obj%iteration, 5) .eq. 0) THEN
           ! AtSync to start load balancing
          call BalanceMe_atSync(objPtr%aid, myIndex)
        ELSE
          call BalanceMe_contribute(objPtr%aid, myIndex, sizeof(myIndex), myIndex, CHARM_SUM_INT, BalanceMe_ReductionTarget_barrier)
        ENDIF
      END SUBROUTINE

      SUBROUTINE BalanceMe_Entry_nextStep(objPtr, myIndex)
        USE LdbDemoMod
        IMPLICIT NONE

        TYPE(LdbDemoPtr) objPtr
        integer myIndex

        call SendTo_BalanceMe_nbrData(objPtr%aid, objPtr%obj%next, objPtr%obj%n, objPtr%obj%myData, myIndex)

      END SUBROUTINE

      SUBROUTINE BalanceMe_Entry_barrier(objPtr, myIndex, result)
        USE LdbDemoMod
        IMPLICIT NONE

        TYPE(LdbDemoPtr) objPtr
        INTEGER myIndex
        INTEGER result

        INTEGER mype
        double precision  t1
        INTEGER nElements

        call CkMyPe(mype)
        print *, mype, result

        objPtr%obj%iterations = objPtr%obj%iterations + 1
        IF (objPtr%obj%iterations .eq. 18) THEN
          t1 = CmiWallTimer()
          call CkPrintf("ALL done in %F seconds.\n$$", t1-objPtr%obj%t0)
          call CkExit()
        ELSE
           ! broadcast using "-1"
          call SendTo_BalanceMe_nextStep(objPtr%aid, -1);
        ENDIF
      END SUBROUTINE


!   MAIN subroutine, user MUST write
!   called once only on processor 0
      SUBROUTINE f90charmmain()
        call set_nElements(5)
        call BalanceMe_CkNew(5)
      END SUBROUTINE
