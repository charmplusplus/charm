      MODULE  CaseTestMod

      Type CaseTestArray
      END TYPE

      Type CaseTestArrayPtr
        TYPE (CaseTestArray), POINTER :: obj
        integer*8 aid
      END TYPE
      END MODULE

      SUBROUTINE CaseTestArray_allocate(objPtr, aid, index)
        USE CaseTestMod
        IMPLICIT NONE
        TYPE(CaseTestArrayPtr) objPtr
        integer*8 aid
        integer index

        allocate(objPtr%obj)
        objPtr%aid = aid
        ! initialize Chare data here in constructor
      END SUBROUTINE

      SUBROUTINE CaseTestArray_pup(p, objPtr, aid)
        USE CaseTestMod
        IMPLICIT NONE
        INCLUDE 'pupf.h'
        INTEGER :: p
        TYPE(CaseTestArrayPtr),target :: objPtr
        integer*8 aid

      END SUBROUTINE

      SUBROUTINE CaseTestArray_resumefromsync(objPtr, aid, index)
        USE CaseTestMod
        TYPE(CaseTestArrayPtr) objPtr
        integer*8 aid
        integer index

      END SUBROUTINE

      SUBROUTINE f90charmmain()
        USE CaseTestMod
        IMPLICIT NONE
        integer*8  aid

        call CaseTestArray_CkNew(1, aid)
        call CaseTestArray_Invoke_run(aid, 0)
      END SUBROUTINE
