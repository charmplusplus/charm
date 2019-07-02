module charm
  integer*8, external :: CmiMemoryUsage
  integer*8, external :: CmiMaxMemoryUsage
  real*8,    external :: CmiWallTimer
  real*8,    external :: CkWallTimer
  real*8,    external :: CmiCpuTimer
  real*8,    external :: CkCpuTimer

! KEEPINSYNC: ckreduction.h
  integer, parameter :: CHARM_NOP = 1
  integer, parameter :: CHARM_SUM_CHAR = 2
  integer, parameter :: CHARM_SUM_SHORT = 3
  integer, parameter :: CHARM_SUM_INT = 4
  integer, parameter :: CHARM_SUM_LONG = 5
  integer, parameter :: CHARM_SUM_LONG_LONG = 6
  integer, parameter :: CHARM_SUM_UCHAR = 7
  integer, parameter :: CHARM_SUM_USHORT = 8
  integer, parameter :: CHARM_SUM_UINT = 9
  integer, parameter :: CHARM_SUM_ULONG = 10
  integer, parameter :: CHARM_SUM_ULONG_LONG = 11
  integer, parameter :: CHARM_SUM_FLOAT = 12
  integer, parameter :: CHARM_SUM_DOUBLE = 13
  integer, parameter :: CHARM_PRODUCT_CHAR = 14
  integer, parameter :: CHARM_PRODUCT_SHORT = 15
  integer, parameter :: CHARM_PRODUCT_INT = 16
  integer, parameter :: CHARM_PRODUCT_LONG = 17
  integer, parameter :: CHARM_PRODUCT_LONG_LONG = 18
  integer, parameter :: CHARM_PRODUCT_UCHAR = 19
  integer, parameter :: CHARM_PRODUCT_USHORT = 20
  integer, parameter :: CHARM_PRODUCT_UINT = 21
  integer, parameter :: CHARM_PRODUCT_ULONG = 22
  integer, parameter :: CHARM_PRODUCT_ULONG_LONG = 23
  integer, parameter :: CHARM_PRODUCT_FLOAT = 24
  integer, parameter :: CHARM_PRODUCT_DOUBLE = 25
  integer, parameter :: CHARM_MAX_CHAR = 26
  integer, parameter :: CHARM_MAX_SHORT = 27
  integer, parameter :: CHARM_MAX_INT = 28
  integer, parameter :: CHARM_MAX_LONG = 29
  integer, parameter :: CHARM_MAX_LONG_LONG = 30
  integer, parameter :: CHARM_MAX_UCHAR = 31
  integer, parameter :: CHARM_MAX_USHORT = 32
  integer, parameter :: CHARM_MAX_UINT = 33
  integer, parameter :: CHARM_MAX_ULONG = 34
  integer, parameter :: CHARM_MAX_ULONG_LONG = 35
  integer, parameter :: CHARM_MAX_FLOAT = 36
  integer, parameter :: CHARM_MAX_DOUBLE = 37
  integer, parameter :: CHARM_MIN_CHAR = 38
  integer, parameter :: CHARM_MIN_SHORT = 39
  integer, parameter :: CHARM_MIN_INT = 40
  integer, parameter :: CHARM_MIN_LONG = 41
  integer, parameter :: CHARM_MIN_LONG_LONG = 42
  integer, parameter :: CHARM_MIN_UCHAR = 43
  integer, parameter :: CHARM_MIN_USHORT = 44
  integer, parameter :: CHARM_MIN_UINT = 45
  integer, parameter :: CHARM_MIN_ULONG = 46
  integer, parameter :: CHARM_MIN_ULONG_LONG = 47
  integer, parameter :: CHARM_MIN_FLOAT = 48
  integer, parameter :: CHARM_MIN_DOUBLE = 49
!  integer, parameter :: CHARM_LOGICAL_AND = 50
  integer, parameter :: CHARM_LOGICAL_AND_INT = 51
  integer, parameter :: CHARM_LOGICAL_AND_BOOL = 52
!  integer, parameter :: CHARM_LOGICAL_OR = 53
  integer, parameter :: CHARM_LOGICAL_OR_INT = 54
  integer, parameter :: CHARM_LOGICAL_OR_BOOL = 55
! CHARM_LOGICAL_XOR does not exist
  integer, parameter :: CHARM_LOGICAL_XOR_INT = 56
  integer, parameter :: CHARM_LOGICAL_XOR_BOOL = 57
!  integer, parameter :: CHARM_BITVEC_AND = 58
  integer, parameter :: CHARM_BITVEC_AND_INT = 59
  integer, parameter :: CHARM_BITVEC_AND_BOOL = 60
!  integer, parameter :: CHARM_BITVEC_OR = 61
  integer, parameter :: CHARM_BITVEC_OR_INT = 62
  integer, parameter :: CHARM_BITVEC_OR_BOOL = 63
!  integer, parameter :: CHARM_BITVEC_XOR = 64
  integer, parameter :: CHARM_BITVEC_XOR_INT = 65
  integer, parameter :: CHARM_BITVEC_XOR_BOOL = 66
  integer, parameter :: CHARM_RANDOM = 67

  INTERFACE
      SUBROUTINE initbigsimtrace(outputParams, outputtiming)
         INTEGER outputParams, outputtiming
      END SUBROUTINE
      SUBROUTINE endtracebigsim1(e,step,p1) 
         CHARACTER* (*)  e
         INTEGER step
         DOUBLE PRECISION p1
      END SUBROUTINE 
      SUBROUTINE endtracebigsim2(e,step,p1,p2) 
         CHARACTER* (*)  e
         INTEGER step
         DOUBLE PRECISION p1,p2
      END SUBROUTINE 
      SUBROUTINE endtracebigsim3(e,step,p1,p2,p3) 
         CHARACTER* (*)  e
         INTEGER step
         DOUBLE PRECISION p1,p2,p3
      END SUBROUTINE 
      SUBROUTINE endtracebigsim4(e,step,p1,p2,p3,p4) 
         CHARACTER* (*)  e
         INTEGER step
         DOUBLE PRECISION p1,p2,p3,p4
      END SUBROUTINE 
      SUBROUTINE endtracebigsim5(e,step,p1,p2,p3,p4,p5) 
         CHARACTER* (*)  e
         INTEGER step
         DOUBLE PRECISION p1,p2,p3,p4,p5
      END SUBROUTINE 
      SUBROUTINE endtracebigsim6(e,step,p1,p2,p3,p4,p5,p6)
         CHARACTER* (*)  e
         INTEGER step
         DOUBLE PRECISION p1,p2,p3,p4,p5,p6
      END SUBROUTINE 
      SUBROUTINE endtracebigsim7(e,step,p1,p2,p3,p4,p5,p6,p7)
         CHARACTER* (*)  e
         INTEGER step
         DOUBLE PRECISION p1,p2,p3,p4,p5,p6,p7
      END SUBROUTINE 
      SUBROUTINE endtracebigsim8(e,step,p1,p2,p3,p4,p5,p6,p7,p8)
         CHARACTER* (*)  e
         INTEGER step
         DOUBLE PRECISION p1,p2,p3,p4,p5,p6,p7,p8
      END SUBROUTINE 
      SUBROUTINE endtracebigsim9(e,step,p1,p2,p3,p4,p5,p6,p7,p8,p9)
         CHARACTER* (*)  e
         INTEGER step
         DOUBLE PRECISION p1,p2,p3,p4,p5,p6,p7,p8,p9
      END SUBROUTINE 
      SUBROUTINE endtracebigsim10(e,step,p1,p2,p3,p4,p5,p6,p7,p8,p9,p10)
         CHARACTER* (*)  e
         INTEGER step
         DOUBLE PRECISION p1,p2,p3,p4,p5,p6,p7,p8,p9,p10
      END SUBROUTINE 
      SUBROUTINE endtracebigsim11(e,step,p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11)
         CHARACTER* (*)  e
         INTEGER step
         DOUBLE PRECISION p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11
      END SUBROUTINE 
   END INTERFACE 
end module charm
