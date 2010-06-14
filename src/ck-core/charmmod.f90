module charm
  integer*8, external :: CmiMemoryUsage
  integer*8, external :: CmiMaxMemoryUsage
  real*8,    external :: CmiWallTimer
  real*8,    external :: CkWallTimer
  real*8,    external :: CmiCpuTimer
  real*8,    external :: CkCpuTimer

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
