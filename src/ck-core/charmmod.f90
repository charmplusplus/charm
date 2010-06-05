module charm
  integer*8, external :: CmiMemoryUsage
  integer*8, external :: CmiMaxMemoryUsage
  real*8,    external :: CmiWallTimer
  real*8,    external :: CkWallTimer
  real*8,    external :: CmiCpuTimer
  real*8,    external :: CkCpuTimer

  INTERFACE
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
   END INTERFACE 
end module charm
