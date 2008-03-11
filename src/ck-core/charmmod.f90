module charm
  integer*8, external :: CmiMemoryUsage
  integer*8, external :: CmiMaxMemoryUsage
  real*8,    external :: CmiWallTimer
  real*8,    external :: CkWallTimer
  real*8,    external :: CmiCpuTimer
  real*8,    external :: CkCpuTimer
end module charm
