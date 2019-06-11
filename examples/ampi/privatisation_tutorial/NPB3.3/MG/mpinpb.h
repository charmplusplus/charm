
c---------------------------------------------------------------------
c---------------------------------------------------------------------

      include           'mpif.h'

      integer           me, nprocs, root, dp_type
      common /mpistuff/ me, nprocs, root, dp_type

     
c-- Privatizing common block mpistuff --------------------------------

!$OMP THREADPRIVATE(/mpistuff/)

