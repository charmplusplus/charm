
       integer, parameter :: ILSI_PARAM=20
       
       external ILSI_CG_Solver
       
       interface
       
       subroutine ILSI_Param_new(param) 
           integer,intent(inout) :: param(ILSI_PARAM)
       end subroutine
       
       end interface

