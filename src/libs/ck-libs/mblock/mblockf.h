
       integer, parameter :: MBLK_BYTE=0
       integer, parameter :: MBLK_INT=1
       integer, parameter :: MBLK_REAL=2
       integer, parameter :: MBLK_DOUBLE=3

       integer, parameter :: MBLK_SUM=0
       integer, parameter :: MBLK_MAX=1
       integer, parameter :: MBLK_MIN=2

       integer, parameter :: MBLK_SUCCESS=1
       integer, parameter :: MBLK_FAILURE=2

       integer, parameter :: MBLK_DONE=1
       integer, parameter :: MBLK_NOTDONE=0

       integer, external :: offsetof

       external :: MBLK_Set_prefix
       external :: MBLK_Set_nblocks
       external :: MBLK_Set_dim

       external :: MBLK_Get_nblocks
       external :: MBLK_Get_myblock
       external :: MBLK_Get_blocksize
       external :: MBLK_Get_extent
       real*8, external :: MBLK_Timer
       external :: MBLK_Print
       external :: MBLK_Print_block

       external :: MBLK_Create_field
       external :: MBLK_Update_field
       external :: MBLK_Iupdate_field
       external :: MBLK_Test_update
       external :: MBLK_Wait_update

       external :: MBLK_Reduce_field
       external :: MBLK_Reduce

       external :: MBLK_Register_bc
       external :: MBLK_Aply_bc
       external :: MBLK_Apply_bc_all
       external :: MBLK_Get_boundary_extent

       external :: MBLK_Register
       external :: MBLK_Migrate
