       
       integer, parameter :: IDXL_FIRST_DATATYPE=1510000000
       integer, parameter :: IDXL_BYTE=(IDXL_FIRST_DATATYPE+0)
       integer, parameter :: IDXL_INT=(IDXL_FIRST_DATATYPE+1)
       integer, parameter :: IDXL_REAL=(IDXL_FIRST_DATATYPE+2)
       integer, parameter :: IDXL_DOUBLE=(IDXL_FIRST_DATATYPE+3)
       integer, parameter :: IDXL_INDEX_0=(IDXL_FIRST_DATATYPE+4)
       integer, parameter :: IDXL_INDEX_1=(IDXL_FIRST_DATATYPE+5)
       
       integer, parameter :: IDXL_FIRST_REDTYPE=1520000000
       integer, parameter :: IDXL_SUM=(IDXL_FIRST_REDTYPE+0)
       integer, parameter :: IDXL_PROD=(IDXL_FIRST_REDTYPE+1)
       integer, parameter :: IDXL_MAX=(IDXL_FIRST_REDTYPE+2)
       integer, parameter :: IDXL_MIN=(IDXL_FIRST_REDTYPE+3)
       
       external IDXL_Comm_sendrecv
       external IDXL_Comm_sendsum
       
       external IDXL_Comm_send
       external IDXL_Comm_recv
       external IDXL_Comm_sum
       
       
       interface
       function IDXL_Create()
         integer :: IDXL_Create
       end function
       subroutine IDXL_Print(l)
         integer,intent (in) :: l
       end subroutine    
       subroutine IDXL_Copy(l,src)
         integer,intent (in) :: l, src
       end subroutine    
       subroutine IDXL_Shift(l,startSend,startRecv)
         integer, intent (in) :: l
	 integer, intent(in) :: startSend, startRecv
       end subroutine    
       subroutine IDXL_Combine(l,src,startSend,startRecv)
         integer, intent (in) :: l, src
	 integer, intent(in) :: startSend, startRecv
       end subroutine    
       subroutine IDXL_Add_entity(l,newIdx,nBetween,between)
         integer, intent (in) :: l
	 integer, intent(in) :: newIdx, nBetween
	 integer, intent(in) :: between(nBetween)
       end subroutine  
       subroutine IDXL_Sort_2d(l,coord2d)
         integer, intent (in) :: l
	 double precision, intent(in) :: coord2d(:)
       end subroutine    
       subroutine IDXL_Sort_3d(l,coord3d)
         integer, intent (in) :: l
	 double precision, intent(in) :: coord3d(:)
       end subroutine    
       subroutine IDXL_Destroy(l)
         integer,intent (in) :: l
       end subroutine    
       
       function IDXL_Get_source(l,localNo)
          integer, intent(in) :: l,localNo
          integer :: IDXL_Get_source
       end function

       function IDXL_Layout_create(type,width)
          integer, intent(in) :: type, width
          integer :: IDXL_Layout_create
       end function
       function IDXL_Layout_offset(type,width,off,dist,skew)
          integer, intent(in) :: type, width, off, dist, skew
          integer :: IDXL_Layout_offset
       end function
       function IDXL_Get_layout_type(l)
          integer, intent(in) :: l
          integer :: IDXL_Get_layout_type
       end function
       function IDXL_Get_layout_width(l)
          integer, intent(in) :: l
          integer :: IDXL_Get_layout_width
       end function
       function IDXL_Get_layout_distance(l)
          integer, intent(in) :: l
          integer :: IDXL_Get_layout_distance
       end function
       subroutine IDXL_Layout_destroy(l)
         integer,intent (in) :: l
       end subroutine
       
       function IDXL_Comm_begin(tag,context) 
         integer,intent (in) :: tag,context
         integer :: IDXL_Comm_begin
       end function
       subroutine IDXL_Comm_flush(comm)
         integer,intent (in) :: comm
       end subroutine
       subroutine IDXL_Comm_wait(comm)
         integer,intent (in) :: comm
       end subroutine
       
       end interface
