
       integer, parameter :: FEM_BYTE=0
       integer, parameter :: FEM_INT=1
       integer, parameter :: FEM_REAL=2
       integer, parameter :: FEM_DOUBLE=3

       integer, parameter :: FEM_SUM=0
       integer, parameter :: FEM_MAX=1
       integer, parameter :: FEM_MIN=2

       integer, parameter :: FEM_TRIANGULAR=3
       integer, parameter :: FEM_TETRAHEDRAL=4
       integer, parameter :: FEM_HEXAHEDRAL=8
       integer, parameter :: FEM_QUADRILATERAL=4

       external FEM_Attach

       external FEM_Set_Mesh

       external FEM_Set_Node
       external FEM_Get_Node
       external FEM_Set_Elem
       external FEM_Get_Elem
       external FEM_Set_Elem_Conn_r
       external FEM_Get_Elem_Conn_r
       external FEM_Set_Elem_Conn_c
       external FEM_Get_Elem_Conn_c
       external FEM_Set_Node_Data_r
       external FEM_Get_Node_Data_r
       external FEM_Set_Elem_Data_r
       external FEM_Get_Elem_Data_r
       external FEM_Set_Node_Data_c
       external FEM_Get_Node_Data_c
       external FEM_Set_Elem_Data_c
       external FEM_Get_Elem_Data_c

       external FEM_Set_Partition
       external FEM_Serial_Split
       external FEM_Serial_Begin

       external FEM_Get_Elem_Numbers
       external FEM_Get_Node_Numbers

       external FEM_Add_Ghost_Layer
       external FEM_Add_Ghost_Elem

       external FEM_Get_Comm_Nodes

       external FEM_Reduce_Field
       external FEM_Reduce
       external FEM_Update_Field
       external FEM_Update_Ghost_Field
       external FEM_Read_Field
       external FEM_Print
       external FEM_Print_Partition
       integer, external :: offsetof

       external FEM_Barrier
       external FEM_Get_Ghost_List

       integer, external :: FEM_Register
       external FEM_Migrate 

       interface
       function FEM_Get_Node_Ghost()
         integer :: FEM_Get_Node_Ghost
       end function
       function FEM_Get_Elem_Ghost(elemType)
	 integer, intent(in) :: elemType
         integer :: FEM_Get_Elem_Ghost
       end function

       subroutine FEM_Add_Node(idx,nBetween,between)
         integer,intent (in) :: idx, nBetween
         integer, intent(in) :: between(nBetween)
       end subroutine       

       subroutine FEM_Exchange_Ghost_Lists(elemType,nIdx,localIdx)
         integer,intent (in) :: elemType,nIdx
         integer, intent(in) :: localIdx(nIdx)
       end subroutine

       function FEM_Get_Ghost_List_Length()
         integer :: FEM_Get_Ghost_List_Length
       end function
       
       function FEM_Get_Comm_Partners()
         integer :: FEM_Get_Comm_Partners
       end function
       function FEM_Get_Comm_Partner(partnerNo)
         integer, intent(in) :: partnerNo
         integer :: FEM_Get_Comm_Partner
       end function
       function FEM_Get_Comm_Count(partnerNo)
         integer, intent(in) :: partnerNo
         integer :: FEM_Get_Comm_Count
       end function

       function FEM_Create_Field(base_type, vec_len, init_offset, distance)
          integer, intent(in)  :: base_type, vec_len, init_offset, distance
          integer  :: FEM_Create_Field
       end function 

       function FEM_My_Partition()
          integer  :: FEM_My_Partition
       end function FEM_My_Partition

       function FEM_Num_Partitions()
         integer  :: FEM_Num_Partitions
       end function 

       function FEM_Timer()
         double precision  :: FEM_Timer
       end function

       end interface







