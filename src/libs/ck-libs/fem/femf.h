
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

      external FEM_Reduce_Field
      external FEM_Reduce
      external FEM_Update_Field
      external FEM_Read_Field
      external FEM_Print
      external FEM_Print_Partition
      integer, external :: offsetof

      integer, external :: FEM_Register
      external FEM_Migrate

      interface

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







