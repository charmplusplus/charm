
  integer, parameter :: FEM_BYTE=0
  integer, parameter :: FEM_INT=1
  integer, parameter :: FEM_REAL=2
  integer, parameter :: FEM_DOUBLE=3

  integer, parameter :: FEM_SUM=0
  integer, parameter :: FEM_MAX=1
  integer, parameter :: FEM_MIN=2

  integer, parameter :: FEM_TRIANGULAR=1
  integer, parameter :: FEM_TETRAHEDRAL=2
  integer, parameter :: FEM_HEXAHEDRAL=3
  integer, parameter :: FEM_QUADRILATERAL=4


  external FEM_Set_Mesh
  external FEM_Reduce_Field
  external FEM_Reduce
  external FEM_Update_Field
  external FEM_Read_Field
  external FEM_Set_Mesh_Transform
  external FEM_Print_Partition
  integer, external :: offsetof

  interface
  function FEM_Create_Field(base_type, vec_len, init_offset, distance)
     integer  :: base_type, vec_len, init_offset, distance
     integer  :: FEM_Create_Field
  end function FEM_Create_Field

  function FEM_My_Partition()
     integer  :: FEM_My_Partition
  end function FEM_My_Partition

  function FEM_Num_Partitions()
    integer  :: FEM_Num_Partitions
  end function FEM_Num_Partitions

  function FEM_Timer()
    double precision  :: FEM_Timer
  end function FEM_Timer

  end interface
