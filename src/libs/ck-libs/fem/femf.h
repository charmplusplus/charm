
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


  interface
  function FEM_Create_Field(base_type, vec_len, init_offset, distance)
     integer  :: base_type, vec_len, init_offset, distance
     integer  :: FEM_Create_Field
  end function FEM_Create_Field

  subroutine FEM_Update_Field(fid, nodes)
     integer  :: fid
     double precision         :: nodes
  end subroutine FEM_Update_Field

  subroutine FEM_Reduce_Field(fid, nodes, outbuf, op)
     integer  :: fid, op
     double precision :: nodes
     double precision :: outbut
  end subroutine FEM_Reduce_Field

  subroutine FEM_Reduce(fid, inbuf, outbuf, op)
     integer  :: fid, op
     double precision         :: inbuf, outbuf
  end subroutine FEM_Reduce

  function FEM_My_Partition()
     integer  :: FEM_My_Partition
  end function FEM_My_Partition

  function FEM_Num_Partitions()
    integer  :: FEM_Num_Partitions
  end function FEM_Num_Partitions

  subroutine FEM_Read_Field(fid, nodes, fname)
    integer  :: fid
    character*20        :: fname
    double precision        :: nodes
  end subroutine FEM_Read_Field

  function offsetof(first, second)
     integer  :: offsetof
     double precision        :: first, second
  end function offsetof

  subroutine FEM_Set_Mesh(nelem, nnodes, ctype, mesh)
    integer :: nelem, nnodes, ctype
    integer, dimension(:,:) :: mesh
  end subroutine FEM_Set_Mesh

  subroutine FEM_Print_Partition()
  end subroutine FEM_Print_Partition

  end interface
