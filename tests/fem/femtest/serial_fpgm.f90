program main
  implicit none
  include 'femf.h'
  include 'tcharmf.h'
  integer npieces,i
  call TCharm_Init
  call init
  npieces=3
  call FEM_Serial_Split(npieces)
  do i=1,npieces
    call FEM_Serial_Begin(i)
    call driver
  end do
end program

subroutine init()
implicit none
include 'femf.h'

  integer :: i, j, nelems, nnodes, esize
  integer, dimension(:,:), allocatable:: conn
  double precision, dimension(:,:), allocatable:: nodeData

  call FEM_Print('init called')
  open(20, file='fmesh.dat')
  read(20,*) nelems, nnodes, esize
  allocate(conn(nelems, esize))
  do i=1,nelems
    read(20,*) (conn(i,j),j=1,esize)
  enddo
  close(20)
  call FEM_Set_Elem(1,nelems,0,esize)
  call FEM_Set_Elem_Conn_c(1,conn)
  
  allocate(nodeData(nnodes,2))
  do i=1,nnodes
     nodeData(i,1)=0
     nodeData(i,2)=0
  enddo
  nodeData(1,1)=1
  nodeData(1,2)=0.25
  nodeData(2,2)=0.25
  nodeData(4,2)=0.25
  nodeData(5,2)=0.25
  call FEM_Set_Node(nnodes,2)
  call FEM_Set_Node_Data_c(nodeData)
end subroutine init

subroutine driver()
implicit none
include 'femf.h'

  integer  :: nnodes,nodeDataP, nelems, elemData,npere
  integer, dimension(:,:), allocatable:: conn
  double precision, dimension(:,:), allocatable:: nodeData
  integer :: i, j, fid
  logical :: failed
  double precision :: sum
  double precision, dimension(:), allocatable:: nodes
  double precision, dimension(:), allocatable:: elements


  call FEM_Get_Elem(1,nelems,elemData,npere)
  call FEM_Get_Node(nnodes,nodeDataP)

  allocate(conn(nelems, npere))
  call FEM_Get_Elem_Conn_c(1,conn)
  allocate(nodeData(nnodes,nodeDataP))
  call FEM_Get_Node_Data_c(nodeData);

  allocate(nodes(nnodes))
  allocate(elements(nelems))

  call FEM_Print_Partition()

end subroutine driver

subroutine mesh_updated(param)
  implicit none
  integer :: param
  include 'femf.h'
  call FEM_Print('mesh_updated called')
end subroutine

subroutine finalize()
implicit none
include 'femf.h'
  call FEM_Print('finalize called')
end subroutine
