subroutine init()
implicit none
include 'femf.h'

  integer :: i, j, nelems, nnodes, esize
  integer, dimension(:,:), allocatable:: conn
  double precision, dimension(:,:), allocatable:: nodeData

  call FEM_Print('init called')
  open(20, file='fmesh.dat')
  read(20,*) nelems, nnodes, esize
  
  allocate(nodeData(2,nnodes))
  do i=1,nnodes
     nodeData(1,i)=0
     nodeData(2,i)=0
  enddo
  nodeData(1,1)=1
  nodeData(2,1)=0.25
  nodeData(2,2)=0.25
  nodeData(2,4)=0.25
  nodeData(2,5)=0.25
  call FEM_Set_node(nnodes,2)
  call FEM_Set_node_data_r(nodeData)
  
  allocate(conn(esize,nelems))
  do i=1,nelems
    read(20,*) (conn(j,i),j=1,esize)
  enddo
  close(20)
  call FEM_Set_elem(1,nelems,0,esize)
  call FEM_Set_elem_conn_r(1,conn)
  
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


  call FEM_Get_elem(1,nelems,elemData,npere)
  call FEM_Get_node(nnodes,nodeDataP)

  allocate(conn(npere, nelems))
  call FEM_Get_elem_conn_r(1,conn)
  allocate(nodeData(nodeDataP, nnodes))
  call FEM_Get_node_data_r(nodeData);

  allocate(nodes(nnodes))
  allocate(elements(nelems))

  call FEM_Print_partition()

  nodes = 0.0
  elements = 0.0
  do i=1,nnodes
     nodes(i)=nodeData(1,i)
  enddo
  fid = FEM_Create_field(FEM_DOUBLE, 1, 0, 8)
  do i=1,nelems
    do j=1,npere
      elements(i) = elements(i) + nodes(conn(j,i))
    enddo
    elements(i) = elements(i)/npere
  enddo
  nodes = 0.0
  do i=1,nelems
    do j=1,npere
      nodes(conn(j,i)) = nodes(conn(j,i)) + elements(i)
    enddo
  enddo
  call FEM_Update_field(fid, nodes(1))
  failed = .FALSE.
  do i=1,nnodes
    if (nodes(i) .ne. nodeData(2,i)) failed= .TRUE.
  enddo
  if (failed) then
    call FEM_Print('update_field test failed.')
  else
    call FEM_Print('update_field test passed.')
  endif
  sum = 0.0
  call FEM_Reduce_field(fid, nodes(1), sum, FEM_SUM)
  if (sum .eq. 1.0) then
    call FEM_Print('reduce_field test passed.')
  else
    call FEM_Print('reduce_field test failed.')
  endif
  sum = 1.0
  call FEM_Reduce(fid, sum, sum, FEM_SUM)
  if (sum .eq. FEM_Num_partitions()) then
    call FEM_Print('reduce test passed.')
  else
    call FEM_Print('reduce test failed.')
  endif
  call FEM_Done()
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
