subroutine init()
implicit none
include 'femf.h'

  integer :: i, j, nelems, nnodes, ctype, esize
  integer, dimension(:,:), allocatable:: conn

  call FEM_Print('init called')
  open(20, file='fmesh.dat')
  read(20,*) nelems, nnodes, ctype
  if (ctype .eq. FEM_TRIANGULAR) then
    esize = 3
  else 
    if(ctype .eq. FEM_HEXAHEDRAL) then
      esize = 8
    else
      esize = 4
    endif
  endif
  allocate(conn(nelems, esize))
  do i=1,nelems
    read(20,*) (conn(i,j),j=1,esize)
  enddo
  close(20)
  call FEM_Set_Mesh(nelems, nnodes, ctype, conn)
end subroutine init

subroutine driver(nnodes, nnums, nelems, enums, npere, conn)
implicit none
include 'femf.h'

  integer  :: nnodes, nelems, npere
  integer, dimension(nnodes) :: nnums
  integer, dimension(nelems) :: enums
  integer, dimension(nelems, npere) :: conn

  integer :: i, j, fid
  logical :: failed
  double precision :: sum
  double precision, dimension(nnodes) :: nodes
  double precision, dimension(nelems) :: elements

  !call FEM_Print_Partition()

  nodes = 0.0
  elements = 0.0
  do i=1,nnodes
    if (nnums(i) .eq. 1) nodes(i) = 1.0
  enddo
  fid = FEM_Create_Field(FEM_DOUBLE, 1, 0, offsetof(nodes(1),nodes(2)))
  do i=1,nelems
    do j=1,npere
      elements(i) = elements(i) + nodes(conn(i,j))
    enddo
    elements(i) = elements(i)/npere
  enddo
  nodes = 0.0
  do i=1,nelems
    do j=1,npere
      nodes(conn(i,j)) = nodes(conn(i,j)) + elements(i)
    enddo
  enddo
  call FEM_Update_Field(fid, nodes(1))
  failed = .FALSE.
  do i=1,nnodes
    if (nnums(i).eq.1 .or. nnums(i).eq.2 .or. &
&       nnums(i).eq.4 .or. nnums(i).eq.5) then 
      if(nodes(i) .ne. 0.25) failed = .TRUE.
    else
      if (nodes(i) .ne. 0.0) failed = .TRUE.
    endif
  enddo
  if (failed) then
    call FEM_Print('update_field test failed.')
  else
    call FEM_Print('update_field test passed.')
  endif
  sum = 0.0
  call FEM_Reduce_Field(fid, nodes(1), sum, FEM_SUM)
  if (sum .eq. 1.0) then
    call FEM_Print('reduce_field test passed.')
  else
    call FEM_Print('reduce_field test failed.')
  endif
  sum = 1.0
  call FEM_Reduce(fid, sum, sum, FEM_SUM)
  if (sum .eq. FEM_Num_Partitions()) then
    call FEM_Print('reduce test passed.')
  else
    call FEM_Print('reduce test failed.')
  endif
  call FEM_Done()
end subroutine driver

subroutine finalize()
implicit none
include 'femf.h'
  call FEM_Print('finalize called')
end subroutine
