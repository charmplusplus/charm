! This file contains ONLY the definitions of routines that take
! a string argument whose length AMPI needs to know.

! Please modify ampif.h and ampif.C to make any other changes.

subroutine MPI_Comm_set_name(comm, comm_name, ierr)
  implicit none
  integer, intent(in) :: comm
  character(len=*), intent(in) :: comm_name
  integer, intent(out) :: ierr

  call ampif_comm_set_name(comm, comm_name, len(comm_name), ierr)

end subroutine MPI_Comm_set_name

subroutine MPI_Type_set_name(datatype, datatype_name, ierr)
  implicit none
  integer, intent(in) :: datatype
  character(len=*), intent(in) :: datatype_name
  integer, intent(out) :: ierr

  call ampif_type_set_name(datatype, datatype_name, len(datatype_name), ierr)

end subroutine MPI_Type_set_name

subroutine MPI_Win_set_name(win, win_name, ierr)
  implicit none
  integer, intent(in) :: win
  character(len=*), intent(in) :: win_name
  integer, intent(out) :: ierr

  call ampif_win_set_name(win, win_name, len(win_name), ierr)

end subroutine MPI_Win_set_name

subroutine MPI_Info_set(info, key, val, ierr)
  implicit none
  integer, intent(in) :: info
  character(len=*), intent(in) :: key, val
  integer, intent(out) :: ierr

  call ampif_info_set(info, key, val, len(key), len(val), ierr)

end subroutine MPI_Info_set

subroutine MPI_Info_delete(info, key, ierr)
  implicit none
  integer, intent(in) :: info
  character(len=*), intent(in) :: key
  integer, intent(out) :: ierr

  call ampif_info_delete(info, key, len(key), ierr)

end subroutine MPI_Info_delete

subroutine MPI_Info_get(info, key, vlen, val, flag, ierr)
  implicit none
  integer, intent(in) :: info
  character(len=*), intent(in) :: key
  integer, intent(in) :: vlen
  character(len=vlen), intent(out) :: val
  logical, intent(out) :: flag
  integer, intent(out) :: ierr

  call ampif_info_get(info, key, vlen, val, flag, len(key), ierr)

end subroutine MPI_Info_get

subroutine MPI_Info_get_valuelen(info, key, vlen, flag, ierr)
  implicit none
  integer, intent(in) :: info
  character(len=*), intent(in) :: key
  integer, intent(out) :: vlen
  logical, intent(out) :: flag
  integer, intent(out) :: ierr

  call ampif_info_get_valuelen(info, key, vlen, flag, len(key), ierr)

end subroutine MPI_Info_get_valuelen

subroutine MPI_Add_error_string(errorcode, string, ierr)
  implicit none
  integer, intent(in) :: errorcode
  character(len=*), intent(in) :: string
  integer, intent(out) :: ierr

  call ampif_add_error_string(errorcode, string, len(string), ierr)

end subroutine MPI_Add_error_string

subroutine AMPI_Print(string, ierr)
  implicit none
  character(len=*), intent(in) :: string
  integer, intent(out) :: ierr

  call ampif_print(string, len(string), ierr)

end subroutine AMPI_Print
