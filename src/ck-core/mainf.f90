      program charmxx

        use iso_c_binding, only:c_char, c_ptr, c_null_ptr, c_loc
        implicit none
        integer :: i, argc, ierr
        integer, parameter :: arg_len = 256
        character(kind=c_char, len=arg_len), dimension(:), allocatable, &
     &  target :: raw_arguments
        type(c_ptr), dimension(:), allocatable :: argv

        argc = command_argument_count() + 1

        allocate (raw_arguments(argc))
        allocate (argv(argc+1))

        do i = 1, argc
          call get_command_argument(i - 1, raw_arguments(i))
        end do
        do i = 1, argc
          raw_arguments(i) = trim(adjustl(raw_arguments(i)))//char(0)
        end do
        do i = 1, argc
          argv(i) = c_loc(raw_arguments(i))
        end do
        argv(argc+1) = c_null_ptr

        call charm_main_fortran_wrapper(argc, argv)

        if (allocated(argv)) deallocate(argv)
        if (allocated(raw_arguments)) deallocate(raw_arguments)

      end program charmxx
