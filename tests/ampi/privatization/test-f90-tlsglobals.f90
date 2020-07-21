      !!! Fortran module variables, implicit and explicit save variables,
      !!! and common blocks are all unsafe if mutable, and must be privatized.
      !!! This is done by declaring all mutable global/static variables with
      !!! OpenMP's threadprivate attribute and compiling with ampif90's -tlsglobals
      !!! option. Since parameter variables are immutable, they need not be privatized.

      module test_mod_tlsglobals

        implicit none

        integer, parameter :: parameter_variable = 0
        integer, target :: module_variable
        !$omp threadprivate(module_variable)

      end module test_mod_tlsglobals


      subroutine about_to_migrate

        implicit none
        include 'mpif.h'

        integer :: rank, ierr

        call mpi_comm_rank(MPI_COMM_WORLD, rank, ierr)
        ! Uncomment here and below when issue #2932 is fixed.
        ! print 1000, rank
        ! 1000 format ('[', I0, '] About to migrate.')

      end subroutine about_to_migrate

      subroutine just_migrated

        implicit none
        include 'mpif.h'

        integer :: rank, ierr

        call mpi_comm_rank(MPI_COMM_WORLD, rank, ierr)
        ! print 2000, rank
        ! 2000 format ('[', I0, '] Just migrated.')

      end subroutine just_migrated


      subroutine mpi_main

        implicit none
        include 'mpif.h'

        external about_to_migrate
        external just_migrated

        integer :: ierr

        call mpi_init(ierr)

        call ampi_register_about_to_migrate(about_to_migrate, ierr)
        call ampi_register_just_migrated(just_migrated, ierr)

        call privatization_test_framework()

        call mpi_finalize(ierr)

      end subroutine mpi_main


      subroutine subroutine_save(failed, test, rank, my_wth, operation)

        implicit none
        save

        integer :: failed, test, rank, my_wth, operation
        integer, target :: save_variable3
        !$omp threadprivate(save_variable3)

        call test_privatization(failed, test, rank, my_wth, operation, save_variable3)

      end subroutine subroutine_save


      subroutine perform_test_batch(failed, test, rank, my_wth, operation)

        use test_mod_tlsglobals
        implicit none

        integer :: failed, test, rank, my_wth, operation
        integer, target :: save_variable1 = 0
        integer, save, target :: save_variable2
        integer, target :: common_variable
        common /commons/ common_variable
        !$omp threadprivate(save_variable1)
        !$omp threadprivate(save_variable2)
        !$omp threadprivate(/commons/)

        call print_test_fortran(test, rank, 'module variable')
        call test_privatization(failed, test, rank, my_wth, operation, module_variable)

        call print_test_fortran(test, rank, 'implicit save variable')
        call test_privatization(failed, test, rank, my_wth, operation, save_variable1)

        call print_test_fortran(test, rank, 'explicit save variable')
        call test_privatization(failed, test, rank, my_wth, operation, save_variable2)

        call print_test_fortran(test, rank, 'subroutine save variable')
        call subroutine_save(failed, test, rank, my_wth, operation)

        call print_test_fortran(test, rank, 'common block variable')
        call test_privatization(failed, test, rank, my_wth, operation, common_variable)

      end subroutine perform_test_batch
