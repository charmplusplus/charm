      subroutine mpi_main

        implicit none
        include 'mpif.h'
        integer :: ierr

        call mpi_init(ierr)

        call privatization_test_framework()

        call mpi_finalize(ierr)

      end subroutine mpi_main


      subroutine perform_test_batch(failed, rank, my_wth)

        implicit none
        integer failed, rank, my_wth
        integer global_variable
        common /globals/ global_variable

        call test_privatization(failed, rank, my_wth, global_variable)

      end subroutine perform_test_batch
