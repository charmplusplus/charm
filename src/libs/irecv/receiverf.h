  integer, parameter :: AMPI_DOUBLE_PRECISION=0
  integer, parameter :: AMPI_INTEGER=1
  integer, parameter :: AMPI_REAL=2
  integer, parameter :: AMPI_COMPLEX=3
  integer, parameter :: AMPI_LOGICAL=4
  integer, parameter :: AMPI_CHARACTER=5
  integer, parameter :: AMPI_BYTE=6
  integer, parameter :: AMPI_PACKED=7

  INTERFACE
    DOUBLE PRECISION  FUNCTION IMPI_Wtime()
    END FUNCTION
  END INTERFACE
