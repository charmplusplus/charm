  integer, parameter :: AMPI_COMM_WORLD=0
  integer, parameter :: AMPI_STATUS_SIZE=3

  integer, parameter :: AMPI_DOUBLE_PRECISION=0
  integer, parameter :: AMPI_INTEGER=1
  integer, parameter :: AMPI_REAL=2
  integer, parameter :: AMPI_COMPLEX=3
  integer, parameter :: AMPI_LOGICAL=4
  integer, parameter :: AMPI_CHARACTER=5
  integer, parameter :: AMPI_BYTE=6
  integer, parameter :: AMPI_PACKED=7

  integer, parameter :: AMPI_SHORT=8
  integer, parameter :: AMPI_LONG=9
  integer, parameter :: AMPI_UNSIGNED_CHAR=10
  integer, parameter :: AMPI_UNSIGNED_SHORT=11
  integer, parameter :: AMPI_UNSIGNED=12
  integer, parameter :: AMPI_UNSIGNED_LONG=13
  integer, parameter :: AMPI_LONG_DOUBLE=14

  integer, parameter :: AMPI_MAX=1
  integer, parameter :: AMPI_MIN=2
  integer, parameter :: AMPI_SUM=3
  integer, parameter :: AMPI_PROD=4

  integer, external :: AMPI_Register
  double precision, external :: AMPI_Wtime
