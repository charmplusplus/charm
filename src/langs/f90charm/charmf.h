#include "charm-api.h"

#define CMPI_DOUBLE_PRECISION 0
#define CMPI_INTEGER 1
#define CMPI_REAL 2
#define CMPI_COMPLEX 3
#define CMPI_LOGICAL 4
#define CMPI_CHAR 5
#define CMPI_BYTE 6
#define CMPI_PACKED 7 

extern "C" int typesize(int type, int count);

