
/****************************************************************************
             Fortran API for common LBDB functions
****************************************************************************/

#include "charm++.h"
#include "charm-api.h"


FDECL {

#define flbturninstrumentoff   FTN_NAME(FLBTURNINSTRUMENTATIONOFF, flbturninstrumentationoff)
#define flbturninstrumenton    FTN_NAME(FLBTURNINSTRUMENTATIONON, flbturninstrumentationon)

void flbturninstrumenton()
{
  LBTurnInstrumentOn();
}

void flbturninstrumentoff()
{
  LBTurnInstrumentOff();
}

}  // FDECL
