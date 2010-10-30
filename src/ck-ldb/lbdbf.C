
/****************************************************************************
             Fortran API for common LBDB functions
****************************************************************************/

#include "charm++.h"
#include "charm-api.h"


#define flbturninstrumentoff  FTN_NAME(FLBTURNINSTRUMENTOFF, flbturninstrumentoff)
#define flbturninstrumenton   FTN_NAME(FLBTURNINSTRUMENTON, flbturninstrumenton)

FDECL void flbturninstrumenton()
{
  LBTurnInstrumentOn();
}

FDECL void flbturninstrumentoff()
{
  LBTurnInstrumentOff();
}

