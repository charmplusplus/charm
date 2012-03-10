/**
 * \addtogroup CkLdb
*/
/*@{*/

/**
  a module which determines if a load balancing cycle is profitable
*/

#include "BaseLB.h"

class LBProfit
{
public:
  virtual int profitable(BaseLB::ProcStats *procArray, int np);
};


/*@}*/
