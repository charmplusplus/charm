/** \file ckgraph.h
 *  Author: Abhinav S Bhatele
 *  Date Created: October 29th, 2010
 *  E-mail: bhatele@illinois.edu
 *
 */

/**
 *  \addtogroup CkLdb
 */

/*@{*/

#ifndef _CKGRAPHTEMP_H_
#define _CKGRAPHTEMP_H_

#include <vector>
#include "BaseLB.h"
#include "ckgraph.h"

class ProcArrayTemp : ProcArray {
  public:
    ProcArrayTemp(BaseLB::LDStats *stats);
    ~ProcArrayTemp() { }
};

#endif // _CKGRAPHTEMP_H_

/*@}*/

