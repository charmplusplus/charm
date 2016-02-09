/**
 * \addtogroup CkDvfs
*/
/*@{*/

#include "energyOpt.h"

CkGroupID _energyOptimizer;

static inline void _energyOptInit(char **argv){
    //CkpvInitialize(FreqController *, _freqController);
    //CkpvAccess(_freqController) = new FreqController;
}
extern void energyCharmInit(char **argv){
    _energyOptInit(argv);
}


#include "energyOpt.def.h"
