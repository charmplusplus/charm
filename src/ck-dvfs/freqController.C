#include "freqController.h"

CkpvDeclare(FreqController*, _freqController);

static inline void _energyOptInit(char **argv){
	CkpvInitialize(FreqController *, _freqController);
	CkpvAccess(_freqController) = new FreqController;
}
extern void energyCharmInit(char **argv){
	_energyOptInit(argv);
}


