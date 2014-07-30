#include "CkSmpCoordinator.h"
#include "CkSmpCoordinationMsg.h"
#include "ckmulticast.h"

CkReduction::reducerType CkSmpCoordinationReducerId;

// custom reducer
CkReductionMsg *CkSmpCoordinationReducer(int nmsgs, CkReductionMsg **msgs){
  CkAssert(nmsgs > 0);
  CkSmpCoordinatorLeaderInfo *accum = (CkSmpCoordinatorLeaderInfo *) (msgs[0]->getData());

  for(int i = 1; i < nmsgs; i++){
    CkAssert(msgs[i]->getSize() == sizeof(CkSmpCoordinatorLeaderInfo));
    const int *compare = (int *) msgs[i]->getData();
    const CkSmpCoordinatorLeaderInfo *other = (CkSmpCoordinatorLeaderInfo *) msgs[i]->getData();

    *accum += *other;
  }

  return CkReductionMsg::buildNew(sizeof(CkSmpCoordinatorLeaderInfo), accum);
}

void CkSmpCoordinationRegisterReducers(){
  CkSmpCoordinationReducerId = CkReduction::addReducer(CkSmpCoordinationReducer);
}

#include "CkSmpCoordination.def.h"
