#ifndef COMMLIBARRAYLISTENER_H
#define COMMLIBARRAYLISTENER_H

#include "ComlibStrategy.h"
#include "ckhashtable.h"

class ComlibArrayListener : public CkArrayListener{
    int nElements;
    CkArrayID thisArrayID;
    CkVec <StrategyTableEntry *> strategyList;
    CkHashtableT<CkArrayIndexMax, CkArrayIndexMax*> *ht;
    
    int isRegistered(ArrayElement *elt, CharmStrategy *astrat);
    void addElement(ArrayElement *elt, CmiBool mogration_flag);
    void deleteElement(ArrayElement *elt, CmiBool migration_flag);
    
 public:
    ComlibArrayListener();
    ComlibArrayListener(CkMigrateMessage *);

    void ckElementCreating(ArrayElement *elt);
    void ckElementDied(ArrayElement *elt);
    
    void ckElementLeaving(ArrayElement *elt);
    CmiBool ckElementArriving(ArrayElement *elt);
    
    void registerStrategy(StrategyTableEntry *);

    void getLocalIndices(CkVec<CkArrayIndexMax> &vec);

    void pup(PUP::er &p);
    PUPable_decl(ComlibArrayListener);
};

#endif
