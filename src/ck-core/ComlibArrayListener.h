#ifndef COMMLIBARRAYLISTENER_H
#define COMMLIBARRAYLISTENER_H

#include "ComlibStrategy.h"
#include "ckhashtable.h"

class ComlibArrayListener : public CkArrayListener{
    int nElements;
    CkArrayID thisArrayID;
    CkVec <StrategyTable *> strategyList;
    CkHashtableT<CkArrayIndexMax, CkArrayIndexMax*> *ht;
    
    int isRegistered(ArrayElement *elt, Strategy *strat);
    void addElement(ArrayElement *elt);
    void deleteElement(ArrayElement *elt);
    
 public:
    ComlibArrayListener();
    ComlibArrayListener(CkMigrateMessage *);

    void ckElementCreating(ArrayElement *elt);
    void ckElementDied(ArrayElement *elt);
    
    void ckElementLeaving(ArrayElement *elt);
    CmiBool ckElementArriving(ArrayElement *elt);
    
    void registerStrategy(StrategyTable *);

    void getLocalIndices(CkVec<CkArrayIndexMax> &vec);

    void pup(PUP::er &p);
    PUPable_decl(ComlibArrayListener);
};

#endif
