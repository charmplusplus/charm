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
    
    //Add strategy to listening list, strategy will get an the number
    //of array elements lying on that processor
    void registerStrategy(StrategyTableEntry *);

    //remove strategy from table, and now it will not get updates
    //about this array
    void unregisterStrategy(StrategyTableEntry *entry) {
        for(size_t count = 0; count < strategyList.size(); count++)
            if(strategyList[count] == entry)
                strategyList.remove(count);
    }

    void getLocalIndices(CkVec<CkArrayIndexMax> &vec);

    void pup(PUP::er &p);
    PUPable_decl(ComlibArrayListener);
};

#endif
