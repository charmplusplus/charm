#include "charm++.h"
#include "envelope.h"
#include "ckhashtable.h"

ComlibArrayListener::ComlibArrayListener () 
    : CkArrayListener(0){ //Carry 1 int for the sid, not used yet
    nElements = 0;
    ht = new CkHashtableT<CkArrayIndexMax, CkArrayIndexMax *>;
    //    CkPrintf("Creating Array Listener\n");
}

ComlibArrayListener::ComlibArrayListener (CkMigrateMessage *m)
    :CkArrayListener(m) {
    nElements = 0;
    ht = new CkHashtableT<CkArrayIndexMax, CkArrayIndexMax *>;
}

void ComlibArrayListener::pup(PUP::er &p) {}

void ComlibArrayListener::ckElementCreating(ArrayElement *elt){
    addElement(elt);
    //CkPrintf("[%d] Element Created\n", CkMyPe());
}

void ComlibArrayListener::ckElementDied(ArrayElement *elt){
    deleteElement(elt);
}

void ComlibArrayListener::ckElementLeaving(ArrayElement *elt){
    deleteElement(elt);
}

CmiBool ComlibArrayListener::ckElementArriving(ArrayElement *elt){
    addElement(elt);
    return CmiTrue;
}

void ComlibArrayListener::addElement(ArrayElement *elt){
    if(nElements == 0)
        thisArrayID = elt->ckGetArrayID();

    ht->put(elt->thisIndexMax) = &(elt->thisIndexMax);
    //elt->thisIndexMax.print();
    nElements ++;

    for(int count = 0; count < strategyList.length(); count++){
        Strategy *strategy = strategyList[count]->strategy;
        if(isRegistered(elt, strategy)) {
            strategyList[count]->numElements ++;
            //strategyList[count]->strategy->insertLocalIndex(elt->thisIndexMax);
        }
    }   
}

void ComlibArrayListener::deleteElement(ArrayElement *elt){
    ht->remove(elt->thisIndexMax);
    nElements --;
    
    for(int count = 0; count < strategyList.length(); count++){
        Strategy *strategy = strategyList[count]->strategy;
        if(isRegistered(elt, strategy)) {
            strategyList[count]->numElements --;
            //strategyList[count]->strategy->removeLocalIndex(elt->thisIndexMax);
        }
    }   
}

int ComlibArrayListener::isRegistered(ArrayElement *elt, Strategy *strat){
    CkArrayIndexMax idx = elt->thisIndexMax;

    CkArrayID st_aid;
    int st_nelements;
    CkArrayIndexMax *st_elem;
    strat->getSourceArray(st_aid, st_elem, st_nelements);

    if(st_nelements <= 0)
        return 1;

    for(int count = 0; count < st_nelements; count ++)
        if(st_elem[count].compare(idx))
            return 1;

    return 0;
}
 
//Assumes strategy is already present in the strategy table   
void ComlibArrayListener::registerStrategy(StrategyTable *stable_entry){    
    strategyList.insertAtEnd(stable_entry);

    Strategy *strat = stable_entry->strategy;

    CkArrayID st_aid;
    int st_nelements;
    CkArrayIndexMax *st_elem;
    strat->getSourceArray(st_aid, st_elem, st_nelements);

    if(st_nelements <= 0) {//All elements of array in strategy
        stable_entry->numElements += nElements;
        /*
        CkHashtableIterator *ht_iterator = ht->iterator();
        ht_iterator->seekStart();
        while(ht_iterator->hasNext()){
            CkArrayIndexMax *idx;
            ht_iterator->next((void **)&idx);
            stable_entry->strategy->insertLocalIndex(*idx);       
        }
        */
    }
    else { //Only some elements belong to strategy
        for(int count = 0; count < st_nelements; count ++)
            if(ht->get(st_elem[count]) != NULL) {
                stable_entry->numElements ++;
                //stable_entry->strategy->insertLocalIndex(st_elem[count]);
            }
    }
}

void ComlibArrayListener::getLocalIndices(CkVec<CkArrayIndexMax> &vec){
    
    CkHashtableIterator *ht_iterator = ht->iterator();
    ht_iterator->seekStart();
    while(ht_iterator->hasNext()){
        CkArrayIndexMax *idx;
        ht_iterator->next((void **)&idx);
        vec.insertAtEnd(*idx);       
    }
}


