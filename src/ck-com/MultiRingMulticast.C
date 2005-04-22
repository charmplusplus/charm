
#include "MultiRingMulticast.h"

//Array Constructor
MultiRingMulticast::MultiRingMulticast(CkArrayID dest_aid, int flag)
    : DirectMulticastStrategy(dest_aid, flag){
}


void MultiRingMulticast::pup(PUP::er &p){

    DirectMulticastStrategy::pup(p);
}

void MultiRingMulticast::beginProcessing(int  nelements){

    DirectMulticastStrategy::beginProcessing(nelements);
}


inline int getMyId(int *pelist, int npes, int mype) {
    int myid = -1;

    for(int count = 0; count < npes; count ++)
        if(mype == pelist[count])
            myid = count;

    if(myid == -1)
        CkPrintf("Warning myid = -1\n");

    return myid;
}


//Assumes a sorted input. Returns the next processor greater a given
//current pe
inline void getNextPe(int *pelist, int npes, int mype, int &nextpe) {
    
    nextpe = pelist[0];
    
    int count= 0;
    for(count = 0; count < npes; count++)
        if(pelist[count] > mype)
            break;
    
    if(count < npes) 
        nextpe = pelist[count];
    
    return;
}


inline int getMidPe(int *pelist, int npes, int src_pe) {
    
    int my_id = 0;
    int mid_pe = -1;
    
    my_id = getMyId(pelist, npes, src_pe);    
    
    CkAssert(my_id >= 0 && my_id < npes);

    if(my_id < npes/2)
        mid_pe = pelist[npes/2 + my_id];        
    else
        mid_pe = pelist[my_id % (npes/2)];
    
    if(mid_pe == -1)
        CkPrintf("Warning midpe = -1\n");

    return mid_pe;
}

//Unlike ring the source here sends two or more messages while all
//elements along the ring only send one.

ComlibSectionHashObject *MultiRingMulticast::createObjectOnSrcPe
(int nelements, CkArrayIndexMax *elements){

    ComlibSectionHashObject *obj = new ComlibSectionHashObject();

    obj->npes = 0;
    obj->pelist = 0;

    int *pelist;
    int npes;
    sinfo.getRemotePelist(nelements, elements, npes, pelist);
    
    sinfo.getLocalIndices(nelements, elements, obj->indices);

    if(npes == 0)
        return obj;

    if(npes == 1) {        
        obj->npes = 1;        
        obj->pelist = pelist;

        return obj;
    }

    if(npes == 2) {        
        obj->npes = 2;        
        obj->pelist = pelist;
        
        return obj;
    }
    
    pelist[npes ++] = CkMyPe();
    qsort(pelist, npes, sizeof(int), intCompare);

    char dump[2560];
    sprintf(dump, "Section on %d : ", CkMyPe());
    for(int count = 0; count < npes; count ++) {
        sprintf(dump, "%s, %d", dump, pelist[count]);
    }
    
    CkPrintf("%s\n\n", dump);

    int myid = getMyId(pelist, npes, CkMyPe());    

    CkAssert(myid >= 0 && myid < npes);

    int nextpe = -1;
    
    if(myid < npes / 2) 
        getNextPe(pelist, npes/2, CkMyPe(), nextpe);
    else 
        getNextPe(pelist+npes/2, npes - npes/2, CkMyPe(), nextpe);
    
    int mid_pe = -1;
    mid_pe = getMidPe(pelist, npes, CkMyPe());
    
    delete [] pelist;

    if(nextpe != CkMyPe()) {
        obj->pelist = new int[2];
        obj->npes = 2;
        
        obj->pelist[0] = nextpe;
        obj->pelist[1] = mid_pe;
    }
    else {
        CkPrintf("Warning Should not be here !!!!!!!!!\n");
        obj->pelist = new int[1];
        obj->npes = 1;
        
        obj->pelist[0] = mid_pe;
    }
    
    CkPrintf("%d Src = %d Next = %d Mid Pe =%d\n", CkMyPe(), CkMyPe(), nextpe, mid_pe);    
    
    return obj;
}


ComlibSectionHashObject *MultiRingMulticast::createObjectOnIntermediatePe
(int nelements, CkArrayIndexMax *elements, int src_pe){

    ComlibSectionHashObject *obj = new ComlibSectionHashObject();

    int *pelist;
    int npes;
    sinfo.getRemotePelist(nelements, elements, npes, pelist);
    
    obj->pelist = 0;
    obj->npes = 0;   
    sinfo.getLocalIndices(nelements, elements, obj->indices);

    pelist[npes ++] = CkMyPe();

    if(npes <= 3)
        return obj;
    
    qsort(pelist, npes, sizeof(int), intCompare);
    
    int myid = getMyId(pelist, npes, CkMyPe());
    
    CkAssert(myid >= 0 && myid < npes);

    int src_id = getMyId(pelist, npes, src_pe);
    
    if(src_id == -1) { //Src isnt the receipient of the multicast
        pelist[npes ++] = src_pe;
        qsort(pelist, npes, sizeof(int), intCompare);
        src_id = getMyId(pelist, npes, src_pe);
        myid = getMyId(pelist, npes, CkMyPe());
        
        CkAssert(src_id >= 0 && src_id < npes);
    }

    int nextpe = -1;
    int new_src_pe = src_pe;
    int mid_pe = getMidPe(pelist, npes, src_pe);
    
    if(myid < npes / 2) {
        getNextPe(pelist, npes/2, CkMyPe(), nextpe);

        //If source is in the other half I have to change to the
        //middle guy
        if(src_id >= npes/2)
            new_src_pe = mid_pe;
    }
    else {
        getNextPe(pelist + (npes/2), npes - npes/2, CkMyPe(), nextpe);

        //If source is in the other half I have to change to the
        //middle guy
        if(src_id < npes/2)
            new_src_pe = mid_pe;
    }

    bool end_flag = isEndOfRing(nextpe, new_src_pe);

    if(!end_flag) {
        obj->pelist = new int[1];
        obj->npes = 1;
        obj->pelist[0] = nextpe;
    }
    
    CkPrintf("%d: Src = %d Next = %d end = %d Midpe = %d\n", CkMyPe(), src_pe, 
             nextpe, end_flag, mid_pe);    
    
    delete [] pelist;
    return obj;
}

//We need to end the ring,
//    if next_pe is the same as the source_pe, or
//    if next_pe is the first processor in the ring, greater than srouce_pe.
//Both these comparisons are done in a 'cyclic' way with wraparounds.

int MultiRingMulticast::isEndOfRing(int next_pe, int src_pe){
    
    ComlibPrintf("[%d] isEndofring %d, %d\n", CkMyPe(), next_pe, src_pe);
    
    if((next_pe != CkMyPe()) && (next_pe != src_pe)) 
        return 0;
    
    return 1;
}

