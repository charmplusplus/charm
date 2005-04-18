
#include "RingMulticastStrategy.h"

//Array Constructor
RingMulticastStrategy::RingMulticastStrategy(CkArrayID dest_aid, int flag)
  : DirectMulticastStrategy(dest_aid, flag){
}


void RingMulticastStrategy::pup(PUP::er &p){

    DirectMulticastStrategy::pup(p);
}

void RingMulticastStrategy::beginProcessing(int  nelements){

    DirectMulticastStrategy::beginProcessing(nelements);
}

ComlibSectionHashObject *RingMulticastStrategy::createObjectOnSrcPe
(int nelements, CkArrayIndexMax *elements){

    ComlibSectionHashObject *robj = new ComlibSectionHashObject;

    int next_pe = CkNumPes();
    int acount = 0;
    int min_dest = CkNumPes();
    
    //Equivalent to sorting the list of destination processors and
    //sending the message to the next processor greater than MyPE.
    //If MyPE is the largest processor send it to minpe
    for(acount = 0; acount < nelements; acount++){
        
        CkArrayID dest;
        int nidx;
        CkArrayIndexMax *idx_list;        
        ainfo.getDestinationArray(dest, idx_list, nidx);

        int p = ComlibGetLastKnown(dest, elements[acount]);
        
        //Find the smallest destination
        if(p < min_dest)
            min_dest = p;
        
        //If there is a processor greater than me and less than next_pe
        //then he is my next_pe
        if(p > CkMyPe() && next_pe > p) 
            next_pe = p;       

        if (p == CkMyPe())
            robj->indices.insertAtEnd(elements[acount]);
    }
    
    //Recycle the destination pelist and start from the begining
    if(next_pe == CkNumPes() && min_dest != CkMyPe())        
        next_pe = min_dest;
    
    if(next_pe == CkNumPes())
        next_pe = -1;

    if(next_pe != -1) {
        robj->pelist = new int[1];
        robj->npes = 1;
        robj->pelist[0] = next_pe;
    }
    else {
        robj->pelist = NULL;
        robj->npes = 0;
    }        
    
    return robj;
}


ComlibSectionHashObject *RingMulticastStrategy::createObjectOnIntermediatePe
(int nelements, CkArrayIndexMax *elements, int src_pe){

    ComlibSectionHashObject *obj;

    obj = createObjectOnSrcPe(nelements, elements);
    
    //here we check if have reached the end of the ring
    if(obj->npes > 0 && isEndOfRing(*obj->pelist, src_pe)) {
        delete obj->pelist;
        obj->pelist = NULL;
        obj->npes =0;
    }

    return obj;
}

//We need to end the ring, 
//    if next_pe is the same as the source_pe, or
//    if next_pe is the first processor in the ring, greater than srouce_pe.
//Both these comparisons are done in a 'cyclic' way with wraparounds.

int RingMulticastStrategy::isEndOfRing(int next_pe, int src_pe){
    
    if(next_pe < 0)
        return 1;
    
    ComlibPrintf("[%d] isEndofring %d, %d\n", CkMyPe(), next_pe, src_pe);
    
    if(next_pe > CkMyPe()){
        if(src_pe <= next_pe && src_pe > CkMyPe())
            return 1;
        
        return 0;
    }
    
    if(src_pe > CkMyPe() || src_pe <= next_pe)
        return 1;
    
    return 0;
}
