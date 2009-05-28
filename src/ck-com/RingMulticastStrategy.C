/** 
    @addtogroup ComlibCharmStrategy
    @{
    @file 
*/

#include "RingMulticastStrategy.h"


void RingMulticastStrategy::createObjectOnSrcPe(ComlibSectionHashObject *obj, int npes, ComlibMulticastIndexCount *pelist) {

  int next_pe = CkNumPes();
  int acount = 0;
  int min_dest = CkNumPes();
    
  //Equivalent to sorting the list of destination processors and
  //sending the message to the next processor greater than MyPE.
  //If MyPE is the largest processor send it to minpe
  for(acount = 0; acount < npes; acount++){
        
    int p = pelist[acount].pe;
        
    //Find the smallest destination
    if(p < min_dest) min_dest = p;
        
    //If there is a processor greater than me and less than next_pe
    //then he is my next_pe
    if(p > CkMyPe() && next_pe > p) next_pe = p;       

  }
  
  //Recycle the destination pelist and start from the begining
  if(next_pe == CkNumPes() && min_dest != CkMyPe()) next_pe = min_dest;
  
  if(next_pe == CkNumPes()) next_pe = -1;

  if(next_pe != -1) {
    obj->pelist = new int[1];
    obj->npes = 1;
    obj->pelist[0] = next_pe;
  }
  else {
    obj->pelist = NULL;
    obj->npes = 0;
  }
}

void RingMulticastStrategy::createObjectOnIntermediatePe(ComlibSectionHashObject *obj,  int npes, ComlibMulticastIndexCount *counts, int src_pe) {

  obj->pelist = new int[1];
  obj->npes = 1;
  obj->pelist[0] = CkMyPe(); // this is neutral for the if inside next loop

  // find the next processor in the ring
  for (int i=0; i<npes; ++i) {
    if (obj->pelist[0] > CkMyPe()) { // we have already found a processor greater
                                     // than us, find the smallest among them
      if (counts[i].pe > CkMyPe() && counts[i].pe < obj->pelist[0])
	obj->pelist[0] = counts[i].pe;
    } else {  // we have not yet found a processor greater than us, stick with
	      // the smallest, or one greater than us
      if (counts[i].pe < obj->pelist[0] || counts[i].pe > CkMyPe())
	obj->pelist[0] = counts[i].pe;
    }
  }
    
  //here we check if have reached the end of the ring
  if(obj->npes > 0 && isEndOfRing(*obj->pelist, src_pe)) {
    delete [] obj->pelist;
    obj->pelist = NULL;
    obj->npes = 0;
  }
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

/*@}*/
