/********************************************************
OLD DESCRIPTION
        Section multicast strategy suite. DirectMulticast and its
        derivatives, multicast messages to a section of array elements
        created on the fly. The section is invoked by calling a
        section proxy. These strategies can also multicast to a subset
        of processors for groups.

        These strategies are non-bracketed. When the first request is
        made a route is dynamically built on the section. The route
        information is stored in

 - Sameer Kumar

**********************************************/

/**
   @addtogroup ComlibCharmStrategy
   @{
   @file 
*/

#include "DirectMulticastStrategy.h"


void DirectMulticastStrategy::createObjectOnSrcPe(ComlibSectionHashObject *obj, int npes, ComlibMulticastIndexCount *pelist) {
	ComlibPrintf("[%d] old createObjectOnSrcPe() npes=%d\n", CkMyPe(), npes);
	
	obj->pelist = new int[npes];
	obj->npes = npes;
	for (int i=0; i<npes; ++i) {
		obj->pelist[i] = pelist[i].pe;
	}
}


void DirectMulticastStrategy::createObjectOnIntermediatePe(ComlibSectionHashObject *obj,
							   int npes,
							   ComlibMulticastIndexCount *counts,
							   int srcpe) {
	ComlibPrintf("[%d] old createObjectOnIntermediatePe()\n", CkMyPe());

    obj->pelist = 0;
    obj->npes = 0;
}


/*@}*/
