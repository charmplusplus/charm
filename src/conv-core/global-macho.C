/*  File: global-mach.C
 *  Author: Isaac Dooley
 *  
 *  This file is not yet fully functional. The makefile is not aware of it
 *  yet, but it is included if we decide to complete this functionality for
 *  the Mach-O executable format and loader.
 *
 *  This code currently can swap in and out entire DATA segments as required,
 *  but it needs some way of identifying the Cpv_* variables which should not
 *  be swapped. Unfortunately there is no Global Offset Table as there is with
 *  ELF, so we cannot just scan the variables and build our own table.
 *  
 */

#include "converse.h"
#include "cklists.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <strings.h>
#include <errno.h>
#include <assert.h>

#include "converse.h"


#if CMK_HAS_MACHO_GETSECT_H


/* problems:
 * the Cpv_mapCreated_[_Cmi_myrank] apparently gets unset after one call to install
 * I'm not sure what is going on with it. Can I copy the entire DATA segment?
 */


#include <mach-o/getsect.h>

/* A global magic cookie to make sure no routine swaps in some completely wrong DATA SEGMENT */
/* A check like this may be used also to verify that the segement is correctly manipulated */
long magic_num_machoassert = 0x0123456;
#define CHECK_MAGIC_NUM {assert(magic_num_machoassert == 0x0123456);printf("_origCtg=%p\n", _origCtg);}


/****************** Global Variable Storage and Swapping *********************/
CpvStaticDeclare(CtgGlobals,_curCtg);
CtgGlobals _origCtg; // should probably be registered like the others, but I couldn't get it to work, somehow it would be undefined in the first call to Install(g=0x0);


/** Initialize the globals support (called on each processor). */
void CtgInit(void) {
	CHECK_MAGIC_NUM;
  
    CmiPrintf("CtgInit()\n");
    
    if (CmiMyNode()==0) {
        CmiPrintf("CHARM> -swapglobals enabled, but not yet finished for Mach-O\n");
    }
    
    const struct segment_command *seg = getsegbyname("__DATA");
    _origCtg = new CtgGlobalStruct;
    _origCtg->allocate(seg->vmsize);
    CHECK_MAGIC_NUM;
    assert(_origCtg->data_seg);
    memcpy(_origCtg->data_seg, (void *)seg->vmaddr, seg->vmsize);
    CHECK_MAGIC_NUM;
    CmiPrintf("_origCtg initialized\n");
    assert(_origCtg);
    CHECK_MAGIC_NUM;
    
    CtgGlobals cur = new CtgGlobalStruct;
    cur->allocate(seg->vmsize);
    CHECK_MAGIC_NUM;
    assert(cur);
    assert(cur->data_seg);
    memcpy(cur->data_seg, (void *)seg->vmaddr, seg->vmsize);
    CHECK_MAGIC_NUM;
    CpvAccess(_curCtg) = cur;
    assert(CpvAccess(_curCtg));
    assert(CpvAccess(_curCtg)->data_seg);
    assert(CpvAccess(_curCtg)->data_seg == cur->data_seg);
    
    CmiPrintf("_curCtg initialized\n");

}

size_t CtgGetSize()
{
  const struct segment_command *seg = getsegbyname("__DATA");
  assert(seg);
  CHECK_MAGIC_NUM;
  return seg->vmsize;
}

/** Copy the current globals into this new set */
CtgGlobals CtgCreate(void * buffer) {
    CtgGlobalStruct g = { buffer };
    const struct segment_command *seg = getsegbyname("__DATA");
    CHECK_MAGIC_NUM;
    printf("CtgCreate()\n");
    assert(seg);
    CHECK_MAGIC_NUM;
    memcpy(g->data_seg, (void *)seg->vmaddr, seg->vmsize);
    CHECK_MAGIC_NUM;
    return g;
}

/** Install this set of globals. */
void CtgInstall(CtgGlobals g) {
    printf("CtgInstall()\n");
    CHECK_MAGIC_NUM;

    printf("installing g=%p\n", g);
    
    CtgGlobals g_old = CtgCurrentGlobals();
    const struct segment_command *seg = getsegbyname("__DATA");
    assert(seg);
    CHECK_MAGIC_NUM;

    // First must uninstall the old data segment, if one is loaded
    if(g_old){
        memcpy(g_old->data_seg, (void *)seg->vmaddr, seg->vmsize);
    }
    else {
        printf("no current data segment to copy out into\n");
    }
    
    // Install the new data segment
    memcpy((void *)seg->vmaddr, g->data_seg, seg->vmsize);
    CpvAccess(_curCtg) = g;
    CHECK_MAGIC_NUM;

}

void CtgUninstall()
{
  CtgInstall(_origCtg);
}

CtgGlobals CtgCurrentGlobals(void){
    CHECK_MAGIC_NUM;
	return CpvAccess(_curCtg);
}

#else

#include "global-nop.C"

#endif

