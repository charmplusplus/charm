/**
   @addtogroup CharmComlib
   @{
   @file
   This file is compiled and linked into all Charm++ programs, 
   even those that do not use -module comlib. Thus, a few functions
   are implemented here that are referenced from parts of the charm++ 
   core. These functions can handle both the case where the comlib
   module is used and where it is not used.
*/

#include "ComlibStrategy.h"

//calls ComlibNotifyMigrationDone(). Even compiles when -module comlib
//is not included. Hack to make loadbalancer work without comlib
//currently.
CkpvDeclare(int, migrationDoneHandlerID);


void ComlibNotifyMigrationDone() {
    if(CkpvInitialized(migrationDoneHandlerID)) 
        if(CkpvAccess(migrationDoneHandlerID) > 0) {
            char *msg = (char *)CmiAlloc(CmiReservedHeaderSize);
            CmiSetHandler(msg, CkpvAccess(migrationDoneHandlerID));
#if CMK_BIGSIM_CHARM
	    // bluegene charm should avoid directly calling converse
            CmiSyncSendAndFree(CkMyPe(), CmiReservedHeaderSize, msg);
#else
            CmiHandleMessage(msg);
#endif
        }
}

/*@}*/
