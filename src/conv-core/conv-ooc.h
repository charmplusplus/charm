/**
 * Converse Out-Of-Core (OOC) extended memory support.
 * The basic idea is to manually shuffle objects
 * in and out of memory to disk files.  We can do this
 * better than the OS's virtual memory can, because we
 * know about the message queue, and hence have more
 * information about what pages will be needed soon.
 *
 * OOC Implemented by Mani Potnuru, 1/2003
 */
#ifndef __CMI_COMMON_OOC_H
#define __CMI_COMMON_OOC_H

#include <stdio.h> /* for FILE */

#ifdef __cplusplus
 extern "C" {
#endif

/**
 * This struct is our representation for 
 * out-of-core "managers", which actually talk to
 * out-of-core capable objects.  For example, the
 * Charm++ Array Manager is a Prefetch Manager.
 */
typedef struct _CooPrefetchManager {
	/**
 	 * Return the out-of-core objid (from CooRegisterObject)
	 * that this Converse message will access.  If the message
	 * will not access an object, return -1.
	 */
        int (* msg2ObjId) (void *msg);
	
	/**
	 * Write this object (registered with RegisterObject)
	 * to this writable file.
	 */
        void (* writeToSwap) (FILE *swapfile,void *objptr);
	
	/**
	 * Read this object (registered with RegisterObject)
	 * from this readable file.
	 */
        void (* readFromSwap) (FILE *swapfile,void *objptr);
} CooPrefetchManager;

/**
 * Register a new Out-Of-Core manager at this Converse
 * handler index.
 *   @param pf The new object manager to register.
 *   @param handlerIdx The Converse handler to override.
 */
extern void CooRegisterManager(CooPrefetchManager *pf,int handlerIdx);


/**
 * Register a new prefetchable out-of-core object into the
 * prefetch table.  Returns the object's new objid.
 *   @param pf The new object's manager, which must previously have been
 *             passed to RegisterManager.
 *   @param objsize The new object's (current) memory size, in bytes.
 *   @param objptr The new object's location.  This pointer is 
 *                only used to pass back to writeToSwap or readFromSwap.
 */
extern int CooRegisterObject(CooPrefetchManager *pf,int objsize,void *objptr);

/**
 * Delete this object from the prefetch tables.  This can
 * happen when the object is destroyed, or when it migrates away.
 */
extern void CooDeregisterObject(int objid);


/**
 * This object's size on disk just changed.
 *   @param objid The object, as returned by CooRegisterObject.
 *   @param newsize The object's new storage size, in bytes.
 */
extern void CooSetSize(int objid,int newsize); 

/** 
 * This object is needed in memory--load it from disk
 * unless it's already in memory.
 *   @param objid The object, as returned by CooRegisterObject.
 */
extern void CooBringIn(int objid); 


#ifdef __cplusplus
 };
#endif

#endif
