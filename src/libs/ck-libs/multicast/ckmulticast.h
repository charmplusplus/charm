#ifndef _MULTICAST
#define _MULTICAST

#include "pup.h"
class mCastEntry;

class multicastSetupMsg;
class multicastGrpMsg;
class cookieMsg;
class CkMcastBaseMsg;
class reductionInfo;

typedef mCastEntry * mCastEntryPtr;
PUPbytes(mCastEntryPtr)

#define MAXMCASTCHILDREN 2

#include "CkMulticast.decl.h"

typedef void (*redClientFn)(CkSectionInfo sid, void *param,int dataSize,void *data);

/// Retrieve section info from a multicast msg. Part of API
extern void CkGetSectionInfo(CkSectionInfo &id, void *msg);



/**
 * A multicast manager group that is a CkDelegateMgr. Can manage all sections of different 
 * chare arrays, so all functions need a CkSectionInfo parameter to tell CkMulticastMgr which 
 * array section it should work on.
 */
class CkMulticastMgr: public CkDelegateMgr 
{
    private:
        int factor;           // spanning tree factor, can be negative
        unsigned int split_size;
        unsigned int split_threshold;
        
    public:
        // ------------------------- Cons/Des-tructors ------------------------
        CkMulticastMgr(CkMigrateMessage *m)  {}
        CkMulticastMgr(int _factor = 2, unsigned int _split_size = 8192, unsigned int _split_threshold = 8192):
            factor(_factor),
            split_size(_split_size),
            split_threshold(_split_threshold) {}
        int useDefCtor(void){ return 1; }
        void pup(PUP::er &p){ 
		CkDelegateMgr::pup(p);
		p|factor;
		p|split_size;
		p|split_threshold;
	}

        // ------------------------- Spanning Tree Setup ------------------------
        /// Stuff section member info into CkSectionInfo and call initCookie for the tree building
        void setSection(CkSectionInfo &id, CkArrayID aid, CkArrayIndex *, int n);
        /// Call initCookie to start the tree build
        void setSection(CkSectionInfo &id);
        /// @deprecated { Use the other setSection methods }
        void setSection(CProxySection_ArrayElement &proxy);
        /// entry Start the build of a (branch of a) spanning tree rooted at you
        void setup(multicastSetupMsg *);
        /// entry My direct children in the tree use this to tell me that they are ready
        void recvCookie(CkSectionInfo sid, CkSectionInfo child);
        /// Notify my tree parent (if any) that I am are ready
        void childrenReady(mCastEntry *entry);
        // ------------------------- Spanning Tree Teardown ------------------------
        /// entry Marks tree as obsolete, releases buffered msgs and propagates the call to children
        void teardown(CkSectionInfo s);
        /// entry Same as teardown, but also resets the root section info
        void retire(CkSectionInfo s, CkSectionInfo root);
        /// entry Actually frees the old spanning tree. Propagates the call to children
        void freeup(CkSectionInfo s);
        // ------------------------- Section Cookie Management ------------------------
        /// entry 
        void retrieveCookie(CkSectionInfo s, CkSectionInfo srcInfo);
        /// entry 
        void recvCookieInfo(CkSectionInfo s, int red);
        // ------------------------- Multicasts ------------------------
        /// entry
        void recvMsg(multicastGrpMsg *m);
        /// entry
        void sendToLocal(multicastGrpMsg *m);
        /// entry
        void recvPacket(CkSectionInfo &_cookie, int offset, int n, char *data, int seqno, int count, int totalsize, int fromBuffer);
        // ------------------------- Reductions ------------------------
        /// entry Accept a redn msg from a child in the spanning tree
        void recvRedMsg(CkReductionMsg *msg);
        /// entry Update the current completed redn num to input value
        void updateRedNo(mCastEntryPtr, int red);
        /// Configure a client to accept the reduction result
        void setReductionClient(CProxySection_ArrayElement &, redClientFn fn,void *param=NULL);
        /// Configure a client to accept the reduction result
        void setReductionClient(CProxySection_ArrayElement &, CkCallback *cb);
        /// reduction trigger
        void contribute(int dataSize,void *data,CkReduction::reducerType type, CkSectionInfo &sid, int userData=-1, int fragSize=-1);
        /// reduction trigger with a callback
        void contribute(int dataSize,void *data,CkReduction::reducerType type, CkSectionInfo &sid, CkCallback &cb, int userData=-1, int fragSize=-1);
        /// @note: User should be careful while passing non-default value of fragSize. fragSize%sizeof(data_type) should be zero


        /// Recreate the section when root migrate
        void resetSection(CProxySection_ArrayElement &proxy);  // called by root
        /// Implement the CkDelegateMgr interface to accept the delegation of a section proxy
        virtual void initDelegateMgr(CProxy *proxy);
        /// To implement the CkDelegateMgr interface for section mcasts
        void ArraySectionSend(CkDelegateData *pd,int ep,void *m, int nsid, CkSectionID *s, int opts);
        /// Send individually to each section member. Used when tree is out-of-date and needs a rebuild
        void SimpleSend(int ep,void *m, CkArrayID a, CkSectionID &sid, int opts);
        /// Retire and rebuild the spanning tree when one of the intermediate vertices migrates
        void rebuild(CkSectionInfo &);

    private:
        /// Fill the SectionInfo cookie in the SectionID obj with relevant info
        void prepareCookie(mCastEntry *entry, CkSectionID &sid, const CkArrayIndex *al, int count, CkArrayID aid);
        /// Get info from the CkSectionInfo and call setup() to start the spanning tree build
        void initCookie(CkSectionInfo sid);
        /// Actually trigger the multicast to a section of a chare array
        void sendToSection(CkDelegateData *pd,int ep,void *m, CkSectionID *sid, int opts);
        /// Mark old cookie spanning tree as old and build a new one
        void resetCookie(CkSectionInfo sid);
        ///
        void releaseBufferedReduceMsgs(mCastEntryPtr entry);
        /// Release buffered redn msgs from later reductions which arrived early (out of order)
        void releaseFutureReduceMsgs(mCastEntryPtr entry);
        ///
        inline CkReductionMsg *buildContributeMsg(int dataSize,void *data,CkReduction::reducerType type, CkSectionInfo &id, CkCallback &cb, int userFlag=-1);
        /// Reduce one fragment of a reduction msg and handle appropriately (transmit up the tree, buffer, combine etc)
        void reduceFragment (int index, CkSectionInfo& id, mCastEntry* entry, reductionInfo& redInfo, int currentTreeUp);
        /// At the tree root: Combine all msg fragments for final delivery to the client
        CkReductionMsg* combineFrags (CkSectionInfo& id, mCastEntry* entry, reductionInfo& redInfo);
};

#endif
