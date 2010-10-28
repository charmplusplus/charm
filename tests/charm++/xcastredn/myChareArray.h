#include "reductionBenchmark.h"

#ifndef MY_CHARE_ARRAY_H
#define MY_CHARE_ARRAY_H

/// A chare array that participates in the xcast/redn loop
class MyChareArray: public CBase_MyChareArray
{
    public:
        MyChareArray(CkGroupID grpID);
        MyChareArray(CkMigrateMessage *msg) {}
        void pup(PUP::er &p) {}
        /// @entry Receives data and toys with it. Returns confirmation via a reduction
        void crunchData(DataMsg *msg);

    private:
        int msgNum;
        CkGroupID mcastGrpID;
        CkMulticastMgr *mcastMgr;
        CkSectionInfo sid;
        double *returnData;
};

#endif // MY_CHARE_ARRAY_H

