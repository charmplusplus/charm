#include <cstdlib>
#include <objectstore.h>
#include "charm++.h"

#include "ckobjectstore.decl.h"

CProxy_CkObjectStore objectStoreProxy;

class CkObjectStore : public CBase_CkObjectStore
{
private:
    int numStores;
    ObjectStore* objStores;
public:
    CkObjectStore(int numStores_);
    ~CkObjectStore();

    inline ObjectStore& getObjectStore(ObjectId objId);
    char* lookupObject(ObjectId objId);
    void bulkSendObject(ObjectId objId, int size, char* obj, std::vector<int> &pes);
    
    void updateLocation(ObjectId objId, int pe);
    void updateLocation(ObjectId objId, std::vector<int> pes);

    void requestLocationObject(ObjectId objId, int requestingPe);
    void requestLocation(ObjectId objId, int requestingPe);
    void receiveRemoteObject(ObjectId obj_id, int size, char* obj);
    void requestObject(ObjectId objId, int requestingPe);
    void requestObject(ObjectId objId, std::vector<int> requestingPes);

    void createObject(ObjectId objId, int size, char* obj);
};

#include "ckobjectstore.def.h"