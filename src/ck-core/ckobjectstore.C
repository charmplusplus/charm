#include <ckobjectstore.h>

CkObjectStore::CkObjectStore(int numStores_)
{
    objectStoreProxy = thisProxy;
    numStores = numStores_;
    objStores = new ObjectStore[numStores];
}

CkObjectStore::~CkObjectStore()
{
    delete objStores;
}

inline ObjectStore& CkObjectStore::getObjectStore(ObjectId objId)
{
    return objStores[objId % numStores];
}

char* CkObjectStore::lookupObject(ObjectId objId)
{
    ObjectStore& store = getObjectStore(objId);
    return store.lookupObject(objId).second;
}

void CkObjectStore::updateLocation(ObjectId objId, int pe)
{
    ObjectStore& store = getObjectStore(objId);
    if (store.updateLocation(objId, pe))
    {
        // TODO: check location buffer
        std::vector<int> &pes = *store.checkObjLocRequestsBuffer(objId);
        thisProxy[pe].requestObject(objId, pes);
    }
}

void CkObjectStore::updateLocation(ObjectId objId, std::vector<int> pes)
{
    // TODO this is only implemented to request location before scheduling a task
    // so that multiple locations can be returned and the scheduler can choose 
    // the best location based on the locations of all other objects
    ObjectStore& store = getObjectStore(objId);
    //store.updateLocation(objId, pes);
}

void CkObjectStore::requestLocationObject(ObjectId objId, int requestingPe)
{
    ObjectStore& store = getObjectStore(objId);
    int pe = store.lookupLocation(objId);
    if (pe == -1)
        store.bufferObjLocRequest(objId, requestingPe);
    else
    {
        thisProxy[pe].requestObject(objId, requestingPe);
        store.addLocation(objId, requestingPe);
    }
}

void CkObjectStore::requestLocation(ObjectId objId, int requestingPe)
{
    ObjectStore& store = getObjectStore(objId);
    int pe = store.lookupLocation(objId);
    if (pe == -1)
        store.bufferLocRequest(objId, requestingPe);
    else
        thisProxy[requestingPe].updateLocation(objId, pe);
}

void CkObjectStore::requestObject(ObjectId objId, int requestingPe)
{
    ObjectStore& store = getObjectStore(objId);
    ObjectData objData = store.lookupObject(objId);
    if (objData.second == nullptr)
        store.bufferObjRequest(objId, requestingPe);
    else
        thisProxy[requestingPe].receiveRemoteObject(objId, objData.first, objData.second);
}

void CkObjectStore::requestObject(ObjectId objId, std::vector<int> requestingPes)
{
    ObjectStore& store = getObjectStore(objId);
    ObjectData objData = store.lookupObject(objId);
    if (objData.second == nullptr)
        store.bufferObjRequest(objId, requestingPes);
    else
        bulkSendObject(objId, objData.first, objData.second, requestingPes);
}

void CkObjectStore::bulkSendObject(ObjectId objId, int size, char* obj, std::vector<int> &pes)
{
    for (int i = 0; i < pes.size(); i++)
        thisProxy[pes[i]].receiveRemoteObject(objId, size, obj);
}

void CkObjectStore::receiveRemoteObject(ObjectId objId, int size, char* obj)
{
    ObjectStore& store = getObjectStore(objId);
    store.insertObject(objId, size, obj);
    std::vector<int> &pes = *store.checkObjRequestsBuffer(objId);
    bulkSendObject(objId, size, obj, pes);

    // check futures waiting buffer
    FutureMap::iterator it = CsvAccess(futuresWaiting).find(objId);
    if (it != CsvAccess(futuresWaiting).end())
    {
        CkSendToFuture(it->second, (void*) obj);
        CsvAccess(futuresWaiting).erase(it);
    }
}

void CkObjectStore::createObject(ObjectId objId, int size, char* obj)
{
    ObjectStore& store = getObjectStore(objId);
    store.insertObject(objId, size, obj);
    thisProxy[objId % CkNumNodes()].updateLocation(objId, CkMyNode());
    bulkSendObject(objId, size, obj, *store.checkObjRequestsBuffer(objId));
}