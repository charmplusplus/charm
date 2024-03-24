#include <objectstore.h>

void deletePEMap(ObjectPEMap &peMap)
{
    for (ObjectPEMapIterator it = peMap.begin(); it != peMap.end(); it++)
        delete it->second;
}

ObjectStore::ObjectStore() 
{
    nullObject = std::make_pair(0, nullptr);
    nullVector = new std::vector<int>();
}

ObjectStore::~ObjectStore()
{
    deletePEMap(locationMap);
    deletePEMap(locReqBuffer);
    deletePEMap(objReqBuffer);
    deletePEMap(objLocReqBuffer);
    delete nullVector;
}

void ObjectStore::bufferRequest(ObjectId objId, int requestingPe, ObjectPEMap &buffer, std::shared_mutex &mutex)
{
    std::unique_lock lock(mutex);
    ObjectPEMapIterator it = buffer.find(objId);
    if (it == buffer.end())
        it->second = new std::vector<int>();
    it->second->push_back(requestingPe);
}

inline void ObjectStore::bufferObjRequest(ObjectId objId, std::vector<int> &requestingPes)
{
    std::unique_lock lock(mutexObjReqBuffer_);
    ObjectPEMapIterator it = objReqBuffer.find(objId);
    if (it == objReqBuffer.end())
        it->second = new std::vector<int>();
    it->second->insert(it->second->end(), requestingPes.begin(), requestingPes.end());
}

inline void ObjectStore::bufferObjRequest(ObjectId objId, int requestingPe)
{
    bufferRequest(objId, requestingPe, objReqBuffer, mutexObjReqBuffer_);
}

inline void ObjectStore::bufferLocRequest(ObjectId objId, int requestingPe)
{
    bufferRequest(objId, requestingPe, locReqBuffer, mutexLocReqBuffer_);
}

inline void ObjectStore::bufferObjLocRequest(ObjectId objId, int requestingPe)
{
    bufferRequest(objId, requestingPe, objLocReqBuffer, mutexObjLocReqBuffer_);
}

std::vector<int>* ObjectStore::checkBuffer(ObjectId objId, ObjectPEMap &buffer, std::shared_mutex &mutex)
{
    std::shared_lock sharedLock(mutex);
    ObjectPEMapIterator it = buffer.find(objId);
    if (it == buffer.end())
        return nullVector;
    sharedLock.unlock();
    std::unique_lock uniqueLock(mutex);
    buffer.erase(objId);
    return it->second;
}

inline std::vector<int>* ObjectStore::checkObjRequestsBuffer(ObjectId objId)
{
    return checkBuffer(objId, objReqBuffer, mutexObjReqBuffer_);
}

inline std::vector<int>* ObjectStore::checkLocRequestsBuffer(ObjectId objId)
{
    return checkBuffer(objId, locReqBuffer, mutexLocReqBuffer_);
}

inline std::vector<int>* ObjectStore::checkObjLocRequestsBuffer(ObjectId objId)
{
    return checkBuffer(objId, objLocReqBuffer, mutexObjLocReqBuffer_);
}

ObjectData ObjectStore::lookupObject(ObjectId objId)
{
    std::shared_lock lock(mutexObjectMap_);
    ObjectMapIterator it = objectMap.find(objId);
    if (it == objectMap.end())
        return nullObject;
    return it->second;
}

inline int ObjectStore::choose(std::vector<int> &nodeList)
{
    return nodeList[replica++ % nodeList.size()];
}

int ObjectStore::lookupLocation(ObjectId objId)
{
    std::shared_lock lock(mutexLocationMap_);
    ObjectPEMapIterator it = locationMap.find(objId);
    if (it == locationMap.end())
        return -1;
    return choose(*(it->second));
}

inline void ObjectStore::insertObject(ObjectId objId, int size, char* obj)
{
    std::unique_lock lock(mutexObjectMap_);
    objectMap[objId] = std::make_pair(size, obj);
}

ObjectData ObjectStore::popObject(ObjectId objId)
{
    std::unique_lock lock(mutexObjectMap_);
    ObjectMapIterator it = objectMap.find(objId);
    ObjectData obj = it->second;
    objectMap.erase(it);
    return obj;
}

bool ObjectStore::updateLocation(ObjectId objId, int pe)
{
    std::unique_lock lock(mutexLocationMap_);
    bool newEntry = false;
    ObjectPEMapIterator it = locationMap.find(objId);
    if (it == locationMap.end())
        it->second = new std::vector<int>();
        newEntry = true;
    it->second->push_back(pe);
    return newEntry;
}

inline void ObjectStore::addLocation(ObjectId objId, int pe)
{
    std::unique_lock lock(mutexLocationMap_);
    locationMap[objId]->push_back(pe);
}