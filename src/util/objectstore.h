#include <unordered_map>
#include <vector>
#include <utility>
#include <list>
#include <stdint.h>
#include <mutex>
#include <shared_mutex>

typedef uint64_t ObjectId;
typedef std::pair<int, char*> ObjectData; 
typedef std::unordered_map<ObjectId, ObjectData> ObjectMap;
typedef ObjectMap::iterator ObjectMapIterator;
typedef std::unordered_map<ObjectId, std::vector<int>*> ObjectPEMap;
typedef ObjectPEMap::iterator ObjectPEMapIterator;

typedef std::pair<char*, int> MessageDependency;
typedef std::list<MessageDependency*> DependencyList;
typedef DependencyList::iterator DependencyListIterator;
typedef std::unordered_map<ObjectId, DependencyList> DependencyMap;
typedef DependencyMap::iterator DependencyMapIterator;


class ObjectStore
{
private:
    uint64_t replica;
    std::vector<int>* nullVector;
    ObjectData nullObject;
    ObjectMap objectMap;
    ObjectPEMap locationMap;
    ObjectPEMap objReqBuffer;
    ObjectPEMap locReqBuffer;
    ObjectPEMap objLocReqBuffer;
    mutable std::shared_mutex mutexObjectMap_;
    mutable std::shared_mutex mutexLocationMap_;
    mutable std::shared_mutex mutexObjReqBuffer_;
    mutable std::shared_mutex mutexLocReqBuffer_;
    mutable std::shared_mutex mutexObjLocReqBuffer_;
    // cdef object proxy

public:
    ObjectStore();
    ~ObjectStore();
    void bufferRequest(ObjectId objId, int requestingPe, ObjectPEMap &buffer, std::shared_mutex &mutex);
    inline void bufferObjRequest(ObjectId objId, int requestingPe);
    inline void bufferObjRequest(ObjectId objId, std::vector<int> &requestingPes);
    inline void bufferLocRequest(ObjectId objId, int requestingPe);
    inline void bufferObjLocRequest(ObjectId objId, int requestingPe);
    std::vector<int>* checkBuffer(ObjectId objId, ObjectPEMap &buffer, std::shared_mutex &mutex);
    inline std::vector<int>* checkObjRequestsBuffer(ObjectId objId);
    inline std::vector<int>* checkLocRequestsBuffer(ObjectId objId);
    inline std::vector<int>* checkObjLocRequestsBuffer(ObjectId objId);
    
    inline int choose(std::vector<int>& nodeList);
    ObjectData lookupObject(ObjectId objId);
    int lookupLocation(ObjectId objId);
    inline void insertObject(ObjectId objId, int size, char* obj);
    ObjectData popObject(ObjectId objId);

    bool updateLocation(ObjectId objId, int pe);
    void addLocation(ObjectId objId, int pe);
};