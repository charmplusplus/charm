#include <unordered_map>
#include <vector>
#include <utility>
#include <list>
#include <stdint.h>

typedef uint64_t ObjectId;
typedef std::unordered_map<ObjectId, void*> ObjectMap;
typedef ObjectMap::iterator ObjectMapIterator;
typedef std::unordered_map<ObjectId, std::vector<int>> ObjectPEMap;
typedef ObjectPEMap::iterator ObjectPEMapIterator;

typedef std::pair<void*, int> MessageDependency;
typedef std::list<MessageDependency*> DependencyList;
typedef DependencyList::iterator DependencyListIterator;
typedef std::unordered_map<ObjectId, DependencyList> DependencyMap;
typedef DependencyMap::iterator DependencyMapIterator;

class CkObjectStore
{
private:
    uint64_t replicaChoice;
    ObjectMap objectMap;
    ObjectPEMap locationMap;
    ObjectPEMap objReqBuffer;
    ObjectPEMap locReqBuffer;
    ObjectPEMap objLocReqBuffer;
    // cdef object proxy

public:
    void buffer_obj_request(ObjectId objId, int requestingPe);
    cdef void buffer_loc_request(self, ObjectId obj_id, int requesting_pe)
    cdef void buffer_obj_loc_request(self, ObjectId obj_id, int requesting_pe)
    cdef void check_obj_requests_buffer(self, ObjectId obj_id)
    cdef void check_loc_requests_buffer(self, ObjectId obj_id)
    cdef void check_obj_loc_requests_buffer(self, ObjectId obj_id)

    cpdef object lookup_object(self, ObjectId obj_id)
    cpdef int lookup_location(self, ObjectId obj_id, bint fetch=*)
    cdef void insert_object(self, ObjectId obj_id, object obj)
    cdef void delete_object(self, ObjectId obj_id)

    cdef int choose_pe(self, vector[int] &node_list)

    cpdef void update_location(self, ObjectId obj_id, int pe)
    cpdef void request_location_object(self, ObjectId obj_id, int requesting_pe)
    cpdef void request_location(self, ObjectId obj_id, int requesting_pe)
    cpdef void receive_remote_object(self, ObjectId obj_id, object obj)
    cpdef void request_object(self, ObjectId obj_id, int requesting_pe)
    cpdef void bulk_send_object(self, ObjectId obj_id, np.ndarray[int, ndim=1] requesting_pes)
    cpdef void create_object(self, ObjectId obj_id, object obj)

}