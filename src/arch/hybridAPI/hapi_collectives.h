// #include "hapi_collectives.decl.h"
// #include <nccl.h>
// #include <unordered_map>
// #include <vector>
// #include "charm++.h"
// #include "hapi.h"

// class NCCLManager : public CBase_NCCLManager {
// public:
//     ncclComm_t baseComm;
//     std::unordered_map<CkArrayID, ncclComm_t> comms;
//     std::unordered_map<CkArrayID, std::unordered_map<int, std::vector<void*>>> localBuffers;
//     std::unordered_map<CkArrayID, std::unordered_map<int, std::vector<void*>>> localResultBuffers;
//     std::unordered_map<CkArrayID, std::unordered_map<int, int>> readyCounts;
//     std::unordered_map<CkArrayID, int> localChares;
//     hapiStream_t* streams;
//     ncclUniqueId id;

//     int nbuffers;

//     NCCLManager(int k);
//     void recvNCCLId(int size, char* id_buf);
// };
