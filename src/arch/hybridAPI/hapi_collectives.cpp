// #include "hapi_collective_types.h"
// #include "hapi_collectives.h"
// #include "hapi_collectives.def.h"

// template <typename T>
// extern void callLocalReduce(T** d_inputs, T* d_result, int n, int numInputs, HAPIReducerType op, cudaStream_t stream);

// class RedCallbackMsg : public CMessage_RedCallbackMsg {
// public:
// 	void* val;
// 	RedCallbackMsg(void* v) :val(v) {}
// };

// template <typename T>
// ncclDataType_t getNCCLDataType()
// {
//     if (std::is_same<T, float>::value) {
//         return ncclFloat;
//     } else if (std::is_same<T, double>::value) {
//         return ncclDouble;
//     } else if (std::is_same<T, int>::value) {
//         return ncclInt;
//     } else if (std::is_same<T, long>::value) {
//         return ncclInt64;
//     }
//     // Add more types as needed
//     throw std::runtime_error("Unsupported data type for NCCL");
// }

// ncclRedOp_t getNCCLRedOp(HAPIReducerType op)
// {
//     switch (op) {
//         case HAPIReducerType::Sum:
//             return ncclSum;
//         case HAPIReducerType::Prod:
//             return ncclProd;
//         case HAPIReducerType::Max:
//             return ncclMax;
//         case HAPIReducerType::Min:
//             return ncclMin;
//         // Add more operations as needed
//         default:
//             throw std::runtime_error("Unsupported reduction operation for NCCL");
//     }
// }

// NCCLManager::NCCLManager(int k) {
//     nbuffers = k;
//     streams = new cudaStream_t[nbuffers];
//     for (int i = 0; i < nbuffers; i++) {
//         cudaStreamCreate(&streams[i]);
//     }
//     if (CkMyPe() == 0) {
//         ncclGetUniqueId(&id);
//         thisProxy.recvNCCLId(sizeof(ncclUniqueId), (char*)id.internal);
//     }
// }

// void NCCLManager::recvNCCLId(int size, char* id_buf) {
//     memcpy(id.internal, id_buf, sizeof(ncclUniqueId));
//     ncclCommInitRank(&baseComm, CkNumPes(), id, CkMyPe());
// }

// template <typename T>
// void NCCLManager::reduce(CkArrayID id, int redNo, int root, T* d_input, int n, HAPIReducerType op, cudaStream_t stream, CkCallback cb) {
//     auto it = localBuffers.find(id);
//     if (it == localBuffers.end()) {
//         localBuffers[id] = std::unordered_map<int, std::vector<void*>>();
//     }
//     auto& redMap = localBuffers[id];
//     auto redIt = redMap.find(redNo);
//     if (redIt == redMap.end()) {
//         redMap[redNo] = std::vector<void*>();
//     }
//     auto& buffers = redMap[redNo];
//     buffers.push_back(d_input);

//     auto resIt = localResultBuffers.find(id);
//     if (resIt == localResultBuffers.end()) {
//         localResultBuffers[id] = std::unordered_map<int, std::vector<void*>>();
//     }
//     auto& resMap = localResultBuffers[id];
//     auto resIt2 = resMap.find(redNo);
//     if (resIt2 == resMap.end()) {
//         std::vector<void*> d_result;
//         for (int i = 0; i < nbuffers; i++) {
//             T* d_res;
//             hapiCheck(cudaMalloc(&d_res, n * sizeof(T)));
//             d_result.push_back(d_res);
//         }
//         resMap[redNo] = d_result;
//     }
//     std::vector<void*> d_result = resMap[redNo];

//     if (buffers.size() == localChares[id]) {
//         T** d_inputs;
//         hapiCheck(cudaMalloc(&d_inputs, buffers.size() * sizeof(T*)));
//         hapiCheck(cudaMemcpy(d_inputs, buffers.data(), buffers.size() * sizeof(T*), cudaMemcpyHostToDevice));
//         callLocalReduce<T>(d_inputs, d_result, n, buffers.size(), op, stream);
//         buffers.clear();

//         T* d_globalResult;
//         hapiCheck(cudaMalloc(&d_globalResult, n * sizeof(T)));
//         ncclReduce((const void*)d_result, (void*)d_globalResult, n,
//                    getNCCLDataType<T>(), getNCCLRedOp(op), root, baseComm, stream);
//         hapiAddCallback(stream, cb, new RedCallbackMsg(d_globalResult));
//         resMap.erase(redNo);
//     }
// }