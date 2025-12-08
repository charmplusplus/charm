// #include "hapi_collective_types.h"

// template <typename T>
// struct AddOperator {
//     __device__ inline static T compute(T a, T b) {
//         return a + b;
//     }
// };

// template <typename T>
// struct MulOperator {
//     __device__ inline static T compute(T a, T b) {
//         return a * b;
//     }
// };

// template <typename T>
// struct MaxOperator {
//     __device__ inline static T compute(T a, T b) {
//         return (a > b) ? a : b;
//     }
// };

// template <typename T>
// struct MinOperator {
//     __device__ inline static T compute(T a, T b) {
//         return (a < b) ? a : b;
//     }
// };

// template <typename T, typename Operator>
// __global__ void localKernel(T** inputs, T* result, int n, int numInputs) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx < n) {
//         T sum = 0.0f;
//         for (int i = 0; i < numInputs; i++) {
//             sum = Operator::compute(sum, inputs[i][idx]);
//         }
//         result[idx] = sum;
//     }
// }

// template <typename T>
// void callLocalReduce(T** d_inputs, T* d_result, int n, int numInputs, HAPIReducerType op, cudaStream_t stream) {
//     int blockSize = 256;
//     int numBlocks = (n + blockSize - 1) / blockSize;

//     switch (op)
//     {
//     case HAPIReducerType::Sum:
//         localKernel<T, AddOperator<T>><<<numBlocks, blockSize, 0, stream>>>(d_inputs, d_result, n, numInputs);
//         break;
//     case HAPIReducerType::Prod:
//         localKernel<T, MulOperator<T>><<<numBlocks, blockSize, 0, stream>>>(d_inputs, d_result, n, numInputs);
//         break;
//     case HAPIReducerType::Max:
//         localKernel<T, MaxOperator<T>><<<numBlocks, blockSize, 0, stream>>>(d_inputs, d_result, n, numInputs);
//         break;
//     case HAPIReducerType::Min:
//         localKernel<T, MinOperator<T>><<<numBlocks, blockSize, 0, stream>>>(d_inputs, d_result, n, numInputs);
//         break;
//     default:
//         fprintf(stderr, "Unsupported reduction operator\n");
//         break;
//     }
//     //cudaDeviceSynchronize();
// }