#include "hapi.h"
#include "jacobi3d.h"

#define TILE_SIZE_3D 8
#define TILE_SIZE_2D 16
#define DIVIDEBY7 0.142857

__global__ void initKernel(DataType* temperature, int block_width,
    int block_height, int block_depth) {
  int i = blockDim.x*blockIdx.x+threadIdx.x;
  int j = blockDim.y*blockIdx.y+threadIdx.y;
  int k = blockDim.z*blockIdx.z+threadIdx.z;
  if (i < block_width+2 && j < block_height+2 && k < block_depth+2) {
    temperature[IDX(i,j,k)] = 0;
  }
}

__global__ void ghostInitKernel(DataType* ghost, int ghost_count) {
  int i = blockDim.x*blockIdx.x+threadIdx.x;
  if (i < ghost_count) {
    ghost[i] = 0;
  }
}

__global__ void leftBoundaryKernel(DataType* temperature, int block_width,
    int block_height, int block_depth) {
  int j = blockDim.x*blockIdx.x+threadIdx.x;
  int k = blockDim.y*blockIdx.y+threadIdx.y;
  if (j < block_height && k < block_depth) {
    temperature[IDX(0,1+j,1+k)] = 1;
  }
}

__global__ void rightBoundaryKernel(DataType* temperature, int block_width,
    int block_height, int block_depth) {
  int j = blockDim.x*blockIdx.x+threadIdx.x;
  int k = blockDim.y*blockIdx.y+threadIdx.y;
  if (j < block_height && k < block_depth) {
    temperature[IDX(block_width+1,1+j,1+k)] = 1;
  }
}

__global__ void topBoundaryKernel(DataType* temperature, int block_width,
    int block_height, int block_depth) {
  int i = blockDim.x*blockIdx.x+threadIdx.x;
  int k = blockDim.y*blockIdx.y+threadIdx.y;
  if (i < block_width && k < block_depth) {
    temperature[IDX(1+i,0,1+k)] = 1;
  }
}

__global__ void bottomBoundaryKernel(DataType* temperature, int block_width,
    int block_height, int block_depth) {
  int i = blockDim.x*blockIdx.x+threadIdx.x;
  int k = blockDim.y*blockIdx.y+threadIdx.y;
  if (i < block_width && k < block_depth) {
    temperature[IDX(1+i,block_height+1,1+k)] = 1;
  }
}

__global__ void frontBoundaryKernel(DataType* temperature, int block_width,
    int block_height, int block_depth) {
  int i = blockDim.x*blockIdx.x+threadIdx.x;
  int j = blockDim.y*blockIdx.y+threadIdx.y;
  if (i < block_width && j < block_height) {
    temperature[IDX(1+i,1+j,0)] = 1;
  }
}

__global__ void backBoundaryKernel(DataType* temperature, int block_width,
    int block_height, int block_depth) {
  int i = blockDim.x*blockIdx.x+threadIdx.x;
  int j = blockDim.y*blockIdx.y+threadIdx.y;
  if (i < block_width && j < block_height) {
    temperature[IDX(1+i,1+j,block_depth+1)] = 1;
  }
}

__global__ void jacobiKernel(DataType* temperature, DataType* new_temperature,
    int block_width, int block_height, int block_depth) {
  int i = (blockDim.x*blockIdx.x+threadIdx.x)+1;
  int j = (blockDim.y*blockIdx.y+threadIdx.y)+1;
  int k = (blockDim.z*blockIdx.z+threadIdx.z)+1;

  if (i <= block_width && j <= block_height && k <= block_depth) {
#ifdef TEST_CORRECTNESS
    new_temperature[IDX(i,j,k)] = (temperature[IDX(i,j,k)] +
      temperature[IDX(i-1,j,k)] + temperature[IDX(i+1,j,k)] +
      temperature[IDX(i,j-1,k)] + temperature[IDX(i,j+1,k)] +
      temperature[IDX(i,j,k-1)] + temperature[IDX(i,j,k+1)]) % 10000;
#else
    new_temperature[IDX(i,j,k)] = (temperature[IDX(i,j,k)] +
      temperature[IDX(i-1,j,k)] + temperature[IDX(i+1,j,k)] +
      temperature[IDX(i,j-1,k)] + temperature[IDX(i,j+1,k)] +
      temperature[IDX(i,j,k-1)] + temperature[IDX(i,j,k+1)]) * DIVIDEBY7;
#endif
  }
}

__global__ void jacobiFusedPackingKernel(DataType* temperature, DataType* new_temperature,
    DataType** ghosts, bool* bounds, int block_width, int block_height, int block_depth) {
  int i = (blockDim.x*blockIdx.x+threadIdx.x)+1;
  int j = (blockDim.y*blockIdx.y+threadIdx.y)+1;
  int k = (blockDim.z*blockIdx.z+threadIdx.z)+1;

  DataType* left_ghost = ghosts[LEFT];
  DataType* right_ghost = ghosts[RIGHT];
  DataType* top_ghost = ghosts[TOP];
  DataType* bottom_ghost = ghosts[BOTTOM];
  DataType* front_ghost = ghosts[FRONT];
  DataType* back_ghost = ghosts[BACK];
  bool left_bound = bounds[LEFT];
  bool right_bound = bounds[RIGHT];
  bool top_bound = bounds[TOP];
  bool bottom_bound = bounds[BOTTOM];
  bool front_bound = bounds[FRONT];
  bool back_bound = bounds[BACK];

  if (i <= block_width && j <= block_height && k <= block_depth) {
    // Interior Jacobi update
#ifdef TEST_CORRECTNESS
    new_temperature[IDX(i,j,k)] = (temperature[IDX(i,j,k)] +
      temperature[IDX(i-1,j,k)] + temperature[IDX(i+1,j,k)] +
      temperature[IDX(i,j-1,k)] + temperature[IDX(i,j+1,k)] +
      temperature[IDX(i,j,k-1)] + temperature[IDX(i,j,k+1)]) % 10000;
#else
    new_temperature[IDX(i,j,k)] = (temperature[IDX(i,j,k)] +
      temperature[IDX(i-1,j,k)] + temperature[IDX(i+1,j,k)] +
      temperature[IDX(i,j-1,k)] + temperature[IDX(i,j+1,k)] +
      temperature[IDX(i,j,k-1)] + temperature[IDX(i,j,k+1)]) * DIVIDEBY7;
#endif

    // Pack ghosts
    if (!left_bound && i == 1) {
      left_ghost[block_height*(k-1)+(j-1)] = new_temperature[IDX(i,j,k)];
    }
    if (!right_bound && i == block_width) {
      right_ghost[block_height*(k-1)+(j-1)] = new_temperature[IDX(i,j,k)];
    }
    if (!top_bound && j == 1) {
      top_ghost[block_width*(k-1)+(i-1)] = new_temperature[IDX(i,j,k)];
    }
    if (!bottom_bound && j == block_height) {
      bottom_ghost[block_width*(k-1)+(i-1)] = new_temperature[IDX(i,j,k)];
    }
    if (!front_bound && k == 1) {
      front_ghost[block_width*(j-1)+(i-1)] = new_temperature[IDX(i,j,k)];
    }
    if (!back_bound && k == block_depth) {
      back_ghost[block_width*(j-1)+(i-1)] = new_temperature[IDX(i,j,k)];
    }
  }
}

__global__ void jacobiFusedAllKernel(DataType* temperature, DataType* new_temperature,
    DataType** send_ghosts, DataType** recv_ghosts, bool* bounds,
    int block_width, int block_height, int block_depth) {
  int i = (blockDim.x*blockIdx.x+threadIdx.x)+1;
  int j = (blockDim.y*blockIdx.y+threadIdx.y)+1;
  int k = (blockDim.z*blockIdx.z+threadIdx.z)+1;

  DataType* send_left_ghost   = send_ghosts[LEFT];
  DataType* send_right_ghost  = send_ghosts[RIGHT];
  DataType* send_top_ghost    = send_ghosts[TOP];
  DataType* send_bottom_ghost = send_ghosts[BOTTOM];
  DataType* send_front_ghost  = send_ghosts[FRONT];
  DataType* send_back_ghost   = send_ghosts[BACK];

  DataType* recv_left_ghost   = recv_ghosts[LEFT];
  DataType* recv_right_ghost  = recv_ghosts[RIGHT];
  DataType* recv_top_ghost    = recv_ghosts[TOP];
  DataType* recv_bottom_ghost = recv_ghosts[BOTTOM];
  DataType* recv_front_ghost  = recv_ghosts[FRONT];
  DataType* recv_back_ghost   = recv_ghosts[BACK];

  bool left_bound   = bounds[LEFT];
  bool right_bound  = bounds[RIGHT];
  bool top_bound    = bounds[TOP];
  bool bottom_bound = bounds[BOTTOM];
  bool front_bound  = bounds[FRONT];
  bool back_bound   = bounds[BACK];

  if (i <= block_width && j <= block_height && k <= block_depth) {
    // Unpack ghosts
    if (!left_bound && i == 1) {
      temperature[IDX(i-1,j,k)] = recv_left_ghost[block_height*(k-1)+(j-1)];
    }
    if (!right_bound && i == block_width) {
      temperature[IDX(i+1,j,k)] = recv_right_ghost[block_height*(k-1)+(j-1)];
    }
    if (!top_bound && j == 1) {
      temperature[IDX(i,j-1,k)] = recv_top_ghost[block_width*(k-1)+(i-1)];
    }
    if (!bottom_bound && j == block_height) {
      temperature[IDX(i,j+1,k)] = recv_bottom_ghost[block_width*(k-1)+(i-1)];
    }
    if (!front_bound && k == 1) {
      temperature[IDX(i,j,k-1)] = recv_front_ghost[block_width*(j-1)+(i-1)];
    }
    if (!back_bound && k == block_depth) {
      temperature[IDX(i,j,k+1)] = recv_back_ghost[block_width*(j-1)+(i-1)];
    }

    // Interior Jacobi update
#ifdef TEST_CORRECTNESS
    new_temperature[IDX(i,j,k)] = (temperature[IDX(i,j,k)] +
      temperature[IDX(i-1,j,k)] + temperature[IDX(i+1,j,k)] +
      temperature[IDX(i,j-1,k)] + temperature[IDX(i,j+1,k)] +
      temperature[IDX(i,j,k-1)] + temperature[IDX(i,j,k+1)]) % 10000;
#else
    new_temperature[IDX(i,j,k)] = (temperature[IDX(i,j,k)] +
      temperature[IDX(i-1,j,k)] + temperature[IDX(i+1,j,k)] +
      temperature[IDX(i,j-1,k)] + temperature[IDX(i,j+1,k)] +
      temperature[IDX(i,j,k-1)] + temperature[IDX(i,j,k+1)]) * DIVIDEBY7;
#endif

    // Pack ghosts
    if (!left_bound && i == 1) {
      send_left_ghost[block_height*(k-1)+(j-1)] = new_temperature[IDX(i,j,k)];
    }
    if (!right_bound && i == block_width) {
      send_right_ghost[block_height*(k-1)+(j-1)] = new_temperature[IDX(i,j,k)];
    }
    if (!top_bound && j == 1) {
      send_top_ghost[block_width*(k-1)+(i-1)] = new_temperature[IDX(i,j,k)];
    }
    if (!bottom_bound && j == block_height) {
      send_bottom_ghost[block_width*(k-1)+(i-1)] = new_temperature[IDX(i,j,k)];
    }
    if (!front_bound && k == 1) {
      send_front_ghost[block_width*(j-1)+(i-1)] = new_temperature[IDX(i,j,k)];
    }
    if (!back_bound && k == block_depth) {
      send_back_ghost[block_width*(j-1)+(i-1)] = new_temperature[IDX(i,j,k)];
    }
  }
}

__global__ void leftPackingKernel(DataType* temperature, DataType* ghost,
    int block_width, int block_height, int block_depth) {
  int j = blockDim.x*blockIdx.x+threadIdx.x;
  int k = blockDim.y*blockIdx.y+threadIdx.y;
  if (j < block_height && k < block_depth) {
    ghost[block_height*k+j] = temperature[IDX(1,1+j,1+k)];
  }
}

__global__ void rightPackingKernel(DataType* temperature, DataType* ghost,
    int block_width, int block_height, int block_depth) {
  int j = blockDim.x*blockIdx.x+threadIdx.x;
  int k = blockDim.y*blockIdx.y+threadIdx.y;
  if (j < block_height && k < block_depth) {
    ghost[block_height*k+j] = temperature[IDX(block_width,1+j,1+k)];
  }
}

__global__ void topPackingKernel(DataType* temperature, DataType* ghost,
    int block_width, int block_height, int block_depth) {
  int i = blockDim.x*blockIdx.x+threadIdx.x;
  int k = blockDim.y*blockIdx.y+threadIdx.y;
  if (i < block_width && k < block_depth) {
    ghost[block_width*k+i] = temperature[IDX(1+i,1,1+k)];
  }
}

__global__ void bottomPackingKernel(DataType* temperature, DataType* ghost,
    int block_width, int block_height, int block_depth) {
  int i = blockDim.x*blockIdx.x+threadIdx.x;
  int k = blockDim.y*blockIdx.y+threadIdx.y;
  if (i < block_width && k < block_depth) {
    ghost[block_width*k+i] = temperature[IDX(1+i,block_height,1+k)];
  }
}

__global__ void frontPackingKernel(DataType* temperature, DataType* ghost,
    int block_width, int block_height, int block_depth) {
  int i = blockDim.x*blockIdx.x+threadIdx.x;
  int j = blockDim.y*blockIdx.y+threadIdx.y;
  if (i < block_width && j < block_height) {
    ghost[block_width*j+i] = temperature[IDX(1+i,1+j,1)];
  }
}

__global__ void backPackingKernel(DataType* temperature, DataType* ghost,
    int block_width, int block_height, int block_depth) {
  int i = blockDim.x*blockIdx.x+threadIdx.x;
  int j = blockDim.y*blockIdx.y+threadIdx.y;
  if (i < block_width && j < block_height) {
    ghost[block_width*j+i] = temperature[IDX(1+i,1+j,block_depth)];
  }
}

__global__ void fusedPackingKernel(DataType* temperature, DataType** ghosts,
    bool* bounds, int block_width, int block_height, int block_depth) {
  int t = blockDim.x*blockIdx.x+threadIdx.x;

  int my_ghost = -1;
  int left_cut = block_height * block_depth;
  int right_cut = left_cut + block_height * block_depth;
  int top_cut = right_cut + block_width * block_depth;
  int bottom_cut = top_cut + block_width * block_depth;
  int front_cut = bottom_cut + block_width * block_height;
  int back_cut = front_cut + block_width * block_height;
  if (t >= 0 && t < left_cut) {
    my_ghost = LEFT;
  } else if (t >= left_cut && t < right_cut) {
    my_ghost = RIGHT;
    t -= left_cut;
  } else if (t >= right_cut && t < top_cut) {
    my_ghost = TOP;
    t -= right_cut;
  } else if (t >= top_cut && t < bottom_cut) {
    my_ghost = BOTTOM;
    t -= top_cut;
  } else if (t >= bottom_cut && t < front_cut) {
    my_ghost = FRONT;
    t -= bottom_cut;
  } else if (t >= front_cut && t < back_cut) {
    my_ghost = BACK;
    t -= front_cut;
  } else {
    return;
  }

  DataType* left_ghost   = ghosts[LEFT];
  DataType* right_ghost  = ghosts[RIGHT];
  DataType* top_ghost    = ghosts[TOP];
  DataType* bottom_ghost = ghosts[BOTTOM];
  DataType* front_ghost  = ghosts[FRONT];
  DataType* back_ghost   = ghosts[BACK];

  bool left_bound   = bounds[LEFT];
  bool right_bound  = bounds[RIGHT];
  bool top_bound    = bounds[TOP];
  bool bottom_bound = bounds[BOTTOM];
  bool front_bound  = bounds[FRONT];
  bool back_bound   = bounds[BACK];

  if (my_ghost == LEFT && !left_bound) {
    int j = t % block_height;
    int k = t / block_height;
    left_ghost[t] = temperature[IDX(1,1+j,1+k)];
  }
  if (my_ghost == RIGHT && !right_bound) {
    int j = t % block_height;
    int k = t / block_height;
    right_ghost[t] = temperature[IDX(block_width,1+j,1+k)];
  }
  if (my_ghost == TOP && !top_bound) {
    int i = t % block_width;
    int k = t / block_width;
    top_ghost[t] = temperature[IDX(1+i,1,1+k)];
  }
  if (my_ghost == BOTTOM && !bottom_bound) {
    int i = t % block_width;
    int k = t / block_width;
    bottom_ghost[t] = temperature[IDX(1+i,block_height,1+k)];
  }
  if (my_ghost == FRONT && !front_bound) {
    int i = t % block_width;
    int j = t / block_width;
    front_ghost[t] = temperature[IDX(1+i,1+j,1)];
  }
  if (my_ghost == BACK && !back_bound) {
    int i = t % block_width;
    int j = t / block_width;
    back_ghost[t] = temperature[IDX(1+i,1+j,block_depth)];
  }
}

__global__ void leftUnpackingKernel(DataType* temperature, DataType* ghost,
    int block_width, int block_height, int block_depth) {
  int j = blockDim.x*blockIdx.x+threadIdx.x;
  int k = blockDim.y*blockIdx.y+threadIdx.y;
  if (j < block_height && k < block_depth) {
    temperature[IDX(0,1+j,1+k)] = ghost[block_height*k+j];
  }
}

__global__ void rightUnpackingKernel(DataType* temperature, DataType* ghost,
    int block_width, int block_height, int block_depth) {
  int j = blockDim.x*blockIdx.x+threadIdx.x;
  int k = blockDim.y*blockIdx.y+threadIdx.y;
  if (j < block_height && k < block_depth) {
    temperature[IDX(block_width+1,1+j,1+k)] = ghost[block_height*k+j];
  }
}

__global__ void topUnpackingKernel(DataType* temperature, DataType* ghost,
    int block_width, int block_height, int block_depth) {
  int i = blockDim.x*blockIdx.x+threadIdx.x;
  int k = blockDim.y*blockIdx.y+threadIdx.y;
  if (i < block_width && k < block_depth) {
    temperature[IDX(1+i,0,1+k)] = ghost[block_width*k+i];
  }
}

__global__ void bottomUnpackingKernel(DataType* temperature, DataType* ghost,
    int block_width, int block_height, int block_depth) {
  int i = blockDim.x*blockIdx.x+threadIdx.x;
  int k = blockDim.y*blockIdx.y+threadIdx.y;
  if (i < block_width && k < block_depth) {
    temperature[IDX(1+i,block_height+1,1+k)] = ghost[block_width*k+i];
  }
}

__global__ void frontUnpackingKernel(DataType* temperature, DataType* ghost,
    int block_width, int block_height, int block_depth) {
  int i = blockDim.x*blockIdx.x+threadIdx.x;
  int j = blockDim.y*blockIdx.y+threadIdx.y;
  if (i < block_width && j < block_height) {
    temperature[IDX(1+i,1+j,0)] = ghost[block_width*j+i];
  }
}

__global__ void backUnpackingKernel(DataType* temperature, DataType* ghost,
    int block_width, int block_height, int block_depth) {
  int i = blockDim.x*blockIdx.x+threadIdx.x;
  int j = blockDim.y*blockIdx.y+threadIdx.y;
  if (i < block_width && j < block_height) {
    temperature[IDX(1+i,1+j,block_depth+1)] = ghost[block_width*j+i];
  }
}

__global__ void fusedUnpackingKernel(DataType* temperature, DataType** ghosts,
    bool* bounds, int block_width, int block_height, int block_depth) {
  int t = blockDim.x*blockIdx.x+threadIdx.x;

  int my_ghost = -1;
  int left_cut = block_height * block_depth;
  int right_cut = left_cut + block_height * block_depth;
  int top_cut = right_cut + block_width * block_depth;
  int bottom_cut = top_cut + block_width * block_depth;
  int front_cut = bottom_cut + block_width * block_height;
  int back_cut = front_cut + block_width * block_height;
  if (t >= 0 && t < left_cut) {
    my_ghost = LEFT;
  } else if (t >= left_cut && t < right_cut) {
    my_ghost = RIGHT;
    t -= left_cut;
  } else if (t >= right_cut && t < top_cut) {
    my_ghost = TOP;
    t -= right_cut;
  } else if (t >= top_cut && t < bottom_cut) {
    my_ghost = BOTTOM;
    t -= top_cut;
  } else if (t >= bottom_cut && t < front_cut) {
    my_ghost = FRONT;
    t -= bottom_cut;
  } else if (t >= front_cut && t < back_cut) {
    my_ghost = BACK;
    t -= front_cut;
  } else {
    return;
  }

  DataType* left_ghost   = ghosts[LEFT];
  DataType* right_ghost  = ghosts[RIGHT];
  DataType* top_ghost    = ghosts[TOP];
  DataType* bottom_ghost = ghosts[BOTTOM];
  DataType* front_ghost  = ghosts[FRONT];
  DataType* back_ghost   = ghosts[BACK];

  bool left_bound   = bounds[LEFT];
  bool right_bound  = bounds[RIGHT];
  bool top_bound    = bounds[TOP];
  bool bottom_bound = bounds[BOTTOM];
  bool front_bound  = bounds[FRONT];
  bool back_bound   = bounds[BACK];

  if (my_ghost == LEFT && !left_bound) {
    int j = t % block_height;
    int k = t / block_height;
    temperature[IDX(0,1+j,1+k)] = left_ghost[t];
  }
  if (my_ghost == RIGHT && !right_bound) {
    int j = t % block_height;
    int k = t / block_height;
    temperature[IDX(block_width+1,1+j,1+k)] = right_ghost[t];
  }
  if (my_ghost == TOP && !top_bound) {
    int i = t % block_width;
    int k = t / block_width;
    temperature[IDX(1+i,0,1+k)] = top_ghost[t];
  }
  if (my_ghost == BOTTOM && !bottom_bound) {
    int i = t % block_width;
    int k = t / block_width;
    temperature[IDX(1+i,block_height+1,1+k)] = bottom_ghost[t];
  }
  if (my_ghost == FRONT && !front_bound) {
    int i = t % block_width;
    int j = t / block_width;
    temperature[IDX(1+i,1+j,0)] = front_ghost[t];
  }
  if (my_ghost == BACK && !back_bound) {
    int i = t % block_width;
    int j = t / block_width;
    temperature[IDX(1+i,1+j,block_depth+1)] = back_ghost[t];
  }
}

void invokeInitKernel(DataType* d_temperature, int block_width, int block_height,
    int block_depth, cudaStream_t stream) {
  dim3 block_dim(TILE_SIZE_3D, TILE_SIZE_3D, TILE_SIZE_3D);
  dim3 grid_dim(((block_width+2)+(block_dim.x-1))/block_dim.x,
      ((block_height+2)+(block_dim.y-1))/block_dim.y,
      ((block_depth+2)+(block_dim.z-1))/block_dim.z);

  initKernel<<<grid_dim, block_dim, 0, stream>>>(d_temperature, block_width,
      block_height, block_depth);
  hapiCheck(cudaPeekAtLastError());
}

void invokeGhostInitKernels(const std::vector<DataType*>& ghosts,
    const std::vector<int>& ghost_counts, cudaStream_t stream) {
  dim3 block_dim(256);
  for (int i = 0; i < ghosts.size(); i++) {
    DataType* ghost = ghosts[i];
    int ghost_count = ghost_counts[i];

    dim3 grid_dim((ghost_count+block_dim.x-1)/block_dim.x);

    ghostInitKernel<<<grid_dim, block_dim, 0, stream>>>(ghost,
        ghost_count);
    hapiCheck(cudaPeekAtLastError());
  }
}

void invokeBoundaryKernels(DataType* d_temperature, int block_width,
    int block_height, int block_depth, bool bounds[], cudaStream_t stream) {
  dim3 block_dim(TILE_SIZE_2D, TILE_SIZE_2D);

  if (bounds[LEFT]) {
    dim3 grid_dim((block_height+(block_dim.x-1))/block_dim.x,
        (block_depth+(block_dim.y-1))/block_dim.y);
    leftBoundaryKernel<<<grid_dim, block_dim, 0, stream>>>(d_temperature,
        block_width, block_height, block_depth);
  }
  if (bounds[RIGHT]) {
    dim3 grid_dim((block_height+(block_dim.x-1))/block_dim.x,
        (block_depth+(block_dim.y-1))/block_dim.y);
    rightBoundaryKernel<<<grid_dim, block_dim, 0, stream>>>(d_temperature,
        block_width, block_height, block_depth);
  }
  if (bounds[TOP]) {
    dim3 grid_dim((block_width+(block_dim.x-1))/block_dim.x,
      (block_depth+(block_dim.y-1))/block_dim.y);
    topBoundaryKernel<<<grid_dim, block_dim, 0, stream>>>(d_temperature,
        block_width, block_height, block_depth);
  }
  if (bounds[BOTTOM]) {
    dim3 grid_dim((block_width+(block_dim.x-1))/block_dim.x,
      (block_depth+(block_dim.y-1))/block_dim.y);
    bottomBoundaryKernel<<<grid_dim, block_dim, 0, stream>>>(d_temperature,
        block_width, block_height, block_depth);
  }
  if (bounds[FRONT]) {
    dim3 grid_dim((block_width+(block_dim.x-1))/block_dim.x,
        (block_height+(block_dim.y-1))/block_dim.y);
    frontBoundaryKernel<<<grid_dim, block_dim, 0, stream>>>(d_temperature,
        block_width, block_height, block_depth);
  }
  if (bounds[BACK]) {
    dim3 grid_dim((block_width+(block_dim.x-1))/block_dim.x,
        (block_height+(block_dim.y-1))/block_dim.y);
    backBoundaryKernel<<<grid_dim, block_dim, 0, stream>>>(d_temperature,
        block_width, block_height, block_depth);
  }
  hapiCheck(cudaPeekAtLastError());
}

void invokeJacobiKernel(DataType* d_temperature, DataType* d_new_temperature,
    DataType** d_send_ghosts, DataType** d_recv_ghosts, bool* d_bounds,
    int block_width, int block_height, int block_depth, cudaStream_t stream,
    bool fuse_update_pack, bool fuse_update_all) {
  dim3 block_dim(TILE_SIZE_3D, TILE_SIZE_3D, TILE_SIZE_3D);
  dim3 grid_dim((block_width+(block_dim.x-1))/block_dim.x,
      (block_height+(block_dim.y-1))/block_dim.y,
      (block_depth+(block_dim.z-1))/block_dim.z);

  if (fuse_update_pack) {
    jacobiFusedPackingKernel<<<grid_dim, block_dim, 0, stream>>>(d_temperature, d_new_temperature,
        d_send_ghosts, d_bounds, block_width, block_height, block_depth);
  } else if (fuse_update_all) {
    jacobiFusedAllKernel<<<grid_dim, block_dim, 0, stream>>>(d_temperature, d_new_temperature,
        d_send_ghosts, d_recv_ghosts, d_bounds, block_width, block_height, block_depth);
  } else {
    jacobiKernel<<<grid_dim, block_dim, 0, stream>>>(d_temperature, d_new_temperature,
        block_width, block_height, block_depth);
  }
  hapiCheck(cudaPeekAtLastError());
}

void packGhostsDevice(DataType* d_temperature,
    DataType* d_ghosts[], DataType* h_ghosts[], bool bounds[],
    int block_width, int block_height, int block_depth,
    size_t x_surf_size, size_t y_surf_size, size_t z_surf_size,
    cudaStream_t comm_stream, cudaStream_t d2h_stream, cudaEvent_t pack_events[],
    bool use_channel) {
  dim3 block_dim(TILE_SIZE_2D, TILE_SIZE_2D);
  if (!bounds[LEFT]) {
    dim3 grid_dim((block_height+(block_dim.x-1))/block_dim.x,
        (block_depth+(block_dim.y-1))/block_dim.y);
    leftPackingKernel<<<grid_dim, block_dim, 0, comm_stream>>>(d_temperature,
        d_ghosts[LEFT], block_width, block_height, block_depth);
    if (!use_channel) {
      cudaEventRecord(pack_events[LEFT], comm_stream);
      cudaStreamWaitEvent(d2h_stream, pack_events[LEFT], 0);
      cudaMemcpyAsync(h_ghosts[LEFT], d_ghosts[LEFT], x_surf_size,
          cudaMemcpyDeviceToHost, d2h_stream);
    }
  }
  if (!bounds[RIGHT]) {
    dim3 grid_dim((block_height+(block_dim.x-1))/block_dim.x,
        (block_depth+(block_dim.y-1))/block_dim.y);
    rightPackingKernel<<<grid_dim, block_dim, 0, comm_stream>>>(d_temperature,
        d_ghosts[RIGHT], block_width, block_height, block_depth);
    if (!use_channel) {
      cudaEventRecord(pack_events[RIGHT], comm_stream);
      cudaStreamWaitEvent(d2h_stream, pack_events[RIGHT], 0);
      cudaMemcpyAsync(h_ghosts[RIGHT], d_ghosts[RIGHT], x_surf_size,
          cudaMemcpyDeviceToHost, d2h_stream);
    }
  }
  if (!bounds[TOP]) {
    dim3 grid_dim((block_width+(block_dim.x-1))/block_dim.x,
        (block_depth+(block_dim.y-1))/block_dim.y);
    topPackingKernel<<<grid_dim, block_dim, 0, comm_stream>>>(d_temperature,
        d_ghosts[TOP], block_width, block_height, block_depth);
    if (!use_channel) {
      cudaEventRecord(pack_events[TOP], comm_stream);
      cudaStreamWaitEvent(d2h_stream, pack_events[TOP], 0);
      cudaMemcpyAsync(h_ghosts[TOP], d_ghosts[TOP], y_surf_size,
          cudaMemcpyDeviceToHost, d2h_stream);
    }
  }
  if (!bounds[BOTTOM]) {
    dim3 grid_dim((block_width+(block_dim.x-1))/block_dim.x,
        (block_depth+(block_dim.y-1))/block_dim.y);
    bottomPackingKernel<<<grid_dim, block_dim, 0, comm_stream>>>(d_temperature,
        d_ghosts[BOTTOM], block_width, block_height, block_depth);
    if (!use_channel) {
      cudaEventRecord(pack_events[BOTTOM], comm_stream);
      cudaStreamWaitEvent(d2h_stream, pack_events[BOTTOM], 0);
      cudaMemcpyAsync(h_ghosts[BOTTOM], d_ghosts[BOTTOM], y_surf_size,
          cudaMemcpyDeviceToHost, d2h_stream);
    }
  }
  if (!bounds[FRONT]) {
    dim3 grid_dim((block_width+(block_dim.x-1))/block_dim.x,
        (block_height+(block_dim.y-1))/block_dim.y);
    frontPackingKernel<<<grid_dim, block_dim, 0, comm_stream>>>(d_temperature,
        d_ghosts[FRONT], block_width, block_height, block_depth);
    if (!use_channel) {
      cudaEventRecord(pack_events[FRONT], comm_stream);
      cudaStreamWaitEvent(d2h_stream, pack_events[FRONT], 0);
      cudaMemcpyAsync(h_ghosts[FRONT], d_ghosts[FRONT], z_surf_size,
          cudaMemcpyDeviceToHost, d2h_stream);
    }
  }
  if (!bounds[BACK]) {
    dim3 grid_dim((block_width+(block_dim.x-1))/block_dim.x,
        (block_height+(block_dim.y-1))/block_dim.y);
    backPackingKernel<<<grid_dim, block_dim, 0, comm_stream>>>(d_temperature,
        d_ghosts[BACK], block_width, block_height, block_depth);
    if (!use_channel) {
      cudaEventRecord(pack_events[BACK], comm_stream);
      cudaStreamWaitEvent(d2h_stream, pack_events[BACK], 0);
      cudaMemcpyAsync(h_ghosts[BACK], d_ghosts[BACK], z_surf_size,
          cudaMemcpyDeviceToHost, d2h_stream);
    }
  }
  hapiCheck(cudaPeekAtLastError());
}

void packGhostsFusedDevice(DataType* d_temperature, DataType** d_send_ghosts,
    bool* d_bounds, int block_width, int block_height, int block_depth,
    cudaStream_t comm_stream) {
  int n_elems = 2*(block_height*block_depth+block_width*block_depth+block_width*block_height);
  dim3 block_dim(256);
  dim3 grid_dim((n_elems+block_dim.x-1)/block_dim.x);
  fusedPackingKernel<<<grid_dim, block_dim, 0, comm_stream>>>(d_temperature,
      d_send_ghosts, d_bounds, block_width, block_height, block_depth);
  hapiCheck(cudaPeekAtLastError());
}

void unpackGhostDevice(DataType* d_temperature, DataType* d_ghost, DataType* h_ghost,
    int dir, int block_width, int block_height, int block_depth, size_t ghost_size,
    cudaStream_t comm_stream, cudaStream_t h2d_stream, cudaEvent_t unpack_events[],
    bool use_channel) {
  if (!use_channel) {
    cudaMemcpyAsync(d_ghost, h_ghost, ghost_size, cudaMemcpyHostToDevice,
        h2d_stream);
    cudaEventRecord(unpack_events[dir], h2d_stream);
    cudaStreamWaitEvent(comm_stream, unpack_events[dir], 0);
  }

  dim3 block_dim(TILE_SIZE_2D, TILE_SIZE_2D);
  if (dir == LEFT) {
    dim3 grid_dim((block_height+(block_dim.x-1))/block_dim.x,
        (block_depth+(block_dim.y-1))/block_dim.y);
    leftUnpackingKernel<<<grid_dim, block_dim, 0, comm_stream>>>(d_temperature,
        d_ghost, block_width, block_height, block_depth);
  } else if (dir == RIGHT) {
    dim3 grid_dim((block_height+(block_dim.x-1))/block_dim.x,
        (block_depth+(block_dim.y-1))/block_dim.y);
    rightUnpackingKernel<<<grid_dim, block_dim, 0, comm_stream>>>(d_temperature,
        d_ghost, block_width, block_height, block_depth);
  } else if (dir == TOP) {
    dim3 grid_dim((block_width+(block_dim.x-1))/block_dim.x,
        (block_depth+(block_dim.y-1))/block_dim.y);
    topUnpackingKernel<<<grid_dim, block_dim, 0, comm_stream>>>(d_temperature,
        d_ghost, block_width, block_height, block_depth);
  } else if (dir == BOTTOM) {
    dim3 grid_dim((block_width+(block_dim.x-1))/block_dim.x,
        (block_depth+(block_dim.y-1))/block_dim.y);
    bottomUnpackingKernel<<<grid_dim, block_dim, 0, comm_stream>>>(d_temperature,
        d_ghost, block_width, block_height, block_depth);
  } else if (dir == FRONT) {
    dim3 grid_dim((block_width+(block_dim.x-1))/block_dim.x,
        (block_height+(block_dim.y-1))/block_dim.y);
    frontUnpackingKernel<<<grid_dim, block_dim, 0, comm_stream>>>(d_temperature,
        d_ghost, block_width, block_height, block_depth);
  } else if (dir == BACK) {
    dim3 grid_dim((block_width+(block_dim.x-1))/block_dim.x,
        (block_height+(block_dim.y-1))/block_dim.y);
    backUnpackingKernel<<<grid_dim, block_dim, 0, comm_stream>>>(d_temperature,
        d_ghost, block_width, block_height, block_depth);
  }
  hapiCheck(cudaPeekAtLastError());
}

void unpackGhostsFusedDevice(DataType* d_temperature, DataType** d_recv_ghosts,
    bool* d_bounds, int block_width, int block_height, int block_depth,
    cudaStream_t comm_stream) {
  int n_elems = 2*(block_height*block_depth+block_width*block_depth+block_width*block_height);
  dim3 block_dim(256);
  dim3 grid_dim((n_elems+block_dim.x-1)/block_dim.x);
  fusedUnpackingKernel<<<grid_dim, block_dim, 0, comm_stream>>>(d_temperature,
      d_recv_ghosts, d_bounds, block_width, block_height, block_depth);
  hapiCheck(cudaPeekAtLastError());
}
