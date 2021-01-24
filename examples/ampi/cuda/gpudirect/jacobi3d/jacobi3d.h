#ifndef __JACOBI3D_H_
#define __JACOBI3D_H_

#ifdef TEST_CORRECTNESS
typedef int DataType;
#else
typedef double DataType;
#endif

enum Direction { LEFT = 0, RIGHT, TOP, BOTTOM, FRONT, BACK, DIR_COUNT };

#define IDX(i,j,k) ((block_width+2)*(block_height+2)*(k)+(block_width+2)*(j)+(i))

#endif // __JACOBI3D_H_
