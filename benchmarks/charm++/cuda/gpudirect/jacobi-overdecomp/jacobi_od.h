#ifndef __GPUDIRECT_JACOBI_OD_H_
#define __GPUDIRECT_JACOBI_OD_H_

#ifdef TEST_CORRECTNESS
typedef int DataType;
#else
typedef double DataType;
#endif

#define IDX(x,y) ((block_width+2)*(y)+(x))

#endif // __GPUDIRECT_JACOBI_OD_H_
