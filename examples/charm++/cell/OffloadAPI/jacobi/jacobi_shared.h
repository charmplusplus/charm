#ifndef __HELLO_SHARED_H__
#define __HELLO_SHARED_H__


#define ELEM_TYPE             float

#define MATRIX_ROWS           (1024 * 4 * 4 * 3)  // Number of rows in the matrix
#define MATRIX_COLS           (1024)  // Number of cols in the matrix
#define ROWS_PER_WORKREQUEST  (12)    // Number of rows going into each work request
//#define MATRIX_ROWS           (64 * 8)  // Number of rows in the matrix
//#define MATRIX_COLS           (64)  // Number of cols in the matrix
//#define ROWS_PER_WORKREQUEST  (64)    // Number of rows going into each work request

#define MAX_ERROR             ((ELEM_TYPE)(0.001))

#define BUFFER_ROWS           (MATRIX_ROWS + 2)
#define BUFFER_COLS           (MATRIX_COLS + 4)

#define NUM_WORKREQUESTS      (MATRIX_ROWS / ROWS_PER_WORKREQUEST)

#define FUNC_CALCROWS   1

#define GET_I(x,y)  (x + (BUFFER_COLS * y))

typedef struct __wrInfo {
  volatile int wrIndex;
  volatile ELEM_TYPE maxError;
} WRInfo;


#define SIZEOF_16(s)   ( (((sizeof(s) & 0x0000000F)) == (0x00)) ? (int)(sizeof(s)) : (int)((sizeof(s) & 0xFFFFFFF0) + (0x10)) )
#define ROUNDUP_16(s)  ( ((((s) & 0x0000000F)) == (0x00)) ? (int)(s) : (int)(((s) & 0xFFFFFFF0) + (0x10)) )


#ifndef NULL
  #define NULL  (0)
#endif

#ifndef TRUE
  #define TRUE  (-1)
#endif

#ifndef FALSE
  #define FALSE (0)
#endif


#endif //__HELLO_SHARED_H__
