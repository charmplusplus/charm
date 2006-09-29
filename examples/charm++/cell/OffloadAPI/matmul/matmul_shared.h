#ifndef __HELLO_SHARED_H__
#define __HELLO_SHARED_H__


#define USE_DOUBLE            0   // Set to zero to use float, non-zero to use double (and use ELEM_TYPE as type)
#if USE_DOUBLE != 0
  #define ELEM_TYPE             double
#else
  #define ELEM_TYPE             float
#endif

#define MATRIX_A_ROWS         (1024 * 2)            // 'this * sizeof(ELEM_TYPE)' should be a multiple of 16
#define MATRIX_A_COLS         (1024 * 2)            // 'this * sizeof(ELEM_TYPE)' should be a multiple of 16
#define MATRIX_B_ROWS         (MATRIX_A_COLS)
#define MATRIX_B_COLS         (1024 * 2)            // 'this * sizeof(ELEM_TYPE)' should be a multiple of 16
#define MATRIX_C_ROWS         (MATRIX_A_ROWS)
#define MATRIX_C_COLS         (MATRIX_B_COLS)

#define NUM_ROWS_PER_WR       (4)
#define NUM_COLS_PER_WR       (4)   // 
// 'NUM_ROWS_PER_WR * NUM_COLS_PER_WR * sizeof(ELEM_TYPE)' should be a multiple of 16

#define REPEAT_COUNT          (1)

#define DISPLAY_MATRICES        (0)
#define DISPLAY_WR_FINISH_FREQ  (1024 * 512)

#define NUM_WRS_PER_ITER      ((MATRIX_C_ROWS / NUM_ROWS_PER_WR) * (MATRIX_C_COLS / NUM_COLS_PER_WR))

#define FUNC_CALC             1

typedef struct __wr_record {
  ELEM_TYPE C[NUM_ROWS_PER_WR * NUM_COLS_PER_WR];
  int startRow;
  int startCol;
} WRRecord;

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
