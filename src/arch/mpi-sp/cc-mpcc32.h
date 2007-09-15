/* no defines needed. */

#if ! CMK_LONG_LONG_DEFINED
#error "long long not defined!"
#endif

#undef CMK_TYPEDEF_INT8
#undef CMK_TYPEDEF_UINT8
#define CMK_TYPEDEF_INT8 long long
#define CMK_TYPEDEF_UINT8 unsigned long long

#ifdef CMK_64BIT
#undef CMK_64BIT
#endif
#define CMK_64BIT    0

