#define CMK_BPROC                                          1

#undef CMK_ASYNC_NOT_NEEDED
#undef CMK_ASYNC_USE_F_SETFL_AND_F_SETOWN
#define CMK_ASYNC_NOT_NEEDED                               1
#define CMK_ASYNC_USE_F_SETFL_AND_F_SETOWN                 0

#undef CMK_SSH_NOT_NEEDED
#define CMK_SSH_NOT_NEEDED				   1
