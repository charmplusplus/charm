#define CMK_BPROC                                          1

#undef CMK_ASYNC_NOT_NEEDED
#undef CMK_ASYNC_USE_F_SETFL_AND_F_SETOWN
#define CMK_ASYNC_NOT_NEEDED                               1
#define CMK_ASYNC_USE_F_SETFL_AND_F_SETOWN                 0

#undef CMK_RSH_NOT_NEEDED
#define CMK_RSH_NOT_NEEDED				   1

/* poll does not work with kill STOP/CONT in on-demand queueing system */
#undef CMK_USE_POLL
#define CMK_USE_POLL                                       0
