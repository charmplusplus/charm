
#undef CMK_USE_TCP
#define CMK_USE_TCP                                         1

#if CMK_SMP
#undef CMK_USE_POLL
#endif
