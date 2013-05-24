
#undef CMK_USE_TCP
#define CMK_USE_TCP                                         1

#if CMK_SMP
#undef CMK_USE_POLL
#endif

#define CMK_NETPOLL					    1

/*
#undef CMK_ASYNC_USE_F_SETFL_AND_F_SETOWN
#undef CMK_ASYNC_NOT_NEEDED
#define CMK_ASYNC_NOT_NEEDED				    1
*/

#undef CMK_BROADCAST_SPANNING_TREE
#undef CMK_BROADCAST_HYPERCUBE
#define CMK_BROADCAST_SPANNING_TREE    			    0
#define CMK_BROADCAST_HYPERCUBE        			    1
