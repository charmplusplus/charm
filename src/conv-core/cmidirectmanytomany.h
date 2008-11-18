
#ifndef   __CMID_MANY_TO_MANY_H__
#define   __CMID_MANY_TO_MANY_H__

#ifdef __cplusplus
extern "C" {
#endif

  typedef  void (* CmiDirectM2mHandler) (void *context);
  void * CmiDirect_manytomany_allocate_handle ();
  
  void   CmiDirect_manytomany_initialize_recvbase ( void                 * handle,
						    unsigned               tag,
						    char                 * rcvbuf,
						    CmiDirectM2mHandler    donecb,
						    void                 * context,
						    unsigned               nranks,
						    unsigned               myIdx );
  
  void   CmiDirect_manytomany_initialize_recv ( void          * handle,
						unsigned        tag,
						unsigned        index,
						unsigned        displ,
						unsigned        bytes,
						unsigned        rank );
  
  void   CmiDirect_manytomany_initialize_sendbase  ( void                 * handle,
						     unsigned               tag,
						     CmiDirectM2mHandler    donecb,
						     void                 * context,
						     char                 * sndbuf,
						     unsigned               nranks,
						     unsigned               myIdx );
  
  void   CmiDirect_manytomany_initialize_send ( void        * handle,
						unsigned      tag, 
						unsigned      idx,
						unsigned      displ,
						unsigned      bytes,
						unsigned      rank );
  
  void   CmiDirect_manytomany_start ( void       * handle,
				      unsigned     tag );
  
#ifdef __cplusplus
}
#endif

#endif
