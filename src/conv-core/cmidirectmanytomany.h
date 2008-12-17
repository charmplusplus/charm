
#ifndef   __CMID_MANY_TO_MANY_H__
#define   __CMID_MANY_TO_MANY_H__

#ifndef CMI_DIRECT_MANY_TO_MANY_DEFINED
#define CmiDirect_manytomany_allocate_handle()   NULL

#define CmiDirect_manytomany_initialize_recvbase(a, b, c, d, e, f, g)

#define CmiDirect_manytomany_initialize_recv(a, b, c, d, e, f)

#define CmiDirect_manytomany_initialize_sendbase(a, b, c, d, e, f, g) 

#define CmiDirect_manytomany_initialize_send(a, b, c, d, e, f)

#define CmiDirect_manytomany_start(a, b) 
#else

#ifdef __cplusplus
extern "C" {
#endif

  typedef  void (* CmiDirectM2mHandler) (void *context);
  void * CmiDirect_manytomany_allocate_handle ();
  
  void   CmiDirect_manytomany_initialize_recvbase ( void                 * handle,
						    unsigned               tag,
						    CmiDirectM2mHandler    donecb,
						    void                 * context,
						    char                 * rcvbuf,
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

#endif    //end CMI_DIRECT_MANY_TO_MANY_DEFINED

#endif    //end __CMID_MANY_TO_MANY_H__
