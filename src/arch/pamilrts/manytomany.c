
#include "converse.h"
#include "cmidirectmanytomany.h"
#define MAX_CONN   8

#define M2M_PAMI_S8DISPATCH 13
#define M2M_PAMI_SDISPATCH  14
#define M2M_PAMI_DISPATCH   15

typedef struct _pami_m2mhdr {
  int8_t    dstrank;
  int8_t    connid;
  int32_t   srcindex;
} PAMI_M2mHeader; 

typedef struct _pami_m2m_work {
  pami_work_t    work;
  int            start;
  int            end;
  void         * handle;
  pami_context_t context;
} PAMI_M2mWork_t;

typedef struct _m2m_completionmsg {
  char  hdr [CmiMsgHeaderSizeBytes];
  void  *handle;
  int    rank;
} M2mCompletionMsg;

#define MAX_NWORK 8

typedef struct _pami_cmidhandle {
  int                   myrank;
  unsigned              m2m_rcvcounter ;
  unsigned              m2m_nzrcvranks;  
  char                * m2m_rcvbuf     ;
  unsigned            * m2m_rcvlens    ;
  unsigned            * m2m_rdispls    ;

  unsigned              m2m_nsndranks;
  unsigned              m2m_srankIndex;		      
  char                * m2m_sndbuf     ;
  unsigned            * m2m_sndlens    ;
  unsigned            * m2m_sdispls    ;
  unsigned              m2m_sndcounter ;
  unsigned            * m2m_permutation;
  unsigned            * m2m_lranks     ;
  pami_endpoint_t     * m2m_node_eps;

  PAMI_M2mWork_t        swork[MAX_NWORK];  
  int                   n_work;

  CmiDirectM2mHandler   m2m_rdone;
  void                * m2m_rdonecontext;
  PAMI_M2mHeader      * m2m_hdrs;
  M2mCompletionMsg      cmsg;

  unsigned              m2m_ntotalrcvranks;
  unsigned              m2m_initialized;  
  unsigned              m2m_rrankIndex; 
  CmiDirectM2mHandler   m2m_sdone;
  void                * m2m_sdonecontext;
} PAMICmiDirectM2mHandle;  

CpvDeclare(PAMICmiDirectM2mHandle*, _handle);
CpvDeclare(int, _completion_handler);

static void m2m_recv_done(pami_context_t ctxt, void *clientdata, pami_result_t result) 
{
  PAMICmiDirectM2mHandle *handle = (PAMICmiDirectM2mHandle *)clientdata;  
  //acquire lock if processed by many comm threads and contexts?
  handle->m2m_rcvcounter ++;
    
  if (handle->m2m_rcvcounter == handle->m2m_nzrcvranks) {
    //printf ("Calling manytomany rdone for handle %p on rank %d counter %d nexp %d\n", 
    //    handle, CmiMyPe(),
    //    handle->m2m_rcvcounter, handle->m2m_nzrcvranks);
    handle->m2m_rcvcounter = 0;
#if CMK_SMP && CMK_ENABLE_ASYNC_PROGRESS
    //Called from comm thread
    CmiSendPeer (handle->myrank, sizeof(M2mCompletionMsg), (char*)&handle->cmsg);
#else
    if (handle->m2m_rdone)
      handle->m2m_rdone(handle->m2m_rdonecontext);
#endif
  }
}

static void m2m_send_done(pami_context_t ctxt, void *clientdata, pami_result_t result) 
{
  PAMICmiDirectM2mHandle *handle = (PAMICmiDirectM2mHandle *)clientdata;  
  //acquire lock if processed by many comm threads and contexts?
  handle->m2m_sndcounter ++;
  if (handle->m2m_sndcounter == handle->m2m_nsndranks) {
    //if in comm thread send a converse message
    //else
    handle->m2m_sndcounter = 0;
    if (handle->m2m_sdone)
      handle->m2m_sdone(handle->m2m_sdonecontext); 
  }
}

static void m2m_rdone_mainthread (void *m) {
  M2mCompletionMsg *msg = (M2mCompletionMsg *) m;
  PAMICmiDirectM2mHandle *handle = (PAMICmiDirectM2mHandle *)msg->handle;
  if (handle->m2m_rdone)
    handle->m2m_rdone(handle->m2m_rdonecontext);
}

static void m2m_s8_dispatch (pami_context_t       context,  
			     void               * clientdata,
			     const void         * header_addr, 
			     size_t               header_size, 
			     const void         * pipe_addr,   
			     size_t               pipe_size,   
			     pami_endpoint_t      origin,
			     pami_recv_t         * recv)       
{
  PAMI_M2mHeader *hdr = (PAMI_M2mHeader *) header_addr;
  PAMICmiDirectM2mHandle *handlevec = CpvAccessOther(_handle, hdr->dstrank);  
  PAMICmiDirectM2mHandle *handle = &handlevec[hdr->connid];
  char *buffer = handle->m2m_rcvbuf + handle->m2m_rdispls[hdr->srcindex];

  //Copy 8 bytes
  *(uint64_t *)buffer = *(uint64_t*)pipe_addr;
  m2m_recv_done (context, handle, PAMI_SUCCESS);
}


static void m2m_spkt_dispatch (pami_context_t       context,  
			      void               * clientdata,
			      const void         * header_addr, 
			      size_t               header_size, 
			      const void         * pipe_addr,   
			      size_t               pipe_size,   
			      pami_endpoint_t      origin,
			      pami_recv_t         * recv)       
{
  PAMI_M2mHeader *hdr = (PAMI_M2mHeader *) header_addr;
  PAMICmiDirectM2mHandle *handlevec = CpvAccessOther(_handle, hdr->dstrank);   
  PAMICmiDirectM2mHandle *handle = &handlevec[hdr->connid];

  char *buffer = handle->m2m_rcvbuf + handle->m2m_rdispls[hdr->srcindex];
  memcpy (buffer, pipe_addr, pipe_size);
  m2m_recv_done (context, handle, PAMI_SUCCESS);
}



static void m2m_pkt_dispatch (pami_context_t       context,  
			      void               * clientdata,
			      const void         * header_addr, 
			      size_t               header_size, 
			      const void         * pipe_addr,   
			      size_t               pipe_size,   
			      pami_endpoint_t      origin,
			      pami_recv_t         * recv)       
{
  PAMI_M2mHeader *hdr = (PAMI_M2mHeader *) header_addr;

  //CmiAssert (hdr->dstrank < CmiMyNodeSize());
  //CmiAssert (hdr->connid  < MAX_CONN);

  PAMICmiDirectM2mHandle *handlevec = CpvAccessOther(_handle, hdr->dstrank);
  //CmiAssert (handlevec != NULL);
  
  //fprintf(stderr, "m2m_pkt_dispatch: mype %d connid %d dstrank %d handlevec %p\n",
  //  CmiMyPe(), hdr->connid, hdr->dstrank, handlevec);
  
  PAMICmiDirectM2mHandle *handle = &handlevec[hdr->connid];

  char *buffer = handle->m2m_rcvbuf + handle->m2m_rdispls[hdr->srcindex];

  if (recv) {
    recv->local_fn = m2m_recv_done;
    recv->cookie   = handle;
    recv->type     = PAMI_TYPE_BYTE;
    recv->addr     = buffer;
    recv->offset   = 0;
    recv->data_fn  = PAMI_DATA_COPY;
  }
  else {
    memcpy (buffer, pipe_addr, pipe_size);
    m2m_recv_done (context, handle, PAMI_SUCCESS);
  }
}


void * CmiDirect_manytomany_allocate_handle () {  
#if CMK_SMP && !CMK_ENABLE_ASYNC_PROGRESS
    CmiAbort("!!!!!!!!!Please build Charm++ with async in order to use many-to-many interface\n");
#else 
  if (!CpvInitialized(_handle))
    CpvInitialize(PAMICmiDirectM2mHandle*, _handle);
  if (!CpvInitialized(_completion_handler))
    CpvInitialize(int, _completion_handler);  
  ppc_msync();
  
  if (CpvAccess(_handle) == NULL) {
    CpvAccess(_handle) = (PAMICmiDirectM2mHandle *)malloc (MAX_CONN *sizeof(PAMICmiDirectM2mHandle));
    memset (CpvAccess(_handle),0,MAX_CONN*sizeof (PAMICmiDirectM2mHandle));
    CpvAccess(_completion_handler) = CmiRegisterHandler(m2m_rdone_mainthread);
  }
  
  //printf ("allocate_handle on rank %d %p\n", CmiMyPe(), CpvAccess(_handle));
  return CpvAccess(_handle);
#endif
}


void   CmiDirect_manytomany_initialize_recvbase(void                 * h,
						unsigned               tag,
						CmiDirectM2mHandler    donecb,
						void                 * context,
						char                 * rcvbuf,
						unsigned               nranks,
						unsigned               myIdx )
{
#if CMK_SMP && !CMK_ENABLE_ASYNC_PROGRESS
    CmiAbort("!!!!!!!!!Please build Charm++ with async in order to use many-to-many interface\n");
#else 
  PAMICmiDirectM2mHandle *handle = &(((PAMICmiDirectM2mHandle *) h)[tag]);
  //PAMICmiDirectM2mHandle *handle = &(CpvAccess(_handle)[tag]);

  //printf ("manytomany recvbase on rank %d handle %p conn %d nranks %d\n", 
  //  CmiMyPe(), handle, tag, nranks);

  handle->myrank = CmiMyRank();
  handle->cmsg.handle = handle;
  CmiSetHandler (&handle->cmsg, CpvAccess(_completion_handler));

  handle->m2m_initialized = 1;
  assert ( tag < MAX_CONN  );
  handle->m2m_rcvbuf   = rcvbuf;

  handle->m2m_rdone        = donecb;
  handle->m2m_rdonecontext = context;
  handle->m2m_ntotalrcvranks    = nranks;
  
  //Receiver is not sender
  //if (myIdx == (unsigned)-1) 
  //(handle->m2m_ntotalrcvranks)++;
    
  handle->m2m_rcvlens   = malloc (sizeof(int) * handle->m2m_ntotalrcvranks);
  handle->m2m_rdispls   = malloc (sizeof(int) * handle->m2m_ntotalrcvranks);
  
  assert (handle->m2m_rcvlens != NULL);
  
  memset (handle->m2m_rcvlens, 0, handle->m2m_ntotalrcvranks * sizeof(int));
  memset (handle->m2m_rdispls, 0, handle->m2m_ntotalrcvranks * sizeof(int));
  
  //Receiver is not sender
  //if (myIdx == (unsigned)-1) {
  //Receiver doesnt send any data
  //  myIdx =     handle->m2m_ntotalrcvranks - 1;
  //CmiDirect_manytomany_initialize_recv (h, tag,  myIdx, 0, 0, CmiMyPe());
  //}
  handle->m2m_rrankIndex = myIdx;
#endif
}

void   CmiDirect_manytomany_initialize_recv ( void          * h,
					      unsigned        tag,
					      unsigned        idx,
					      unsigned        displ,
					      unsigned        bytes,
					      unsigned        rank )
{
#if CMK_SMP && !CMK_ENABLE_ASYNC_PROGRESS
    CmiAbort("!!!!!!!!!Please build Charm++ with async in order to use many-to-many interface\n");
#else 
  PAMICmiDirectM2mHandle *handle = &(((PAMICmiDirectM2mHandle *) h)[tag]);
  assert ( tag < MAX_CONN  );
  
  if (handle->m2m_rcvlens[idx] == 0 && bytes > 0)
    handle->m2m_nzrcvranks ++;

  handle->m2m_rcvlens  [idx]   = bytes;
  handle->m2m_rdispls  [idx]   = displ;
#endif
}


void   CmiDirect_manytomany_initialize_sendbase( void                 * h,
						 unsigned               tag,
						 CmiDirectM2mHandler    donecb,
						 void                 * context,
						 char                 * sndbuf,
						 unsigned               nranks,
						 unsigned               myIdx )
{
#if CMK_SMP && !CMK_ENABLE_ASYNC_PROGRESS
    CmiAbort("!!!!!!!!!Please build Charm++ with async in order to use many-to-many interface\n");
#else 
  PAMICmiDirectM2mHandle *handle = &(((PAMICmiDirectM2mHandle *) h)[tag]);
  assert ( tag < MAX_CONN  );
  handle->m2m_sndbuf       = sndbuf;
  handle->m2m_sdone        = donecb;
  handle->m2m_sdonecontext = context;
  
  handle->m2m_nsndranks    = nranks;
  handle->m2m_srankIndex   = myIdx;  
  handle->m2m_sndlens      = (unsigned int *) malloc (sizeof(unsigned int) * nranks);
  handle->m2m_sdispls      = (unsigned int *) malloc (sizeof(unsigned int) * nranks);
  handle->m2m_lranks       = (unsigned int *) malloc (sizeof(unsigned int) * nranks);
  handle->m2m_node_eps     = (pami_endpoint_t *) malloc (sizeof(pami_endpoint_t) * nranks);
  handle->m2m_permutation  = (unsigned int *) malloc (sizeof(unsigned int) * nranks);
  handle->m2m_hdrs = (PAMI_M2mHeader *) malloc(sizeof(PAMI_M2mHeader) * nranks);

  memset (handle->m2m_sndlens,    0, nranks * sizeof(int));
  memset (handle->m2m_sdispls,    0, nranks * sizeof(int));
  memset (handle->m2m_lranks,     0, nranks * sizeof(int));
  memset (handle->m2m_node_eps,   0, nranks * sizeof(pami_endpoint_t));
  memset (handle->m2m_permutation,0, nranks * sizeof(int));  

#if CMK_SMP && CMK_ENABLE_ASYNC_PROGRESS
  //we have a completion callback
  if (handle->m2m_sdone != NULL) {
    handle->swork[0].start = 0;
    handle->swork[0].end   = handle->m2m_nsndranks;   
    handle->swork[0].handle = handle;
    handle->n_work = 1;

    int context_id = MY_CONTEXT_ID();
    context_id ++;
    if (context_id >= cmi_pami_numcontexts)
      context_id = 0;	      
    pami_context_t context = cmi_pami_contexts[context_id];    
    handle->swork[0].context = context;
  }
  else {
    int i = 0;
    int context_id = MY_CONTEXT_ID();
    pami_context_t context = NULL;
    int start = 0, nranks = 0;
    int ncontexts = cmi_pami_numcontexts;
    if (ncontexts > MAX_NWORK)
      ncontexts = MAX_NWORK;
    if (ncontexts > handle->m2m_nsndranks)
      ncontexts = handle->m2m_nsndranks;
    handle->n_work = ncontexts;

    nranks = handle->m2m_nsndranks / ncontexts;   
    for (i = 0; i < ncontexts; ++i) {
      handle->swork[i].start  = start;
      handle->swork[i].end    = start + nranks;   
      handle->swork[i].handle = handle;
      start += nranks;
      if (i == ncontexts - 1)
	handle->swork[i].end  = handle->m2m_nsndranks;

      context_id ++;
      if (context_id >= cmi_pami_numcontexts)
	context_id = 0;	      
      context = cmi_pami_contexts[context_id];
      handle->swork[i].context = context;
    }
  }
#else
  PAMIX_CONTEXT_LOCK(MY_CONTEXT());
  handle->swork[0].start = 0;
  handle->swork[0].end   = handle->m2m_nsndranks;   
  handle->swork[0].handle = handle;
  handle->n_work = 1;
  handle->swork[0].context = MY_CONTEXT();
  PAMIX_CONTEXT_UNLOCK(MY_CONTEXT());
#endif
#endif
}

#define PRIME_A  3010349UL
#define PRIME_B  3571UL

void   CmiDirect_manytomany_initialize_send ( void        * h,
					      unsigned      tag, 
					      unsigned      idx,
					      unsigned      displ,
					      unsigned      bytes,
					      unsigned      pe )
{
#if CMK_SMP && !CMK_ENABLE_ASYNC_PROGRESS
    CmiAbort("!!!!!!!!!Please build Charm++ with async in order to use many-to-many interface\n");
#else 
  PAMICmiDirectM2mHandle *handle = &(((PAMICmiDirectM2mHandle *) h)[tag]);
  assert ( tag < MAX_CONN  );  
  handle->m2m_sndlens    [idx]   = bytes;
  handle->m2m_sdispls    [idx]   = displ;
  
  int lrank                      = CmiRankOf(pe);
  handle->m2m_lranks     [idx]   = lrank;
  
  pami_endpoint_t target;
  //get the destination context
#if CMK_PAMI_MULTI_CONTEXT 
  size_t dst_context = (lrank>>LTPS);
#else
  size_t dst_context = 0;
#endif
  PAMI_Endpoint_create (cmi_pami_client, (pami_task_t)CmiGetNodeGlobal(CmiNodeOf(pe),CmiMyPartition()),
                        dst_context, &target);
  handle->m2m_node_eps   [idx]   = target;

  //uint64_t p_rand = ((uint64_t)idx+1)*PRIME_A + PRIME_B*(CmiMyPe()+1);
  unsigned seed = CmiMyPe()+1;
  //start at a random location and move linearly from there
  uint64_t p_rand = rand_r(&seed) + idx + 1;
  //uint64_t p_rand = (uint64_t)idx + 1 + CmiMyPe();
  //uint64_t p_rand   =  idx + 1;
  handle->m2m_permutation[idx]   = (uint32_t)(p_rand%handle->m2m_nsndranks);
  handle->m2m_hdrs[idx].connid   = tag;  
  handle->m2m_hdrs[idx].dstrank  = lrank; 
  handle->m2m_hdrs[idx].srcindex = handle->m2m_srankIndex;
#endif
}

static void  _internal_machine_send   ( pami_context_t      context, 
					pami_endpoint_t     target_ep, 
					int                 rank, 
					int                 hdrsize,
					char              * hdr,
					int                 size, 
					char              * msg,
					pami_event_function cb_done,
					void              * cd)
{
  if (size < 128) {
    pami_send_immediate_t parameters;
    parameters.dispatch        = (size == 8)? M2M_PAMI_S8DISPATCH : M2M_PAMI_SDISPATCH;
    //parameters.dispatch        = M2M_PAMI_SDISPATCH;
    parameters.header.iov_base = hdr;
    parameters.header.iov_len  = hdrsize;
    parameters.data.iov_base   = msg;
    parameters.data.iov_len    = size;
    parameters.dest            = target_ep;
    
    PAMI_Send_immediate (context, &parameters);
    //if (cb_done)
    //cb_done (context, cd, PAMI_SUCCESS);
  }
  else {
    pami_send_t parameters;
    parameters.send.dispatch        = M2M_PAMI_DISPATCH;
    parameters.send.header.iov_base = hdr;
    parameters.send.header.iov_len  = hdrsize;
    parameters.send.data.iov_base   = msg;
    parameters.send.data.iov_len    = size;
    parameters.events.cookie        = cd;
    parameters.events.local_fn      = cb_done;
    parameters.events.remote_fn     = NULL;
    memset(&parameters.send.hints, 0, sizeof(parameters.send.hints));
    parameters.send.dest            = target_ep;
    
    PAMI_Send (context, &parameters);
  }
}

pami_result_t   _cmidirect_m2m_send_post_handler (pami_context_t     context,
						  void             * cd) 
{
  PAMI_M2mWork_t  *work = (PAMI_M2mWork_t *) cd;
  PAMICmiDirectM2mHandle *handle = (PAMICmiDirectM2mHandle *)work->handle;
  
  int i = 0;
  int pidx = 0;
  char *buffer = NULL;
  int bytes = NULL;

  pami_event_function cb_done = m2m_send_done;
  void *clientdata = handle;

  if (handle->m2m_sdone == NULL) {
    cb_done     = NULL;
    clientdata  = NULL;
  }

  for (i = work->start; i < work->end; ++i) {
    pidx   = handle->m2m_permutation[i];
    buffer = handle->m2m_sndbuf + handle->m2m_sdispls[pidx];
    bytes  = handle->m2m_sndlens[pidx];
    
    _internal_machine_send(context,
			   handle->m2m_node_eps[pidx],
			   handle->m2m_lranks[pidx],
			   sizeof(PAMI_M2mHeader),
			   (char*)&(handle->m2m_hdrs[pidx]),
			   bytes, 
			   buffer,
			   cb_done,
			   clientdata);
  }  

  return PAMI_SUCCESS;
}


void _cmidirect_m2m_initialize (pami_context_t *contexts, int nc) {
  pami_dispatch_hint_t options = (pami_dispatch_hint_t) {0};
  pami_dispatch_callback_function pfn;
  int i = 0;
  for (i = 0; i < nc; ++i) {
    pfn.p2p = m2m_pkt_dispatch;
    PAMI_Dispatch_set (contexts[i],
		       M2M_PAMI_DISPATCH,
		       pfn,
		       NULL,
		       options);

    pfn.p2p = m2m_spkt_dispatch;
    PAMI_Dispatch_set (contexts[i],
		       M2M_PAMI_SDISPATCH,
		       pfn,
		       NULL,
		       options);

    pfn.p2p = m2m_s8_dispatch;
    PAMI_Dispatch_set (contexts[i],
		       M2M_PAMI_S8DISPATCH,
		       pfn,
		       NULL,
		       options);
  }
}


void   CmiDirect_manytomany_start ( void       * h,
				    unsigned     tag ) {
#if CMK_SMP && !CMK_ENABLE_ASYNC_PROGRESS
    CmiAbort("!!!!!!!!!Please build Charm++ with async in order to use many-to-many interface\n");
#else 
  PAMICmiDirectM2mHandle *handle = &(((PAMICmiDirectM2mHandle *) h)[tag]);
  assert (tag < MAX_CONN);

  //printf ("Calling manytomany_start for conn %d handle %p on rank %d\n", tag, 
  //  handle, CmiMyPe());
  
#if CMK_SMP && CMK_ENABLE_ASYNC_PROGRESS
  //we have a completion callback
  if (handle->m2m_sdone != NULL) {
    PAMI_Context_post ( handle->swork[0].context, 
		       &handle->swork[0].work, 
		       _cmidirect_m2m_send_post_handler,
		       &handle->swork[0]);
  }
  else {
    int i;
    for (i = 0; i < handle->n_work; ++i) {
      PAMI_Context_post( handle->swork[i].context, 
			&handle->swork[i].work, 
			_cmidirect_m2m_send_post_handler,
			&handle->swork[i]);
    }
  }
#else
  PAMIX_CONTEXT_LOCK(MY_CONTEXT());
  _cmidirect_m2m_send_post_handler (MY_CONTEXT(), &handle->swork[0]);
  PAMIX_CONTEXT_UNLOCK(MY_CONTEXT());
#endif
#endif
}
