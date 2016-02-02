
#include "converse.h"
#include "cmidirectmanytomany.h"
#define MAX_CONN   8

#define M2M_PAMI_S8DISPATCH 13
#define M2M_PAMI_SDISPATCH  14
#define M2M_PAMI_DISPATCH   15

typedef struct _pami_m2mhdr {
  uint8_t    dstrank;
  uint8_t    connid;
  uint32_t   srcindex;
} PAMI_M2mHeader;

typedef struct _pami_m2m_work {
  int            start;
  int            end;
  void         * handle;
  pami_context_t context;
  pami_work_t    work;
} PAMI_M2mWork_t;

typedef struct _m2m_completionmsg {
  char  hdr [CmiMsgHeaderSizeBytes];
  void  *handle;
  int    rank;
} M2mCompletionMsg;

typedef struct _m2m_sendinfo {
  char            * buf;
  uint32_t          bytes;
  pami_endpoint_t   ep;
  uint16_t          dispatch;
  PAMI_M2mHeader    hdr;
} M2mSendInfo;

#if CMK_SMP && CMK_ENABLE_ASYNC_PROGRESS
#define M2M_PARALLEL_CONTEXT 1
#elif CMK_SMP
#define M2M_PARALLEL_CONTEXT 1
#else
#define M2M_PARALLEL_CONTEXT 0
#endif

#if M2M_PARALLEL_CONTEXT
#define MAX_NWORK 8
#else
#define MAX_NWORK 1
#endif

typedef struct _pami_cmidhandle {
  int                   myrank;
  unsigned              m2m_rcvcounter;
  unsigned              m2m_nzrcvranks;
  unsigned              m2m_nsndranks;
  char                * m2m_rcvbuf     ;
  unsigned            * m2m_rcvlens    ;
  unsigned            * m2m_rdispls    ;
  M2mSendInfo         * m2m_sndinfo    ;
  PAMI_M2mWork_t        swork[MAX_NWORK];
  int                   n_work;

  //Less frequently used (or unused) during runtime execution
  char                * m2m_sndbuf     ;
  unsigned              m2m_sndcounter ;
  unsigned              m2m_srankIndex;	  //Stored in header

  CmiDirectM2mHandler   m2m_rdone;
  void                * m2m_rdonecontext;
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
  int ntotal = 0;
  PAMICmiDirectM2mHandle *handle = (PAMICmiDirectM2mHandle *)clientdata;
  //acquire lock if processed by many comm threads and contexts?
  handle->m2m_rcvcounter ++;
  ntotal = handle->m2m_rcvcounter;

  if (ntotal == handle->m2m_nzrcvranks) {
    //printf ("Calling manytomany rdone for handle %p on rank %d counter %d nexp %d\n",
    //    handle, CmiMyPe(),
    //    handle->m2m_rcvcounter, handle->m2m_nzrcvranks);
    handle->m2m_rcvcounter = 0;
#if CMK_SMP && (M2M_PARALLEL_CONTEXT || LTPS)
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
#if CMK_SMP && (M2M_PARALLEL_CONTEXT || LTPS)
  PAMICmiDirectM2mHandle *handlevec = CpvAccessOther(_handle, hdr->dstrank);
#else
  PAMICmiDirectM2mHandle *handlevec = CpvAccess(_handle);
#endif
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
#if CMK_SMP && (M2M_PARALLEL_CONTEXT || LTPS)
  PAMICmiDirectM2mHandle *handlevec = CpvAccessOther(_handle, hdr->dstrank);
#else
  PAMICmiDirectM2mHandle *handlevec = CpvAccess(_handle);
#endif
  PAMICmiDirectM2mHandle *handle = &handlevec[hdr->connid];

  char *buffer = handle->m2m_rcvbuf + handle->m2m_rdispls[hdr->srcindex];
  if (pipe_size == 32) {
    uint64_t *src = (uint64_t *)pipe_addr;
    uint64_t *dst = (uint64_t *)buffer;

    dst[0] = src[0];
    dst[1] = src[1];
    dst[2] = src[2];
    dst[3] = src[3];
  }
  else
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

#if CMK_SMP && (M2M_PARALLEL_CONTEXT || LTPS)
  PAMICmiDirectM2mHandle *handlevec = CpvAccessOther(_handle, hdr->dstrank);
#else
  PAMICmiDirectM2mHandle *handlevec = CpvAccess(_handle);
#endif

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
  if (!CpvInitialized(_handle))
    CpvInitialize(PAMICmiDirectM2mHandle*, _handle);
  if (!CpvInitialized(_completion_handler))
    CpvInitialize(int, _completion_handler);

  if (CpvAccess(_handle) == NULL) {
    CpvAccess(_handle) = (PAMICmiDirectM2mHandle *)malloc (MAX_CONN *sizeof(PAMICmiDirectM2mHandle));
    memset (CpvAccess(_handle),0,MAX_CONN*sizeof (PAMICmiDirectM2mHandle));
    CpvAccess(_completion_handler) = CmiRegisterHandler(m2m_rdone_mainthread);
  }

  //printf ("allocate_handle on rank %d %p\n", CmiMyPe(), CpvAccess(_handle));
  return CpvAccess(_handle);
}


void   CmiDirect_manytomany_initialize_recvbase(void                 * h,
						unsigned               tag,
						CmiDirectM2mHandler    donecb,
						void                 * context,
						char                 * rcvbuf,
						unsigned               nranks,
						unsigned               myIdx )
{
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
}

void   CmiDirect_manytomany_initialize_recv ( void          * h,
					      unsigned        tag,
					      unsigned        idx,
					      unsigned        displ,
					      unsigned        bytes,
					      unsigned        rank )
{
  PAMICmiDirectM2mHandle *handle = &(((PAMICmiDirectM2mHandle *) h)[tag]);
  assert ( tag < MAX_CONN  );

  if (handle->m2m_rcvlens[idx] == 0 && bytes > 0)
    handle->m2m_nzrcvranks ++;

  handle->m2m_rcvlens  [idx]   = bytes;
  handle->m2m_rdispls  [idx]   = displ;
}


void   CmiDirect_manytomany_initialize_sendbase( void                 * h,
						 unsigned               tag,
						 CmiDirectM2mHandler    donecb,
						 void                 * context,
						 char                 * sndbuf,
						 unsigned               nranks,
						 unsigned               myIdx )
{
  PAMICmiDirectM2mHandle *handle = &(((PAMICmiDirectM2mHandle *) h)[tag]);
  assert ( tag < MAX_CONN  );
  handle->m2m_sndbuf       = sndbuf;
  handle->m2m_sdone        = donecb;
  handle->m2m_sdonecontext = context;

  handle->m2m_nsndranks    = nranks;
  handle->m2m_srankIndex   = myIdx;
  handle->m2m_sndinfo = (M2mSendInfo *)malloc(nranks * sizeof(M2mSendInfo));
  memset (handle->m2m_sndinfo,0, nranks * sizeof(M2mSendInfo));

#if M2M_PARALLEL_CONTEXT
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
  PAMICmiDirectM2mHandle *handle = &(((PAMICmiDirectM2mHandle *) h)[tag]);
  assert ( tag < MAX_CONN  );

  int lrank                      = CmiRankOf(pe);
  pami_endpoint_t target;
  //get the destination context
#if CMK_PAMI_MULTI_CONTEXT
  size_t dst_context = (lrank>>LTPS);
#else
  size_t dst_context = 0;
#endif
  PAMI_Endpoint_create (cmi_pami_client, (pami_task_t)CmiNodeOf(pe),
			dst_context, &target);

  unsigned seed = CmiMyPe()+1;
  //start at a random location and move linearly from there
  //uint64_t p_rand = rand_r(&seed) + idx + 1;
  uint64_t p_rand = ((uint64_t)idx+1)*PRIME_A + PRIME_B*(CmiMyPe()+1);
  uint32_t pidx = (uint32_t)(p_rand%handle->m2m_nsndranks);

  char *buffer = handle->m2m_sndbuf + displ;
  handle->m2m_sndinfo[pidx].buf    = buffer;
  handle->m2m_sndinfo[pidx].bytes  = bytes;
  handle->m2m_sndinfo[pidx].ep     = target;
  handle->m2m_sndinfo[pidx].hdr.connid   = tag;
  handle->m2m_sndinfo[pidx].hdr.dstrank  = lrank;
  handle->m2m_sndinfo[pidx].hdr.srcindex = handle->m2m_srankIndex;

  if (bytes == 8)
    handle->m2m_sndinfo[pidx].dispatch = M2M_PAMI_S8DISPATCH;
  else if (bytes < 128)
    handle->m2m_sndinfo[pidx].dispatch = M2M_PAMI_SDISPATCH;
  else
    handle->m2m_sndinfo[pidx].dispatch = M2M_PAMI_DISPATCH;
}

pami_result_t   _cmidirect_m2m_send_post_handler (pami_context_t     context,
						  void             * cd)
{
  PAMI_M2mWork_t  *work = (PAMI_M2mWork_t *) cd;
  PAMICmiDirectM2mHandle *handle = (PAMICmiDirectM2mHandle *)work->handle;

#if CMK_TRACE_ENABLED
  double starttime = CmiWallTimer();
#endif

  int i = 0;
  CmiAssert(handle->m2m_sdone == NULL);
  pami_send_t  parameters;

  parameters.send.header.iov_len  = sizeof(PAMI_M2mHeader);
  parameters.events.cookie        = NULL;
  parameters.events.local_fn      = NULL;
  parameters.events.remote_fn     = NULL;
  memset(&parameters.send.hints, 0, sizeof(parameters.send.hints));

  for (i = work->start; i < work->end; ++i) {
    M2mSendInfo *sndinfo = &handle->m2m_sndinfo[i];
    parameters.send.data.iov_base   = sndinfo->buf;
    parameters.send.data.iov_len    = sndinfo->bytes;
    parameters.send.dest            = sndinfo->ep;
    parameters.send.header.iov_base = &sndinfo->hdr;
    parameters.send.dispatch        = sndinfo->dispatch;

    if (sndinfo->bytes < 128)
      PAMI_Send_immediate(context, &parameters.send);
    else
      PAMI_Send (context, &parameters);
  }

#if CMK_TRACE_ENABLED
  traceUserBracketEvent(30006, starttime, CmiWallTimer());
#endif

  return PAMI_SUCCESS;
}


void _cmidirect_m2m_initialize (pami_context_t *contexts, int nc) {
  pami_dispatch_hint_t soptions = (pami_dispatch_hint_t) {0};
  pami_dispatch_hint_t loptions = (pami_dispatch_hint_t) {0};

  soptions.long_header    = PAMI_HINT_DISABLE;
  soptions.recv_immediate = PAMI_HINT_ENABLE;
  soptions.use_rdma       = PAMI_HINT_DISABLE;

  loptions.long_header     = PAMI_HINT_DISABLE;
  loptions.recv_contiguous = PAMI_HINT_ENABLE;
  loptions.recv_copy       = PAMI_HINT_ENABLE;

  pami_dispatch_callback_function pfn;
  int i = 0;
  for (i = 0; i < nc; ++i) {
    pfn.p2p = m2m_pkt_dispatch;
    PAMI_Dispatch_set (contexts[i],
		       M2M_PAMI_DISPATCH,
		       pfn,
		       NULL,
		       loptions);

    pfn.p2p = m2m_spkt_dispatch;
    PAMI_Dispatch_set (contexts[i],
		       M2M_PAMI_SDISPATCH,
		       pfn,
		       NULL,
		       soptions);

    pfn.p2p = m2m_s8_dispatch;
    PAMI_Dispatch_set (contexts[i],
		       M2M_PAMI_S8DISPATCH,
		       pfn,
		       NULL,
		       soptions);
  }
}


void   CmiDirect_manytomany_start ( void       * h,
				    unsigned     tag ) {
  PAMICmiDirectM2mHandle *handle = &(((PAMICmiDirectM2mHandle *) h)[tag]);
  assert (tag < MAX_CONN);

  //printf ("Calling manytomany_start for conn %d handle %p on rank %d\n", tag,
  //  handle, CmiMyPe());

#if M2M_PARALLEL_CONTEXT
  //we have a completion callback
  if (handle->m2m_sdone != NULL) {
    PAMI_Context_post ( handle->swork[0].context,
			&handle->swork[0].work,
			_cmidirect_m2m_send_post_handler,
			&handle->swork[0]);
  }
  else {
    int i;
#if CMK_TRACE_ENABLED
    double starttime = CmiWallTimer();
#endif
    for (i = 0; i < handle->n_work; ++i)
#if !CMK_ENABLE_ASYNC_PROGRESS
      if (handle->swork[i].context != MY_CONTEXT())
#endif
	PAMI_Context_post( handle->swork[i].context,
			   &handle->swork[i].work,
			   _cmidirect_m2m_send_post_handler,
			   &handle->swork[i]);

#if CMK_TRACE_ENABLED
    traceUserBracketEvent(30007, starttime, CmiWallTimer());
#endif

#if !CMK_ENABLE_ASYNC_PROGRESS
    for (i = 0; i < handle->n_work; ++i)
      if (handle->swork[i].context == MY_CONTEXT()) {
	PAMIX_CONTEXT_LOCK(MY_CONTEXT());
	_cmidirect_m2m_send_post_handler (MY_CONTEXT(), &handle->swork[i]);
	PAMIX_CONTEXT_UNLOCK(MY_CONTEXT());
      }
#endif
  }
#else
  PAMIX_CONTEXT_LOCK(MY_CONTEXT());
  _cmidirect_m2m_send_post_handler (MY_CONTEXT(), &handle->swork[0]);
  PAMIX_CONTEXT_UNLOCK(MY_CONTEXT());
#endif
}
