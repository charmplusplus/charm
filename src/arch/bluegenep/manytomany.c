
#include <converse.h>
#include <cmidirectmanytomany.h>
#include <dcmf.h>
#include <dcmf_multisend.h>

#define MAX_CONN   8

typedef struct _bgp_cmidhandle {
  DCMF_Protocol_t  m2m_protocol            __attribute__((__aligned__(16)));
  DCMF_Request_t   send_request[MAX_CONN]  __attribute__((__aligned__(16)));
  DCMF_Request_t   recv_request[MAX_CONN]  __attribute__((__aligned__(16)));
  
  char          * m2m_rcvbuf      [MAX_CONN];
  char          * m2m_sndbuf      [MAX_CONN];
  unsigned      * m2m_rcvlens     [MAX_CONN];
  unsigned      * m2m_rdispls     [MAX_CONN];
  unsigned      * m2m_rcvcounters [MAX_CONN];
  unsigned      * m2m_sndlens     [MAX_CONN];
  unsigned      * m2m_sdispls     [MAX_CONN];
  unsigned      * m2m_sndcounters [MAX_CONN];
  unsigned      * m2m_permutation [MAX_CONN];
  unsigned      * m2m_ranks       [MAX_CONN];
  unsigned        m2m_initialized [MAX_CONN];
  
  unsigned       m2m_nsndranks   [MAX_CONN];
  unsigned       m2m_nrcvranks   [MAX_CONN];
  unsigned       m2m_srankIndex  [MAX_CONN];
  unsigned       m2m_rrankIndex  [MAX_CONN];
  
  DCMF_Callback_t  m2m_rcb_done [MAX_CONN];
  DCMF_Callback_t  m2m_scb_done [MAX_CONN];  
} BGPCmiDirectM2mHandle;  

BGPCmiDirectM2mHandle  *_handle;

DCMF_Request_t *cb_Manytomany ( unsigned          conn_id,
				void            * arg,
				char           ** rcvbuf,
				unsigned       ** rcvlens,
				unsigned       ** rcvdispls,
				unsigned       ** rcvcounters,
				unsigned        * nranks,
				unsigned        * rankIndex,
				DCMF_Callback_t * cb_done ) 
{  
  assert (conn_id < MAX_CONN);  
  BGPCmiDirectM2mHandle *handle =   (BGPCmiDirectM2mHandle *) arg;
  assert (handle->m2m_initialized[conn_id] != 0);
  
  * rcvbuf       =  handle->m2m_rcvbuf[conn_id];
  * rcvlens      =  handle->m2m_rcvlens[conn_id];
  * rcvdispls    =  handle->m2m_rdispls[conn_id];
  * rcvcounters  =  handle->m2m_rcvcounters[conn_id];
  * nranks       =  handle->m2m_nrcvranks [conn_id];
  * rankIndex    =  handle->m2m_rrankIndex[conn_id];
  * cb_done      =  handle->m2m_rcb_done[conn_id];

  return &handle->recv_request [conn_id];
}

void * CmiDirect_manytomany_allocate_handle () {
  if (!_handle) {
    _handle = (BGPCmiDirectM2mHandle *)malloc (sizeof (BGPCmiDirectM2mHandle));
    memset (_handle, 0, (sizeof (BGPCmiDirectM2mHandle)));
    
    DCMF_Manytomany_Configuration_t mconfig;
    
    mconfig.protocol     =  DCMF_MEMFIFO_DMA_M2M_PROTOCOL;
    mconfig.cb_recv      =  cb_Manytomany;
    mconfig.arg          =  _handle;
    mconfig.nconnections =  MAX_CONN;
    
    DCMF_Manytomany_register (&_handle->m2m_protocol, &mconfig);
  }
  return _handle;
}


void   CmiDirect_manytomany_initialize_recvbase ( void                 * h,
						  unsigned               tag,
						  CmiDirectM2mHandler    donecb,
						  void                 * context,
						  char                 * rcvbuf,
						  unsigned               nranks,
						  unsigned               myIdx )
{
  //printf ("%d In m2mrecvbase tag %d \n", DCMF_Messager_rank(), tag);

  BGPCmiDirectM2mHandle *handle = (BGPCmiDirectM2mHandle *) h;

  handle->m2m_initialized[tag] = 1;

  assert ( tag < MAX_CONN  );
  handle->m2m_rcvbuf  [tag] = rcvbuf;

#if (DCMF_VERSION_MAJOR >= 2)
  handle->m2m_rcb_done[tag].function   = (void (*)(void*, DCMF_Error_t *)) donecb;
#else
  handle->m2m_rcb_done[tag].function   = donecb;
#endif

  handle->m2m_rcb_done[tag].clientdata = (void *) context;
  handle->m2m_nrcvranks  [tag] = nranks;
  
  //Receiver is not sender
  if (myIdx == (unsigned)-1) 
    (handle->m2m_nrcvranks [tag])++;
    
  handle->m2m_rcvlens[tag]     = malloc (sizeof(int) * handle->m2m_nrcvranks[tag]);
  handle->m2m_rdispls[tag]     = malloc (sizeof(int) * handle->m2m_nrcvranks[tag]);
  handle->m2m_rcvcounters[tag] = malloc (sizeof(int) * handle->m2m_nrcvranks[tag]);
  
  assert (handle->m2m_rcvlens[tag] != NULL);
  
  memset (handle->m2m_rcvlens[tag], 0, handle->m2m_nrcvranks[tag] * sizeof(int));
  memset (handle->m2m_rdispls[tag], 0, handle->m2m_nrcvranks[tag] * sizeof(int));
  memset (handle->m2m_rcvcounters[tag], 0, handle->m2m_nrcvranks[tag] * sizeof(int));
  
  //Receiver is not sender
  if (myIdx == (unsigned)-1) {
    //Receiver doesnt send any data
    myIdx =     handle->m2m_nrcvranks  [tag] - 1;
    CmiDirect_manytomany_initialize_recv (h, tag,  myIdx, 0, 0, CmiMyPe());
  }
  handle->m2m_rrankIndex [tag] = myIdx;
}

void   CmiDirect_manytomany_initialize_recv ( void          * h,
					      unsigned        tag,
					      unsigned        idx,
					      unsigned        displ,
					      unsigned        bytes,
					      unsigned        rank )
{
  BGPCmiDirectM2mHandle *handle = (BGPCmiDirectM2mHandle *) h;

  //printf ("%d In m2mrecv tag %d idx %d\n", DCMF_Messager_rank(), tag, idx);
  //assert ( idx < MAX_NODES );
  assert ( tag < MAX_CONN  );
  
  if (handle->m2m_rcvlens[tag] == NULL)
    printf ("%d: rcvlens == NULL for tag = %d\n", DCMF_Messager_rank(), tag);

  handle->m2m_rcvlens    [tag][idx]   = bytes;
  handle->m2m_rdispls    [tag][idx]   = displ;
}


void   CmiDirect_manytomany_initialize_sendbase  ( void                 * h,
						   unsigned               tag,
						   CmiDirectM2mHandler    donecb,
						   void                 * context,
						   char                 * sndbuf,
						   unsigned               nranks,
						   unsigned               myIdx )
{
  BGPCmiDirectM2mHandle *handle = (BGPCmiDirectM2mHandle *) h;

  assert ( tag < MAX_CONN  );
  handle->m2m_sndbuf[tag] = sndbuf;

#if (DCMF_VERSION_MAJOR >= 2)
  handle->m2m_scb_done[tag].function   = (void (*)(void*, DCMF_Error_t *)) donecb;
#else
  handle->m2m_scb_done[tag].function   = donecb;
#endif
  handle->m2m_scb_done[tag].clientdata = context;
  handle->m2m_nsndranks  [tag] = nranks;
  handle->m2m_srankIndex [tag] = myIdx;
  
  handle->m2m_sndlens    [tag] = (unsigned int *) malloc (sizeof(unsigned int) * nranks);
  handle->m2m_sdispls    [tag] = (unsigned int *) malloc (sizeof(unsigned int) * nranks);
  handle->m2m_sndcounters[tag] = (unsigned int *) malloc (sizeof(unsigned int) * nranks);
  handle->m2m_ranks      [tag] = (unsigned int *) malloc (sizeof(unsigned int) * nranks);
  handle->m2m_permutation[tag] = (unsigned int *) malloc (sizeof(unsigned int) * nranks);

  memset (handle->m2m_sndlens[tag], 0, nranks * sizeof(int));
  memset (handle->m2m_sdispls[tag], 0, nranks * sizeof(int));
  memset (handle->m2m_sndcounters[tag], 0, nranks * sizeof(int));
  memset (handle->m2m_ranks[tag], 0, nranks * sizeof(int));
  memset (handle->m2m_permutation[tag], 0, nranks * sizeof(int));
  
  //assert (nranks < MAX_NODES);
}

void   CmiDirect_manytomany_initialize_send ( void        * h,
					      unsigned      tag, 
					      unsigned      idx,
					      unsigned      displ,
					      unsigned      bytes,
					      unsigned      rank )
{
  BGPCmiDirectM2mHandle *handle = (BGPCmiDirectM2mHandle *) h;

  assert ( tag < MAX_CONN  );
  
  handle->m2m_sndlens    [tag][idx]   = bytes;
  handle->m2m_sdispls    [tag][idx]   = displ;
  handle->m2m_ranks      [tag][idx]   = CmiGetNodeGlobal(CmiNodeOf(rank),CmiMyPartition());
  handle->m2m_permutation[tag][idx]   = (idx+1)%handle->m2m_nsndranks[tag];
}



void   CmiDirect_manytomany_start ( void       * h,
				    unsigned     tag ) {
  BGPCmiDirectM2mHandle *handle = (BGPCmiDirectM2mHandle *) h;
  assert (tag < MAX_CONN);
  
  DCMF_Manytomany 
    (& handle->m2m_protocol,
     & handle->send_request[tag],
     handle->m2m_scb_done[tag],
     DCMF_MATCH_CONSISTENCY,
     tag,
     0,
     handle->m2m_srankIndex[tag],
     NULL,
     handle->m2m_sndbuf[tag],
     handle->m2m_sndlens[tag],
     handle->m2m_sdispls[tag],
     handle->m2m_sndcounters[tag],
     handle->m2m_ranks[tag],
     handle->m2m_permutation[tag],
     handle->m2m_nsndranks[tag]);  
}
