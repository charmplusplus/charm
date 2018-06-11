void _initOnesided( pami_context_t *contexts, int nc);

/* Support for Nocopy Direct API */

// Machine specific information for a nocopy pointer
typedef struct _cmi_pami_rzv_rdma_ptr {
  pami_memregion_t    mregion;
  size_t              offset;
} CmiPAMIRzvRdmaPtr_t;

void rdma_direct_get_dispatch (
    pami_context_t       context,
    void               * clientdata,
    const void         * header_addr,
    size_t               header_size,
    const void         * pipe_addr,
    size_t               pipe_size,
    pami_endpoint_t      origin,
    pami_recv_t         * recv);

/* Compiler checks to ensure that CMK_NOCOPY_DIRECT_BYTES in conv-common.h
 * is set to sizeof(CmiPAMIRzvRdmaPtr_t). CMK_NOCOPY_DIRECT_BYTES is used in
 * ckrdma.h to reserve bytes for source or destination metadata info.           */
#define DUMB_STATIC_ASSERT(test) typedef char sizeCheckAssertion[(!!(test))*2-1]

/* Check the value of CMK_NOCOPY_DIRECT_BYTES if the compiler reports an
 * error with the message "the size of an array must be greater than zero" */
DUMB_STATIC_ASSERT(sizeof(CmiPAMIRzvRdmaPtr_t) == CMK_NOCOPY_DIRECT_BYTES);

// Set the machine specific information for a nocopy pointer
void LrtsSetRdmaBufferInfo(void *info, const void *ptr, int size, unsigned short int mode){
  CmiPAMIRzvRdmaPtr_t *rdmaInfo = (CmiPAMIRzvRdmaPtr_t *)info;
#if CMK_BLUEGENEQ
  rdmaInfo->offset = (size_t)(ptr) - (size_t)cmi_pami_memregion[0].baseVA;
  memcpy(rdmaInfo->mregion, &cmi_pami_memregion[0].mregion, sizeof(pami_memregion_t));
#else // ppc64le
  rdmaInfo->offset = (size_t)(ptr);
  size_t bytes_out;
  pami_memregion_t mregion;

#if CMK_SMP && CMK_ENABLE_ASYNC_PROGRESS
  int c = CmiMyNode() % cmi_pami_numcontexts;
  pami_context_t my_context = cmi_pami_contexts[c];
#else
  pami_context_t my_context= MY_CONTEXT();
#endif // end of CMK_SMP && CMK_ENABLE_ASYNC_PROGRESS

  PAMI_Memregion_create (my_context,
                         (void *)ptr,
                         size,
                         &bytes_out,
                         &mregion);
#endif
}
