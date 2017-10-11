/*
 * Copyright (c) 2017, Intel Corporation. All rights reserved.
 * See LICENSE in this directory.
 */

#ifndef REQUEST_H_
#define REQUEST_H_

#include <rdma/fabric.h>
#include <rdma/fi_eq.h>

#include "converse.h"

#define sizeofa(x)         (sizeof(x) / sizeof(*(x)))
#define REQUEST_CACHE_SIZE 32768

#define ZERO_REQUEST(r)            \
    do                             \
    {                              \
        (r)->mr               = 0; \
        (r)->callback         = 0; \
        (r)->destNode        = -1; \
        (r)->destPE          = -1; \
        (r)->size             = 0; \
        (r)->mode             = 0; \
        (r)->data.recv_buffer = 0; \
    } while (0)

typedef enum request_state
{
    rs_none     = 0, /* message object/slot is empty */
    rs_progress = 1  /* slot is in progress */
} request_state;

struct OFIRmaHeader;
struct OFIRmaAck;
struct OFILongMsg;

/**
 * OFI Request
 * Structure representing data movement operations.
 *  - context: fi_context
 *  - state: writing rs_progress value to this value must be done atomically
 *  - index: request index
 *  - callback: Request callback called upon completion
 *  - destNode: destination NodeNo
 *  - destPE: destination PE
 *  - size: message size
 *  - mode: sending mode (unused at this time)
 *  - mr: memory region associated with RMA buffer
 *  - data: Pointer to data associated with the request
 *      - recv_buffer: used when posting a receive buffer
 *      - rma_header: used when an OFIRmaHeader was received or sent
 *      - rma_ack: used when an OFIRmaAck was received
 *      - long_msg: used when an RMA Read operation completed
 *      - short_msg: used when a short message was sent
 *      - rma_ncpy_info: used when an RMA Read operation completed through the Nocopy API
 *      - rma_ncpy_ack: used when an OFIRmaAck was received through the Nocopy API
 */
typedef struct OFIRequest
{
    struct fi_context  context;
    request_state      state;
    int                index;
    void               (*callback)(struct fi_cq_tagged_entry*, struct OFIRequest*);
    int                destNode;
    int                destPE;
    int                size;
    int                mode;
    struct fid_mr      *mr;
    union
    {
        void                *recv_buffer;
        struct OFIRmaHeader *rma_header;
        struct OFIRmaAck    *rma_ack;
        struct OFILongMsg   *long_msg;
        void                *short_msg;
#if CMK_ONESIDED_IMPL
        void                *rma_ncpy_info;
        void                *rma_ncpy_ack;
#endif
    } data;
} OFIRequest;

#if USE_OFIREQUEST_CACHE
typedef struct request_cache_t
{
    /** Array of OFIRequest */
    OFIRequest request[REQUEST_CACHE_SIZE];

    /**
     * Index of first request in current cache. This is used to find the
     * request by index.
     */
    int index;

    /** Pointer to next cache */
    struct request_cache_t* next;
} request_cache_t;

static inline request_cache_t* create_request_cache()
{
    request_cache_t* cache = malloc(sizeof(*cache));
    if (cache)
    {
        cache->next = 0;
        cache->index = 0;
        int i;
        for (i = 0; i < sizeofa(cache->request); i++)
            cache->request[i].state = rs_none;
        return cache;
    }
    else
        return 0;
}

static inline void destroy_request_cache(request_cache_t* cache)
{
    while (cache)
    {
        struct request_cache_t* c = cache->next;
        free(cache);
        cache = c;
    }
}

static inline void free_request(OFIRequest* req)
{
    CmiAssert(req);
    CmiAssert(req->state == rs_progress);
    if (req)
    {
        ZERO_REQUEST(req);
        __atomic_store_n(&req->state, rs_none, __ATOMIC_RELEASE);
    }
}

/**
 * Find/create empty request slot. This function is thread safe thanks to the
 * use of atomic primitives to locate/update entries.
 */
static inline OFIRequest* alloc_request(request_cache_t* req_cache)
{
    CmiAssert(req_cache);
    int i;
    OFIRequest* request = 0;
    /* try to find free request (state == rs_none) */
    struct request_cache_t* cache = req_cache;
    while (cache)
    {
        for (i = 0; i < sizeofa(cache->request); i++)
        {
            if (__sync_bool_compare_and_swap(&cache->request[i].state, rs_none, rs_progress))
            {
                /* Found one entry */
                cache->request[i].index = cache->index + i;
                ZERO_REQUEST(&cache->request[i]);
                request = cache->request + i;
                goto fn_exit;
            }
        }
        /* no entries in current cache element, try next one... */
        cache = cache->next;
    }

    /* still no free entries; create new cache entry */
    cache = create_request_cache();
    CmiAssert(cache);
    if (cache) {
        /* use first request entry in the new cache */
        cache->request[0].state = rs_progress;
        /* append new cache entry to list */
        struct request_cache_t* c = req_cache;
        /* here is the trick: __sync_val_compare_and_swap updates c->next only if
         * it was 0. otherwise, keep iterating. */
        for (i = 0; c; i++)
        {
            /* set index value to 'current' count */
            cache->index = (i + 1) * sizeofa(c->request);
            c = __sync_val_compare_and_swap(&c->next, 0, cache);
        }
        cache->request->index = cache->index;
        ZERO_REQUEST(cache->request);
        request = cache->request;
        goto fn_exit;
    }

fn_exit:
    return request;
}

/**
 * Lookup request by index: list all cache arrays till index is inside the
 * current one.
 */
static inline OFIRequest* lookup_request(request_cache_t* cache, int index)
{
    while (cache)
    {
        if (index >= cache->index && index < cache->index + sizeofa(cache->request))
            return cache->request + (index - cache->index);
        cache = cache->next;
    }
    return 0;
}
#endif /* USE_OFIREQUEST_CACHE */

#endif /* REQUEST_H_ */
