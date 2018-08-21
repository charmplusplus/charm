/*
 * Copyright (c) 2019, Mellanox Technologies. All rights reserved.
 * See LICENSE in this directory.
 */

#include <stdlib.h>
#include <string.h>

#include <pmix.h>

#define UCX_PREFIX "ucx"

static pmix_proc_t myproc;
static pmix_proc_t proc;

int runtime_init(int *rank, int *jobsize)
{
    int ret;
    pmix_value_t *val;

    if (PMIX_SUCCESS != (ret = PMIx_Init(&myproc, NULL, 0))) {
        fprintf(stderr, "NS %s rank %d: PMIx_Init failed: %d\n",
                myproc.nspace, myproc.rank, ret);
        return -1;
    }

    /* job-related info is found in our nspace, assigned to the
     * wildcard rank as it doesn't relate to a specific rank. Setup
     * a name to retrieve such values */
    PMIX_PROC_CONSTRUCT(&proc);
    (void)strncpy(proc.nspace, myproc.nspace, PMIX_MAX_NSLEN);
    proc.rank = PMIX_RANK_WILDCARD;

    if (PMIX_SUCCESS != (ret = PMIx_Get(&proc, PMIX_JOB_SIZE, NULL, 0, &val))) {
        fprintf(stderr, "Client ns %s rank %d: PMIx_Get job size failed: %d\n",
                myproc.nspace, myproc.rank, ret);
        return -2;
    }

    *jobsize = val->data.uint32;
    *rank    = myproc.rank;

    PMIX_VALUE_RELEASE(val);
    return 0;
}

int runtime_fini()
{
    int ret;

    if (PMIX_SUCCESS != (ret = PMIx_Finalize(NULL, 0))) {
        fprintf(stderr, "Client ns %s rank %d:PMIx_Finalize failed: %d\n",
                myproc.nspace, myproc.rank, ret);
        return -1;
    }

    return 0;
}

int runtime_kvs_put(int id, const void *v, int vlen)
{
    int ret;
    int keylen;
    char key[PMIX_MAX_KEYLEN + 1];
    pmix_value_t value;

    value.type          = PMIX_BYTE_OBJECT;
    value.data.bo.bytes = (char*)v;
    value.data.bo.size  = vlen;

    if (snprintf(key, PMIX_MAX_KEYLEN + 1, "%s-%s-%d",
                 UCX_PREFIX, myproc.nspace, id) < 0) {
        fprintf(stderr, "Client ns %s rank %d: snprintf failed\n",
                myproc.nspace, myproc.rank);
        return -1;
    }

    if (PMIX_SUCCESS != (ret = PMIx_Put(PMIX_GLOBAL, key, &value))) {
        fprintf(stderr, "Client ns %s rank %d: PMIx_Put local failed: %d\n",
                myproc.nspace, myproc.rank, ret);
        return -2;
    }

    if (PMIX_SUCCESS != (ret = PMIx_Commit())) {
        fprintf(stderr, "Client ns %s rank %d: PMIx_Commit failed: %d\n",
                myproc.nspace, myproc.rank, ret);
        return -3;
    }

    return 0;
}

int runtime_kvs_get(int id, void *v, int vlen)
{
    int ret;
    pmix_value_t *value;
    char key[PMIX_MAX_KEYLEN + 1];

    if (snprintf(key, PMIX_MAX_KEYLEN + 1, "%s-%s-%d",
                 UCX_PREFIX, myproc.nspace, id) < 0) {
        fprintf(stderr, "Client ns %s rank %d: snprintf failed\n",
                myproc.nspace, myproc.rank);
        return -1;
    }

    proc.rank = id;
    if (PMIX_SUCCESS != (ret = PMIx_Get(&proc, key, NULL, 0, &value))) {
        fprintf(stderr, "Client ns %s rank %d: PMIx_Get %s failed: %d\n",
                myproc.nspace, myproc.rank, key, ret);
        return -2;
    }

    if (value->type != PMIX_BYTE_OBJECT) {
        fprintf(stderr, "Client ns %s rank %d: PMIx_Get %s returned wrong type: %d\n",
                myproc.nspace, myproc.rank, key, value->type);
        PMIX_VALUE_RELEASE(value);
        return -3;
    }

    memcpy(v, value->data.bo.bytes, value->data.bo.size);

    PMIX_VALUE_RELEASE(value);

    return 0;
}

int runtime_barrier()
{
    int ret;

    if (PMIX_SUCCESS != (ret = PMIx_Fence(NULL, 0, NULL, 0))) {
        fprintf(stderr, "Client ns %s rank %d: PMIx_Fence failed: %d\n",
                myproc.nspace, myproc.rank, ret);
        return -1;
    }

    return 0;
}

int runtime_fence()
{
    bool flag = true;
    int ret;
    pmix_info_t *info;

    PMIX_INFO_CREATE(info, 1);
    PMIX_INFO_LOAD(info, PMIX_COLLECT_DATA, &flag, PMIX_BOOL);

    if (PMIX_SUCCESS != (ret = PMIx_Fence(NULL, 0, info, 1))) {
        fprintf(stderr, "Client ns %s rank %d: PMIx_Fence failed: %d\n",
                myproc.nspace, myproc.rank, ret);
        PMIX_INFO_FREE(info, 1);
        return -2;
    }

    PMIX_INFO_FREE(info, 1);
    return 0;
}
