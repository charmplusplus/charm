/*
 * Copyright (c) 2017, Intel Corporation. All rights reserved.
 * See LICENSE in this directory.
 *
 * Runtime functions used by OFI LRTS machine layer to exchange
 * addresses during the initialization.
 *
 * This example uses the PMI API as described in pmi.h.
 */
#include <stdlib.h>
#include <string.h>

#include <pmi.h>

#include "runtime.h"

#if CMK_OFI_USE_SIMPLEPMI
#include "simple_pmi.C"
#include "simple_pmiutil.C"
#endif

/* For encode/decode functions */
#include "runtime-codec.h"

static int initialized;
static int max_keylen;
static int max_valuelen;
static char *kvsname;
static char *key;
static char *value;

int runtime_init(int *rank, int *jobsize)
{
    int ret;
    int first_spawned;
    int max_kvsnamelen;

    ret = PMI_Init(&first_spawned);
    if (PMI_SUCCESS != ret) {
        return 1;
    }

    ret = PMI_Get_size(jobsize);
    if (PMI_SUCCESS != ret) {
        return 2;
    }

    ret = PMI_Get_rank(rank);
    if (PMI_SUCCESS != ret) {
        return 3;
    }

    ret = PMI_KVS_Get_name_length_max(&max_kvsnamelen);
    if (PMI_SUCCESS != ret) {
        return 4;
    }

    kvsname = (char *)calloc(max_kvsnamelen, sizeof(char));
    if (!kvsname) {
        return 5;
    }

    ret = PMI_KVS_Get_my_name(kvsname, max_kvsnamelen);
    if (PMI_SUCCESS != ret) {
        free(kvsname);
        return 6;
    }

    ret = PMI_KVS_Get_key_length_max(&max_keylen);
    if (PMI_SUCCESS != ret) {
        free(kvsname);
        return 7;
    }

    key = (char *)calloc(max_keylen, sizeof(char));
    if (!key) {
        free(kvsname);
        return 8;
    }

    ret = PMI_KVS_Get_value_length_max(&max_valuelen);
    if (PMI_SUCCESS != ret) {
        free(key);
        free(kvsname);
        return 9;
    }

    value = (char *)calloc(max_valuelen, sizeof(char));
    if (!value) {
        free(key);
        free(kvsname);
        return 10;
    }

    initialized = 1;
    return 0;
}

int runtime_fini()
{
    int ret;

    if (initialized) {
        ret = PMI_Finalize();
        if (PMI_SUCCESS != ret) {
            return 1;
        }
    }

    if (value) {
        free(value);
        value = NULL;
    }
    if (key) {
        free(key);
        key = NULL;
    }
    if (kvsname) {
        free(kvsname);
        kvsname = NULL;
    }

    initialized = 0;
    return 0;
}

int runtime_get_max_keylen(int *len)
{
    if (!initialized) {
        return 1;
    }
    *len = max_keylen;
    return 0;
}

int runtime_kvs_put(const char *k, const void *v, int vlen)
{
    int ret;
    int keylen;

    if (!initialized) {
        return 1;
    }

    keylen = strlen(k);
    if (keylen > max_keylen) {
        return 2;
    }

    if (vlen > max_valuelen) {
        return 3;
    }

    ret = encode(v, vlen, value, max_valuelen);
    if (ret) {
        return 4;
    }

    ret = PMI_KVS_Put(kvsname, k, value);
    if (ret) {
        return 5;
    }

    ret = PMI_KVS_Commit(kvsname);
    if (ret) {
        return 6;
    }

    return 0;
}

int runtime_kvs_get(const char *k, void *v, int vlen)
{
    int ret;

    if (!initialized) {
        return 1;
    }

    ret = PMI_KVS_Get(kvsname, k, value, max_valuelen);
    if (ret) {
        return 2;
    }

    ret = decode(value, v, vlen);
    if (ret) {
        return 3;
    }

    return 0;
}

int runtime_barrier()
{
    int ret;

    if (!initialized) {
        return 1;
    }

    ret = PMI_Barrier();
    if (ret) {
        return 2;
    }
    return 0;
}
