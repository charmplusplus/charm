/*
 * Copyright (c) 2017, Intel Corporation. All rights reserved.
 * See LICENSE in this src/arch/ofi/.
 *
 * Runtime functions used by OFI LRTS machine layer to exchange
 * addresses during the initialization.
 *
 * The idea is that there could be multiple ways of implementing
 * these functions. The example provided in runtime-pmi.C uses PMI.
 */
int runtime_init(int *rank, int *jobsize);
int runtime_fini();

int runtime_get_max_keylen(int *len);
int runtime_kvs_put(const char *k, const void *v, int vlen);
int runtime_kvs_get(const char *k, void *v, int vlen);
int runtime_barrier();
