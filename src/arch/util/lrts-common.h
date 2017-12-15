/* This header is included in conv-config.h, which is included in the
 * machine layer implementations through converse.h
 */

// Use CMA for intra node shared memory communication on all machines where it is supported, except for Multicore
#define CMK_USE_CMA                    (CMK_HAS_CMA && !CMK_MULTICORE)
