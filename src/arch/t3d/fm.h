/* Copyright (c) 1994-95
 * The University of Illinois Board of Trustees.
 *      All Rights Reserved.
 *
 * CONFIDENTIAL INFORMATION. Distribution
 * restricted under license agreement.
 *
 * Authors: Vijay Karamcheti
 * Contributing Author:
 * Project Manager and Principal Investigator:
 *      Andrew A. Chien (achien@cs.uiuc.edu)
 *
 * -------------------------------------------
 *  Header file for FM library on Cray T3D.
 * -------------------------------------------
 *
 *	VK	Tue Feb 21 11:35:18 EST 1995
 *		separated out FMf_initialize and FMs_initialize
 *		provided FMf_barrier and FMs_barrier
 *
 *	VK	Mon Apr  3 14:34:56 EDT 1995
 *		provided FMf_nested_extract and FMf_nested_barrier
 *
 *	VK	Mon Jul  3 15:36:13 EDT 1995
 *		provided FMs_reclaim_mbufs
 */


#ifndef FM_H
#define FM_H

/* macros defining parameter ids */

#define MAX_MSG_SIZE_FINC	1
#define MSG_BUFFER_SIZE_FINC	2


/* initialization and parameter setup */

void FM_initialize(void);
void FMf_initialize(void);
void FMs_initialize(void);
void FM_set_parameter(int param_id, int value);

typedef void FM_shandler();
typedef void FM_lhandler(void *, int);


/* fetch-and-increment primitives */

void FMf_send_4(int node, FM_shandler *fptr, ...);
void FMf_send_4i(int node, FM_shandler **fptr, ...);
void FMf_send(int node, FM_lhandler *fptr, void *buf, int byte_count);
int FMf_extract(void);
int FMf_nested_extract(void);
int FMf_extract_1(void);	
void FMf_barrier(int);
void FMf_nested_barrier(int);


/* atomic-swap primitives */

void FMs_send_4(int node, FM_shandler *fptr, ...);
void FMs_send_4i(int node, FM_shandler **fptr, ...);
void FMs_send(int node, FM_lhandler *fptr, void *buf, int byte_count);
int FMs_extract(void);
int FMs_extract_1(void);
void FMs_complete_send(void);
void FMs_barrier(int);
int FMs_reclaim_mbufs(void);


#endif	/* FM_H */
