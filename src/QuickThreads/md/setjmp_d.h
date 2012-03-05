/*
 * QuickThreads -- Threads-building toolkit.
 * Copyright (c) 1993 by David Keppel
 *
 * Permission to use, copy, modify and distribute this software and
 * its documentation for any purpose and without fee is hereby
 * granted, provided that the above copyright notice and this notice
 * appear in all copies.  This software is provided as a
 * proof-of-concept and for demonstration purposes; there is no
 * representation about the suitability of this software for any
 * purpose.
 *
 * Must define: QT_STKALIGN, QT_ARGS, QT_VARGS, QT_BLOCK, QT_BLOCKI, QT_ABORT
 *
 */

#ifndef QT_GENERIC1_H
#define QT_GENERIC1_H

typedef unsigned long qt_word_t;

#define QT_GROW_DOWN
#define QT_STKALIGN	(8)   /* real alignment is done elsewhere */
#define QT_STKBASE	(8)   /* this is a dummy constant */

typedef struct qt_t qt_t1;
 
typedef void *(qt_userf_t1)(void *pu);
typedef void (qt_only_t1)(void *pu, void *pt, qt_userf_t1 *userf);
typedef void *(qt_helper_t1)(qt_t1 *old, void *a0, void *a1);
 
qt_t1 *qt_args(qt_t1 *,void *,void *,qt_userf_t1 *,qt_only_t1*);
void *qt_block(qt_helper_t1 *, void *, void *, qt_t1 *);
void *qt_abort(qt_helper_t1 *, void *, void *, qt_t1 *);
 
#define QT_ARGS(sp,u,t,uf,on) (qt_args(sp,u,t,(qt_userf_t1*)(uf),(qt_only_t1*)(on)))
#define QT_VARGS(sp,nb,vargs,pt,startup,vuserf,cleanup) (qt_error(),(void*)0)
#define QT_BLOCK(help,old,new,sp) (qt_block(help,old,new,sp))
#define QT_BLOCKI(help,old,new,sp) (qt_block(help,old,new,sp))
#define QT_ABORT(help,old,new,sp) (qt_abort(help,old,new,sp))

#endif /* ndef QT_GENERIC1_H */
