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
 * This is a stub file.  All the quickthreads functions just print
 * out "quickthreads not implemented."  This simplifies testing when
 * you plan on implementing the threads package but haven't gotten
 * around to it.
 *
 */

#ifndef QT_STUB_H
#define QT_STUB_H

typedef unsigned long qt_word_t;

#define QT_STKALIGN	(8)
#define QT_STKBASE      (8)
#define QT_GROW_DOWN    1

void *qt_ni(void);

#define QT_ARGS(sp,u,t,uf,on) (qt_ni())
#define QT_VARGS(sp,nb,vargs,pt,startup,vuserf,cleanup) (qt_ni())
#define QT_BLOCK(help,old,new,sp) (qt_ni())
#define QT_BLOCKI(help,old,new,sp) (qt_ni())
#define QT_ABORT(help,old,new,sp) (qt_ni())

#endif /* ndef QT_STUB_H */
