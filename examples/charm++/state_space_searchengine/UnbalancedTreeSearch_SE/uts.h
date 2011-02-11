/*
 *         ---- The Unbalanced Tree Search (UTS) Benchmark ----
 *  
 *  Copyright (c) 2010 See AUTHORS file for copyright holders
 *
 *  This file is part of the unbalanced tree search benchmark.  This
 *  project is licensed under the MIT Open Source license.  See the LICENSE
 *  file for copyright and licensing information.
 *
 *  UTS is a collaborative project between researchers at the University of
 *  Maryland, the University of North Carolina at Chapel Hill, and the Ohio
 *  State University.  See AUTHORS file for more information.
 *
 */

#ifndef _UTS_H
#define _UTS_H

extern "C" {

#include "rng/rng.h"

#define UTS_VERSION "2.1"

/***********************************************************
 *  Tree node descriptor and statistics                    *
 ***********************************************************/

#define MAXNUMCHILDREN    100  // cap on children (BIN root is exempt)

/* Tree type
 *   Trees are generated using a Galton-Watson process, in 
 *   which the branching factor of each node is a random 
 *   variable.
 *   
 *   The random variable can follow a binomial distribution
 *   or a geometric distribution.  Hybrid tree are
 *   generated with geometric distributions near the
 *   root and binomial distributions towards the leaves.
 */
#define BIN 0
#define GEO 1
#define HYBRID 2
#define BALANCED 3
#define LINEAR 0
#define  EXPDEC 1
#define CYCLIC 2
#define FIXED 3
//enum   uts_trees_e    { BIN = 0, GEO, HYBRID, BALANCED };
//enum   uts_geoshape_e { LINEAR = 0, EXPDEC, CYCLIC, FIXED };

//typedef enum uts_trees_e    tree_t;
//typedef enum uts_geoshape_e geoshape_t;

/* Strings for the above enums */
extern const char * uts_trees_str[];
extern const char * uts_geoshapes_str[];


/* Tree  parameters */
extern int type;
extern double     b_0;
extern int        rootId;
extern int        nonLeafBF;
extern double     nonLeafProb;
extern int        gen_mx;
extern int shape_fn;
extern double     shiftDepth;         

/* Benchmark parameters */
extern int    computeGranularity;
extern int    debug;
extern int    verbose;

/* For stats generation: */
typedef unsigned long long counter_t;

/* Utility Functions */
#define __max(a,b) (((a) > (b)) ? (a) : (b))
#define __min(a,b) (((a) < (b)) ? (a) : (b))

void   uts_parseParams(int argc, char **argv);
int    uts_paramsToStr(char *strBuf, int ind);
void   uts_printParams();
void   uts_helpMessage();

double rng_toProb(int n);

/* Implementation Specific Functions */
int    impl_paramsToStr(char *strBuf, int ind);
int    impl_parseParam(char *param, char *value);
void   impl_helpMessage();

}

#endif /* _UTS_H */
