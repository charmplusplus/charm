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

/*
* Unbalanced Tree Search 
*
* alfg.c
*   Additive Lagged Fibonacci Generator
*    - splittable pseudorandom number generator for UTS
*
*/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "alfg.h"


void rng_init(RNG_state *mystate, int seed)
{
  /************************************************/
  /* state_size < 0: this is the special case     */
  /* where we initialize the root sequence state. */
  /* (every subsequent state uses information     */
  /* copied from its parent state.)               */
  /************************************************/
  int i,j,l,k,cbit,runup,state_size,tmp[861];
  int *reg, *node;

  state_size=2*UTS_ALFG_L-1+N_SCALARS;
  l=UTS_ALFG_L;
	
  k = 0;
  if(l ==    7) { k =   4 ; cbit =   1 ; runup = k*71;   }
  if(l ==   17) { k =  12 ; cbit =   5 ; runup = k*65;   }
  if(l ==   55) { k =  31 ; cbit =  18 ; runup = k*93;   }
  if(l ==  159) { k = 128 ; cbit =  63 ; runup = k*179;  }
  if(l ==  607) { k = 334 ; cbit = 166 ; runup = k*628;  }
  if(l == 1279) { k = 861 ; cbit = 233 ; runup = k*1291; }
  if(!k) { printf("\n\n    invalid UTS_ALFG_L value: %d\n\n",l); exit(0); }
	
  mystate[J_STATE_SIZE] = state_size;
  mystate[J_L         ] = l;
  mystate[J_SEED      ] = seed;
	
  mystate[J_K         ] = k;
  mystate[J_CBIT      ] = cbit;
  mystate[J_RUNUP     ] = runup;
	
  mystate[J_LP        ] = l-1;
  mystate[J_KP        ] = k-1;
  mystate[J_ZP        ] = 0;

  node       = &(mystate[NODE0]);
  reg        = &(mystate[REG0 ]);

	
  for(i=0; i<l-1; i++) node[i] = 0; 
  node[0]    = 2;

  for(i=0; i<l  ; i++) reg [i] = 0; 
  reg[cbit] += 1;
	
	/********************************************/
	/* adjust initial value of seed up one bit, */
	/* and xor into reg[0]:                     */
	/********************************************/

  mystate[J_SEED] = ((mystate[J_SEED] << 1) & POS_MASK);
  reg[0]          = 2^mystate[J_SEED];
	
  k = mystate[J_K];

  for(j=0; j<mystate[J_RUNUP]; j+=k)
    {
      for(i=0; i<k; i++) tmp[i]  = reg[i+l-k];
      for(i=k; i<l; i++) reg[i]  = reg[i-k];
      for(i=0; i<k; i++) reg[i] += tmp[i];
    }
}

void rng_spawn(RNG_state *mystate, RNG_state *newstate, int spawnNumber)
{
	
  /**********************************************************************/
  /* This routine initializes the state associated with a new           */
  /* child sequence, newstate, using the state of the parent            */
  /* sequence, mystate.                                                 */
  /*                                                                    */
  /* The state of a sequence consists of an array of type int (which    */
  /* can be passed as an MPI message) containing an L-long subarray,    */
  /* reg, an (L-1)-long subarray, node, and a handful of scalar         */
  /* quantities. Together the data in the state array is enough to be a */
  /* self-contained unit of work that can be pushed onto a stack,       */
  /* passed off to another thread, etc.                                 */
  /*                                                                    */
  /* There is a node on the tree for every independent cycle of the     */
  /* fibonacci shift register --- enough nodes that it takes (L-1)      */
  /* type int locations to uniquely specify a node index.               */
  /* The generation number of each sequence is not a function of        */
  /* its beginning node number, but is simply one more than the         */
  /* generation number of its parent sequence. The root sequence is     */
  /* in generation zero. The ''high threshold'' of a sequence depends   */
  /* only on its generation number and the high threshold of the root   */
  /* sequence.                                                          */
  /*                                                                    */
  /* When a parent state spawns a child state, the parent moves to a    */
  /* new node on the tree, so that there cannot be any revisiting of    */
  /* nodes. If the parent is at node m, then it will spawn a child at   */
  /* node 2m and move itself to node 2m+1. When it moves to a new node, */
  /* it does not change its current shift register state (reg).  The    */
  /* shift register contents of the child state will start out as a     */
  /* shifted copy of its node number, added to by the ''canonical bit'' */
  /* (cbit) associated with its L and K values. The seed is XOR-ed      */
  /* into the position 0 of the initial register state,  giving 2^31    */
  /* variations on each run of fixed L and K.                           */
  /*                                                                    */
  /* Several of the scalar state variables will be adjusted in the      */
  /* child state from their values in the parent state. Finally, the    */
  /* fibonacci rgesiter, reg, will be stepped enough times to get       */
  /* past the initial correlation that one sees when any two of these   */
  /* generators are seeded with nearly identical states. (The runup     */
  /* and cbit values have been determined elsewhere.)                   */
  /*                                                                    */
  /**********************************************************************/
  int i,j,l,k,zp,state_size,tmp[861];
  int *reg, *node;
	
  state_size = mystate[J_STATE_SIZE];
  l          = mystate[J_L];
  node       = &(mystate[NODE0]);
  reg        = &(mystate[REG0 ]);

  /*************************************************/
  /* first, multiply node number by 2 in MYstate:  */
  /* check for overflow:                           */
  /*************************************************/
  zp = mystate[J_ZP];
  zp = zp-1; if(zp < 0) zp += (l-1);
  if(node[zp] & HIGH_BIT)
    {
      printf("\n\nALFG:  node limit exceeded\n\n");
      exit(0);
    }
  node[zp] <<= 1;
  mystate[J_ZP] = zp;
	
  /***********************************************************/
  /* allocate new state and copy all but reg from old state: */
  /***********************************************************/
  for(i=0; i<state_size-l; i++) newstate[i] = mystate[i];
	
  /*****************************/
  /* add 1 to MY node number:  */
  /* i.e. in next-to-least bit */
  /* increment MY child count: */
  /*****************************/
  node[zp] += 2;
  //mystate[J_NCHILDREN]++;
	
  /*****************************************************/
  /* now, make necessary changes to newstate:          */
  /* first fill reg with node number, reg[l-1] with 0; */
  /* set cbit; adjust reg[0] by xor-ing with seed;     */
  /* adjust my_gen, my_th_h, lp, kp, child count:      */
  /*****************************************************/
  node = &(newstate[NODE0]);
  reg  = &(newstate[REG0 ]);
  j    = zp;

  for(i=0; i<l-1; i++) { reg[i] = node[j]; if(++j == (l-1)) j = 0; }
	
  reg[newstate[J_CBIT]] += 1;
  reg[0] ^= newstate[J_SEED];
  reg[l-1] = 0;
	
  newstate[J_LP       ] = newstate[J_L] - 1;
  newstate[J_KP       ] = newstate[J_K] - 1;
	
  k = newstate[J_K];

  for(j=0; j<newstate[J_RUNUP]; j+=k)
    {
      for(i=0; i<k; i++) tmp[i]  = reg[i+l-k];
      for(i=k; i<l; i++) reg[i]  = reg[i-k];
      for(i=0; i<k; i++) reg[i] += tmp[i];
    }
	
}

/* extract random value from current state of ALFG
 * do not advance state 
 */
int rng_rand(RNG_state *mystate)
{
  int n,l,lp,*reg;
			
  l       = mystate[J_L      ];			
  lp      = mystate[J_LP     ];
  reg     = &(mystate[REG0]);
  n = (reg[lp] >> 1) & POS_MASK;

  return n;
}

/* advance ALFG and extract random value */
int rng_nextrand(RNG_state *mystate)
{
  int n,l,lp,kp,*reg;
			
  l       = mystate[J_L      ];
  lp      = mystate[J_LP     ];
  kp      = mystate[J_KP     ];
  reg     = &(mystate[REG0]);
	
  reg[lp] +=  reg[kp];
  n = (reg[lp] >> 1) & POS_MASK;
  lp--; if(lp < 0) lp = l-1;
  kp--; if(kp < 0) kp = l-1;
	
  mystate[J_LP   ]  = lp;
  mystate[J_KP   ]  = kp;
	
  return n;
}

/* condense state into string to display in debugging */
char * rng_showstate(RNG_state *mystate, char *s){
  int n,l,lp,*reg;

  l       = mystate[J_L      ];
  lp      = mystate[J_LP     ];
  reg     = &(mystate[REG0]);
  n = (reg[lp] >> 1) & POS_MASK;

  sprintf(s,"%.8X...", n);
  return s;
}

/* describe random number generator type into string */
int rng_showtype(char *strBuf, int ind) {
  ind += sprintf(strBuf+ind, "ALFG (state size = %dB, L = %d)",
                 sizeof(struct state_t), UTS_ALFG_L);
  return ind;
}






