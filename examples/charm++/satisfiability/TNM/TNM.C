/* 
Copyright (C) 2009
Wanxia WEI (weiwanxia@gmail.com), 
Chumin LI (chu-min.li@u-picardie.fr)

   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 2 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program; if not, write to the Free Software
   Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
*/

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>

#include <sys/times.h>
#include <sys/types.h>
#include <limits.h>

#include "TNM.h"

#include <vector>

using namespace std;

typedef signed int my_type;
typedef unsigned int my_unsigned_type;

//#define lookahead_length 30
//int decreasing_vars_stack1[lookahead_length+1];
#define WORD_LENGTH 100 
#define TRUE 1
#define FALSE 0
#define NONE -1
#define SATISFIABLE 2
#define walk_satisfiable() (MY_CLAUSE_STACK_fill_pointer == 0)

#define WEIGTH 4
#define T 10

/* the tables of variables and clauses are statically allocated. Modify the 
   parameters tab_variable_size and tab_clause_size before compilation if 
   necessary */

#define tab_variable_size  100000
#define tab_clause_size 500000

#define tab_unitclause_size \
 ((tab_clause_size/4<2000) ? 2000 : tab_clause_size/4)
#define my_tab_variable_size \
 ((tab_variable_size/2<1000) ? 1000 : tab_variable_size/2)
#define my_tab_clause_size \
 ((tab_clause_size/2<2000) ? 2000 : tab_clause_size/2)
#define my_tab_unitclause_size \
 ((tab_unitclause_size/2<1000) ? 1000 : tab_unitclause_size/2)
#define tab_literal_size 2*tab_variable_size
#define double_tab_clause_size 2*tab_clause_size
#define positive(literal) literal<NB_VAR
#define negative(literal) literal>=NB_VAR
#define get_var_from_lit(negative_literal) negative_literal-NB_VAR
#define RESOLVANT_LENGTH 3
#define RESOLVANT_SEARCH_THRESHOLD 5000
#define complement(lit1, lit2) \
 ((lit1<lit2) ? lit2-lit1 == NB_VAR : lit1-lit2 == NB_VAR)

#define inverse_signe(signe) \
 (signe == POSITIVE) ? NEGATIVE : POSITIVE
#define unsat(val) (val==0)?"UNS":"SAT"
#define pop(stack) stack[--stack ## _fill_pointer]
#define push(item, stack) stack[stack ## _fill_pointer++] = item
#define satisfiable() CLAUSE_STACK_fill_pointer == NB_CLAUSE

#define NEGATIVE 0
#define POSITIVE 1
#define PASSIVE 0
#define ACTIVE 1
#define INUTILE 2
#define MAX_NODE_NUMBER 6000
struct node {
  int clause;
  struct node *next;
};

int *neg_in[tab_variable_size];
int *pos_in[tab_variable_size];
int neg_nb[tab_variable_size];
int pos_nb[tab_variable_size];
my_type *var_current_value;//[tab_variable_size];
my_type *var_rest_value;//[tab_variable_size];
my_type *var_state;//[tab_variable_size];


float *reduce_if_negative_nb;//[tab_variable_size];
float *reduce_if_positive_nb;//[tab_variable_size];

int *sat[tab_clause_size];
int *var_sign[tab_clause_size];
my_type *clause_state;//[tab_clause_size];
my_type *clause_length;//[tab_clause_size];
int *most_recent;//[tab_clause_size];
int *most_recent_count;//[tab_clause_size];

int VARIABLE_STACK_fill_pointer = 0;
int CLAUSE_STACK_fill_pointer = 0;
int UNITCLAUSE_STACK_fill_pointer = 0;
int MANAGEDCLAUSE_STACK_fill_pointer = 0;


int *VARIABLE_STACK;//[tab_variable_size];
int *CLAUSE_STACK;//[tab_clause_size];
int *UNITCLAUSE_STACK;//[tab_unitclause_size];
int *MANAGEDCLAUSE_STACK;//[tab_clause_size];
//int TESTED_VAR_STACK[tab_variable_size];
//int TESTED_VAR_STACK_fill_pointer=0;


int tab_variable_nb_unit[2*tab_variable_size]={0};
int clause_possible[tab_clause_size]={0};
//int SAVED_VAR_VARIABLE_STACK[tab_variable_size];
int SAVED_VAR_VARIABLE_STACK_fill_pointer = 0;
//int SAVED_DEB_CLAUSE_STACK[tab_variable_size];
int SAVED_DEB_CLAUSE_STACK_fill_pointer;
//int SAVED_FIN_CLAUSE_STACK[tab_variable_size];
int SAVED_FIN_CLAUSE_STACK_fill_pointer;
//int SAVED_VARIABLE_STACK[tab_variable_size];
int SAVED_VARIABLE_STACK_fill_pointer = 0;
//int SAVED_MANAGEDCLAUSE_STACK[tab_clause_size];
int SAVED_MANAGEDCLAUSE_STACK_fill_pointer = 0;
//int SAVED_CLAUSE_STACK[tab_clause_size];
int SAVED_CLAUSE_STACK_fill_pointer = 0;
int PREVIOUS_MANAGEDCLAUSE_STACK_fill_pointer = 0;

int CPT=0;

int NB_VAR;
int NB_ACTIVE_VAR;
int NB_CLAUSE;
int INIT_NB_CLAUSE;
my_type R = 3;

double *counter;//[tab_variable_size];
double max_counter, min_counter, ave_counter, old_ave_counter;
int global_j;

double Interval, Adjustment, Sigma;
double coefficient_for_prm;
int Intensity;

long NB_UNIT=1, NB_MONO=0, NB_BRANCHE=0, NB_BACK = 0;

#define double_tab_clause_size 2*tab_clause_size

struct var_node {
  int var;
  int weight;
  struct var_node *next;
};

#define VAR_NODES1_nb 6
int VAR_NODES1_index=0;
struct var_node VAR_NODES1[10*VAR_NODES1_nb];
struct var_node *VAR_FOR_TEST1=NULL;

struct var_node *allocate_var_node1() {
  return &VAR_NODES1[VAR_NODES1_index++];
}

int *test_flag;//[tab_variable_size];

int MAX_REDUCED;
int T_SEUIL;

unsigned int SEED;
int SEED_FLAG=FALSE, BUILD_FLAG=TRUE;
char saved_input_file[WORD_LENGTH];
char *INPUT_FILE;

unsigned long IMPLIED_LIT_FLAG=0;
int IMPLIED_LIT_STACK_fill_pointer=0;
int *IMPLIED_LIT_STACK;//[tab_variable_size];
unsigned long LIT_IMPLIED[tab_variable_size]={0};

long NB_SECOND_SEARCH=0;
long NB_SECOND_FIXED = 0;

void remove_clauses(int var) {
  register int clause;
  register int *clauses;
  if (var_current_value[var] == POSITIVE) clauses = pos_in[var];
  else clauses = neg_in[var];
  for(clause=*clauses; clause!=NONE; clause=*(++clauses)) {
    if (clause_state[clause] == ACTIVE) {
      clause_state[clause] = PASSIVE;
      push(clause, CLAUSE_STACK);
    }
  }
}

int manage_clauses(int var) {
  register int clause;
  register int *clauses;
  if (var_current_value[var] == POSITIVE) clauses = neg_in[var];
  else clauses = pos_in[var];
  for(clause=*clauses; clause!=NONE; clause=*(++clauses)) {
    if (clause_state[clause] == ACTIVE) {
      switch (clause_length[clause]) {
      case 1: return FALSE;
      case 2: push(clause, UNITCLAUSE_STACK);
	push(clause, MANAGEDCLAUSE_STACK);
	clause_length[clause]--; break;
      default: clause_possible[clause] = 1;
	clause_length[clause]--;
	push(clause, MANAGEDCLAUSE_STACK);
      }
    }
  }
  return TRUE;
}
void print_values(int nb_var) {
  FILE* fp_out;
  int i;
  fp_out = fopen("satx.sol", "w");
  for (i=0; i<nb_var; i++) {
    if (var_current_value[i] == 1) 
      fprintf(fp_out, "%d ", i+1);
    else
      fprintf(fp_out, "%d ", 0-i-1);
  }
  fprintf(fp_out, "\n");
  fclose(fp_out);			
} 
#include "inputbis.C"

int verify_solution() {
  int i, var, *vars_signs, clause_truth,cpt;

  for (i=0; i<NB_CLAUSE; i++) {
    clause_truth = FALSE;
    vars_signs = var_sign[i];
    for(var=*vars_signs; var!=NONE; var=*(vars_signs+=2))
      if (*(vars_signs+1) == var_current_value[var] ) {
	clause_truth = TRUE;
	break;
      }
    if (clause_truth == FALSE) return FALSE;
  }
  return TRUE;
}

long NB_SEARCH = 0; long NB_FIXED = 0;
  
int unitclause_process() {
  int  i, unitclause, var, *vars_signs, unitclause_position,cpt;
  
  for (unitclause_position = 0; 
       unitclause_position < UNITCLAUSE_STACK_fill_pointer;
       unitclause_position++) {
    unitclause = UNITCLAUSE_STACK[unitclause_position];
    if (clause_state[unitclause] == ACTIVE) {
      NB_UNIT++;
      vars_signs = var_sign[unitclause];
      for(var=*vars_signs; var!=NONE; var=*(vars_signs+=2)) {
	if (var_state[var] == ACTIVE ){
	  var_current_value[var] = *(vars_signs+1);
	  var_rest_value[var] = NONE;
	  if (manage_clauses(var)==TRUE) {
	    var_state[var] = PASSIVE;
	    push(var, VARIABLE_STACK);
	    remove_clauses(var);
	    break;
	  }
	  else {
	    return NONE;
	  }
	}
      }
    }     
  }
  UNITCLAUSE_STACK_fill_pointer = 0;
  return TRUE;
}
int PREVIOUS_UNITCLAUSE_STACK_fill_pointer = 0;

/*
my_type build(int build_flag, char* input_file) {
  if (build_flag==TRUE)
    return build_simple_sat_instance(input_file);
  else return build_simple_sat_instance(input_file);
}
*/


#define DECREASING 1
#define INCREASING 2
#define PLATEAU 0
int var_count[tab_variable_size]={1};
int *neibor_stack;//[tab_variable_size];
int neibor_stack_fill_pointer=0;
int **neibor_relations[tab_variable_size];
int *neibor[tab_variable_size];
int *score;//[tab_variable_size];
int *make;//[tab_variable_size];
int *break_value;//[tab_variable_size];
int *tmp_score;//[tab_variable_size];
int *decreasing_vars_stack;//[tab_variable_size];
int decreasing_vars_stack_fill_pointer=0;
int tabu[tab_variable_size]={FALSE};
int *tabu_stack;//[tab_variable_size];
int tabu_stack_fill_pointer=0;
int *tabu_list;//[tab_variable_size];
int *tendance;//[tab_variable_size];
int MY_CLAUSE_STACK_fill_pointer=0;
int *MY_CLAUSE_STACK;//[tab_clause_size];
int *nb_lit_true;//[tab_clause_size];
int *clause_truth;//[tab_clause_size];
int *dommage_if_flip;//[tab_variable_size];
int *zerodommage;//[tab_variable_size];
int *zerodommage_vars_stack;//[tab_variable_size];
int zerodommage_vars_stack_fill_pointer=0;
int *flip_time;//[tab_variable_size];
int *enter_stack_time;//[tab_variable_size];
int *walk_time;//[tab_variable_size];
int MAXTRIES=10000;
int MAXSTEPS=2000000000;
int NOISE=50;
int NOISE1=50;
int LNOISE=5;
int saved_var_current_value[tab_variable_size];

int index_in_MY_CLAUSE_STACK[tab_clause_size];

void clause_value() {
  int clause, var, *vars_signs, nb_true;
   
  MY_CLAUSE_STACK_fill_pointer=0;  
  for (clause=0; clause<NB_CLAUSE; clause++) {
    nb_true=0;
    vars_signs = var_sign[clause];
    for(var=*vars_signs; var!=NONE; var=*(vars_signs+=2))
      if (var_current_value[var]==*(vars_signs+1))  
	nb_true++;        
    nb_lit_true[clause]=nb_true;
    if (nb_true ==0 ) {
      clause_truth[clause]=FALSE;
      //push(clause, MY_CLAUSE_STACK);
 
      index_in_MY_CLAUSE_STACK[clause]=MY_CLAUSE_STACK_fill_pointer;
      push(clause, MY_CLAUSE_STACK);
    }
    else
      clause_truth[clause]=TRUE;    
  } 
}

void pass(int the_var, int *clauses) {
  int clause, var, *vars_signs;
  for(clause=*clauses; clause!=NONE; clause=*(++clauses)) {
    if (clause_state[clause] == ACTIVE) {
      vars_signs = var_sign[clause];
      for(var=*vars_signs; var!=NONE; var=*(vars_signs+=2)) {
	if (var_state[var]==ACTIVE && var != the_var) {
	  if (var_count[var]==1) push(var, neibor_stack);
	  // nb of cells for other variables (than var and the_var: length-2)
	  //and a sign for dxdy and a separator from the next clauses
	  var_count[var]+=2*(clause_length[clause]-1);
	}
      }
    }
  }
}

int *build_neibor_relation(int var, int neibor_var, 
			  int *neibor_relations, int *clauses) {
  int i, clause, var1, *vars_signs, present, *relations, sign1, sign2;

  for(clause=*clauses; clause!=NONE; clause=*(++clauses)) {
    if (clause_state[clause] == ACTIVE) {
      vars_signs = var_sign[clause];
      present=FALSE;
      for(var1=*vars_signs; var1!=NONE; var1=*(vars_signs+=2)) {
	if (var1==neibor_var) {present=TRUE; break;}
      }
      if (present==TRUE) {
	relations=neibor_relations; neibor_relations++;
	vars_signs = var_sign[clause];
	for(var1=*vars_signs; var1!=NONE; var1=*(vars_signs+=2)) {
	  if (var_state[var1]==ACTIVE) {
	    if (var1==var) sign1=*(vars_signs+1);
	    else if (var1==neibor_var) sign2=*(vars_signs+1);
	    else {
	      *neibor_relations=var1; 
	      *(neibor_relations+1)=*(vars_signs+1);
	      neibor_relations+=2;
	    }
	  }
	}
	*relations=(sign1==sign2);
	*neibor_relations=NONE;
	neibor_relations++;
      }
    }
  }
  return neibor_relations;
}
      

void preprocess() {
  int var, neibor_var, i, *vector;

  for (var=0; var<NB_VAR; var++) {
    var_count[var]=1;
    tabu[var]=FALSE;
    tabu_list[var]=FALSE;
  }

  for (var=0; var<NB_VAR; var++) {
    for(i=0; i<neibor_stack_fill_pointer; i++) var_count[neibor_stack[i]]=1;
    neibor_stack_fill_pointer=0;
    pass(var, neg_in[var]);
    pass(var, pos_in[var]);
    vector=(int *)malloc((neibor_stack_fill_pointer+1)*sizeof(int));
    // vector_relations=(int **)malloc((neibor_stack_fill_pointer+1)*sizeof(int));
    for(i=0; i<neibor_stack_fill_pointer; i++) {
      neibor_var=neibor_stack[i];
      vector[i]=neibor_var;
      // neibor_var occupies var_count cells for all clauses
      // in which var and neibor_var both occur
      //vector_relations[i]=(int *)malloc((var_count[neibor_var])*sizeof(int));
      //relations=build_neibor_relation(var, neibor_var, 
      //vector_relations[i], neg_in[var]);
    // relations=build_neibor_relation(var, neibor_var, 
      //			      relations, pos_in[var]);
      //the end
      //*relations=NONE;
    }
    vector[i]=NONE;
    neibor[var]=vector;
// neibor_relations[var]=vector_relations;
  }
}

int random_integer(int max)
{
  unsigned long int RAND;
  RAND=rand();
  return RAND % max;
}//random_integer(max) is from 0 to max-1

//-------------------------------------------------------------------------

//Modifaction du germe du générateur aléatoire

//-------------------------------------------------------------------------
/*
void modify_seed()
{
  struct tms *a_tms;
  int seed=2;
  time_t tp, mess;

  mess=time(&tp);

//  tv=(struct timeval *)malloc(sizeof(struct timeval));
//  tzp=(struct timezone *)malloc(sizeof(struct timezone));
//  gettimeofday(tv,tzp);
//  seed = (( tv->tv_sec & 0177 ) * 1000000) + tv->tv_usec;  

  if (mess==-1) {
    a_tms = ( struct tms *) malloc( sizeof (struct tms));
    mess=times(a_tms);
    seed = a_tms->tms_utime;
  }
  else seed=mess;
  srand(seed);
}
*/

  struct timeval tv;
  struct timezone tzp;

void modify_seed() {
  int seed;
  if (SEED_FLAG==TRUE) {
    srand(SEED); SEED=SEED+17;
    if (SEED==0) SEED=17;
  }
  else {
    gettimeofday(&tv,&tzp);
    seed = (( tv.tv_sec & 0177 ) * 1000000) + tv.tv_usec;
    srand(seed);
  }
}


int get_gradient(int var, int *clauses) {
  int clause, var1, *vars_signs, gradient=0, clause_gradient=1;
  for(clause=*clauses; clause!=NONE; clause=*(++clauses)) {
    if (clause_state[clause] == ACTIVE) {
      vars_signs = var_sign[clause];  clause_gradient=1;
      for(var1=*vars_signs; var1!=NONE; var1=*(vars_signs+=2)) {
	if ((var_state[var1]==ACTIVE) && (var1!=var)) {

	  if (var_current_value[var1]==*(vars_signs+1)) {
	    clause_gradient=0; break;
	  }
	}
      }
      gradient+=clause_gradient;
    }
  }
  return gradient;
}

void free_tabu() {
  int i, var;
  for (i=0; i<tabu_stack_fill_pointer; i++) {
    var=tabu_stack[i];
    tabu[var]=FALSE;
    tabu_list[var]=FALSE;
  }
  tabu_stack_fill_pointer=0;
}

inline int decreasing_var(int var) {
  return (score[var]>0);
}
//int clause_weight[tab_clause_size];
//after vars have initial values, call initialize()
int initialize() {
  int var, gradient, neg_gradient, pos_gradient, clause;

  decreasing_vars_stack_fill_pointer=0;
  zerodommage_vars_stack_fill_pointer=0;
  for (var=0; var<NB_VAR; var++) {
  	counter[var]=0.0; 
    tmp_score[var]=0;
    if (var_state[var]==ACTIVE) {
      neg_gradient=get_gradient(var, neg_in[var]);
      pos_gradient=get_gradient(var, pos_in[var]);
      if (var_current_value[var]==TRUE) {
      	score[var]=neg_gradient-pos_gradient;
      	make[var]=neg_gradient; break_value[var]=pos_gradient;
      }	
      else {
      	score[var]=pos_gradient-neg_gradient;
      	make[var]=pos_gradient; break_value[var]=neg_gradient; 
      }	
      if (var_current_value[var]==TRUE)
	     dommage_if_flip[var]=pos_gradient;
      else dommage_if_flip[var]=neg_gradient;
      if ((dommage_if_flip[var]==0)  && ( score[var] != 0)) {
    	push(var, zerodommage_vars_stack);
		zerodommage[var]=TRUE;
      }
      else zerodommage[var]=FALSE;
      if (decreasing_var(var)) {
	push(var, decreasing_vars_stack);
	tendance[var]=DECREASING;
      }
      else if (score[var]==0)
	tendance[var]=PLATEAU;
      else tendance[var]=INCREASING;
    }
  }
  max_counter=0.0; min_counter=0.0; ave_counter=0.0;
  
  for(clause=0; clause<NB_CLAUSE; clause++) {
    most_recent[clause]=NONE;
    most_recent_count[clause]=0;
   // clause_weight[clause]=0;
  }  
  
  free_tabu();
  return TRUE;
}

int nb_clauses_violated(int *clauses) {
  int clause, nb_violated=0;
    
  for (clause=*clauses;clause!=NONE;clause=*(++clauses))
    if (nb_lit_true[clause]==1)
      nb_violated++;       
  return nb_violated; 
}

int choose_least_flipped_var() {
  int var, chosen_var, i, flip=MAXSTEPS;

  for (i=0; i<VARIABLE_STACK_fill_pointer; i++) {
    var=VARIABLE_STACK[i];
    if (flip_time[var]<flip) {
      chosen_var=var; flip=flip_time[var];
    }
  }
  return chosen_var;
}

void score_for_vars_in_sat_clauses(int var, int *clauses) {
  int clause,  neibor_var, *vars_signs, dommage=0;
  for (clause=*clauses;clause!=NONE;clause=*(++clauses)) {
    vars_signs=var_sign[clause];
    switch(nb_lit_true[clause]) {
    case 0: 
      for(neibor_var=*vars_signs; neibor_var!=NONE; 
	  neibor_var=*(vars_signs+=2)) {
	if (neibor_var!=var) {
	    tmp_score[neibor_var]--;
	}
      }
      break;
    case 1: 
      for(neibor_var=*vars_signs; neibor_var!=NONE; 
	  neibor_var=*(vars_signs+=2)) {
	if ((neibor_var!=var) && 
	    (var_current_value[neibor_var]==*(vars_signs+1))) {
	  tmp_score[neibor_var]++;
	  break;
	}
      }
      break;
    }
  }
}

void score_for_vars_in_unsat_clauses(int var, int *clauses) {
  int clause, neibor_var, *vars_signs;
  for (clause=*clauses;clause!=NONE;clause=*(++clauses)) {
    vars_signs=var_sign[clause];
    switch(nb_lit_true[clause]) {
    case 1: 
      for(neibor_var=*vars_signs; neibor_var!=NONE; 
	  neibor_var=*(vars_signs+=2)) {
	if (neibor_var!=var) {
	  tmp_score[neibor_var]++;
	}
      }
      break;
    case 2: 
      for(neibor_var=*vars_signs; neibor_var!=NONE; 
	  neibor_var=*(vars_signs+=2)) {
	if ((neibor_var!=var) && 
	    (var_current_value[neibor_var]==*(vars_signs+1))) {
	  tmp_score[neibor_var]--;
	  break;
	}
      }
      break;
    }
  }
}

int TIME;

char expchance(int X)
{
   if (X>Intensity) X=Intensity;
   if (X<1) return 1;
   X=1<<X;
   if (rand()%X==0) return 1;
   return 0;
}
      
int get_wp_var(int random_clause_unsat) {
  int number_vars=0, index_of_vars, var;
  //int number_vars=0, index_vars, var, old=MAXSTEPS, old_var=NONE, clause, *clauses, weight;
  register int *vars_signs;
  int all_vars[tab_variable_size];

  vars_signs = var_sign[random_clause_unsat]; 
  
  //weight=MAXSTEPS; old_var=NONE;
  for(var=*vars_signs; var!=NONE; var=*(vars_signs+=2)) {
  	all_vars[number_vars++]=var;
  	//number_vars++;
  }//var is all_vars[0], all_vars[1], ..., all_vars[number_vars-1]
  index_of_vars=random_integer(number_vars);//index_of_vars is from 0 to number_vars-1
  //random_integer(max) is from 0 to max-1
  return all_vars[index_of_vars];
 }
 



int get_var_by_ori_vw() {
  int  random_unsatisfied_clause,  var_to_flip, number_vars=0, ii, one_index, var, best_break, the_break, best_var, best_counter, var_counter;
  int all_vars[tab_variable_size]; 
  register int *vars_signs;
  
  random_unsatisfied_clause=random_integer(MY_CLAUSE_STACK_fill_pointer);
  random_unsatisfied_clause=MY_CLAUSE_STACK[random_unsatisfied_clause];
  vars_signs = var_sign[random_unsatisfied_clause]; 
  for(var=*vars_signs; var!=NONE; var=*(vars_signs+=2)) {
  	all_vars[number_vars++]=var;
  	//number_vars++;
  }//var is all_vars[0], all_vars[1], ..., all_vars[number_vars-1], there are number_vars vars
  //number_vw++;
  one_index=random_integer(number_vars);
  var=all_vars[one_index];
  best_break=break_value[var]; 
  if (best_break==0) return var;
  best_var=var; best_counter=counter[var]; 
  for (ii=1;ii<number_vars;ii++) {
    one_index++; if (one_index>=number_vars) one_index=0;
    var=all_vars[one_index];
    the_break=break_value[var];
  	if (the_break==0) return var;
  	var_counter=counter[var];
  	if (the_break<best_break || (var_counter<best_counter) && (the_break==best_break || expchance(the_break-best_break)))
  	  {
  	   best_var=var; best_counter=var_counter; best_break=the_break; 
  	  }
 }
 return best_var;
}

int get_var_to_flip_in_clause_as_plus(int random_clause_unsat) {
  int var, best_var, second_best_var, nb, max_nb, pos_gradient, second_max=-NB_CLAUSE,
    neg_gradient, real_nb, flip=-1, flip_index, var_to_flip, old=MAXSTEPS, old_var;
  register int *vars_signs;

  vars_signs = var_sign[random_clause_unsat]; max_nb=-NB_CLAUSE;
  for(var=*vars_signs; var!=NONE; var=*(vars_signs+=2)) {
    nb=score[var];
    if ((nb>max_nb) || ((nb==max_nb) && (flip_time[var]<flip_time[best_var]))) {
      second_best_var=best_var; second_max=max_nb; best_var=var; max_nb=nb;
    }
    else if ((nb>second_max) || 
	     ((nb==second_max) && (flip_time[var]<flip_time[second_best_var]))) {
      second_max=nb; second_best_var=var;
    }
    if (flip_time[var]>flip) 
      flip=flip_time[var];
    if (flip_time[var]<old) {
      old=flip_time[var];
      old_var=var;
    }
  }
  
  //if (random_integer(100)<LNOISE)
  //  return old_var;
    
  if (random_integer(100)<LNOISE) {
     return get_wp_var(random_clause_unsat);
  }  
  else {
    if (flip_time[best_var]==flip) {
      if (random_integer(100)<NOISE) 
	    return second_best_var; 
      else return best_var;
      }
    else return best_var;
  }
}



int get_var_to_flip_in_clause_as_new(int random_clause_unsat) {
  int var, best_var, second_best_var, nb, max_nb, pos_gradient, second_max=-NB_CLAUSE,
    neg_gradient, real_nb, flip=-1, flip_index, var_to_flip, old=MAXSTEPS, old_var,
    chosen_var;
  register int *vars_signs;

  vars_signs = var_sign[random_clause_unsat]; max_nb=-NB_CLAUSE;
  for(var=*vars_signs; var!=NONE; var=*(vars_signs+=2)) {
    nb=score[var];
    if ((nb>max_nb) || ((nb==max_nb) && (flip_time[var]<flip_time[best_var]))) {
      second_best_var=best_var; second_max=max_nb; best_var=var; max_nb=nb;
    }
    else if ((nb>second_max) || 
	     ((nb==second_max) && (flip_time[var]<flip_time[second_best_var]))) {
      second_max=nb; second_best_var=var;
    }
    if (flip_time[var]<old) {
      old=flip_time[var];
      old_var=var;
    }
  }
  if (random_integer(100)<LNOISE)
    chosen_var=old_var;
  else {
    if (best_var==most_recent[random_clause_unsat]) {
      switch(most_recent_count[random_clause_unsat]) {
      case 1: NOISE1=20; break;
      case 2: NOISE1=50; break;
      case 3: NOISE1=65; break;
      case 4: NOISE1=72; break;
      case 5: NOISE1=78; break;
      case 6: NOISE1=86; break;
      case 7: NOISE1=90; break;
      case 8: NOISE1=95; break;
      case 9: NOISE1=98; break;
      default: NOISE1=100; break;
      }
      if (random_integer(100)<NOISE1) {
	chosen_var=second_best_var; 
      }
      else 
	chosen_var=best_var;
    }
    else
	chosen_var=best_var;
  }
  return chosen_var;
}

void satisfy_clauses(int var, int *clauses) {
  int clause,  neibor_var, *vars_signs, dommage=0, last_unsatisfied_clause,
    index;
  for (clause=*clauses;clause!=NONE;clause=*(++clauses)) {
    vars_signs=var_sign[clause];
    switch(nb_lit_true[clause]) {
    case 0: clause_truth[clause]=TRUE; nb_lit_true[clause]++;
      dommage++; //clause_weight[clause]=TIME;
      last_unsatisfied_clause=pop(MY_CLAUSE_STACK);
      index=index_in_MY_CLAUSE_STACK[clause];
      MY_CLAUSE_STACK[index]=last_unsatisfied_clause;
      index_in_MY_CLAUSE_STACK[last_unsatisfied_clause]=index;

      for(neibor_var=*vars_signs; neibor_var!=NONE; 
	  neibor_var=*(vars_signs+=2)) {
	if (neibor_var!=var) {
	    tmp_score[neibor_var]--;
	    make[neibor_var]--;// break_value[neibor_var] unchanged
	}
      }
      break;
    case 1: nb_lit_true[clause]++;
      for(neibor_var=*vars_signs; neibor_var!=NONE; 
	  neibor_var=*(vars_signs+=2)) {
	if ((neibor_var!=var) && 
	    (var_current_value[neibor_var]==*(vars_signs+1))) {
	  // dommage_if_flip[neibor_var]--;
	  tmp_score[neibor_var]++;
	  break_value[neibor_var]--;//make[neibor_var] unchanged, 
	  break;
	}
      }
      break;
    default:  nb_lit_true[clause]++;
    }
  }
  if (dommage==0) 
    printf("c'est curieux...");
  
}

/* let x1 be the new value of x and x0 be the old value of x
   then for each clause in *clauses here, (x1-x0)df/dx is positive */
void unsatisfy_clauses(int var, int *clauses) {
  int clause, neibor_var, *vars_signs;
  for (clause=*clauses;clause!=NONE;clause=*(++clauses)) {
    vars_signs=var_sign[clause];
    switch(nb_lit_true[clause]) {
    	//before flipping var, nb_lit_true[cause]>=1 because var is a positive literal  
    case 1:  clause_truth[clause]=FALSE; nb_lit_true[clause]--;
      //push(clause,MY_CLAUSE_STACK);
      
      if (most_recent[clause]==var) 
	     most_recent_count[clause]++;
      else {
	    most_recent[clause]=var;
	    most_recent_count[clause]=1;
      }
      
      index_in_MY_CLAUSE_STACK[clause]=MY_CLAUSE_STACK_fill_pointer;
      push(clause,MY_CLAUSE_STACK);

      for(neibor_var=*vars_signs; neibor_var!=NONE; 
	  neibor_var=*(vars_signs+=2)) {
	if (neibor_var!=var) {
	  tmp_score[neibor_var]++;
	  make[neibor_var]++;// break_value[neibor_var] unchanged
	}
      }
      break;
    case 2: nb_lit_true[clause]--;
      for(neibor_var=*vars_signs; neibor_var!=NONE; 
	  neibor_var=*(vars_signs+=2)) {
	if ((neibor_var!=var) && 
	    (var_current_value[neibor_var]==*(vars_signs+1))) {
	  tmp_score[neibor_var]--;
	  break_value[neibor_var]++; 
	  // dommage_if_flip[neibor_var]++;
	  break;
	}
      }
      break;
    default:  nb_lit_true[clause]--;
    }
  }
}

void check_implied_clauses(int var, int value, int nb_flip) { 
  int *neibors, neibor_var;
  //neibors=neibor[var];
  //  for(neibor_var=*neibors; neibor_var!=NONE; neibor_var=*(++neibors)) 
  //  tmp_score[neibor_var]=0;
  if (value==TRUE) {
    satisfy_clauses(var, pos_in[var]);
    unsatisfy_clauses(var, neg_in[var]);
  }
  else {
    satisfy_clauses(var, neg_in[var]);
    unsatisfy_clauses(var, pos_in[var]);
  }
  neibors=neibor[var];
  for(neibor_var=*neibors; neibor_var!=NONE; neibor_var=*(++neibors)) {
    if ((score[neibor_var]<=0) && (score[neibor_var]+tmp_score[neibor_var]>0))
      push(neibor_var, decreasing_vars_stack);
    score[neibor_var]+=tmp_score[neibor_var];
    tmp_score[neibor_var]=0;
  }
}

void eliminate_satisfied_clauses() {
  int first_satisfied=NONE,put_in,i;
  
  for (i=0;i<MY_CLAUSE_STACK_fill_pointer;i++)
    if(clause_truth[MY_CLAUSE_STACK[i]]==TRUE) {
      first_satisfied=i;
      break;
    }    
  if (first_satisfied!=NONE) { 
    put_in=first_satisfied;
    for (i=first_satisfied+1;i<MY_CLAUSE_STACK_fill_pointer;i++)
      if (clause_truth[MY_CLAUSE_STACK[i]]!=TRUE) {
	MY_CLAUSE_STACK[put_in]=MY_CLAUSE_STACK[i];
	put_in++;              
      }
    MY_CLAUSE_STACK_fill_pointer=put_in;
  }
}

void push_unsatisfied_clauses(int var, int value) { 
  int *clauses,clause;
   
  if (value==TRUE) clauses=neg_in[var];
  else clauses=pos_in[var];
  for (clause=*clauses;clause!=NONE;clause=*(++clauses)) {
    if (nb_lit_true[clause]==1 ) {
      push(clause,MY_CLAUSE_STACK);
      clause_truth[clause]=FALSE;         
    }
    nb_lit_true[clause]--;
  } 
}

void dec_nb_lit_true(int var, int value) {
  int *clauses,clause;
  if (positive(var))
    clauses=neg_in[var];
  else
    clauses=pos_in[var];
  for (clause=*clauses;clause!=NONE;clause=*(++clauses))    
    nb_lit_true[clause]--;    
}

int simple_eliminate_increasing_vars() {
  int i, first=NONE, put_in, var, current=0, nb=0, chosen_var=NONE, flip=MAXSTEPS;
  for (i=0; i<decreasing_vars_stack_fill_pointer; i++) {
    var=decreasing_vars_stack[i];
    if (score[var]<=0) {
      first=i;
      break;
    }
  }
  if (first !=NONE) {
    put_in=first;
    for (i=first+1; i<decreasing_vars_stack_fill_pointer; i++) {
      var=decreasing_vars_stack[i];
      if (score[var]>0) {
	decreasing_vars_stack[put_in++]=var;
      }
    }
    decreasing_vars_stack_fill_pointer=put_in;
  }
  return chosen_var;
}

void eliminate_dommage_vars() {
  int i, first=NONE, put_in, var, current=-1;
  for (i=0; i<zerodommage_vars_stack_fill_pointer; i++) {
    var=zerodommage_vars_stack[i];
    if (zerodommage[var]==FALSE) {
      first=i;
      break;
    }
  }
  if (first !=NONE) {
    put_in=first;
    for (i=first+1; i<zerodommage_vars_stack_fill_pointer; i++) {
      var=zerodommage_vars_stack[i];
      if (zerodommage[var]==TRUE)
	zerodommage_vars_stack[put_in++]=var;
    }
    zerodommage_vars_stack_fill_pointer=put_in;
  }
}

int choose_best_decreasing_var() {
  int var, chosen_var, i, maxnb=0, flip=MAXSTEPS, nb, considered_length;
  double the_counter;
  considered_length=decreasing_vars_stack_fill_pointer; 
  var=decreasing_vars_stack[0]; 
  //the_counter=counter[var]; 
  flip=flip_time[var]; 
  chosen_var=var;
  for (i=1; i<considered_length; i++) {
     var=decreasing_vars_stack[i];
     if (flip_time[var]<flip) {
          flip=flip_time[var]; chosen_var=var;	
     }
  }
  return chosen_var;
}


  
int update_gradient_and_choose_flip_var(int var) {
  int neibor_var, chosen1, sign, index, for_swap, 
    other_var_in_same_clause, clause_gradient, gradient, chosen_var=NONE;

  score[var]=0-score[var];
  for_swap=make[var];
  make[var]=break_value[var];
  break_value[var]=for_swap;
  simple_eliminate_increasing_vars();
  if (decreasing_vars_stack_fill_pointer>1)
    return choose_best_decreasing_var();
  else if (decreasing_vars_stack_fill_pointer==1)
    return decreasing_vars_stack[0];
  return chosen_var;
}
  

int choose_var_by_random_walk_as_plus() {
  int  random_unsatisfied_clause,  var_to_flip;
  random_unsatisfied_clause=random_integer(MY_CLAUSE_STACK_fill_pointer);
  random_unsatisfied_clause=MY_CLAUSE_STACK[random_unsatisfied_clause];
  var_to_flip=get_var_to_flip_in_clause_as_plus(random_unsatisfied_clause);
  return  var_to_flip;
}

 
 int choose_var_by_random_walk_as_new() {
  int  random_unsatisfied_clause,  var_to_flip;
  random_unsatisfied_clause=random_integer(MY_CLAUSE_STACK_fill_pointer);
  random_unsatisfied_clause=MY_CLAUSE_STACK[random_unsatisfied_clause];
  var_to_flip=get_var_to_flip_in_clause_as_new(random_unsatisfied_clause);
  return  var_to_flip;
}


int flipped_var_stack[tab_variable_size];
int flipped_var_stack_fill_pointer=0;
int flipped[tab_variable_size];

void clean_flipped() {
  int i;
  for(i=0; i<flipped_var_stack_fill_pointer; i++)
    flipped[flipped_var_stack[i]]=0;
  flipped_var_stack_fill_pointer=0;
}

void save_assignment() {
  int i;
  clean_flipped();
  // for(i=0; i<NB_VAR; i++) 
  //  saved_var_current_value[i]=var_current_value[i];
}

int compute_distance() {
  int k, dist=0, var, i;
  for(i=0; i<flipped_var_stack_fill_pointer; i++) {
    var=flipped_var_stack[i];
    if (flipped[var]%2 == 1) 
      dist++;
  }
  //for(k=0;k<NB_VAR;k++) {
  //  if (var_current_value[k] != saved_var_current_value[k])
  //    dist++; 
  //}
  return dist;
}

//int TIME;

#include "adaptnoisebis1at5.C"

void update_counter(int var_to_flip) {
  double the_old_counter, the_new_counter;	
  //if ((j+1)%1000==0) analyse_counter(var_to_flip, counter[var_to_flip]);
  the_old_counter=counter[var_to_flip];
  //ounter[var_to_flip]=(1.0-Sigma)*(counter[var_to_flip]+1.0)+Sigma*(double)(global_j+1);
  counter[var_to_flip]=1.0*(counter[var_to_flip]+1.0);
  the_new_counter=counter[var_to_flip];
  old_ave_counter=ave_counter;
  ave_counter=old_ave_counter+(the_new_counter-the_old_counter)/(double)NB_VAR;
  if (the_new_counter>max_counter) {
  	max_counter=the_new_counter;
  	//var_with_biggest_counter=var_to_flip;
  }
}

bool search() {
  int last_suc_j=MAXSTEPS, i,j,k,SAT=FALSE, var_to_flip=NONE, index, flag, 
    *min, nbwalk=0, var;

  double avgdepth, depth=0;
  //double avgdepths=0, total_min=0, total_last=0, total_nbwalk=0;
  long begintime, endtime, mess;
  struct tms *a_tms;
  
  a_tms = ( struct tms *) malloc( sizeof (struct tms));
  mess=times(a_tms); begintime = a_tms->tms_utime;
  
  min=(int *)malloc(MAXTRIES*sizeof(int));
  preprocess();

  for (i=0;i<MAXTRIES;i++) {
  	//Intensity=1;  Sigma=0.1;  Interval=1000.0;  Adjustment=1000.0;
  	Sigma=0.0;
    modify_seed(); min[i]=NB_CLAUSE; nbwalk=0; SAT=FALSE; depth=0;
    for (k=0;k<NB_VAR;k++) {
      if (var_state[k]==ACTIVE) {
	var_current_value[k]=random_integer(2);
	enter_stack_time[k]=0;
	flip_time[k]=0;
	walk_time[k]=0;
	flipped[k]=0;
      }
    }
    initialize();
    clause_value();
    initNoise();
    if (zerodommage_vars_stack_fill_pointer>0) {
      index=random_integer(zerodommage_vars_stack_fill_pointer);
      var_to_flip=zerodommage_vars_stack[index];
    }
    else 
    if (decreasing_vars_stack_fill_pointer>0) {
      index=random_integer(decreasing_vars_stack_fill_pointer);
      var_to_flip=decreasing_vars_stack[index];
      // var_to_flip=choose_best_decreasing_var();
    }
    else var_to_flip=NONE;
    for (j=0;j<MAXSTEPS;j++) {
      global_j=j;	
      flag=TRUE; TIME=j;
      if (walk_satisfiable()) {
	    SAT=TRUE; last_suc_j=j;
	    break;
      }
     
      if (max_counter>=(double)coefficient_for_prm*ave_counter)
       {
       	//number_uneven++;
       	if (var_to_flip==NONE) {// when there is no non-tabu decreasing var 
        	nbwalk++; 
	    	var_to_flip=choose_var_by_random_walk_as_new();
	        flag=FALSE; 
          }   
       }
      else 
       {
           //number_even++;     
          if (var_to_flip==NONE) {// when there is no non-tabu decreasing var 
        	nbwalk++; 
	    	var_to_flip=choose_var_by_random_walk_as_plus(); 
	        flag=FALSE; 
          }   
        }
           	
      var_current_value[var_to_flip]=1-var_current_value[var_to_flip]; 
      check_implied_clauses(var_to_flip, var_current_value[var_to_flip], j);
      //eliminate_satisfied_clauses();
      flip_time[var_to_flip]=j;
      update_counter(var_to_flip);
      //if ((double)(j+1)>Adjustment) adjust_parameters_from_ori_vw((double)(j+1)); 
      adaptNoveltyNoise(j);
      // flipped_var=var_to_flip;
      var_to_flip=update_gradient_and_choose_flip_var(var_to_flip);
   }
    if (SAT==TRUE) {
      clause_value();
      if (walk_satisfiable()) {
	//fprintf(stdout, "c A solution is found\n");
	break;
      }
      else {
	//fprintf(stdout, "c I'AM SORRY SOMETHING IS WRONG\n");
	SAT=FALSE;
      }
    }
    
 }
  mess=times(a_tms); endtime = a_tms->tms_utime;
   if (SAT==TRUE) {
    //fprintf(stdout, "s SATISFIABLE\n");
    //fprintf(stdout, "v ");
    /*for (var=0; var<NB_VAR; var++) {
      if (var_current_value[var]==TRUE)
	fprintf(stdout, "%d ", var+1);
      else fprintf(stdout, "%d ", -(var+1));
    }*/
    //fprintf(stdout, "\n");
    //fprintf(stdout, "v 0 \n");
  }
  else {
    //fprintf(stdout, "s UNKNOWN\n");
  }
  fprintf(stdout, "c Done (mycputime is %5.3f seconds)\n", 
	  ((double)(endtime-begintime)/((double)CLOCKS_PER_SEC/(double)10000)));
  if (SAT==TRUE)
    return true;
  else return false;
}



int HELP_FLAG=FALSE;


void __initialize_vars()
{
    var_current_value = (my_type*) malloc(tab_variable_size * sizeof(my_type));
    var_rest_value = (my_type*) malloc(tab_variable_size * sizeof(my_type));
    var_state = (my_type*) malloc(tab_variable_size * sizeof(my_type));
    clause_state = (my_type*)malloc(tab_clause_size * sizeof(my_type));
    clause_length = (my_type*)malloc(tab_clause_size * sizeof(my_type));

    test_flag = (int*) malloc(sizeof(int) * tab_variable_size);

    IMPLIED_LIT_STACK = (int*) malloc(sizeof(int) *tab_variable_size);
   neibor_stack = (int*) malloc(sizeof(int) * tab_variable_size);

    counter = (double*) malloc(tab_variable_size * sizeof(double));
    reduce_if_negative_nb = (float*) malloc(sizeof(float) * tab_variable_size);
    reduce_if_positive_nb = (float*) malloc(sizeof(float) * tab_variable_size);
    most_recent = (int*) malloc(sizeof(int) * tab_clause_size);
    most_recent_count = (int*) malloc( sizeof(int) * tab_clause_size);

    VARIABLE_STACK = (int*) malloc(tab_variable_size * sizeof(int));
    CLAUSE_STACK = (int*) malloc(tab_clause_size * sizeof(int));
    UNITCLAUSE_STACK = (int*) malloc(tab_unitclause_size * sizeof(int));

    MANAGEDCLAUSE_STACK = (int*) malloc(tab_clause_size * sizeof(int));


    score = (int*) malloc( sizeof(int) * tab_variable_size);
    make = (int*) malloc(sizeof(int) * tab_variable_size);
    break_value= (int*) malloc( sizeof(int) *tab_variable_size);
    tmp_score = (int*) malloc(sizeof(int) * tab_variable_size);
    decreasing_vars_stack =  (int*) malloc( sizeof(int) *tab_variable_size);
   tabu_stack = (int*) malloc(sizeof(int) * tab_variable_size);
  
   tabu_list = (int*) malloc(sizeof(int) * tab_variable_size);
   tendance = (int*) malloc( sizeof(int) * tab_variable_size);

   MY_CLAUSE_STACK = (int*) malloc(sizeof(int) * tab_clause_size);
   nb_lit_true = (int*) malloc( sizeof(int) * tab_clause_size);
   clause_truth = (int*) malloc(sizeof(int) * tab_clause_size);
   dommage_if_flip = (int*) malloc(sizeof(int) * tab_variable_size);
   zerodommage = (int*) malloc(sizeof(int) *tab_variable_size);
   zerodommage_vars_stack = (int*) malloc(sizeof(int) * tab_variable_size);
   flip_time = (int*) malloc(sizeof(int) * tab_variable_size);
   enter_stack_time = (int*) malloc(sizeof(int) * tab_variable_size);
   walk_time = (int*) malloc(sizeof(int) *tab_variable_size);

}

bool seq_processing(int _var_size, const vector< vector<int> >& clauses)
{

    int NB_VAR, NB_CLAUSE;
    __initialize_vars();
    int ret = build_simple_sat_instance(_var_size, clauses);

    switch(ret)
    {
    case FALSE:
         return false;
    case SATISFIABLE:
         return true;
    case TRUE:
         VARIABLE_STACK_fill_pointer=0;
         CLAUSE_STACK_fill_pointer = 0;
         MANAGEDCLAUSE_STACK_fill_pointer = 0;
         T_SEUIL= 0; 
         return search();
    case NONE:
         return false;
    }

    return false;
}

/*
main(int argc, char *argv[]) {
  int i,  var; 

  if (argc==3) {
    SEED_FLAG=TRUE; 
    if (sscanf(argv[2],"%u",&SEED)!=1) {
      fprintf(stdout, "c Bad argument %s\n", argv[2]);
      fprintf(stdout, "s UNKNOWN\n");
      exit(0);
    }
  } 
  //parse_parameters(argc,argv);
  INPUT_FILE=argv[1];
  coefficient_for_prm = 10.0; 
   switch (build(BUILD_FLAG, INPUT_FILE)) {
  case FALSE: fprintf(stdout, "c Input file error or too large formula\n"); 
    fprintf(stdout, "s UNKNOWN\n");
    exit(0);
    return FALSE;
  case SATISFIABLE:
    if (verify_sol_input(argv[1])==TRUE) {
      fprintf(stdout, "c Satisfied at top\n");
      fprintf(stdout, "s SATISFIABLE\n");
      fprintf(stdout, "v ");

      for (var=0; var<NB_VAR; var++) {
	if (var_current_value[var]==TRUE)
	  fprintf(stdout, "%d ", var+1);
	else fprintf(stdout, "%d ", -(var+1));
      }
      fprintf(stdout, "\n");
      fprintf(stdout, "v 0 \n");
      exit(10);
    }
    else {
      fprintf(stdout, "c problem at top\n");
      fprintf(stdout, "s UNKNOWN\n");
      exit(0);
    }
    return FALSE;
  case TRUE:
    VARIABLE_STACK_fill_pointer=0;
    CLAUSE_STACK_fill_pointer = 0;
    MANAGEDCLAUSE_STACK_fill_pointer = 0;
    T_SEUIL= 0; 
    if (search(argv[1])==TRUE) {
	exit(10);
    }
    else 
      exit(0);
    break;
  case NONE: fprintf(stdout, "c A contradiction is found at top!\n");
    fprintf(stdout, "s UNSATISFIABLE\n");
    exit(20);
  }
  return TRUE;
}

*/
