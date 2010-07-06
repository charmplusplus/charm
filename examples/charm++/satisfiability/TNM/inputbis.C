
#define OLD_CLAUSE_REDUNDANT -77
#define NEW_CLAUSE_REDUNDANT -7


int smaller_than(int lit1, int lit2) {
  return ((lit1<NB_VAR) ? lit1 : lit1-NB_VAR) < 
    ((lit2<NB_VAR) ? lit2 : lit2-NB_VAR);
}

my_type redundant(int *new_clause, int *old_clause) {
  int lit1, lit2, old_clause_diff=0, new_clause_diff=0;
    
  lit1=*old_clause; lit2=*new_clause;
  while ((lit1 != NONE) && (lit2 != NONE)) {
    if (smaller_than(lit1, lit2)) {
      lit1=*(++old_clause); old_clause_diff++;
    }
    else
      if (smaller_than(lit2, lit1)) {
	lit2=*(++new_clause); new_clause_diff++;
      }
      else
	if (complement(lit1, lit2)) {
	  return FALSE; /* old_clause_diff++; new_clause_diff++; j1++; j2++; */
	}
	else {
          lit1=*(++old_clause);  lit2=*(++new_clause);
	}
  }
  if ((lit1 == NONE) && (old_clause_diff == 0))
    /* la nouvelle clause est redondante ou subsumee */
    return NEW_CLAUSE_REDUNDANT;
  if ((lit2 == NONE) && (new_clause_diff == 0))
    /* la old clause est redondante ou subsumee */
    return OLD_CLAUSE_REDUNDANT;
  return FALSE;
}

my_type get_resolvant(int *clause1, int *clause2, int *resolvant) {
  int lit1, lit2, nb_diff1=0,  nb_diff2=0,
    nb_iden=0, nb_opps=0, j1=0, j2=0, j, limited_length;

  while (((lit1=clause1[j1])!=NONE) && ((lit2=clause2[j2]) != NONE)) {
    if (complement(lit1, lit2)) {
      j1++; j2++; nb_opps++;
    }
    else
      if (lit1 == lit2) {
	j1++; j2++; nb_iden++;
      }
      else
	if (smaller_than(lit1, lit2)) {
          nb_diff1++; j1++;
	}
	else {
          nb_diff2++; j2++;
	}
  }
  if (nb_opps ==1) {
    if (clause1[j1] ==NONE) {
      for (; clause2[j2]!= NONE; j2++) nb_diff2++;
    }
    else {
      for (; clause1[j1]!= NONE; j1++) nb_diff1++;
    }
    if ((j1==1) || (j2==1))  limited_length=RESOLVANT_LENGTH; 
    else
      if ((j1==2) && (j2==2))  limited_length=1;
      else
	if (j1<j2) limited_length=((j1<RESOLVANT_LENGTH) ? j1 : RESOLVANT_LENGTH);
	else  limited_length=((j2<RESOLVANT_LENGTH) ? j2 : RESOLVANT_LENGTH);

    if (nb_diff1 + nb_diff2 + nb_iden <= limited_length) {
      j1=0; j2=0; j=0;
      while (((lit1 = clause1[j1])!=NONE) && ((lit2 = clause2[j2]) != NONE)) {
	if (lit1 == lit2) {
	  resolvant[j] = lit1; j1++; j2++; j++;
	}
	else 
	  if (smaller_than(lit1, lit2)) {
	    resolvant[j] = lit1; j1++; j++;
	  }
	  else
	    if (smaller_than(lit2, lit1)) {
	      resolvant[j] = lit2; j2++; j++;
	    }
	    else {
	      j1++; j2++;
	    }
      }
      if (clause1[j1] ==NONE) while ((resolvant[j++] = clause2[j2++]) != NONE);
      else while ((resolvant[j++] = clause1[j1++]) != NONE);
      if (j==0) return NONE;
      if (nb_diff2==0) return 2; /* clause1 is redundant */
      return TRUE;
    }
  }
  return FALSE;
}

int INVOLVED_CLAUSE_STACK[tab_clause_size];
int INVOLVED_CLAUSE_STACK_fill_pointer=0;
int CLAUSE_INVOLVED[tab_clause_size];

void remove_passive_clauses() {
  int  clause, put_in, first=NONE;
  for (clause=0; clause<NB_CLAUSE; clause++) {
    if (clause_state[clause]==PASSIVE) {
      first=clause; break;
    }
  }
  if (first!=NONE) {
    put_in=first;
    for(clause=first+1; clause<NB_CLAUSE; clause++) {
      if (clause_state[clause]==ACTIVE) {
	sat[put_in]=sat[clause]; var_sign[put_in]=var_sign[clause];
	clause_state[put_in]=ACTIVE; 
	clause_length[put_in]=clause_length[clause];
	put_in++;
      }
    }
    NB_CLAUSE=put_in;
  }
}

void remove_passive_vars_in_clause(int clause) {
  int *vars_signs, *vars_signs1, var, var1, first=NONE;
  vars_signs=var_sign[clause];
  for(var=*vars_signs; var!=NONE; var=*(vars_signs+=2)) {
    if (var_state[var]!=ACTIVE) {
      first=var; break;
    }
  }
  if (first!=NONE) {
    for(vars_signs1=vars_signs+2, var1=*vars_signs1; var1!=NONE; 
	var1=*(vars_signs1+=2)) {
      if (var_state[var1]==ACTIVE) {
	*vars_signs=var1; *(vars_signs+1) = *(vars_signs1+1);
	vars_signs+=2;
      }
    }
    *vars_signs=NONE;
  }
}

int clean_structure() {
  int clause, var, *vars_signs;
  remove_passive_clauses();
  if (NB_CLAUSE==0) 
    return SATISFIABLE;
  for (clause=0; clause<NB_CLAUSE; clause++) 
    remove_passive_vars_in_clause(clause);
  NB_ACTIVE_VAR=0;
  for (var=0; var<NB_VAR; var++) { 
    neg_nb[var] = 0;
    pos_nb[var] = 0;
    if (var_state[var]==ACTIVE) NB_ACTIVE_VAR++;
  }
  for (clause=0; clause<NB_CLAUSE; clause++) {
    vars_signs=var_sign[clause];
    for(var=*vars_signs; var!=NONE; var=*(vars_signs+=2)) {
      if (*(vars_signs+1)==POSITIVE) 
	pos_in[var][pos_nb[var]++]=clause;
      else  neg_in[var][neg_nb[var]++]=clause;
    }
  }
  for (var=0; var<NB_VAR; var++) { 
    neg_in[var][neg_nb[var]]=NONE;
    pos_in[var][pos_nb[var]]=NONE;
  }
  return TRUE;
}

int unitclause_process();
int simplify_formula();

int lire_clauses(const vector< vector<int> >& _clauses) {
    int i, j, jj, ii, length, tautologie, lits[1000], lit, lit1;
    for (i=0; i<NB_CLAUSE; i++) {
    
        length = _clauses[i].size();
        for(j=0; j<length; j++)
        {
            lits[j] = _clauses[i][j];
        }
        tautologie = FALSE;
        /* test if some literals are redundant and sort the clause */
        for (ii=0; ii<length-1; ii++) {
            lit = lits[ii];
            for (jj=ii+1; jj<length; jj++) {
                if (abs(lit)>abs(lits[jj])) {
                    lit1=lits[jj]; lits[jj]=lit; lit=lit1;
                }
                else
                    if (lit == lits[jj]) {
                        lits[jj] = lits[length-1]; 
                        jj--; length--; lits[length] = 0;
                    }
                    else
                        if (abs(lit) == abs(lits[jj])) {
                            tautologie = TRUE; break;
                        }
            }
            if (tautologie == TRUE) break;
            else lits[ii] = lit;
        }
        if (tautologie == FALSE) {
            sat[i]= (int *)malloc((length+1) * sizeof(int));
            for (j=0; j<length; j++) {
              if (lits[j] < 0) 
                  sat[i][j] = abs(lits[j]) - 1 + NB_VAR ;
              else 
                  sat[i][j] = lits[j]-1;
          }
          sat[i][length]=NONE;
          clause_length[i]=length;
          clause_state[i] = ACTIVE;
      }
      else { i--; NB_CLAUSE--;}
  }
  return TRUE;
}

void build_structure() {
    int i, j, var, *lits1, length, clause, *vars_signs, lit;
    for (i=0; i<NB_VAR; i++) { 
        neg_nb[i] = 0; pos_nb[i] = 0;
    }
    for (i=0; i<NB_CLAUSE; i++) {
        for(j=0; j<clause_length[i]; j++) {
            if (sat[i][j]>=NB_VAR) {
                var=sat[i][j]-NB_VAR; neg_nb[var]++;
            }
            else {
                var=sat[i][j]; 
                pos_nb[var]++;
            }
        }
        if (sat[i][clause_length[i]] !=NONE)
            printf("erreur ");
    }
    for(clause=0;clause<NB_CLAUSE;clause++) {
        length = clause_length[clause];
        var_sign[clause] = (int *)malloc((2*length+1)*sizeof(int));
        lits1 = sat[clause]; vars_signs = var_sign[clause];
        for(lit=*lits1; lit!=NONE; lit=*(++lits1),(vars_signs+=2)) {
            if (negative(lit)) {
                *(vars_signs+1)= NEGATIVE;
                *vars_signs = get_var_from_lit(lit);
            }
            else {
                *(vars_signs+1)=POSITIVE;
                *vars_signs = lit;
            }
        }
        *vars_signs = NONE;  
    }
    for (i=0; i<NB_VAR; i++) { 
    neg_in[i] = (int *)malloc((neg_nb[i]+1) * sizeof(int));
    pos_in[i] = (int *)malloc((pos_nb[i]+1) * sizeof(int));
    neg_in[i][neg_nb[i]]=NONE; pos_in[i][pos_nb[i]]=NONE;
    neg_nb[i] = 0; pos_nb[i] = 0;
    var_state[i] = ACTIVE;
  }   
  for (i=0; i<NB_CLAUSE; i++) {
    // if (i==774)
    //  printf("kjhsdf");
    lits1 = sat[i];
    //printf("\n");
    for(lit=*lits1; lit!=NONE; lit=*(++lits1)) {

      //  printf("   %d", get_var_from_lit(lit));
        if (positive(lit)) 
            pos_in[lit][pos_nb[lit]++] = i;
        else
            neg_in[get_var_from_lit(lit)]
                [neg_nb[get_var_from_lit(lit)]++] = i;
    }
  }
}


void eliminate_redundance() {
  int *lits, i, lit, *clauses, res, clause;

  for (i=0; i<NB_CLAUSE; i++) {
    if (clause_state[i]==ACTIVE) {
      if (clause_length[i]==1)
	push(i, UNITCLAUSE_STACK);
      lits = sat[i];
      for(lit=*lits; lit!=NONE; lit=*(++lits)) {
	if (positive(lit)) 
	  clauses=pos_in[lit];
	else clauses=neg_in[lit-NB_VAR];
	for(clause=*clauses; clause!=NONE; clause=*(++clauses)) {
	  if ((clause<i) && (clause_state[clause]==ACTIVE)) {
	    res=redundant(sat[i], sat[clause]);
	    if (res==NEW_CLAUSE_REDUNDANT) {
	      clause_state[i]=PASSIVE;
	      break;
	    }
	    else if (res==OLD_CLAUSE_REDUNDANT)
	      clause_state[clause]=PASSIVE;
	  }
	}
	if (res==NEW_CLAUSE_REDUNDANT)
	  break;
      }
    }
  }
}

my_type build_simple_sat_instance(int vars_size,  const vector< vector<int> >&  clauses) {
  
    int i, j, length, NB_CLAUSE1, res, ii, jj, tautologie, lit1,
        *pos_nb, *neg_nb;
 
    NB_VAR = vars_size;
    NB_CLAUSE = clauses.size();

    INIT_NB_CLAUSE = NB_CLAUSE;
    neg_nb=(int *)reduce_if_negative_nb;
    pos_nb=(int *)reduce_if_positive_nb;

    if (lire_clauses(clauses)==FALSE)
        return FALSE;
    build_structure();
    eliminate_redundance();
    if (unitclause_process()==NONE) return NONE;
    res=clean_structure();
    if (res==FALSE)
        return FALSE;
    else if (res==SATISFIABLE)
        return SATISFIABLE;
    return TRUE;
}

int verify_sol_input(char *input_file) {
    FILE* fp_in=fopen(input_file, "r");
  char ch, word2[WORD_LENGTH];
  int i, j, lit, var, nb_var1, nb_clause1;

  if (fp_in == NULL) return FALSE;

  fscanf(fp_in, "%c", &ch);
  while (ch!='p') {
    while (ch!='\n') fscanf(fp_in, "%c", &ch);  
    fscanf(fp_in, "%c", &ch);
  }
  
  fscanf(fp_in, "%s%d%d", word2, &nb_var1, &nb_clause1);
  for (i=0; i<nb_clause1; i++) {
    fscanf(fp_in, "%d", &lit);
    while (lit != 0) {
      if (lit<0) {
	if (var_current_value[abs(lit)-1]==FALSE)
	  break;
      }
      else {
	if (var_current_value[lit-1]==TRUE)
	  break;
      }
      fscanf(fp_in, "%d", &lit);
    }
    if (lit==0) {
      fclose(fp_in);
      return FALSE;
    }
    else {
      do fscanf(fp_in, "%d", &lit);
      while (lit != 0) ;
    }
  }
  fclose(fp_in);
  return TRUE;
}
