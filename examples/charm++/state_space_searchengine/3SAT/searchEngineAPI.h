#ifndef __SEARCHENGINEAPI__
#define __SEARCHENGINEAPI__

class SatStateBase : public StateBase
{
	int             var_size;
	int             clause_size;
	int             level;
	int             assigned_lit;
	int             *clauses;
	int             *occurrences;
	int             *positive_occurrences;

    int             lower;
    int             higher;
    int             current_pointer;
public:
 
/********************** par_SolverState implementation ******/
    void initialize(int vars, int cl, int l);
    int unsolvedClauses() const;
    /* add clause, before adding, check unit conflict */
    bool addClause(int *ps);
    /**** branching strategy */
    int makeDecision() const;

    void copy(SatStateBase* org);
    void copyto(SatStateBase* org) const;
    void printSolution() const;
    void printState() const;
    void printInfo() const;

    int getVarSize() const 				{ return var_size; }
    int getClauseSize() const				{ return clause_size; }
    int getLevel() const				{ return level; }

    int assignedLit() const				{ return assigned_lit; }
    int &assignedLit()					{ return assigned_lit; }

    int clause(int i) const				{ return clauses[i]; }
    int &clause(int i)					{ return clauses[i]; }

    int occurrence(int i) const			{ return occurrences[i]; }
    int &occurrence(int i)				{ return occurrences[i]; }

    int positiveOccurrence(int i) const	{ return positive_occurrences[i]; }
    int &positiveOccurrence(int i)		{ return positive_occurrences[i]; }
private:
    SatStateBase(const SatStateBase&);
    SatStateBase& operator = (const SatStateBase&);
};

#endif
