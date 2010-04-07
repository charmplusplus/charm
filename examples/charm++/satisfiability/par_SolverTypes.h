/***********************************************************************************[SolverTypes.h]
MiniSat -- Copyright (c) 2003-2006, Niklas Een, Niklas Sorensson

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
associated documentation files (the "Software"), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify, merge, publish, distribute,
sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or
substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT
OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
**************************************************************************************************/


#ifndef _PAR_SolverTypes_h
#define _PAR_SolverTypes_h

#include <cassert>
#include <stdint.h>


template<class V> 
int get_max_element(const V& ps) {

    int max_index = -1;
    int max = -1;
    for(int __i=0; __i<ps.size(); __i++)
    {
        if(ps[__i] > max)
        {
            max = ps[__i];
            max_index = __i;
        }
    }
    return max_index;
}


//=================================================================================================
// intiables, literals, lifted booleans, clauses:


// NOTE! intiables are just integers. No abstraction here. They should be chosen from 0..N,
// so that they can be used as array indices.

#define var_Undef (-1)


class par_Lit {
    int     x;
public:
    par_Lit() : x(var_Undef)                                              { }   // (lit_Undef)
    par_Lit(int var) {x = var;};
    //par_Lit(int var, bool sign)   {x=sign?var:-var; }

    // Don't use these for constructing/deconstructing literals. Use the normal constructors instead.
    friend int  toInt       (par_Lit p);  // Guarantees small, positive integers suitable for array indexing.
    friend par_Lit  par_toLit       (int i);  // Inverse of 'toInt()'
    friend par_Lit  operator   ~(par_Lit p);
    //friend bool sign        (Lit p);
    friend int  var         (par_Lit p);
    //friend Lit  unsign      (Lit p);
    //friend Lit  id          (Lit p, bool sgn);

    bool operator == (par_Lit p) const { return x == p.x; }
    bool operator != (par_Lit p) const { return x != p.x; }
    bool operator <  (par_Lit p) const { return x < p.x;  } // '<' guarantees that p, ~p are adjacent in the ordering.

    void pup(PUP::er &p) {
        p|x;
    }
};

inline  int  toInt       (par_Lit p)           { return p.x; }
inline  par_Lit  par_toLit       (int i)           { par_Lit p; p.x = i; return p; }
inline  par_Lit  operator   ~(par_Lit p)           { par_Lit q; q.x = -p.x; return q; }
//inline  bool sign        (Lit p)           { return p.x >1?true:false; }
inline  int  var         (par_Lit p)           { return p.x >> 1; }
//inline  Lit  unsign      (Lit p)           { Lit q; q.x = p.x & ~1; return q; }
//inline  Lit  id          (Lit p, bool sgn) { Lit q; q.x = p.x ^ (int)sgn; return q; }

//const Lit lit_Undef(var_Undef, false);  // }- Useful special constants.
//const Lit lit_Error(var_Undef, true );  // }


//=================================================================================================
// Lifted booleans:

/*
class lbool {
    char     value;
    explicit lbool(int v) : value(v) { }

public:
    lbool()       : value(0) { }
    lbool(bool x) : value((int)x*2-1) { }
    int toInt(void) const { return value; }

    bool  operator == (lbool b) const { return value == b.value; }
    bool  operator != (lbool b) const { return value != b.value; }
    lbool operator ^ (bool b) const { return b ? lbool(-value) : lbool(value); }

    friend int   toInt  (lbool l);
    friend lbool toLbool(int   v);
};
inline int   toInt  (lbool l) { return l.toInt(); }
inline lbool toLbool(int   v) { return lbool(v);  }

const lbool l_True  = toLbool( 1);
const lbool l_False = toLbool(-1);
const lbool l_Undef = toLbool( 0);
*/
//=================================================================================================
// Clause -- a simple class for representing a clause:


class par_Clause {
    CkVec<par_Lit>  data;

public:
   
     par_Clause(){data.removeAll();}
   
     par_Clause(const par_Clause& corg)
     {
         data.resize(corg.size());
        for(int _i=0; _i<data.size(); _i++)
        {
            data[_i] = corg[_i];
        }
     }

    // NOTE: This constructor cannot be used directly (doesn't allocate enough memory).
    template<class V>
    par_Clause(const V& ps, bool learnt) {
        data.resize(ps.size());
        for (int i = 0; i < ps.size(); i++) data[i] = ps[i];
    }
    
    template<class V>
    void attachdata(const V& ps, bool learnt)
    {
        data.resize(ps.size());
        for (int i = 0; i < ps.size(); i++) data[i] = ps[i];
    }

    void remove(int v){
        for(int i=0; i<data.size(); i++)
        {
            if(toInt(data[i]) == v)
            {
                data.remove(i);
                return;
            }
        }
    }
    // -- use this function instead:
    int          size        ()      const   { return data.size(); }
    //void         shrink      (int i)         { }
    //void         pop         ()              { shrink(1); }
    //bool         learnt      ()      const   {  }
    //uint32_t     mark        ()      const   { return 1; }
    //void         mark        (uint32_t m)    {  }
    //const Lit&   last        ()      const   { return data[size()-1]; }

    // NOTE: somewhat unsafe to change the clause in-place! Must manually call 'calcAbstraction' afterwards for
    //       subsumption operations to behave correctly.
    par_Lit&         operator [] (int i)         { return data[i]; }
    par_Lit          operator [] (int i) const   { return data[i]; }
    //operator const Lit* (void) const         { return data; }

    //float       activity    ()              { return 1; }
    //uint32_t     abstraction () const { return 1; }

    //Lit          subsumes    (const Clause& other) const;
    //void         strengthen  (Lit p);


    void        resize  (int __size)  {data.resize(__size); }
    void pup(PUP::er &p) {
        for(int i=0; i<size(); i++)
            p|data[i];
    }
};


/*_________________________________________________________________________________________________
|
|  subsumes : (other : const Clause&)  ->  Lit
|  
|  Description:
|       Checks if clause subsumes 'other', and at the same time, if it can be used to simplify 'other'
|       by subsumption resolution.
|  
|    Result:
|       lit_Error  - No subsumption or simplification
|       lit_Undef  - Clause subsumes 'other'
|       p          - The literal p can be deleted from 'other'
|________________________________________________________________________________________________@*/
/*inline Lit Clause::subsumes(const Clause& other) const
{
        return lit_Error;
}
*/
/*
inline void Clause::strengthen(Lit p)
{
    //remove(*this, p);
    //calcAbstraction();
}
*/
#endif
