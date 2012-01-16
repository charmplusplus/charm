#ifndef __SEARCHENGINEAPI__
#define __SEARCHENGINEAPI__
#include "cmipool.h"
/*   framework for search engine */
#include "searchEngine.h"
#include "rng/rng.h"
#include "uts.h"

#include <vector>
extern int initial_grainsize;

class UtsStateBase : public StateBase
{
public:

    int type;          // distribution governing number of children
    int height;        // depth of this node in the tree
    int numChildren;   // number of children, -1 => not yet determined

    /* for RNG state associated with this node */
    struct state_t state;

	UtsStateBase()
	{
        type = -1;
        height = -1;
        numChildren = -1;    // not yet determined
    }

    void uts_initRoot(int type)
    {
        this->type = type;
        this->height = 0;
        this->numChildren = -1;      // means not yet determined
        rng_init(this->state.state, rootId);
    }

    int uts_numChildren_bin() {
        // distribution is identical everywhere below root
        int    v = rng_rand(state.state);	
        double d = rng_toProb(v);
        return (d < nonLeafProb) ? nonLeafBF : 0;
    }
    int uts_numChildren_geo() {
        double b_i = b_0;
        int depth = height;
        int __numChildren, h;
        double p, u;

        // use shape function to compute target b_i
        if (depth > 0){
            switch (shape_fn) {

                // expected size polynomial in depth
                case EXPDEC:
                    b_i = b_0 * pow((double) depth, -log(b_0)/log((double) gen_mx));
                    break;

                    // cyclic tree size
                    case CYCLIC:
                    if (depth > 5 * gen_mx){
                        b_i = 0.0;
                        break;
                    } 
                    b_i = pow(b_0, 
                        sin(2.0*3.141592653589793*(double) depth / (double) gen_mx));
                    break;

                    // identical distribution at all nodes up to max depth
                    case FIXED:
                    b_i = (depth < gen_mx)? b_0 : 0;
                    break;

                    // linear decrease in b_i
                    case LINEAR:
                    default:
                    b_i =  b_0 * (1.0 - (double)depth / (double) gen_mx);
                    break;
            }
        }

        // given target b_i, find prob p so expected value of 
        // // geometric distribution is b_i.
        p = 1.0 / (1.0 + b_i);

        // get uniform random number on [0,1)
        h = rng_rand(state.state);
        u = rng_toProb(h);

        // max number of children at this cumulative probability
        // // (from inverse geometric cumulative density function)
        __numChildren = (int) floor(log(1 - u) / log(1 - p)); 

        return __numChildren;
    }
    
    void uts_gennumChildren() {
        /* Determine the number of children */
        switch (type) {
        case BIN:
            if (height == 0)
                numChildren = (int) floor(b_0);
            else 
                numChildren = uts_numChildren_bin();
            break;

        case GEO:
            numChildren = uts_numChildren_geo();
            break;

        case HYBRID:
            if (height < shiftDepth * gen_mx)
                numChildren = uts_numChildren_geo();
            else
                numChildren = uts_numChildren_bin();
            break;
        case BALANCED:
            if (height < gen_mx)
                numChildren = (int) b_0;
            break;
        default:
            CkPrintf("parTreeSearch(): Unknown tree type");
        }

        // limit number of children
        // // only a BIN root can have more than MAXNUMCHILDREN
        if (height == 0 && type == BIN) {
            int rootBF = (int) ceil(b_0);
            if (numChildren > rootBF) {
                CkPrintf("*** Number of children of root truncated from %d to %d\n",
                    numChildren, rootBF);
                numChildren = rootBF;
            }
        }
        else if (type != BALANCED) {
            if (numChildren > MAXNUMCHILDREN) {
                CkPrintf("*** Number of children truncated from %d to %d\n", 
                    numChildren, MAXNUMCHILDREN);
                numChildren = MAXNUMCHILDREN;
            }
        }

    }

    int uts_childType() {
        switch (type) {
        case BIN:
            return BIN;
        case GEO:
            return GEO;
        case HYBRID:
            if (height < shiftDepth * gen_mx)
                return GEO;
            else 
                return BIN;
        case BALANCED:
            return BALANCED;
        default:
            CkPrintf("uts_get_childtype(): Unknown tree type");
            return -1;
        }
    }
};


    inline void createInitialChildren(Solver *solver)
    {
        UtsStateBase *root = (UtsStateBase*)solver->registerRootState(sizeof(UtsStateBase), 0, 1);
        root->type = type;
        root->height = 0;
        root->numChildren = -1;      // means not yet determined
        rng_init(root->state.state, rootId);
        root->uts_gennumChildren();
        if( root->numChildren==0)
            solver->reportSolution();
        solver->process(root);
    }

    inline void createChildren( StateBase *_base , Solver* solver, bool parallel)
    {
        UtsStateBase parent = *((UtsStateBase*)_base);
        int childIndex = 0;

        int parentHeight = parent.height;
        int numChildren, childType;

        numChildren = parent.numChildren;
        childType   = parent.uts_childType();

        int i, j;
        for (i = 0; i < numChildren; i++) {
            UtsStateBase *child  = (UtsStateBase*)solver->registerState(sizeof(UtsStateBase), i, numChildren);
            child->type = childType;
            child->height = parentHeight + 1;
            for (j = 0; j < computeGranularity; j++) {
                rng_spawn((parent.state).state, (child->state).state, i);
            }
            child->uts_gennumChildren(); 
            if(child->numChildren==0)
            {
                solver->reportSolution();
            }
            if(parallel)
                solver->process(child);
        }
    }

    inline double cost( )
    {
        return 0;
    }

    double heuristic( )
    {
        return 0;
    }

    double bound( int &l )
    {
        return 0;
    }

    inline bool isGoal(StateBase *s){
    
        return (((UtsStateBase*)s)->numChildren==0)?true:false; 
    }
    inline bool terminate(StateBase *s){
        return (((UtsStateBase*)s)->numChildren==0)?true:false; 
    }
    //Search Engine Option
    inline int parallelLevel()
    {
        return initial_grainsize;
    }
    inline int searchDepthLimit()
    {
        return 1;
    }
    int minimumLevel()
    {
        return 1;
    }

    int maximumLevel()
    {
        return  10;
    }
    inline void searchDepthChangeNotify( int ) {}

    SE_Register(UtsStateBase,createInitialChildren, createChildren, parallelLevel, searchDepthLimit);

/*
    void registerSE() {
        SE_register(createInitialChildren, createChildren, parallelLevel, searchDepthLimit);
    }
*/

#endif
