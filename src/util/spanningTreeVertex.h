#ifndef SPANNING_TREE_VERTEX
#define SPANNING_TREE_VERTEX

namespace topo {

/// Alias for the actual data type of a vertex id (PE/node number)
typedef int vtxType;

/**
 * Contains indices to direct children. childIndex[i]+1 and childIndex[i+1] are the first and
 * beyondLast indices of the sub-tree members of the child at childIndex[i].
 * @note: We're using a (vertex, edge) terminology to talk about spanning trees. Consciously staying
 * away from using "node" to avoid ambiguity with machine nodes and PEs. This is inspite of the fact
 * that typically a vertex of a spanning tree is a machine node / PE.
 */
class SpanningTreeVertex
{
    public:
        /// The id (PE) of the vertex
        vtxType id;
        /// The parent of this vertex. Uncomment if needed
        // vtxType parent;
        /// The machine coordinates of this vertex
        std::vector<int> X;
        /// Relative distance (in the container) from the position of this vertex to direct children (and their sub-tree members)
        std::vector<int> childIndex;
        /// Constructor
        SpanningTreeVertex(const vtxType _id=-1): id(_id) {}

    /// Overload == and < to keep users happy. Note: not member functions
    ///@{
    inline friend bool operator== (const SpanningTreeVertex &obj, const vtxType vtxID)
    { return (obj.id == vtxID); }

    inline friend bool operator== (const vtxType vtxID, const SpanningTreeVertex &obj)
    { return (obj.id == vtxID); }

    inline friend bool operator< (const SpanningTreeVertex &obj, const vtxType vtxID)
    { return (obj.id < vtxID); }

    inline friend bool operator< (const vtxType vtxID, const SpanningTreeVertex &obj)
    { return (vtxID < obj.id); }
    ///@}

    /// Stream inserter. Note: not a member function
    friend std::ostream& operator<< (std::ostream &out, const SpanningTreeVertex &obj)
    {
        out<<" "<<obj.id;
        if (obj.X.size()>0)
        {
            out<<"("<<obj.X[0];
            for (int i=1,cSize=obj.X.size(); i<cSize; i++)
                out<<","<<obj.X[i];
            out<<") ";
        }
        return out;
    }
};

inline int getProcID(const vtxType vtx) { return vtx; }
inline int getProcID(const SpanningTreeVertex &vtx) { return vtx.id; }

/// Return the number of hops (on the machine network) between two vertices in the tree
inline int numHops(const SpanningTreeVertex &vtx1, const SpanningTreeVertex &vtx2)
{
    /// Assert that the dimensions of the coordinate vectors of the two vertices are the same
    //CkAssert(vtx1.X.size() == vtx2.X.size());

    int nHops = 0;
    for (int i=0, nDims=vtx1.X.size(); i<nDims; i++)
        nHops += abs( vtx1.X[i] - vtx2.X[i] );
    return nHops;
}



/// Pick the vertex closes to the parent in the given range
template <typename Iterator>
inline Iterator pickClosest(const SpanningTreeVertex &parent, const Iterator start, const Iterator end)
{
    /// @todo: Static assert that Iterator::value_type == SpanningTreeVertex
    Iterator itr     = start;
    Iterator closest = itr++;
    int      minHops = numHops(parent,*closest); // aTopoMgr.getHopsBetweenRanks( parentPE, (*closest).id );
    /// Loop thro the range and identify the vertex closest to the parent
    for (; itr != end; itr++)
    {
        int hops = numHops(parent,*itr); //aTopoMgr.getHopsBetweenRanks( parentPE, (*itr).id );
        if (hops < minHops)
        {
            closest = itr;
            minHops = hops;
        }
    }
    return closest;
}

} // end namespace topo

#endif // SPANNING_TREE_VERTEX

