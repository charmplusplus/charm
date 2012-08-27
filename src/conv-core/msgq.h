#ifndef MSG_Q_H
#define MSG_Q_H

#include <deque>
#include <queue>
#include <ostream>

#if CMK_HAS_STD_UNORDERED_MAP
#include <unordered_map>
#else
#include <map>
#endif

namespace conv {

typedef void msg_t;

template <typename P>
inline P defaultprio(P *dummy_tag) { return 0; }


template <typename P = int>
class msgQ
{
    public:
        /// The datatype for msg priorities
        typedef P prio_t;
        /// The datatype of the index of the container storing msgs of a given priority
        typedef short bktidx_t;
        /// Yet another typedef. Just for terseness
        typedef typename std::pair<prio_t, bktidx_t> prioidx_t;

        ///
        msgQ(): qSize(0) {}
        ///
        void enq(const msg_t *msg
                ,const prio_t &prio = defaultprio<prio_t>(0)
                ,const bool isFifo = true
                );
        ///
        const msg_t* deq();
        ///
        const msg_t* front() const;
        ///
        inline size_t size() const { return qSize; }
        ///
        inline bool empty() const { return (0 == qSize); }
        ///
        inline prio_t top_priority() const { return prios.top().first; }

        ///
        void enumerate(msg_t **first, msg_t **last) const;
        ///
        friend std::ostream& operator<< (std::ostream &out, const msgQ &q)
        {
            out <<"\nmsgQ[" << q.qSize << "]:";
            out<<"\n";
            return out;
        }

    private:
        ///
        size_t qSize;
        /// Vector of msg buckets (each of them a deq)
        std::vector< std::deque<const msg_t*> > msgbuckets;
        /// A heap of distinct msg priorities
        std::priority_queue<prioidx_t, std::vector<prioidx_t>, std::greater<prioidx_t> > prios;
        /// A mapping between priority values and the bucket indices
        #if CMK_HAS_STD_UNORDERED_MAP
        std::unordered_map<prio_t, int> prio2bktidx;
        #else
        std::map<prio_t, int> prio2bktidx;
        #endif
};



template <typename P>
void msgQ<P>::enq(const msg_t *msg
                 ,const prio_t &prio
                 ,const bool isFifo
                 )
{
    // Find index of / create the bucket holding msgs of this priority
    typename std::map<prio_t, int>::iterator itr = prio2bktidx.find(prio);
    bktidx_t bktidx;
    if (prio2bktidx.end() != itr)
        bktidx = itr->second;
    else
    {
        msgbuckets.push_back( std::deque<const msg_t*>() );
        bktidx = msgbuckets.size() - 1;
        prio2bktidx[prio] = bktidx;
    }

    // Access deq holding msgs of this priority
    std::deque<const msg_t*> &q = msgbuckets[bktidx];
    // If this deq is empty, insert corresponding priority into prioQ
    if (q.empty())
        prios.push( std::make_pair(prio, bktidx) );

    // Enq msg either at front or back of deq
    if (isFifo)
        q.push_back(msg);
    else
        q.push_front(msg);
    // Increment the total number of msgs in this container
    qSize++;
}



template <typename P>
const msg_t* msgQ<P>::deq()
{
    if (prios.empty())
        return NULL;

    // Get the index of the bucket holding the highest priority msgs
    const bktidx_t &bktidx = prios.top().second;
    std::deque<const msg_t*> &q = msgbuckets[bktidx];

    // Assert that there is at least one msg corresponding to this priority
    if (q.empty()) throw;
    const msg_t *msg = q.front();
    q.pop_front();
    // If all msgs of the highest priority have been consumed, pop that priority from the priority Q
    if (q.empty())
        prios.pop();
    // Decrement the total number of msgs in this container
    qSize--;
    return msg;
}



template <typename P>
const msg_t* msgQ<P>::front() const
{
    if (prios.empty())
        return NULL;
    return msgbuckets[prios.top().second].front();
}



template <typename P>
void msgQ<P>::enumerate(msg_t **first, msg_t **last) const
{
    if (first >= last)
        return;

    msg_t **ptr = first;
    for (int i=0; ptr != last && i <= msgbuckets.size(); i++)
    {
        std::deque<const msg_t*>::const_iterator itr = msgbuckets[i].begin();
        while (ptr != last && itr != msgbuckets[i].end())
            *ptr++ = (msg_t*) *itr++;
    }
}

} // end namespace conv

#endif // MSG_Q_H

