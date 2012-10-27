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

// Some day, messages may be handled as something other than void* within the runtime.
// Prepare for that day.This also enhances readability
typedef void msg_t;

/**
 * Charm Message Queue: Holds msg pointers and returns the next message to execute on a PE
 *
 * Templated on the type of the priority field. Defaults to int priority.
 * All scheduling policies are encapsulated behind this queue interface.
 */
template <typename P = int>
class msgQ
{
    public:
        /// The datatype for msg priorities
        typedef P prio_t;

        /// Hardly any initialization required
        msgQ(): qSize(0) {}

        /// Given a message (optionally with a priority and queuing policy), enqueue it for delivery
        void enq(const msg_t *msg
                ,const prio_t &prio = prio_t()
                ,const bool isFifo = true
                );

        /// Pop (and return) the next message to deliver
        const msg_t* deq();

        /// Return ptr to message that is next in line for delivery. Does not deq() the msg
        const msg_t* front() const
        {
            if (prioQ.empty())
                return NULL;
            return msgbuckets[prioQ.top().second].front();
        }

        /// Number of messages in the queue
        inline size_t size() const { return qSize; }

        /// Is the queue empty?
        inline bool empty() const { return (0 == qSize); }

        /** Returns the value of the highest priority amongst all the messages in the queue
         *
         * @note: Depending on scheduling policy, this may or may not be the priority of the
         * next msg in line delivery. However, the default scheduling policy does return a msg
         * of this priority.
         */
        inline prio_t top_priority() const { return prioQ.top().first; }

        /// Just so that we can support CqsEnumerateQueue()
        void enumerate(msg_t **first, msg_t **last) const;

        /// An ostream operator overload, that currently just prints q size
        friend std::ostream& operator<< (std::ostream &out, const msgQ &q)
        {
            out <<"\nmsgQ[" << q.qSize << "]:";
            out<<"\n";
            return out;
        }

    private:
        /// Maintains the size of this message queue
        size_t qSize;

        /// Collection of msg buckets, each holding msgs of a given priority
        std::vector< std::deque<const msg_t*> > msgbuckets;

        /// The type of the index into the container of message buckets
        typedef short bktidx_t;
        /// A key-val pair of a priority value and the index to the bucket of msgs of that priority
        typedef typename std::pair<prio_t, bktidx_t> prioidx_t;

        /// A _min_ heap of distinct msg priorities along with the matching bucket indices
        std::priority_queue<prioidx_t, std::vector<prioidx_t>, std::greater<prioidx_t> > prioQ;

        /// A mapping between priority values and bucket indices, to locate buckets given a priority (used in enq)
        #if CMK_HAS_STD_UNORDERED_MAP
        std::unordered_map<prio_t, bktidx_t> prio2bktidx;
        #else
        std::map<prio_t, bktidx_t> prio2bktidx;
        #endif
};



template <typename P>
void msgQ<P>::enq(const msg_t *msg
                 ,const prio_t &prio
                 ,const bool isFifo
                 )
{
    // Find index of / create the bucket holding msgs of this priority
    typename std::map<prio_t, bktidx_t>::iterator itr = prio2bktidx.find(prio);
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
        prioQ.push( std::make_pair(prio, bktidx) );

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
    if (prioQ.empty())
        return NULL;

    // Get the index of the bucket holding the highest priority msgs
    const bktidx_t &bktidx = prioQ.top().second;
    std::deque<const msg_t*> &q = msgbuckets[bktidx];

    // Assert that there is at least one msg corresponding to this priority
    if (q.empty()) throw;
    const msg_t *msg = q.front();
    q.pop_front();
    // If all msgs of the highest priority have been consumed, pop that priority from the priority Q
    if (q.empty())
        prioQ.pop();
    // Decrement the total number of msgs in this container
    qSize--;
    return msg;
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

