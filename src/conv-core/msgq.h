#ifndef MSG_Q_H
#define MSG_Q_H

#include <deque>
#include <queue>
#include <ostream>
#include <functional>

#if CMK_HAS_STD_UNORDERED_MAP
#include <unordered_map>
#else
#include <map>
#endif

namespace conv {

// Some day, messages may be handled as something other than void* within the runtime.
// Prepare for that day, while enhancing readability today.
typedef void msg_t;

/**
 * Charm Message Queue: Holds msg pointers and returns the next message to execute on a PE
 *
 * Templated on the type of the priority field. Defaults to int priority.
 * All scheduling policies are encapsulated behind this queue interface.
 *
 * All messages of a given priority p are stored in a single container. Since
 * each message can be enqueued either to the front or back of this container,
 * a dequeue is used. Each such dequeue is referred to as a bucket.
 * The set of priority values of all the messages in the container is stored in
 * a min-heap. A deq() operation simply peeks at the most important prio
 * value, and finds the bucket associated with that value. It then dequeues the
 * message at the front of this bucket.
 * A mapping between the priority values and the corresponding buckets is
 * maintained. enq() operations simply find the bucket corresponding to a prio
 * value and place the msg into it.
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
        void enq(const msg_t *msg, const prio_t &prio = prio_t(), const bool isFifo = true);

        /// Pop (and return) the next message to deliver
        const msg_t* deq();

        /// Return ptr to message that is next in line for delivery. Does not deq() the msg
        inline const msg_t* front() const { return empty() ? NULL : prioQ.top().second->front(); }

        /// Number of messages in the queue
        inline size_t size() const { return qSize; }

        /// Is the queue empty?
        inline bool empty() const { return (0 == qSize); }

        /// Returns the value of the highest priority amongst all the messages in the queue
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
        /// Message bucket type
        typedef std::deque<const msg_t*> bkt_t;
        /// A key-val pair of a priority value and a handle to the bucket of msgs of that priority
        typedef typename std::pair<prio_t, bkt_t*> prioidx_t;
        /// A type for mapping between priority values and msg buckets
        #if CMK_HAS_STD_UNORDERED_MAP
        typedef typename std::unordered_map<prio_t, bkt_t> bktmap_t;
        #else
        typedef typename std::map<prio_t, bkt_t> bktmap_t;
        #endif

        /// The size of this message queue
        size_t qSize;
        /// Collection of msg buckets, each holding msgs of a given priority (maps priorities to buckets)
        bktmap_t msgbuckets;
        /// A _min_ heap of distinct msg priorities along with handles to matching buckets
        std::priority_queue<prioidx_t, std::vector<prioidx_t>, std::greater<prioidx_t> > prioQ;
};



template <typename P>
void msgQ<P>::enq(const msg_t *msg, const prio_t &prio, const bool isFifo)
{
    // Find / create the bucket holding msgs of this priority
    bkt_t &bkt = msgbuckets[prio];
    #if ! CMK_RANDOMIZED_MSGQ
    // If this deq is empty, insert corresponding priority into prioQ
    if (bkt.empty())
        prioQ.push( std::make_pair(prio, &bkt) );
    #endif
    // Enq msg either at front or back of deq
    isFifo ? bkt.push_back(msg) : bkt.push_front(msg);
    // Increment the total number of msgs in this container
    qSize++;
}



#if ! CMK_RANDOMIZED_MSGQ // charm NOT built with a randomized queue

// Iterative applications typically have a set of msg priority values that just
// repeat over time. However, the arrival pattern of messages at a given PE is
// unknown. Within a single iteration, the size of a bucket holding msgs of a
// given priority may fluctuate. It may become empty as these messages are
// delivered. Instead of deleting the bucket every time it becomes empty (which
// could be several times in a single iteration), we choose to leave it around
// because it will eventually get filled again. This avoids the costs of
// repeatedly constructing / destroying buckets (dequeues).
//
// However, for non-iterative applications, a given priority value may not
// recur later in the execution. Not deleting buckets may increase container
// sizes, degrading performance over time. Hence, buckets should be deleted
// whenever they become empty (or with a small timeout).
//
// The choice between the two execution modes should be a compile-time
// decision. Ideally, this should be a decision during application compilation
// and be specified programmatically by the application (think template
// parameter). However, since this Q is buried so deep within charm, the
// decision has to be made during charm compilation.
//
// Assume by default that the application is iterative
#define CMK_ITERATIVE_MSG_PRIOS 1

template <typename P>
const msg_t* msgQ<P>::deq()
{
    if (empty())
        return NULL;
    // Find the bucket holding the highest priority msgs
    bkt_t &bkt = * prioQ.top().second;
    // Pop msg from the front of the deque
    const msg_t *msg = bkt.front();
    bkt.pop_front();
    // If all msgs in the bucket have been consumed
    if (bkt.empty())
    {
        #if ! CMK_ITERATIVE_MSG_PRIOS
        // remove the empty bucket from the collection of buckets
        msgbuckets.erase( prioQ.top().first );
        #endif
        // pop corresponding priority from the priority Q
        prioQ.pop();
    }
    // Decrement the total number of msgs in this container
    qSize--;
    return msg;
}

#else // If charm is built with a randomized msg queue

/**
 * For detecting races, and for a general check that applications dont
 * depend on priorities or message ordering for correctness, charm can be built
 * with a randomized scheduler queue. In this case, this container's deq()
 * operation will not actually respect priorities. Instead it simply returns a
 * randomly selected msg.
 */
template <typename P>
const msg_t* msgQ<P>::deq()
{
    if (empty())
        return NULL;

    // Randomly select a bucket and a msg within it
    int rnd          = rand();
    typename bktmap_t::iterator itr = msgbuckets.begin();
    std::advance(itr, rnd % msgbuckets.size());
    bkt_t &bkt       = itr->second;
    int idx          = rnd % bkt.size();
    // Retrieve msg and remove it from bucket
    const msg_t *msg = bkt[idx];
    bkt.erase(bkt.begin() + idx);
    // Remove bucket if its empty
    if (bkt.empty())
        msgbuckets.erase(itr);
    // Decrement the total number of msgs in this container
    qSize--;
    return msg;
}

#endif // CMK_RANDOMIZED_MSGQ



template <typename P>
void msgQ<P>::enumerate(msg_t **first, msg_t **last) const
{
    if (first >= last)
        return;

    msg_t **ptr = first;
    for (typename bktmap_t::const_iterator bktitr = msgbuckets.begin();
            ptr != last && bktitr != msgbuckets.end(); bktitr++)
    {
        bkt_t::const_iterator msgitr = bktitr->second.begin();
        while (ptr != last && msgitr != bktitr->second.end())
            *ptr++ = (msg_t*) *msgitr++;
    }
}

} // end namespace conv

#endif // MSG_Q_H

