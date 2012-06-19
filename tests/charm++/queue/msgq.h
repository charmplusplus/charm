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

typedef void msg_t;

template <typename P>
inline P defaultprio(P *dummy_tag) { return 0; }


template <typename P = int>
class msgQ
{
    public:
        typedef P prio_t;

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
        inline prio_t top_priority() const { return prios.top(); }

        ///
        friend std::ostream& operator<< (std::ostream &out, const msgQ &q)
        {
            out <<"\nmsgQ[" << q.qSize << "]:";
            out<<"\n";
            return out;
        }

    private:
        size_t qSize;
        #if CMK_HAS_STD_UNORDERED_MAP
        std::unordered_map<prio_t, std::deque<const msg_t*> > msgmap;
        #else
        std::map<prio_t, std::deque<const msg_t*> > msgmap;
        #endif
        std::priority_queue<prio_t, std::vector<prio_t>, std::greater<prio_t> > prios;
};



template <typename P>
void msgQ<P>::enq(const msg_t *msg
                 ,const prio_t &prio
                 ,const bool isFifo
                 )
{
    // Create or access deq holding msgs of this priority
    std::deque<const msg_t*> &q = msgmap[prio];
    // If this deq is empty, insert corresponding priority into prioQ
    if (q.empty())
        prios.push(prio);
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

    // Get the top priority in the priority queue
    const prio_t &prio = prios.top();

    std::deque<const msg_t*> &q = msgmap[prio];
    // Assert that there is at least one msg corresponding to this priority, in the msgmap
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
    prio_t &prio = prios.top();
    return msgmap[prio].front();
}

#endif // MSG_Q_H

