#ifndef __AMPIOOQ_H_
#define __AMPIOOQ_H_

#include <string.h>
#include <charm++.h>

#define INITIAL_Q_SIZE 8

template <class T>
class AmpiNode {
public:
  T   m_data;
  int m_next;

  AmpiNode () {
    m_next  = -1;
  }
  
  void pup(PUP::er &p) {
    p|m_data;
    p|m_next;
  }
};

class Que {
public:
  int m_head;
  int m_tail;
  int m_size;

  Que () {
    m_head = -1;
    m_tail = -1;
    m_size = 0;
  }
  
  void pup(PUP::er &p) {
    p|m_head;
    p|m_tail;
    p|m_size;
  }
};

template <class T>
class AmpiOOQ {
  AmpiNode<T>*  m_list;        // list holding virtual queues data
  Que*          m_q;           // pointer to list of virtual queues
  int           m_numP;        // num of virtual queues
  int           m_totalNodes;  // total nodes in the list
  int           m_freeNode;    // index of head of free node list
  int           m_availNodes;  // free nodes

  /**
   * This function should be called when m_availNodes == 0.
   * It adds 'INITIAL_Q_SIZE' more nodes to existing queue.
   */
  void _expand (void) {
    AmpiNode<T>* list;

    list = new AmpiNode<T> [m_totalNodes + INITIAL_Q_SIZE];
    memcpy (list, m_list, sizeof (AmpiNode<T>)*m_totalNodes);

    delete [] m_list;
    m_list = list;

    m_freeNode = m_totalNodes;
    for (int i=m_totalNodes; i<m_totalNodes+INITIAL_Q_SIZE; i++) {
      m_list [i].m_next = i+1;
    }
    m_totalNodes += INITIAL_Q_SIZE;
    m_list [m_totalNodes-1].m_next = -1;
    m_availNodes = INITIAL_Q_SIZE;
  }

public:

  AmpiOOQ () {}
  
  void init (int numP) {
    m_numP       = numP;
    m_totalNodes = INITIAL_Q_SIZE;
    m_freeNode   = 0;
    m_availNodes = m_totalNodes;
    m_q          = new Que [numP];
    m_list       = new AmpiNode<T> [m_totalNodes];
    for (int i=0; i<m_totalNodes; i++) {
      m_list [i].m_next = i+1;
    }
    m_list [m_totalNodes-1].m_next = -1;
  }

  ~AmpiOOQ () {
    delete [] m_list;
    delete [] m_q;
  }
  
  void pup(PUP::er &p) {
    p|m_numP;
    p|m_totalNodes;
    p|m_freeNode;
    p|m_availNodes;
    
    if (p.isUnpacking()) {
      m_q    = new Que [m_numP];
      m_list = new AmpiNode<T> [m_totalNodes];
    }
    p(m_q,m_numP);
    p(m_list,m_totalNodes);
  }

  int length () { return (m_totalNodes - m_availNodes); }

  int length (int p) { return m_q[p].m_size; }

  int isEmpty (int p) { return (0 == m_q[p].m_size); }

  bool isEmpty () { return (m_totalNodes == m_availNodes); }

  T deq (int p) {
    if (-1 != m_q[p].m_head) {
      int index                     = m_q[p].m_head;
      T& ret                        = m_list [index].m_data;
      m_q[p].m_head                 = m_list [index].m_next;
      m_list [index].m_next         = m_freeNode;
      m_freeNode                    = index;
      m_q[p].m_size --;
      m_availNodes ++;
      if (-1 == m_q[p].m_head)
        m_q[p].m_tail = -1;
      return ret;
    } else return T ();
  }

  void insert (int p, int pos, const T& elt) {
    if (-1 == m_freeNode) _expand ();

    if ((0 == m_q[p].m_size) || (pos == m_q[p].m_size)) {
      enq (p, elt);
    } else {
      int index = m_freeNode;

      m_list [index].m_data = elt;
      m_availNodes --;
      m_freeNode = m_list [index].m_next;
      m_q[p].m_size ++;

      // insert the message at proper position
      if (0 == pos) {
        // insert before the current head of queue
        m_list [index].m_next = m_q[p].m_head;
        m_q[p].m_head = index;
      } else {
        // find the position between head and tail
        int curr = m_q[p].m_head;
        int next = m_list[curr].m_next;

        for (int i=0; i<pos-1; i++) {
          curr = next;
          next = m_list[curr].m_next;
        }

        m_list [curr].m_next = index;
        m_list [index].m_next = next;
      }
    }
  }

  void enq (int p, const T& elt) {
    if (-1 == m_freeNode) _expand ();

    m_list [m_freeNode].m_data = elt;
    m_availNodes --;
    m_q[p].m_size ++;
    if (-1 != m_q[p].m_tail) {
      m_list [m_q[p].m_tail].m_next = m_freeNode;
    } else {
      m_q[p].m_head = m_freeNode;
    }
    m_q[p].m_tail = m_freeNode;
    m_freeNode = m_list [m_freeNode].m_next;
    m_list [m_q[p].m_tail].m_next = -1;
  }

  T peek (int p, int pos) {
    int index = m_q[p].m_head;

    if (pos >= m_q[p].m_size)
      return T();
    else {
      for (int i=0; i<pos; i++) {
        index = m_list [index].m_next;
      }
      return m_list [index].m_data;
    }
  }
};
#endif

