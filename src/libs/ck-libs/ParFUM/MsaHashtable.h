#ifndef MSAHASHTABLE_H
#define MSAHASHTABLE_H

#include "ElemList.h"
#include "msa/msa.h"

class Hashnode{
public:
	class tupledata{
		public:
		enum {MAX_TUPLE = 8};
			int nodes[MAX_TUPLE];
			tupledata(int _nodes[MAX_TUPLE]){
				memcpy(nodes,_nodes,sizeof(int)*MAX_TUPLE);
			}
			tupledata(tupledata &rhs){
				memcpy(nodes,rhs.nodes,sizeof(int)*MAX_TUPLE);
			}
			tupledata(const tupledata &rhs){
				memcpy(nodes,rhs.nodes,sizeof(int)*MAX_TUPLE);
			}
			tupledata(){};
			//dont store the returned string
			char *toString(int numnodes,char *str){
				str[0]='\0';
				for(int i=0;i<numnodes;i++){
					sprintf(&str[strlen(str)],"%d ",nodes[i]);
				}
				return str;
			}
			inline int &operator[](int i){
				return nodes[i];
			}
			inline const int &operator[](int i) const {
				return nodes[i];
			}
			virtual void pup(PUP::er &p){
				p(nodes,MAX_TUPLE);
			}
	};
	int numnodes; //number of nodes in this tuple
	//TODO: replace *nodes with the above tupledata class
	tupledata nodes;	//the nodes in the tuple
	int chunk;		//the chunk number to which this element belongs
	int elementNo;		//local number of that element
	Hashnode(){
		numnodes=0;
	};
	Hashnode(int _num,int _chunk,int _elNo,int _nodes[tupledata::MAX_TUPLE]): nodes(_nodes){
		numnodes = _num;
		chunk = _chunk;
		elementNo = _elNo;
	}
	Hashnode(const Hashnode &rhs){
		*this = rhs;
	}
	inline Hashnode &operator=(const Hashnode &rhs){
		numnodes = rhs.numnodes;
		for(int i=0;i<numnodes;i++){
			nodes[i] = rhs.nodes[i];
		}
		chunk = rhs.chunk;
		elementNo = rhs.elementNo;
                return *this;
	}
	inline bool operator==(const Hashnode &rhs){
		if(numnodes != rhs.numnodes){
			return false;
		}
		for(int i=0;i<numnodes;i++){
			if(nodes[i] != rhs.nodes[i]){
				return false;
			}
		}
		if(chunk != rhs.chunk){
			return false;
		}
		if(elementNo != rhs.elementNo){
			return false;
		}
		return true;
	}
	inline bool operator>=(const Hashnode &rhs){
		if(numnodes < rhs.numnodes){
			return false;
		};
		if(numnodes > rhs.numnodes){
			return true;
		}

    for(int i=0;i<numnodes;i++){
      if(nodes[i] < rhs.nodes[i]){
	return false;
      }
      if(nodes[i] > rhs.nodes[i]){
	return true;
      }
    }
    if(chunk < rhs.chunk){
      return false;
    }
    if(chunk > rhs.chunk){
      return true;
    }
    if(elementNo < rhs.elementNo){
      return false;
    }
    if(elementNo > rhs.elementNo){
      return true;
    }
    return true;
  }

  inline bool operator<=(const Hashnode &rhs){
    if(numnodes < rhs.numnodes){
      return true;
    };
    if(numnodes > rhs.numnodes){
      return false;
    }

    for(int i=0;i<numnodes;i++){
      if(nodes[i] < rhs.nodes[i]){
	return true;
      }
      if(nodes[i] > rhs.nodes[i]){
	return false;
      }
    }
    if(chunk < rhs.chunk){
      return true;
    }
    if(chunk > rhs.chunk){
      return false;
    }
    if(elementNo < rhs.elementNo){
      return true;
    }
    if(elementNo > rhs.elementNo){
      return false;
    }
    return true;
  }

  inline bool equals(tupledata &tuple){
    for(int i=0;i<numnodes;i++){
      if(tuple.nodes[i] != nodes[i]){
	return false;
      }
    }
    return true;
  }
  virtual void pup(PUP::er &p){
    p | numnodes;
    p | nodes;
    p | chunk;
    p | elementNo;
  }
};

template <class T, bool PUP_EVERY_ELEMENT=true >
class DefaultListEntry {
    public:
    template<typename U>
    static inline void accumulate(T& a, const U& b) { a += b; }
    // identity for initializing at start of accumulate
    static inline T getIdentity() { return T(); }
    static inline bool pupEveryElement(){ return PUP_EVERY_ELEMENT; }
};

typedef UniqElemList<Hashnode> Hashtuple;
typedef MSA::MSA1D<Hashtuple,DefaultListEntry<Hashtuple,true>,MSA_DEFAULT_ENTRIES_PER_PAGE> MSA1DHASH;

class MsaHashtable
{
    MSA1DHASH msa;
    bool initHandleGiven;

public:
	class Read; class Add;
	friend class Read; friend class Add;
	Add getInitialAdd();
	void pup(PUP::er &p) { p|msa; }
	void enroll(int n) { msa.enroll(n); }

	class Read : private MSA::MSARead<MSA1DHASH>
	{
	public:
	    using MSA::MSARead<MSA1DHASH>::get;

		friend class MsaHashtable;
		friend class MsaHashtable::Add;
		void print();
		Add syncToAdd();

	private:
	Read(MSA1DHASH *m) : MSA::MSARead<MSA1DHASH>(m) { }
	};

	class Add : private MSA::MSAAccum<MSA1DHASH>
	{
	    using MSA::MSAAccum<MSA1DHASH>::accumulate;
	    friend class MsaHashtable;
	    friend class MsaHashtable::Read;
	Add(MSA1DHASH *m) : MSA::MSAAccum<MSA1DHASH>(m) { }
	public:
		int addTuple(int *tuple, int nodesPerTuple, int chunk, int elementNo);
		Read syncToRead();
	};


MsaHashtable(int _numSlots,int numWorkers)
    : msa(_numSlots, numWorkers), initHandleGiven(false) { }
	MsaHashtable(){};
};



#endif // MSAHASHTABLE_H
