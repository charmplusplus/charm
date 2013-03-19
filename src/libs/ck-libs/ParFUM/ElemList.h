#if !defined(ELEM_LIST_H)
#define ELEM_LIST_H

#include <cklists.h>

extern double elemlistaccTime;

template <class T>
class ElemList{
 public:
  CkVec<T> *vec;
  ElemList(){
    vec = new CkVec<T>();
  }
  ~ElemList(){
    delete vec;
  }
  ElemList(const ElemList &rhs){
    vec = new CkVec<T>();
    *this=rhs;
  }
  inline ElemList& operator=(const ElemList& rhs){
    //		 vec = new CkVec<T>();
    *vec = *(rhs.vec);
		return *this;
  }
  inline ElemList& operator+=(const ElemList& rhs){
    /*
      add the new unique elements to the List
    */
    double _start = CkWallTimer();
    for(int i=0;i<rhs.vec->length();i++){
      vec->push_back((*(rhs.vec))[i]);
    }
    //		uniquify();
    elemlistaccTime += (CkWallTimer() - _start);
    return *this;
  }
  ElemList(const T &val){
    vec =new CkVec<T>();
    vec->push_back(val);
  };
  inline virtual void pup(PUP::er &p){
    if(p.isUnpacking()){
      vec = new CkVec<T>();
    }
    pupCkVec(p,*vec);
  }
};

template <class T>
class UniqElemList: public ElemList<T>{
public:
  UniqElemList(const T &val):ElemList<T>(val){};
  UniqElemList():ElemList<T>(){};
  inline void uniquify(){
    CkVec<T> *lvec = this->vec;
    lvec->quickSort(8);
    if(lvec->length() != 0){
      int count=0;
      for(int i=1;i<lvec->length();i++){
	if((*lvec)[count] == (*lvec)[i]){
	}else{
	  count++;
	  if(i != count){
	    (*lvec)[count] = (*lvec)[i];
	  }
	}
      }
      lvec->resize(count+1);
    }
  }
};


#endif
