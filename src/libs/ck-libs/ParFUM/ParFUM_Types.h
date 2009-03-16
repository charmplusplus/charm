#ifndef PARFUM_TYPES_H
#define PARFUM_TYPES_H

#include <pup.h>

// ghosts and spatial symmetries
#define FEM_Is_ghost_index(idx) ((idx)<-1)
#define FEM_To_ghost_index(idx) (-(idx)-2)
#define FEM_From_ghost_index(idx) (-(idx)-2)

	 
/** A reference to an element. Ghost indices are handled correctly here */
  class ElemID{
  public:
    ///type is negative for ghost elements
    int type;
    ///id refers to the index in the entity list
    int id;

    ///default constructor
    ElemID(){
      type=-1;
      id = -1;
    };
    ///constructor - initializer
    ElemID(int _type,int _id){
      if(_id < 0) {
    	  type = -(_type+1);
    	  id = FEM_To_ghost_index(_id);
      }
      else {
    	  type = _type;
    	  id = _id;
      }
    };
    bool operator ==(const ElemID &rhs)const {
      return (type == rhs.type) && (id == rhs.id);
    }
    bool operator < (const ElemID &rhs)const {
       return (type < rhs.type) || ( type == rhs.type && id < rhs.id);
     }
    const ElemID& operator =(const ElemID &rhs) {
      type = rhs.type;
      id = rhs.id;
      return *this;
    }
    virtual void pup(PUP::er &p){
      p | type;
      p | id;
    };

    static ElemID createNodeID(int type,int node){
      ElemID temp(type, node);
      return temp;
    }
    int getSignedId() {
      if(type<0){
    	  return FEM_From_ghost_index(id);
      }
      else return id;
    }
    int getSignedType(){
    	return type;
    }
    /** Return the element's type. This is necessary because the type member itself is negative for ghosts(for some stupid reason) */
    int getUnsignedType(){
    	if(type>=0)
    		return type;
    	else 
    		return -(type+1);
    }
     
    
  };



#endif
