#ifndef _CKOBJID_H_
#define _CKOBJID_H_

union _ObjectID {
	struct {
		CkChareID id;
	} chare;
	struct {
		CkGroupID id;
		int onPE;
	} group; //also used for NodeGroups
	struct s_array{
		CkGroupID id; //array id
		CkArrayIndexBase idx; //index
	} array;
};

extern int totalCompares;

class CkObjID {
public:
	ChareType type;
	_ObjectID data;
	CkObjID(){
		type = TypeInvalid;
	};

	inline operator CkHashCode()const{
		CkHashCode ret=circleShift(type,25);
		switch(type){
		case TypeChare:
		case TypeMainChare:
		    ret += circleShift(data.chare.id.onPE,5);
		    ret += circleShift((CmiInt8)data.chare.id.objPtr,3);
		    break;
		case TypeGroup:
		case TypeNodeGroup:
		    ret += circleShift(data.group.onPE,4);
		    ret += circleShift(data.group.id.idx,6);
		    break;
		case TypeArray:
		    CkHashCode temp = data.array.idx.asChild().hash();
		    //ret = circleShift(ret,13);
		    //ret += circleShift(temp,11);
		    ret += temp;
		    break;
		}
		return ret;
	}
	
	inline bool operator == (const CkObjID &t) const{
		
		if(type != t.type){
			return false;
		}
		switch (type){
			case TypeChare:
				if((data.chare.id.onPE == t.data.chare.id.onPE) && (data.chare.id.objPtr == t.data.chare.id.objPtr)){
					return true;
				}else{
					return false;
				}
				//break; unreachable
			case TypeGroup:
			case TypeNodeGroup:
				if((data.group.onPE == t.data.group.onPE) && (data.group.id == t.data.group.id)){
					return true;
				}else{
					return false;
				}
				//break; unreachable
			case TypeArray:
				bool val;
				if(data.array.id == t.data.array.id && data.array.idx.asChild().compare(t.data.array.idx.asChild())){
					val = true;
				}else{
					val = false;
				}
				return val;
				// break; unreachable
		}
		return false;
	}
	
	void* getObject();

	int guessPE();
	
	char *toString(char *buf) const;

	inline void updatePosition(int PE);
};

PUPbytes(CkObjID)

typedef unsigned int MCount; //Message Count

#endif
