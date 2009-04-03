#ifndef _CKOBJID_H_
#define _CKOBJID_H_

typedef enum{
	TypeInvalid=0,
	TypeChare,
	TypeGroup,
	TypeNodeGroup,
	TypeArray
} _ObjectType;

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
		CkArrayIndexStruct idx; //index
	} array;
};

extern int totalCompares;

class CkObjID {
public:
	_ObjectType type;
	_ObjectID data;
	CkObjID(){
		type = TypeInvalid;
	};

	inline operator CkHashCode()const{
		CkHashCode ret=circleShift(type,25);
		switch(type){
			case TypeChare:
				ret += circleShift(data.chare.id.onPE,5);
				ret += circleShift((long long )data.chare.id.objPtr,3);
			break;
		case TypeGroup:
		case TypeNodeGroup:
			ret += circleShift(data.group.onPE,4);
			ret += circleShift(data.group.id.idx,6);
			break;
		case TypeArray:
			CkArrayIndex &i1= (CkArrayIndex &)data.array.idx.asMax();
			CkHashCode temp = i1.hash();
//			ret = circleShift(ret,13);
//			ret += circleShift(temp,11);
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
				break;
			case TypeGroup:
			case TypeNodeGroup:
				if((data.group.onPE == t.data.group.onPE) && (data.group.id == t.data.group.id)){
					return true;
				}else{
					return false;
				}
				break;
			case TypeArray:
				CkArrayIndex &i1= (CkArrayIndex &)data.array.idx.asMax();
				CkArrayIndex &i2 = (CkArrayIndex &)t.data.array.idx.asMax();
				bool val;
				if(data.array.id == t.data.array.id && i1.compare(i2)){
					val = true;
				}else{
					val = false;
				}
				return val;
				break;
		}
	}
	
	void* getObject();

	inline int guessPE();
	
	char *toString(char *buf) const;

	inline void updatePosition(int PE);
};

PUPbytes(CkObjID);

typedef unsigned int MCount; //Message Count

#endif
