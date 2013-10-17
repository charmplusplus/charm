
/*
Finite Element Method (FEM) Framework for Charm++
Parallel Programming Lab, Univ. of Illinois 2002
Orion Sky Lawlor, olawlor@acm.org, 12/20/2002

FEM Implementation file: mesh creation and user-data manipulation.
*/
#include <assert.h>
#include "fem.h"
#include "fem_impl.h"
#include "charm-api.h" /*for CDECL, FTN_NAME*/
#include "fem_mesh_modify.h"

extern int femVersion;

/*** IDXL Interface ****/
FEM_Comm_Holder::FEM_Comm_Holder(FEM_Comm *sendComm, FEM_Comm *recvComm)
	:comm(sendComm,recvComm)
{
	registered=false;
	idx=-1; 
}
void FEM_Comm_Holder::registerIdx(IDXL_Chunk *c) {
	assert(!registered);
	IDXL_Chunk *owner=c;
	if (idx!=-1) // had an old index: try to get it back
		idx=owner->addStatic(&comm,idx);
	else //No old index:
		idx=owner->addStatic(&comm);
}
void FEM_Comm_Holder::pup(PUP::er &p) {
	p|idx;
	if (p.isUnpacking() && idx!=-1) 
	{ // Try to grab the same index we had on our old processor:
		registerIdx(IDXL_Chunk::get("FEM_Comm_Holder::pup"));
	}
}

FEM_Comm_Holder::~FEM_Comm_Holder(void)
{
	if (registered) 
	{ // Try to unregister from IDXL:
		const char *caller="FEM_Comm_Holder::~FEM_Comm_Holder";
		IDXL_Chunk *owner=IDXL_Chunk::getNULL();
		if (owner) owner->destroy(idx,caller); 
	}
}

/******* FEM_Mesh API ******/

static void checkIsSet(int fem_mesh,bool wantSet,const char *caller) {
	if (FEM_Mesh_is_set(fem_mesh)!=wantSet) {
		const char *msg=wantSet?"This mesh (%d) is not a setting mesh":
			"This mesh (%d) is not a getting mesh";
		FEM_Abort(caller,msg,fem_mesh);
	}
}

/* Connectivity: Map calls to appropriate version of FEM_Mesh_data */
CDECL void 
FEM_Mesh_conn(int fem_mesh,int entity,
  	int *conn, int firstItem, int length, int width) 
{
	FEM_Mesh_data(fem_mesh,entity,FEM_CONN, conn, firstItem,length, FEM_INDEX_0, width);
}
FDECL void 
FTN_NAME(FEM_MESH_CONN,fem_mesh_conn)(int *fem_mesh,int *entity,
  	int *conn, int *firstItem,int *length, int *width)
{
	//Can't just call the C version of this routine, because we use 1-based indices:
	FEM_Mesh_data(*fem_mesh,*entity,FEM_CONN, conn, *firstItem-1,*length, FEM_INDEX_1, *width);
}

CDECL void
FEM_Mesh_set_conn(int fem_mesh,int entity,
  	const int *conn, int firstItem,int length, int width)
{
	checkIsSet(fem_mesh,true,"FEM_Mesh_set_conn");
	FEM_Mesh_conn(fem_mesh,entity,(int *)conn,firstItem,length,width);
}
FDECL void
FTN_NAME(FEM_MESH_SET_CONN,fem_mesh_set_conn)(int *fem_mesh,int *entity,
  	const int *conn, int *firstItem,int *length, int *width)
{
	checkIsSet(*fem_mesh,true,"fem_mesh_set_conn");
	FTN_NAME(FEM_MESH_CONN,fem_mesh_conn)(fem_mesh,entity,(int *)conn,firstItem,length,width);
}

CDECL void
FEM_Mesh_get_conn(int fem_mesh,int entity,
  	int *conn, int firstItem,int length, int width)
{
	checkIsSet(fem_mesh,false,"FEM_Mesh_get_conn");
	FEM_Mesh_conn(fem_mesh,entity,conn,firstItem,length,width);
}
FDECL void
FTN_NAME(FEM_MESH_GET_CONN,fem_mesh_get_conn)(int *fem_mesh,int *entity,
  	int *conn, int *firstItem,int *length, int *width)
{
	checkIsSet(*fem_mesh,false,"fem_mesh_get_conn");
	FTN_NAME(FEM_MESH_CONN,fem_mesh_conn)(fem_mesh,entity,conn,firstItem,length,width);
}


/* Data: map to FEM_Mesh_offset */
CDECL void
FEM_Mesh_data(int fem_mesh,int entity,int attr, 	
  	void *data, int firstItem,int length, int datatype,int width)
{
	IDXL_Layout lo(datatype,width);
	FEM_Mesh_data_layout(fem_mesh,entity,attr,data,firstItem,length,lo);
}

FORTRAN_AS_C(FEM_MESH_DATA,FEM_Mesh_data,fem_mesh_data,
	(int *fem_mesh,int *entity,int *attr,void *data,int *firstItem,int *length,int *datatype,int *width),
	(*fem_mesh,*entity,*attr,data,*firstItem-1,*length,*datatype,*width)
)


CDECL void
FEM_Mesh_set_data(int fem_mesh,int entity,int attr, 	
  	const void *data, int firstItem,int length, int datatype,int width)
{
	checkIsSet(fem_mesh,true,"FEM_Mesh_set_data");
	FEM_Mesh_data(fem_mesh,entity,attr,(void *)data,firstItem,length,datatype,width);
}
FORTRAN_AS_C(FEM_MESH_SET_DATA,FEM_Mesh_set_data,fem_mesh_set_data,
	(int *fem_mesh,int *entity,int *attr,void *data,int *firstItem,int *length,int *datatype,int *width),
	(*fem_mesh,*entity,*attr,data,*firstItem-1,*length,*datatype,*width)
)

CDECL void
FEM_Mesh_get_data(int fem_mesh,int entity,int attr, 	
  	void *data, int firstItem,int length, int datatype,int width)
{
	checkIsSet(fem_mesh,false,"FEM_Mesh_get_data");
	FEM_Mesh_data(fem_mesh,entity,attr,data,firstItem,length,datatype,width);
}
FORTRAN_AS_C(FEM_MESH_GET_DATA,FEM_Mesh_get_data,fem_mesh_get_data,
	(int *fem_mesh,int *entity,int *attr,void *data,int *firstItem,int *length,int *datatype,int *width),
	(*fem_mesh,*entity,*attr,data,*firstItem-1,*length,*datatype,*width)
)

CDECL void
FEM_Mesh_data_layout(int fem_mesh,int entity,int attr, 	
  	void *data, int firstItem,int length, IDXL_Layout_t layout)
{
	const char *caller="FEM_Mesh_data_layout";
	FEM_Mesh_data_layout(fem_mesh,entity,attr,data,firstItem,length,
		IDXL_Layout_List::get().get(layout,caller));
}

FORTRAN_AS_C(FEM_MESH_DATA_LAYOUT,FEM_Mesh_data_layout,fem_mesh_data_layout,
	(int *fem_mesh,int *entity,int *attr,void *data,int *firstItem,int *length,int *layout),
	(*fem_mesh,*entity,*attr,data,*firstItem-1,*length,*layout)
)

CDECL void
FEM_Mesh_data_offset(int fem_mesh,int entity,int attr, 	
  	void *data, int firstItem,int length,
	int type,int width, int offsetBytes,int distanceBytes,int skewBytes)
{
	const char *caller="FEM_Mesh_data_offset";
	FEM_Mesh_data_layout(fem_mesh,entity,attr,data,firstItem,length,
		IDXL_Layout(type,width,offsetBytes,distanceBytes,skewBytes));
}
FORTRAN_AS_C(FEM_MESH_DATA_OFFSET,FEM_Mesh_data_offset,fem_mesh_data_offset,
	(int *fem_mesh,int *entity,int *attr,
	 void *data,int *firstItem,int *length,
	 int *type,int *width,int *offset,int *distance,int *skew),
	(*fem_mesh,*entity,*attr,
	 data,*firstItem-1,*length,
	 *type,*width,*offset,*distance,*skew)
)


void FEM_Register_array(int fem_mesh,int entity,int attr,
	void *data, int datatype,int width,int firstItem){
	IDXL_Layout lo(datatype,width);
/*	if(attr == FEM_CONN){
		printf("CONN width %d \n",width);
		int len = FEM_Mesh_get_length(fem_mesh,entity);
		int *connd = (int *)data;
		for(int i=0;i<len;i++){
			printf("%d -> (%d %d %d) \n",i+1,connd[3*i],connd[3*i+1],connd[3*i+2]);
		}
	}
	printf("firstItem %d \n",firstItem);*/
	FEM_Register_array_layout(fem_mesh,entity,attr,data,firstItem,lo);
}

void FEM_Register_array_layout(int fem_mesh,int entity,int attr, 	
  	void *data, IDXL_Layout_t layout,int firstItem){
	const char *caller="FEM_Register_array_layout";
	FEM_Register_array_layout(fem_mesh,entity,attr,data,firstItem, 
		IDXL_Layout_List::get().get(layout,caller));

}

/*registration api */
CDECL void 
FEM_Register_array(int fem_mesh,int entity,int attr,
	void *data, int datatype,int width)
{	
	FEM_Register_array(fem_mesh,entity,attr,data,datatype,width,0);
}

CDECL void
FEM_Register_array_layout(int fem_mesh,int entity,int attr, 	
  	void *data, IDXL_Layout_t layout){
	FEM_Register_array_layout(fem_mesh,entity,attr,data,layout,0);
}


CDECL void 
FEM_Register_entity(int fem_mesh,int entity,void *data,
		int len,int max,FEM_Mesh_alloc_fn fn) {
		FEM_Register_entity_impl(fem_mesh,entity,data,len,max,fn);
}

/**TODO: add the fortran api for registration*/

FORTRAN_AS_C(FEM_REGISTER_ARRAY,FEM_Register_array,fem_register_array,
	(int *fem_mesh,int *entity,int *attr,void *data,int *datatype,int *width),(*fem_mesh,*entity,*attr,data,*datatype,*width,0))


FORTRAN_AS_C(FEM_REGISTER_ARRAY_LAYOUT,FEM_Register_array_layout,fem_register_array_layout,
	(int *fem_mesh,int *entity,int *attr,void *data,int *layout),(*fem_mesh,*entity,*attr,data,*layout,0))

FORTRAN_AS_C(FEM_REGISTER_ENTITY,FEM_Register_entity,fem_register_entity,
	(int *fem_mesh,int *entity,void *data,int *len,int *max,FEM_Mesh_alloc_fn fn),(*fem_mesh,*entity,data,*len,*max,fn))


// User data API:
CDECL void 
FEM_Mesh_pup(int fem_mesh,int dataTag,FEM_Userdata_fn fn,void *data) {
	const char *caller="FEM_Mesh_pup"; FEMAPI(caller);
	FEM_Mesh *m=FEM_Mesh_lookup(fem_mesh,caller);
	FEM_Userdata_item &i=m->udata.find(dataTag);
	FEM_Userdata_pupfn f(fn,data);
	if (m->isSetting()) i.store(f);
	else /* m->isGetting() */ {
		if (!i.hasStored())
			FEM_Abort(caller,"Never stored any user data at tag %d",dataTag);
		i.restore(f);
	}
}
FORTRAN_AS_C(FEM_MESH_PUP,FEM_Mesh_pup,fem_mesh_pup,
	(int *m,int *t,FEM_Userdata_fn fn,void *data), (*m,*t,fn,data))

// Accessor API:
CDECL void 
FEM_Mesh_set_length(int fem_mesh,int entity,int newLength) {
	const char *caller="FEM_Mesh_set_length"; FEMAPI(caller);
	checkIsSet(fem_mesh,true,caller);
	FEM_Entity_lookup(fem_mesh,entity,caller)->setLength(newLength);
}
FORTRAN_AS_C(FEM_MESH_SET_LENGTH,FEM_Mesh_set_length,fem_mesh_set_length,
	(int *fem_mesh,int *entity,int *newLength),
	(*fem_mesh,*entity,*newLength)
)


CDECL int 
FEM_Mesh_get_length(int fem_mesh,int entity) {
	const char *caller="FEM_Mesh_get_length"; FEMAPI(caller);
	int len=FEM_Entity_lookup(fem_mesh,entity,caller)->size();
	return len;
}
FORTRAN_AS_C_RETURN(int,
	FEM_MESH_GET_LENGTH,FEM_Mesh_get_length,fem_mesh_get_length,
	(int *fem_mesh,int *entity),(*fem_mesh,*entity)
)


CDECL void 
FEM_Mesh_set_width(int fem_mesh,int entity,int attr,int newWidth) {
	const char *caller="FEM_Mesh_set_width";
	FEMAPI(caller);
	checkIsSet(fem_mesh,true,caller);
	FEM_Attribute_lookup(fem_mesh,entity,attr,caller)->setWidth(newWidth,caller);
}
FORTRAN_AS_C(FEM_MESH_SET_WIDTH,FEM_Mesh_set_width,fem_mesh_set_width,
	(int *fem_mesh,int *entity,int *attr,int *newWidth),
	(*fem_mesh,*entity,*attr,*newWidth)
)

CDECL int 
FEM_Mesh_get_width(int fem_mesh,int entity,int attr) {
	const char *caller="FEM_Mesh_get_width";
	FEMAPI(caller);
	return FEM_Attribute_lookup(fem_mesh,entity,attr,caller)->getWidth();
}
FORTRAN_AS_C_RETURN(int,
	FEM_MESH_GET_WIDTH,FEM_Mesh_get_width,fem_mesh_get_width,
	(int *fem_mesh,int *entity,int *attr),(*fem_mesh,*entity,*attr)
)

CDECL int 
FEM_Mesh_get_datatype(int fem_mesh,int entity,int attr) {
	const char *caller="FEM_Mesh_get_datatype";
	FEMAPI(caller);
	return FEM_Attribute_lookup(fem_mesh,entity,attr,caller)->getDatatype();
}
FORTRAN_AS_C_RETURN(int,
	FEM_MESH_GET_DATATYPE,FEM_Mesh_get_datatype,fem_mesh_get_datatype,
	(int *fem_mesh,int *entity,int *attr),(*fem_mesh,*entity,*attr)
)

CDECL int 
FEM_Mesh_is_set(int fem_mesh) /* return 1 if this is a writing mesh */
{
	return (FEM_Mesh_lookup(fem_mesh,"FEM_Mesh_is_get")->isSetting())?1:0;
}
FORTRAN_AS_C_RETURN(int,
	FEM_MESH_IS_SET,FEM_Mesh_is_set,fem_mesh_is_set,
	(int *fem_mesh),(*fem_mesh)
)

CDECL int 
FEM_Mesh_is_get(int fem_mesh) /* return 1 if this is a readable mesh */
{
	return (!FEM_Mesh_lookup(fem_mesh,"FEM_Mesh_is_get")->isSetting())?1:0;
}
FORTRAN_AS_C_RETURN(int,
	FEM_MESH_IS_GET,FEM_Mesh_is_get,fem_mesh_is_get,
	(int *fem_mesh),(*fem_mesh)
)

CDECL void 
FEM_Mesh_become_get(int fem_mesh) /* Make this a readable mesh */
{ FEM_Mesh_lookup(fem_mesh,"FEM_Mesh_become_get")->becomeGetting(); }
FORTRAN_AS_C(FEM_MESH_BECOME_GET,FEM_Mesh_become_get,fem_mesh_become_get, (int *m),(*m))

CDECL void 
FEM_Mesh_become_set(int fem_mesh)
{ FEM_Mesh_lookup(fem_mesh,"FEM_Mesh_become_get")->becomeSetting(); }
FORTRAN_AS_C(FEM_MESH_BECOME_SET,FEM_Mesh_become_set,fem_mesh_become_set, (int *m),(*m))


CDECL IDXL_t 
FEM_Comm_shared(int fem_mesh,int entity) {
	const char *caller="FEM_Comm_shared";
	FEMAPI(caller); 
	if (entity!=FEM_NODE) FEM_Abort(caller,"Only shared nodes supported");
	return FEM_Mesh_lookup(fem_mesh,caller)->node.
		sharedIDXL.getIndex(IDXL_Chunk::get(caller));
}
FORTRAN_AS_C_RETURN(int,
	FEM_COMM_SHARED,FEM_Comm_shared,fem_comm_shared,
	(int *fem_mesh,int *entity),(*fem_mesh,*entity)
)

CDECL IDXL_t 
FEM_Comm_ghost(int fem_mesh,int entity) {
	const char *caller="FEM_Comm_ghost";
	FEMAPI(caller);
	FEM_Entity *e=FEM_Entity_lookup(fem_mesh,entity,caller);
	if (e->isGhost()) FEM_Abort(caller,"Can only call FEM_Comm_ghost on real entity type");
	return e->ghostIDXL.getIndex(IDXL_Chunk::get(caller));
}
FORTRAN_AS_C_RETURN(int,
	FEM_COMM_GHOST,FEM_Comm_ghost,fem_comm_ghost,
	(int *fem_mesh,int *entity),(*fem_mesh,*entity)
)


// Internal API:
void FEM_Mesh_data_layout(int fem_mesh,int entity,int attr, 	
  	void *data, int firstItem,int length, const IDXL_Layout &layout) 
{
	if (femVersion == 0 && length==0) return;
	const char *caller="FEM_Mesh_data";
	FEMAPI(caller);
	FEM_Mesh *m=FEM_Mesh_lookup(fem_mesh,caller);
	FEM_Attribute *a=m->lookup(entity,caller)->lookup(attr,caller);
	
	if (m->isSetting()) 
		a->set(data,firstItem,length,layout,caller);
	else /* m->isGetting()*/
		a->get(data,firstItem,length,layout,caller);
}

/** the internal registration function */
void FEM_Register_array_layout(int fem_mesh,int entity,int attr,void *data,int firstItem,const IDXL_Layout &layout){
	const char *caller="FEM_Register_array";
	FEMAPI(caller);
	FEM_Mesh *m=FEM_Mesh_lookup(fem_mesh,caller);
	FEM_Entity *e = m->lookup(entity,caller);
	int length = e->size();
	//should actually be a call on the entity
	int max = e->getMax();
	FEM_Attribute *a = e->lookup(attr,caller);
	
	
	if(m->isSetting()){
	}else{
		a->get(data,firstItem,length,layout,caller);
	}
	//replace the attribute's data array with the user's data
	a->register_data(data,length,max,layout,caller);
}
void FEM_Register_entity_impl(int fem_mesh,int entity,void *args,int len,int max,FEM_Mesh_alloc_fn fn){
	const char *caller = "FEM_Register_entity";
	FEM_Mesh *m=FEM_Mesh_lookup(fem_mesh,caller);
/*	if(!m->isSetting()){
		CmiAbort("Register entity called on mesh that can't be written into");
	}
*/
	FEM_Entity *e = m->lookup(entity,caller);
	e->setMaxLength(len,max,args,fn);
}

FEM_Entity *FEM_Entity_lookup(int fem_mesh,int entity,const char *caller) {
	return FEM_Mesh_lookup(fem_mesh,caller)->lookup(entity,caller);
}
FEM_Attribute *FEM_Attribute_lookup(int fem_mesh,int entity,int attr,const char *caller) {
	return FEM_Entity_lookup(fem_mesh,entity,caller)->lookup(attr,caller);
}

CDECL int FEM_Mesh_get_entities(int fem_mesh, int *entities) {
	const char *caller="FEM_Mesh_get_entities";
	FEMAPI(caller); 
	return FEM_Mesh_lookup(fem_mesh,caller)->getEntities(entities);
}
FORTRAN_AS_C_RETURN(int,
	FEM_MESH_GET_ENTITIES,FEM_Mesh_get_entities,fem_mesh_get_entities,
	(int *mesh,int *ent), (*mesh,ent)
)

CDECL int FEM_Mesh_get_attributes(int fem_mesh,int entity,int *attributes) {
	const char *caller="FEM_Mesh_get_attributes";
	FEMAPI(caller);
	return FEM_Entity_lookup(fem_mesh,entity,caller)->getAttrs(attributes);
}
FORTRAN_AS_C_RETURN(int,
	FEM_MESH_GET_ATTRIBUTES,FEM_Mesh_get_attributes,fem_mesh_get_attributes,
	(int *mesh,int *ent,int *attrs), (*mesh,*ent,attrs)
)

/************** FEM_Attribute ****************/

CDECL const char *FEM_Get_datatype_name(int datatype,char *storage) {
	switch(datatype) {
	case FEM_BYTE: return "FEM_BYTE";
	case FEM_INT: return "FEM_INT";
	case FEM_FLOAT: return "FEM_FLOAT";
	case FEM_DOUBLE: return "FEM_DOUBLE";
	case FEM_INDEX_0: return "FEM_INDEX_0";
	case FEM_INDEX_1: return "FEM_INDEX_1";
	};
	sprintf(storage,"unknown datatype code (%d)",datatype);
	return storage;
}

/// Return the human-readable version of this FEM_ATTR code.
///  For example, FEM_attr2name(FEM_CONN)=="FEM_CONN".
CDECL const char *FEM_Get_attr_name(int attr,char *storage) 
{
	if (attr<FEM_ATTRIB_TAG_MAX) 
	{ //It's a user tag:
		sprintf(storage,"FEM_DATA+%d",attr-FEM_DATA);
		return storage;
	}
	switch(attr) {
	case FEM_CONN: return "FEM_CONN"; 
	case FEM_SPARSE_ELEM: return "FEM_SPARSE_ELEM";
	case FEM_COOR: return "FEM_COOR";
	case FEM_GLOBALNO: return "FEM_GLOBALNO";
	case FEM_PARTITION: return "FEM_PARTITION";
	case FEM_SYMMETRIES: return "FEM_SYMMETRIES";
	case FEM_NODE_PRIMARY: return "FEM_NODE_PRIMARY";
	case FEM_NODE_ELEM_ADJACENCY: return "FEM_NODE_ELEM_ADJACENCY";
	case FEM_NODE_NODE_ADJACENCY: return "FEM_NODE_NODE_ADJACENCY";
	case FEM_ELEM_ELEM_ADJACENCY: return "FEM_ELEM_ELEM_ADJACENCY";
	case FEM_ELEM_ELEM_ADJ_TYPES: return "FEM_ELEM_ELEM_ADJ_TYPES";
	case FEM_IS_VALID_ATTR: return "FEM_IS_VALID_ATTR";
	case FEM_MESH_SIZING: return "FEM_MESH_SIZING";

	default: break;
	};
	sprintf(storage,"unknown attribute code (%d)",attr);
	return storage;
}

//Abort with a nice error message saying: 
// Our <field> was previously set to <cur>; it cannot now be <operation> <next>
void FEM_Attribute::bad(const char *field,bool forRead,int cur,int next,const char *caller) const
{
	char nameStorage[256];
	const char *name=FEM_Get_attr_name(attr,nameStorage);
	char errBuf[1024];
	const char *cannotBe=NULL;
	if (forRead) {
		if (cur==-1) {
			sprintf(errBuf,"The %s %s %s was never set-- it cannot now be read",
				e->getName(),name,field);
		}
		else /* already had a value */
			cannotBe="read as";
	}
	else /* for write */ {
		cannotBe="set to";
	}
	if (cannotBe!=NULL) /* Use standard ... <something> cannot be <something>'d... error message */
		sprintf(errBuf,"The %s %s %s was previously set to %d; it cannot now be %s %d",
			e->getName(),name,field,cur,cannotBe,next);
	
	FEM_Abort(caller,errBuf);
}


FEM_Attribute::FEM_Attribute(FEM_Entity *e_,int attr_)
		:e(e_),ghost(0),attr(attr_),width(0),datatype(-1), allocated(false)
{
	tryAllocate();
	if (femVersion == 0) width=-1;
}
void FEM_Attribute::pup(PUP::er &p) {
	// e, attr, and ghost are always set by the constructor
	p|width;
	if (p.isUnpacking() && femVersion > 0 && width<0)  width=0;
	p|datatype;
	if (p.isUnpacking()) tryAllocate();
}
void FEM_Attribute::pupSingle(PUP::er &p, int pupindx) {
	// e, attr, and ghost are always set by the constructor
	p|width;
	if (p.isUnpacking() && femVersion > 0 && width<0)  width=0;
	p|datatype;
	if (p.isUnpacking()) tryAllocate();
}
FEM_Attribute::~FEM_Attribute() {}

void FEM_Attribute::setLength(int next,const char *caller) {
	int cur=getLength();
	if (next==cur) return; //Already set--nothing to do 
	if (cur>0) bad("length",false,cur,next, caller);
	e->setLength(next);
	tryAllocate();
}
	
void FEM_Attribute::setWidth(int next,const char *caller) {
	int cur=getWidth();
	if (next==cur) return; //Already set--nothing to do 
	if (cur>0) bad("width",false,cur,next, caller);
	width=next;
	tryAllocate();
	if (ghost) ghost->setWidth(width,caller);
}

void FEM_Attribute::setDatatype(int next,const char *caller) {
	int cur=getDatatype();
	if (next==cur) return; //Already set--nothing to do 
	if (cur!=-1) bad("datatype",false,cur,next, caller);
	datatype=next;
	tryAllocate();
	if (ghost) ghost->setDatatype(datatype,caller);
}

void FEM_Attribute::copyShape(const FEM_Attribute &src) {
	setWidth(src.getWidth());
	if (src.getDatatype()!=-1)
	  setDatatype(src.getDatatype()); //Automatically calls tryAllocate
}
void FEM_Attribute::set(const void *src, int firstItem,int length, 
		const IDXL_Layout &layout, const char *caller) 
{
	if (firstItem!=0) { /* If this isn't the start... */
		if (length!=1) /* And we're not setting one at a time */
			CmiAbort("FEM_Mesh_data: unexpected firstItem");
	}

	if (femVersion == 0 && getRealLength() == -1) setLength(length);
	else if (getLength()==0) setLength(length);
	else if (length!=1 && length!=getLength()) 
		bad("length",false,getLength(),length, caller);
	
	int width=layout.width;
	if (femVersion==0 && getRealWidth()==-1) setWidth(width);
	else if (getWidth()==0) setWidth(width);
	else if (width!=getWidth()) 
		bad("width",false,getWidth(),width, caller);
	
	int datatype=layout.type;
	if (getDatatype()==-1) setDatatype(datatype);
	else if (datatype!=getDatatype()) 
		bad("datatype",false,getDatatype(),datatype, caller);
	
	/* Assert: our storage should be allocated now.
	   Our subclass will actually copy user data */
}

void FEM_Attribute::get(void *dest, int firstItem,int length, 
		const IDXL_Layout &layout, const char *caller)  const
{
	if (length==0) return; //Nothing to get
	if (length!=1 && length!=getLength()) 
		bad("length",true,getLength(),length, caller);
	
	int width=layout.width;
	if (width!=getWidth()) 
		bad("width",true,getWidth(),width, caller);
	
	int datatype=layout.type;
	if (datatype!=getDatatype()) 
		bad("datatype",true,getDatatype(),datatype, caller);
	
	/* our subclass will actually copy into user data */
}

/*check if the layout is the same as earlier */

void FEM_Attribute::register_data(void *user, int length,int max,
	const IDXL_Layout &layout, const char *caller){
	
		int width=layout.width;
		if (femVersion == 0 && getRealWidth()==-1) setWidth(width);
		else if (getWidth()==0){
			setWidth(width);
		}else{
			if (width!=getWidth()){
				bad("width",false,getWidth(),width, caller);
			}
		}	
	
		int datatype=layout.type;
		if (getDatatype()==-1){
			setDatatype(datatype);
		}else{
			if (datatype!=getDatatype()){ 
				bad("datatype",false,getDatatype(),datatype, caller);
			}
		}	
		
}

//Check if all three of length, width, and datatype are set.
// If so, call allocate.
void FEM_Attribute::tryAllocate(void) {
	int lenNull, widthNull;
        if (femVersion == 0) {
	  // version 0 takes -1 as empty
	  lenNull = (getRealLength()==-1);
	  widthNull = (getRealWidth()==-1);
	}
	else {
	  lenNull = (getLength()==0);
	  widthNull = (getWidth()==0);
	}
	if ((!allocated) && !lenNull && !widthNull && getDatatype()!=-1) {
	  allocated=true;
		allocate(getMax(),getWidth(),getDatatype());
	}
}

/*********************** DataAttribute *******************/
FEM_DataAttribute::FEM_DataAttribute(FEM_Entity *e,int myAttr)
	:FEM_Attribute(e,myAttr), 
	 char_data(0),int_data(0),float_data(0),double_data(0)
{
}
void FEM_DataAttribute::pup(PUP::er &p) {
	super::pup(p);
	switch(getDatatype()) {
	case -1: /* not allocated yet */ break;
	case FEM_BYTE:   if (char_data) char_data->pup(p); break;
	case FEM_INT:    if (int_data) int_data->pup(p); break;
	case FEM_FLOAT:  if (float_data) float_data->pup(p); break;
	case FEM_DOUBLE: if (double_data) double_data->pup(p); break;
	default: CkAbort("Invalid datatype in FEM_DataAttribute::pup");
	}
}
void FEM_DataAttribute::pupSingle(PUP::er &p, int pupindx) {
	super::pupSingle(p,pupindx);
	switch(getDatatype()) {
	case -1: /* not allocated yet */ break;
	case FEM_BYTE:   if (char_data) char_data->pupSingle(p,pupindx); break;
	case FEM_INT:    if (int_data) int_data->pupSingle(p,pupindx); break;
	case FEM_FLOAT:  if (float_data) float_data->pupSingle(p,pupindx); break;
	case FEM_DOUBLE: if (double_data) double_data->pupSingle(p,pupindx); break;
	default: CkAbort("Invalid datatype in FEM_DataAttribute::pupSingle");
	}
}
FEM_DataAttribute::~FEM_DataAttribute() {
	if (char_data) delete char_data;
	if (int_data) delete int_data;
	if (float_data) delete float_data;
	if (double_data) delete double_data;
	
}

/// Copy this data out of the user's (layout-formatted) array:
template <class T>
inline void setTableData(const void *user, int firstItem, int length, 
	IDXL_LAYOUT_PARAM, AllocTable2d<T> *table) 
{
	for (int r=0;r<length;r++) {
		register T *tableRow=table->getRow(firstItem+r);
		for (int c=0;c<width;c++)
			tableRow[c]=IDXL_LAYOUT_DEREF(T,user,r,c);
	}
}

/// Copy this data into the user's (layout-formatted) array:
template <class T>
inline void getTableData(void *user, int firstItem, int length, 
	IDXL_LAYOUT_PARAM, const AllocTable2d<T> *table) 
{
	for (int r=0;r<length;r++) {
		register const T *tableRow=table->getRow(firstItem+r);
		for (int c=0;c<width;c++)
			IDXL_LAYOUT_DEREF(T,user,r,c)=tableRow[c];
	}

}

void FEM_DataAttribute::set(const void *u, int f,int l, 
		const IDXL_Layout &layout, const char *caller)
{
	super::set(u,f,l,layout,caller);
	switch(getDatatype()) {
	case FEM_BYTE:  setTableData(u,f,l,IDXL_LAYOUT_CALL(layout),char_data); break;
	case FEM_INT: setTableData(u,f,l,IDXL_LAYOUT_CALL(layout),int_data); break;
	case FEM_FLOAT: setTableData(u,f,l,IDXL_LAYOUT_CALL(layout),float_data); break;
	case FEM_DOUBLE: setTableData(u,f,l,IDXL_LAYOUT_CALL(layout),double_data); break;
	}
}
	
void FEM_DataAttribute::get(void *u, int f,int l,
		const IDXL_Layout &layout, const char *caller) const
{
	super::get(u,f,l,layout,caller);
	switch(getDatatype()) {
	case FEM_BYTE:  getTableData(u,f,l,IDXL_LAYOUT_CALL(layout),char_data); break;
	case FEM_INT: getTableData(u,f,l,IDXL_LAYOUT_CALL(layout),int_data); break;
	case FEM_FLOAT: getTableData(u,f,l,IDXL_LAYOUT_CALL(layout),float_data); break;
	case FEM_DOUBLE: getTableData(u,f,l,IDXL_LAYOUT_CALL(layout),double_data); break;
	}
}

void FEM_DataAttribute::register_data(void *u,int l,int max,
	    const IDXL_Layout &layout, const char *caller)
{
	super::register_data(u,l,max,layout,caller);
	switch(getDatatype()){
		case FEM_BYTE: char_data->register_data((unsigned char *)u,l,max); break;
		case FEM_INT:	 int_data->register_data((int *)u,l,max); break;
		case FEM_FLOAT: float_data->register_data((float *)u,l,max);break;
		case FEM_DOUBLE: double_data->register_data((double *)u,l,max);break;
	}
}

template<class T>
inline AllocTable2d<T> *allocTablePtr(AllocTable2d<T> *src,int len,int wid) {
	if (src==NULL) src=new AllocTable2d<T>;
	src->allocate(wid,len);
	return src;
}
void FEM_DataAttribute::allocate(int l,int w,int datatype)
{
	switch(datatype) {
	case FEM_BYTE:  char_data=allocTablePtr(char_data,l,w); break;
	case FEM_INT: int_data=allocTablePtr(int_data,l,w); break;
	case FEM_FLOAT: float_data=allocTablePtr(float_data,l,w); break;
	case FEM_DOUBLE: double_data=allocTablePtr(double_data,l,w); break;
	default: CkAbort("Invalid datatype in FEM_DataAttribute::allocate");
	};
}
void FEM_DataAttribute::copyEntity(int dstEntity,const FEM_Attribute &src,int srcEntity)
{
	const FEM_DataAttribute *dsrc=(const FEM_DataAttribute *)&src;
	switch(getDatatype()) {
	case FEM_BYTE:  char_data->setRow(dstEntity,dsrc->char_data->getRow(srcEntity)); break;
	case FEM_INT: 
			int_data->setRow(dstEntity,dsrc->int_data->getRow(srcEntity)); break;
	case FEM_FLOAT: float_data->setRow(dstEntity,dsrc->float_data->getRow(srcEntity)); break;
	case FEM_DOUBLE: double_data->setRow(dstEntity,dsrc->double_data->getRow(srcEntity)); break;
	}
}

template<class T>
inline void interpolateAttrs(AllocTable2d<T> *data,int A,int B,int D,double frac,int width){
  T *rowA = data->getRow(A);
  T *rowB = data->getRow(B);
  T *rowD = data->getRow(D);
  for(int i=0;i<width;i++){
    double val = (double )rowA[i];
    val *= (frac);
    val += (1-frac) *((double )rowB[i]);
    rowD[i] = (T )val;
  }
}

template<class T>
inline void interpolateAttrs(AllocTable2d<T> *data,int *iNodes,int rNode,int k,int width){
  T *row[8];
  for (int i=0; i<k; i++) {
    row[i] = data->getRow(iNodes[i]);
  }
  T *rowR = data->getRow(rNode);
  for(int i=0;i<width;i++){
    double val = 0.0;
    for (int j=0; j<k; j++) {
      val += (double)row[j][i];
    }
    val = val/k;
    rowR[i] = (T )val;
  }
}

template<class T>
inline void minAttrs(AllocTable2d<T> *data,int A,int B,int D,double frac,int width){
  T *rowA = data->getRow(A);
  T *rowB = data->getRow(B);
  T *rowD = data->getRow(D);
  for(int i=0;i<width;i++){
    if(rowA[i] < rowB[i]){
      rowD[i] = rowA[i];
    }else{
      rowD[i] = rowB[i];
    }
  }
}

template<class T>
inline void minAttrs(AllocTable2d<T> *data,int *iNodes,int rNode,int k,int width){
  T *row[8];
  for (int i=0; i<k; i++) {
    row[i] = data->getRow(iNodes[i]);
  }
  T *rowR = data->getRow(rNode);
  for(int i=0;i<width;i++){
    rowR[i] = row[0][i]; 
    for (int j=1; j<k; j++) {
      if (row[j][i] < rowR[i]) {
	rowR[i] = row[j][i];
      }
    }
  }
}

void FEM_DataAttribute::interpolate(int A,int B,int D,double frac){
  switch(getDatatype()){
  case FEM_BYTE:
    minAttrs(char_data,A,B,D,frac,getWidth());		
    break;
  case FEM_INT:
    minAttrs(int_data,A,B,D,frac,getWidth());		
    break;
  case FEM_FLOAT:
    interpolateAttrs(float_data,A,B,D,frac,getWidth());		
    break;
  case FEM_DOUBLE:
    interpolateAttrs(double_data,A,B,D,frac,getWidth());		
    break;
  }
}

void FEM_DataAttribute::interpolate(int *iNodes,int rNode,int k){
  switch(getDatatype()){
  case FEM_BYTE:
    minAttrs(char_data,iNodes,rNode,k,getWidth());		
    break;
  case FEM_INT:
    minAttrs(int_data,iNodes,rNode,k,getWidth());		
    break;
  case FEM_FLOAT:
    interpolateAttrs(float_data,iNodes,rNode,k,getWidth());		
    break;
  case FEM_DOUBLE:
    interpolateAttrs(double_data,iNodes,rNode,k,getWidth());		
    break;
  }
}

/*********************** FEM_IndexAttribute *******************/
FEM_IndexAttribute::Checker::~Checker() {}

FEM_IndexAttribute::FEM_IndexAttribute(FEM_Entity *e,int myAttr,FEM_IndexAttribute::Checker *checker_)
	:FEM_Attribute(e,myAttr), idx(0,0,-1), checker(checker_)
{
	setDatatype(FEM_INT);
}
void FEM_IndexAttribute::pup(PUP::er &p) {
	super::pup(p);
	p|idx;
}
void FEM_IndexAttribute::pupSingle(PUP::er &p, int pupindx) {
	super::pupSingle(p,pupindx);
	idx.pupSingle(p,pupindx);
}
FEM_IndexAttribute::~FEM_IndexAttribute() {
	if (checker) delete checker;
}

void FEM_IndexAttribute::allocate(int length,int width,int datatype)
{
	idx.allocate(width,length);
}

/**
 * Convert a datatype, which must be FEM_INDEX_0 or FEM_INDEX_1, to 
 * the first valid index (base index) of that type.  Otherwise 
 * call FEM_Abort, because the datatype is wrong.
 */
static int type2base(int base_type,const char *caller) {
	if (base_type==FEM_INDEX_0) return 0;
	if (base_type==FEM_INDEX_1) return 1;
	FEM_Abort(caller,"You must use the datatype FEM_INDEX_0 or FEM_INDEX_1 with FEM_CONN, not %d",
		base_type);
	return 0; //< for whining compilers
}

/// Copy this data out of the user's (layout-formatted, indexBase) array:
void setIndexTableData(const void *user, int firstItem, int length, 
	IDXL_LAYOUT_PARAM, AllocTable2d<int> *table,int indexBase) 
{
	for (int r=0;r<length;r++) {
		register int *tableRow=table->getRow(firstItem+r);
		for (int c=0;c<width;c++)
			tableRow[c]=IDXL_LAYOUT_DEREF(int,user,r,c)-indexBase;
	}
}

/// Copy this data into the user's (layout-formatted, indexBase) array:
void getIndexTableData(void *user, int firstItem, int length, 
	IDXL_LAYOUT_PARAM, const AllocTable2d<int> *table,int indexBase) 
{
	for (int r=0;r<length;r++) {
		register const int *tableRow=table->getRow(firstItem+r);
		for (int c=0;c<width;c++)
			IDXL_LAYOUT_DEREF(int,user,r,c)=tableRow[c]+indexBase;
	}
}

void FEM_IndexAttribute::set(const void *src, int firstItem,int length,
		const IDXL_Layout &layout,const char *caller)
{
	IDXL_Layout lo=layout; lo.type=FEM_INT; //Pretend it's always int data, not INDEX
	super::set(src,firstItem,length,lo,caller);
	
	int indexBase=type2base(layout.type,caller);
	setIndexTableData(src,firstItem,length,IDXL_LAYOUT_CALL(layout),&idx,indexBase);
	
	if (checker) 
		for (int r=0;r<length;r++)
			checker->check(firstItem+r,idx,caller);
}

void FEM_IndexAttribute::get(void *dest, int firstItem,int length, 
		const IDXL_Layout &layout,const char *caller) const
{
	IDXL_Layout lo=layout; lo.type=FEM_INT; //Pretend it's always int data, not INDEX
	super::get(dest,firstItem,length,lo,caller);
	
	int indexBase=type2base(layout.type,caller);
	getIndexTableData(dest,firstItem,length,IDXL_LAYOUT_CALL(layout),&idx,indexBase);
}

void FEM_IndexAttribute::register_data(void *user, int length,int max,
		const IDXL_Layout &layout, const char *caller){
	IDXL_Layout lo=layout; lo.type=FEM_INT; //Pretend it's always int data, not INDEX
	super::register_data(user,length,max,lo,caller);

	idx.register_data((int *)user,length,max);
}

void FEM_IndexAttribute::copyEntity(int dstEntity,const FEM_Attribute &src,int srcEntity)
{
	const FEM_IndexAttribute *csrc=(const FEM_IndexAttribute *)&src;
	idx.setRow(dstEntity,csrc->idx.getRow(srcEntity));
}

/*************FEM_VarIndexAttribute***************/

FEM_VarIndexAttribute::FEM_VarIndexAttribute(FEM_Entity *e,int myAttr)
	:FEM_Attribute(e,myAttr)
{
  oldlength = 0;
	allocate(getMax(),getWidth(),getDatatype());
	setDatatype(FEM_INT);
}

void FEM_VarIndexAttribute::pup(PUP::er &p){
	super::pup(p);
	p | idx;
}

void FEM_VarIndexAttribute::pupSingle(PUP::er &p, int pupindx){
	super::pupSingle(p,pupindx);
	p|idx[pupindx];
}

void FEM_VarIndexAttribute::set(const void *src,int firstItem,int length,
		const IDXL_Layout &layout,const char *caller){
		printf("set not yet implemented for FEM_VarIndexAttribute \n");
}

void FEM_VarIndexAttribute::get(void *dest, int firstItem,int length,
		const IDXL_Layout &layout, const char *caller) const{
	 printf("get not yet implemented for FEM_VarIndexAttribute \n");
			
}

void FEM_VarIndexAttribute::copyEntity(int dstEntity,const FEM_Attribute &_src,int srcEntity){
	FEM_VarIndexAttribute &src = (FEM_VarIndexAttribute &)_src;
	const CkVec<CkVec<ID> > &srcTable = src.get();
	idx.insert(dstEntity,srcTable[srcEntity]);
}

void FEM_VarIndexAttribute::print(){
	for(int i=0;i<idx.size();i++){
		printf("%d -> ",i);
		for(int j=0;j<idx[i].size();j++){
			printf("(%d %d) ",((idx[i])[j]).type,((idx[i])[j]).id);
		}
		printf("\n");
	}
}

int FEM_VarIndexAttribute::findInRow(int row,const ID &data){
	if(row >= idx.length()){
		return -1;
	}
	CkVec<ID> &rowVec = idx[row];
	for(int i=0;i<rowVec.length();i++){
		if(data == rowVec[i]){
			return i;
		}
	}
	return -1;
}

/********************** Entity **************************/

/// Return the human-readable version of this entity code.
CDECL const char *FEM_Get_entity_name(int entity,char *storage) 
{
	char *dest=storage;
	if (entity<FEM_ENTITY_FIRST || entity>=FEM_ENTITY_LAST) {
		sprintf(dest,"unknown entity code (%d)",entity);
	}
	else {
		if (entity>FEM_ENTITY_FIRST+FEM_GHOST) {
			sprintf(dest,"FEM_GHOST+");
			dest+=strlen(dest); /* we want "FEM_GHOST+foo" */
			entity-=FEM_GHOST;
		}
		if (entity==FEM_NODE)
			sprintf(dest,"FEM_NODE");
		else if (entity>=FEM_SPARSE)
			sprintf(dest,"FEM_SPARSE+%d",entity-FEM_SPARSE);
		else /* entity>=FEM_ELEM */
			sprintf(dest,"FEM_ELEM+%d",entity-FEM_ELEM);
	}
	return storage;
}

FEM_Entity::FEM_Entity(FEM_Entity *ghost_) //Default constructor
  :length(0), max(0),ghost(ghost_), coord(0), sym(0), globalno(0), valid(0), meshSizing(0),
	 ghostIDXL(ghost?&ghostSend:NULL, ghost?&ghost->ghostRecv:NULL),resize(NULL)
{
	//No attributes initially
	if (femVersion == 0) {
		length=-1;
		max=-1;
	}
} 
void FEM_Entity::pup(PUP::er &p) {
	p|length;
	if (p.isUnpacking() && femVersion > 0 && length<0)  length=0;
	p.comment(" Ghosts to send out: ");
	ghostSend.pup(p);
	p.comment(" Ghosts to recv: ");
	ghostRecv.pup(p);
	p.comment(" Ghost IDXL tag: ");
	ghostIDXL.pup(p);
	
	int nAttributes=attributes.size();
	p|nAttributes;
	for (int a=0;a<nAttributes;a++) 
	{
	/* Beautiful hack: the FEM_Attribute objects are normally hideously cross-linked
	   with their owning classes.  Thus instead of trying to rebuild the FEM_Attributes
	   from scratch here, we just use the existing "lookup" method to demand-create those
	   that need it, or just *find* those that are already there.
	   
	   This is a much better fit for this situation than using the general PUP::able.
	 */
		int attr=0; // The attribute type we're pupping
		FEM_Attribute *r=NULL;
		if (!p.isUnpacking()) { //Send side: we already know the source
			r=attributes[a];
			attr=r->getAttr();
		}
		p|attr;
		if (p.isUnpacking()) { //Recv side: create (or recycle) the destination
			r=lookup(attr,"FEM_Entity::pup");
		}
		
		{ //Put the human-readable attribute name in the output file:
			char attrNameStorage[256];
			p.comment(FEM_Get_attr_name(attr,attrNameStorage));
		}
		
		r->pup(p);
	}
	
	if (ghost!=NULL) {
		p.comment(" ---- Ghost attributes ---- ");
		ghost->pup(p);
	}
}
FEM_Entity::~FEM_Entity() 
{
	delete ghost;
	for (int a=0;a<attributes.size();a++)
		delete attributes[a];
}

/// Copy our attributes' widths and data types from this entity.
void FEM_Entity::copyShape(const FEM_Entity &src) {
	for (int a=0;a<src.attributes.size();a++) 
	{ // We need each of his attributes:
		const FEM_Attribute *Asrc=src.attributes[a];
		FEM_Attribute *Adst=lookup(Asrc->getAttr(),"FEM_Entity::copyShape");
		Adst->copyShape(*Asrc);
	}
	if (ghost) ghost->copyShape(*src.ghost);
}

void FEM_Entity::setLength(int newlen) 
{
  if (!resize) {
    if (size() != newlen) {
      length = newlen;
      // Each of our attributes need to be expanded for our new length:
      for (int a=0; a<attributes.size(); a++) {
	CkAssert(attributes[a]->getWidth() < 1000);
	attributes[a]->reallocate();
      }
    }
  }
  else {
    length = newlen;
    if (length > max) {
      if (max > 4) {
	max = max + (max >> 2);
      }
      else {
	max = max + 10;
      }
      for (int a=0;a<attributes.size();a++){
	int code = attributes[a]->getAttr();
	if(!(code <= FEM_ATTRIB_TAG_MAX || code == FEM_CONN || code == FEM_BOUNDARY)){
	  attributes[a]->reallocate();
	}
      }	
      // call resize with args max n;
      CkPrintf("Resize called \n");
      resize(args,&length,&max);
    }
  }
}

void FEM_Entity::allocateValid(void) {
  if (!valid){
	valid=new FEM_DataAttribute(this,FEM_IS_VALID_ATTR);
	add(valid);
	valid->setWidth(1); //Only 1 flag per node
	valid->setLength(size());
	valid->setDatatype(FEM_BYTE);
	valid->reallocate();
	
	// Set all to valid initially
	for(int i=0;i<size();i++)
	  valid->getChar()(i,0)=1;
  first_invalid = last_invalid = 0;
  }

}

void FEM_Entity::set_valid(unsigned int idx, bool isNode){
  if(false) {
    CkAssert(idx < size() && idx >=0);
    valid->getChar()(idx,0)=1;
  }
  else {
    CkAssert(idx < size() && idx >=0 && first_invalid<=last_invalid);
    valid->getChar()(idx,0)=1;
    
    if(idx == first_invalid)
      // Move first_invalid to the next invalid entry	
      while((first_invalid<last_invalid) && is_valid(first_invalid)){
	first_invalid++;
      }
    else if(idx == last_invalid)
      // Move last_invalid to the previous invalid entry	
      while((first_invalid<last_invalid) && is_valid(last_invalid))
	last_invalid--;
    
    // If we have no invalid elements left, then put both pointers to 0
    if( first_invalid == last_invalid && is_valid(first_invalid) )
      first_invalid = last_invalid = 0;
  }
}

void FEM_Entity::set_invalid(unsigned int idx, bool isNode){
  if(false) {
    CkAssert(idx < size() && idx >=0);
    valid->getChar()(idx,0)=0;
  }
  else {
    CkAssert(idx < size() && idx >=0 && first_invalid<=last_invalid);
    valid->getChar()(idx,0)=0;
    
    // If there are currently no invalid entities
    if(first_invalid==0 && last_invalid==0 && is_valid(0)){
      first_invalid = last_invalid = idx;
      return;
    }
    
    if(idx < first_invalid){
      first_invalid = idx;
    }
    
    if(idx > last_invalid){
      last_invalid = idx;
    }
    
    // TODO:
    // We should probably have an algorithm for shrinking the entire attribute 
    // array if we invalidate the final element. In this case we should scan backwards
    // to find the largest indexed valid entity and resize down to it.
    // 
    // It may be necessary to modify the idxl lists if we do this type of shrinking.
    // Someone needs to confirm whether that is necessary. If not, then it should be 
    // simple to allow shrinking of the number of nodes or elements.
    
  }
}

int FEM_Entity::is_valid(unsigned int idx){
  if(false) {
    CkAssert(idx < size() && idx >=0);
    return valid->getChar()(idx,0);
  } else {
    CkAssert(idx < size() && idx >=0 && first_invalid<=last_invalid);
    return valid->getChar()(idx,0);
  }
}

unsigned int FEM_Entity::count_valid(){
  CkAssert(first_invalid<=last_invalid);
  unsigned int count=0;
  for(int i=0;i<size();i++)
	if(is_valid(i)) count++;
  return count;
}

/// Get an entry(entity index) that corresponds to an invalid entity
/// The invalid slot in the tables can then be reused when "creating" a new element or node
/// We either return an empty slot, or resize the array and return a value at the end
/// If someone has a better name for this function, please change it.
unsigned int FEM_Entity::get_next_invalid(FEM_Mesh *m, bool isNode, bool isGhost){
  unsigned int retval;
  if(false) {
    retval = size();
    setLength(retval+1);  
  }
  else {
    CkAssert(!is_valid(first_invalid) || first_invalid==0);
    
    // if we have an invalid entity to return
    bool flag1 = false;
    if(!is_valid(first_invalid)){
      retval = first_invalid;
      if(isNode && !isGhost) { //it is a node & the entity is not a ghost entity
	while(retval <= last_invalid) {
	  if(!is_valid(retval)) {
	    if(m->getfmMM()->fmLockN[retval]->haslocks()) {
	      retval++;
	    }
	    else if(hasConn(retval)) { //has some connectivity
	      retval++;
	    }
	    else {
	      flag1 = true;
	      break;
	    }
	  }
	  else retval++;
	}
      }
      else if(isNode) {
	while(retval <= last_invalid) {
	  if(!is_valid(retval)) {
	    if(hasConn(retval)) { //has some connectivity
	      retval++;
	    }
	    else {
	      flag1 = true;
	      break;
	    }
	  }
	  else retval++;
	}
      }      
      else{
	// resize array and return new entity
	flag1 = true;
      }
    }
    if(!flag1) {
      retval = size();
      setLength(retval+1);
    }
  }
  set_valid(retval,isNode);
  return retval;
}

void FEM_Entity::setMaxLength(int newLen,int newMaxLen,void *pargs,FEM_Mesh_alloc_fn fn){
        CkPrintf("resize fn %p \n",fn);
	max = newMaxLen;
	resize = fn;
	args = pargs;
	setLength(newLen);
}


/// Copy src[srcEntity] into our dstEntity.
void FEM_Entity::copyEntity(int dstEntity,const FEM_Entity &src,int srcEntity) {
	FEM_Entity *dp=this; //Destination entity
	const FEM_Entity *sp=&src;
	for (int a=0;a<sp->attributes.size();a++) 
	{ //We copy each of his attributes:
		const FEM_Attribute *Asrc=sp->attributes[a];
		FEM_Attribute *Adst=dp->lookup(Asrc->getAttr(),"FEM_Entity::copyEntity");
		Adst->copyEntity(dstEntity,*Asrc,srcEntity);
	}
}

/// Add room for one more entity, with initial values from src[srcEntity],
/// and return the new entity's index.
int FEM_Entity::push_back(const FEM_Entity &src,int srcEntity) {
	int dstEntity=size();
	setLength(dstEntity+1);
	copyEntity(dstEntity,src,srcEntity);
	return dstEntity;
}

/// Add this attribute to this kind of Entity.
/// This method is normally called by the default lookup method.
void FEM_Entity::add(FEM_Attribute *attribute) {
	if (ghost!=NULL) 
	{ //Look up (or create) the ghost attribute, too:
		attribute->setGhost(ghost->lookup(attribute->getAttr(),"FEM_Entity::add"));
	}
	attributes.push_back(attribute);
}

/**
 * Find this attribute (from an FEM_ATTR code) of this entity.
 * The default implementation searches the list of userdata attributes;
 * subclasses with other attributes should override this routine.
 */
FEM_Attribute *FEM_Entity::lookup(int attr,const char *caller) {
	//Try to find an existing attribute (FIXME: keep attributes in a map, to speed this up)
	for (int a=0;a<attributes.size();a++) {
		if (attributes[a]->getAttr()==attr)
			return attributes[a];
	}
	
	//If we get here, no existing attribute fits the bill: create one
	create(attr,caller);
	
	// If create did its job, the next lookup should succeed:
	return lookup(attr,caller);
}

/**
 * Create a new attribute from an FEM_ATTR code.
 * The default implementation handles FEM_DATA tags; entity-specific
 * attributes (like FEM_CONN) need to be overridden and created 
 * by subclasses.
 */
void FEM_Entity::create(int attr,const char *caller) {
  if (attr<=FEM_ATTRIB_TAG_MAX) //It's a valid user data tag
	add(new FEM_DataAttribute(this,attr));
  else if (attr==FEM_COORD) 
	allocateCoord();
  else if (attr==FEM_SYMMETRIES) 
	allocateSym();
  else if (attr==FEM_GLOBALNO) 
	allocateGlobalno();
  else if (attr==FEM_IS_VALID_ATTR)
	allocateValid();
  else if (attr==FEM_MESH_SIZING) 
	allocateMeshSizing();
  else if(attr == FEM_CHUNK){
	FEM_IndexAttribute *chunkNo= new FEM_IndexAttribute(this,FEM_CHUNK,NULL);
	add(chunkNo);
	chunkNo->setWidth(1);
  } else if(attr == FEM_BOUNDARY){
	//the boundary attribute for this entity
	allocateBoundary();
  } else {
	//It's an unrecognized tag: abort
	char attrNameStorage[256], msg[1024];
	sprintf(msg,"Could not locate the attribute %s for entity %s",
			FEM_Get_attr_name(attr,attrNameStorage), getName());
	FEM_Abort(caller,msg);
  }
}

void FEM_Entity::allocateCoord(void) {
	if (coord) CkAbort("FEM_Entity::allocateCoord called, but already allocated");
	coord=new FEM_DataAttribute(this,FEM_COORD);
	add(coord); // coord will be deleted by FEM_Entity now
	coord->setDatatype(FEM_DOUBLE);
}

void FEM_Entity::allocateSym(void) {
	if (sym) CkAbort("FEM_Entity::allocateSym called, but already allocated");
	sym=new FEM_DataAttribute(this,FEM_SYMMETRIES);
	add(sym); // sym will be deleted via attributes list now
	sym->setWidth(1);
	sym->setDatatype(FEM_BYTE); //Same as FEM_Symmetries_t
}

void FEM_Entity::setSymmetries(int r,FEM_Symmetries_t s)
{
	if (!sym) {
		if (s==0) return; //Don't bother allocating just for 0
		allocateSym();
	}
	sym->getChar()(r,0)=s;
}


/*
 * Set the coordinates for a node or other entity.
 * Use the appropriate 2d or 3d version.
 * 
 * Note that first the function attempts to find coordinates in
 * the FEM_COORD attribute's array "*coord". If this fails, then
 * it will use the user's FEM_DATA field. Little error checking is
 * done, so the functions may crash if used inappropriately.
 */
inline void FEM_Entity::set_coord(int idx, double x, double y){
  if(coord){
	coord->getDouble()(idx,0)=x;
	coord->getDouble()(idx,1)=y;
  }
  else {
	FEM_DataAttribute* attr = 	(FEM_DataAttribute*)lookup(FEM_DATA,"set_coord");
	attr->getDouble()(idx,0)=x;
	attr->getDouble()(idx,1)=y;
  }
}

inline void FEM_Entity::set_coord(int idx, double x, double y, double z){
  if(coord){
	coord->getDouble()(idx,0)=x;
	coord->getDouble()(idx,1)=y;
	coord->getDouble()(idx,2)=z;
  }
  else {
	FEM_DataAttribute* attr = 	(FEM_DataAttribute*)lookup(FEM_DATA,"set_coord");
	attr->getDouble()(idx,0)=x;
	attr->getDouble()(idx,1)=y;
	attr->getDouble()(idx,2)=z;
  }
}


void FEM_Entity::allocateGlobalno(void) {
	if (globalno) CkAbort("FEM_Entity::allocateGlobalno called, but already allocated");
	globalno=new FEM_IndexAttribute(this,FEM_GLOBALNO,NULL);
	add(globalno); // globalno will be deleted via attributes list now
	globalno->setWidth(1);
}

void FEM_Entity::allocateMeshSizing(void) {
  if (meshSizing) 
    CkAbort("FEM_Entity::allocateMeshSizing called, but already allocated");
  meshSizing=new FEM_DataAttribute(this,FEM_MESH_SIZING);
  add(meshSizing); // meshSizing will be deleted via attributes list now
  meshSizing->setWidth(1);
  meshSizing->setDatatype(FEM_DOUBLE);
}

double FEM_Entity::getMeshSizing(int r) {
  if (!meshSizing) {
    allocateMeshSizing();
    return -1.0;
  }
  if (r >= 0)  return meshSizing->getDouble()(r,0);
  else  return ghost->meshSizing->getDouble()(FEM_To_ghost_index(r),0);
}

void FEM_Entity::setMeshSizing(int r,double s)
{
  if (!meshSizing) allocateMeshSizing();
  if (s <= 0.0) return;
  if (r >= 0)  meshSizing->getDouble()(r,0)=s;
  else ghost->meshSizing->getDouble()(FEM_To_ghost_index(r),0)=s;
}

void FEM_Entity::setMeshSizing(double *sf)
{
  if (!meshSizing) allocateMeshSizing();
  int len = size();
  for (int i=0; i<len; i++)
    meshSizing->getDouble()(i,0)=sf[i];
}

void FEM_Entity::allocateBoundary(){
	FEM_DataAttribute *bound = new FEM_DataAttribute(this,FEM_BOUNDARY);
	add(bound);
	bound->setWidth(1);
	bound->setDatatype(FEM_INT);
}

void FEM_Entity::setGlobalno(int r,int g) {
	if (!globalno) allocateGlobalno();
	globalno->get()(r,0)=g;
}
void FEM_Entity::setAscendingGlobalno(void) {
	if (!globalno) {
		allocateGlobalno();
		int len=size();
		for (int i=0;i<len;i++) globalno->get()(i,0)=i;
	}
}
void FEM_Entity::setAscendingGlobalno(int base) {
	if (!globalno) {
		allocateGlobalno();
		int len=size();
		for (int i=0;i<len;i++) globalno->get()(i,0)=i+base;
	}
}
void FEM_Entity::copyOldGlobalno(const FEM_Entity &e) {
	if ((!hasGlobalno()) && e.hasGlobalno() && size()>=e.size()) {
		for (int i=0;i<size();i++) 
			setGlobalno(i,e.getGlobalno(i));
	}
}

/********************** Node *****************/
FEM_Node::FEM_Node(FEM_Node *ghost_) 
  :FEM_Entity(ghost_), primary(0), sharedIDXL(&shared,&shared),
   elemAdjacency(0),nodeAdjacency(0)
{}

void FEM_Node::allocatePrimary(void) {
  if (primary) CkAbort("FEM_Node::allocatePrimary called, but already allocated");
  primary=new FEM_DataAttribute(this,FEM_NODE_PRIMARY);
  add(primary); // primary will be deleted by FEM_Entity now
  primary->setWidth(1); //Only 1 flag per node
  primary->setDatatype(FEM_BYTE);
}

bool FEM_Node::hasConn(unsigned int idx) {
  if((elemAdjacency->get()[idx].length() > 0)||(nodeAdjacency->get()[idx].length() > 0))
    return true;
  else return false;
}

void FEM_Node::pup(PUP::er &p) {
	p.comment(" ---------------- Nodes ------------------ ");	
	super::pup(p);
	p.comment(" ---- Shared nodes ----- ");	
	shared.pup(p);
	p.comment(" shared nodes IDXL ");
	sharedIDXL.pup(p);
}
FEM_Node::~FEM_Node() {
}


const char *FEM_Node::getName(void) const {return "FEM_NODE";}

void FEM_Node::create(int attr,const char *caller) {
  if (attr==FEM_NODE_PRIMARY)
	allocatePrimary();
  else if(attr == FEM_NODE_ELEM_ADJACENCY)
	allocateElemAdjacency();
  else if(attr == FEM_NODE_NODE_ADJACENCY)
	allocateNodeAdjacency();
  else
	super::create(attr,caller);
}


/********************** Elem *****************/
/// This checker verifies that FEM_Elem::conn's entries are valid node indices.
class FEM_Elem_Conn_Checker : public FEM_IndexAttribute::Checker {
	const FEM_Entity &sizeSrc;
	const FEM_Entity *sizeSrc2;
public:
	FEM_Elem_Conn_Checker(const FEM_Entity &sizeSrc_,const FEM_Entity *sizeSrc2_) 
		:sizeSrc(sizeSrc_), sizeSrc2(sizeSrc2_) {}
	
	void check(int row,const BasicTable2d<int> &table,const char *caller) const {
		const int *idx=table.getRow(row);
		int n=table.width();
		int max=sizeSrc.size();
		if (sizeSrc2) max+=sizeSrc2->size();
		for (int i=0;i<n;i++) 
			if ((idx[i]<0) || (idx[i]>=max))
			{ /* This index is out of bounds: */
				if (idx[i]<0)
					FEM_Abort(caller,"Connectivity entry %d's value, %d, is negative",row,idx[i]);
				else /* (idx[i]>=max) */
					FEM_Abort(caller,
						"Connectivity entry %d's value, %d, should be less than the number of nodes, %d",
						row,idx[i],max);
			}
	}
};

FEM_Elem::FEM_Elem(const FEM_Mesh &mesh, FEM_Elem *ghost_) 
  :FEM_Entity(ghost_), elemAdjacency(0), elemAdjacencyTypes(0)
{
	FEM_IndexAttribute::Checker *c;
	if (isGhost()) // Ghost elements can point to both real as well as ghost nodes
		c=new FEM_Elem_Conn_Checker(mesh.node, mesh.node.getGhost());
	else /* is real */ //Real elements only point to real nodes
		c=new FEM_Elem_Conn_Checker(mesh.node, NULL);
	conn=new FEM_IndexAttribute(this,FEM_CONN,c);
	add(conn); // conn will be deleted by FEM_Entity now
}
void FEM_Elem::pup(PUP::er &p) {
	p.comment(" ------------- Element data ---------- ");
	FEM_Entity::pup(p);
}
FEM_Elem::~FEM_Elem() {
}


void FEM_Elem::create(int attr,const char *caller) {
  // We need to catch both FEM_ELEM_ELEM_ADJACENCY and FEM_ELEM_ELEM_ADJ_TYPES, 
  // since if either one falls through to 
  // the super::create(), it will not know what to do, and will fail
  //
  // Note: allocateElemAdjacency() will create both attribute fields since they
  //       should always be used together.
  
  if(attr == FEM_ELEM_ELEM_ADJACENCY)
    allocateElemAdjacency();
  else if(attr == FEM_ELEM_ELEM_ADJ_TYPES)
    allocateElemAdjacency();
  else
    super::create(attr,caller);
}


const char *FEM_Elem::getName(void) const {
	return "FEM_ELEM";
}

bool FEM_Elem::hasConn(unsigned int idx) {
  return false;
}

/********************* Sparse ******************/
/**
 * This checker makes sure FEM_Sparse::elem's two element indices
 * (element type, element index) are valid.
 */
class FEM_Sparse_Elem_Checker : public FEM_IndexAttribute::Checker {
	const FEM_Mesh &mesh;
public:
	FEM_Sparse_Elem_Checker(const FEM_Mesh &mesh_) :mesh(mesh_) {}
	
	void check(int row,const BasicTable2d<int> &table,const char *caller) const {
		//assert: table.getWidth==2
		const int *elem=table.getRow(row);
		int maxT=mesh.elem.size();
		if ((elem[0]<0) || (elem[1]<0))
			FEM_Abort(caller,"Sparse element entry %d's values, %d and %d, are negative",
				row,elem[0],elem[1]);
		int t=elem[0];
		if (t>=maxT)
			FEM_Abort(caller,"Sparse element entry %d's element type, %d, is too big",
				row,elem[0]);
		if (elem[1]>=mesh.elem[t].size())
			FEM_Abort(caller,"Sparse element entry %d's element index, %d, is too big",
				row,elem[1]);
	}
};

FEM_Sparse::FEM_Sparse(const FEM_Mesh &mesh_,FEM_Sparse *ghost_) 
	:FEM_Elem(mesh_,ghost_), elem(0), mesh(mesh_)
{
}
void FEM_Sparse::allocateElem(void) {
	if (elem) CkAbort("FEM_Sparse::allocateElem called, but already allocated");
	FEM_IndexAttribute::Checker *checker=new FEM_Sparse_Elem_Checker(mesh);
	elem=new FEM_IndexAttribute(this,FEM_SPARSE_ELEM,checker);
	add(elem); //FEM_Entity will delete elem now
	elem->setWidth(2); //SPARSE_ELEM consists of pairs: element type, element number
}
void FEM_Sparse::pup(PUP::er &p) {
	p.comment(" ------------- Sparse Element ---------- ");
	super::pup(p);
}
FEM_Sparse::~FEM_Sparse() {
}

const char *FEM_Sparse::getName(void) const { return "FEM_SPARSE"; }

void FEM_Sparse::create(int attr,const char *caller) {
	if (attr==FEM_SPARSE_ELEM)
		allocateElem();
	else /*super*/ FEM_Entity::create(attr,caller);
}


/******************* Mesh *********************/
FEM_Mesh::FEM_Mesh() 
	:node(new FEM_Node(NULL)),
	 elem(*this,"FEM_ELEM"),
	 sparse(*this,"FEM_SPARSE"),
	 lastElemAdjLayer(NULL)
{
	m_isSetting=true; //Meshes start out setting
	lastElemAdjLayer=NULL; // Will be created on demand
}
FEM_Mesh::~FEM_Mesh() {
}

FEM_Entity *FEM_Mesh::lookup(int entity,const char *caller) {
	FEM_Entity *e=NULL;
	if (entity>=FEM_ENTITY_FIRST && entity<FEM_ENTITY_LAST) 
	{ //It's in the right range for an entity code:
		bool isGhost=false;
		if (entity-FEM_ENTITY_FIRST>=FEM_GHOST) {
			entity-=FEM_GHOST;
			isGhost=true;
		}
		if (entity==FEM_NODE)
			e=&node;
		else if (entity>=FEM_ELEM && entity<FEM_ELEM+100) 
		{ //It's a kind of element:
			int elType=entity-FEM_ELEM;
			e=&elem.set(elType);
		}
		else if (entity>=FEM_SPARSE && entity<FEM_SPARSE+100) 
		{ //It's a kind of sparse:
			int sID=entity-FEM_SPARSE;
			e=&sparse.set(sID);
		}
		
		if (isGhost) //Move from the real to the ghost entity
			e=e->getGhost();
	}
	
	if (e==NULL) //We didn't find an entity!
		FEM_Abort(caller,"Expected an entity type (FEM_NODE, FEM_ELEM, etc.) but got %d",entity);
	return e;
}
const FEM_Entity *FEM_Mesh::lookup(int entity,const char *caller) const {
	/// FIXME: the const version is quite similar to the above, 
	/// but it should *not* create new Entity types...
	return ((FEM_Mesh *)this)->lookup(entity,caller);
}

void FEM_Mesh::setFemMeshModify(femMeshModify *m){
  fmMM = m;
}


femMeshModify *FEM_Mesh::getfmMM(){
  return fmMM;
}

void FEM_Mesh::pup(PUP::er &p)  //For migration
{
	p.comment(" ------------- Node Data ---------- ");
	node.pup(p);

	p.comment(" ------------- Element Types ---------- ");
	elem.pup(p);
	
	p.comment("-------------- Sparse Types ------------");
	sparse.pup(p);
	
	p.comment("-------------- Symmetries ------------");
	symList.pup(p);
	
	p|m_isSetting;

	p.comment("-------------- Mesh data --------------");
	udata.pup(p);

/* NOTE: for backward file compatability (fem_mesh_vp files),
   be sure to add new stuff at the *end* of this routine--
   it will be read as zeros for old files. */
}

int FEM_Mesh::chkET(int elType) const {
	if ((elType<0)||(elType>=elem.size())) {
		CkError("FEM Error! Bad element type %d!\n",elType);
		CkAbort("FEM Error! Bad element type used!\n");
	}
	return elType;
}

int FEM_Mesh::nElems(int t_max) const //Return total number of elements before type t_max
{
#if CMK_ERROR_CHECKING
	if (t_max<0 || t_max>elem.size()) {
		CkPrintf("FEM> Invalid element type %d used!\n");
		CkAbort("FEM> Invalid element type");
	}
#endif
	int ret=0;
	for (int t=0;t<t_max;t++){ 
		if (elem.has(t)){
			ret+=elem.get(t).size();
		}
	}	
	return ret;
}

int FEM_Mesh::getGlobalElem(int elType,int elNo) const
{
	int base=nElems(elType); //Global number of first element of this type
#if CMK_ERROR_CHECKING
	if (elNo<0 || elNo>=elem[elType].size()) {
		CkPrintf("FEM> Element number %d is invalid-- element type %d has only %d elements\n",
			elNo,elType,elem[elType].size());
		CkAbort("FEM> Invalid element number, probably passed via FEM_Set_Sparse_elem");
	}
#endif
	return base+elNo;
}

/// Set our global numbers as 0...n-1 for nodes, elements, and sparse
void FEM_Mesh::setAscendingGlobalno(void) {
	node.setAscendingGlobalno();
	for (int e=0;e<elem.size();e++)
		if (elem.has(e)) elem[e].setAscendingGlobalno();
	for (int s=0;s<sparse.size();s++)
		if (sparse.has(s)) sparse[s].setAscendingGlobalno();
}
void FEM_Mesh::setAbsoluteGlobalno(){
	node.setAscendingGlobalno();
	for (int e=0;e<elem.size();e++){
		if (elem.has(e)) elem[e].setAscendingGlobalno(nElems(e));
	}	
}

void FEM_Mesh::copyOldGlobalno(const FEM_Mesh &m) {
	node.copyOldGlobalno(m.node);
	for (int e=0;e<m.elem.size();e++)
		if (m.elem.has(e) && e<elem.size() && elem.has(e)) 
			elem[e].copyOldGlobalno(m.elem[e]);
	for (int s=0;s<m.sparse.size();s++)
		if (m.sparse.has(s) && s<sparse.size() && sparse.has(s)) 
			sparse[s].copyOldGlobalno(m.sparse[s]);
}

void FEM_Index_Check(const char *caller,const char *entityType,int type,int maxType) {
	if (type<0 || type>maxType) {
		char msg[1024];
		sprintf(msg,"%s %d is not a valid entity type (it must be between %d and %d)",
			entityType,type, 0, maxType-1);
		FEM_Abort(caller,msg);
	}
}
void FEM_Is_NULL(const char *caller,const char *entityType,int type) {
	char msg[1024];
	sprintf(msg,"%s %d was never set--it cannot now be read",entityType,type);
	FEM_Abort(caller,msg);
}

void FEM_Mesh::copyShape(const FEM_Mesh &src)
{
	node.copyShape(src.node);
	for (int t=0;t<src.elem.size();t++) 
		if (src.elem.has(t)) elem.set(t).copyShape(src.elem.get(t));
	
	for (int s=0;s<src.sparse.size();s++)
		if (src.sparse.has(s)) sparse.set(s).copyShape(src.sparse.get(s));
	
	setSymList(src.getSymList());
}

/// Extract a list of our entities:
int FEM_Mesh::getEntities(int *entities) {
	int len=0;
	entities[len++]=FEM_NODE;
	for (int t=0;t<elem.size();t++) 
		if (elem.has(t)) entities[len++]=FEM_ELEM+t;
	for (int s=0;s<sparse.size();s++)
		if (sparse.has(s)) entities[len++]=FEM_SPARSE+s;
	return len;
}


FILE *FEM_openMeshFile(const char *prefix,int chunkNo,int nchunks,bool forRead)
{
    char fname[256];
    static const char *meshFileNames="%s_vp%d_%d.dat";
    sprintf(fname, meshFileNames, prefix, nchunks, chunkNo);
    FILE *fp = fopen(fname, forRead?"r":"w");
    CkPrintf("FEM> %s %s...\n",forRead?"Reading":"Writing",fname);  
    if(fp==0) {
      FEM_Abort(forRead?"FEM: unable to open input file"
      	:"FEM: unable to create output file.\n");
    }
    return fp;
}

static inline void read_version()
{
    FILE *f = fopen("FEMVERSION", "r");
    if (f!=NULL)  {
	fscanf(f, "%d", &femVersion);
	if (CkMyPe()==0) CkPrintf("FEM> femVersion detected: %d\n", femVersion);
	fclose(f);
    }
}

static inline void write_version()
{
    FILE *f = fopen("FEMVERSION", "w");
    if (f!=NULL)  {
		fprintf(f, "%d", femVersion);
		fclose(f);
    }
}

FEM_Mesh *FEM_readMesh(const char *prefix,int chunkNo,int nChunks)
{
	// find FEM file version number
	static int version_checked = 0;
	if (!version_checked) {
	    version_checked=1;
	    read_version();
	}

	FEM_Mesh *ret=new FEM_Mesh;
	ret->becomeGetting();
        FILE *fp = FEM_openMeshFile(prefix,chunkNo,nChunks,true);
	PUP::fromTextFile p(fp);
	ret->pup(p);
  	fclose(fp);

#ifdef PRINT_SHARED_NODE_INFO
        FEM_Comm &shared = ret->node.shared;
        int numNeighborVPs = shared.size();
        CkPrintf("COMM DATA %d %d ", chunkNo, numNeighborVPs);
        
        for(int i=0;i<numNeighborVPs;i++) {
	  IDXL_List list = shared.getLocalList(i);
          CkPrintf("%d %d ", list.getDest(), list.size()); 
        }
	CkPrintf("\n");
#endif

	return ret;
}

void FEM_writeMesh(FEM_Mesh *m,const char *prefix,int chunkNo,int nChunks)
{
	if (chunkNo == 0) {
	   write_version();
	}
        FILE *fp = FEM_openMeshFile(prefix,chunkNo,nChunks,false);
	PUP::toTextFile p(fp);
	m->pup(p);
	fclose(fp);
}


// Setup the entity FEM_IS_VALID tables
CDECL void FEM_Mesh_allocate_valid_attr(int fem_mesh, int entity_type){  
  FEM_Mesh *m=FEM_Mesh_lookup(fem_mesh,"FEM_Mesh_create_valid_elem");
  FEM_Entity *entity = m->lookup(entity_type,"FEM_Mesh_allocate_valid_attr");
  entity->allocateValid();
}
FORTRAN_AS_C(FEM_MESH_ALLOCATE_VALID_ATTR,
             FEM_Mesh_allocate_valid_attr,
             fem_mesh_allocate_valid_attr, 
             (int *fem_mesh, int *entity_type),  (*fem_mesh, *entity_type) )


// mark an entity as valid
CDECL void FEM_set_entity_valid(int mesh, int entityType, int entityIdx){
  FEM_Mesh *m=FEM_Mesh_lookup(mesh,"FEM_");
  FEM_Entity *entity = m->lookup(entityType,"FEM_");
  entity->set_valid(entityIdx,true);
}
FORTRAN_AS_C(FEM_SET_ENTITY_VALID, 
             FEM_set_entity_valid, 
             fem_set_entity_valid,  
             (int *fem_mesh, int *entity_type, int *entityIdx),  (*fem_mesh, *entity_type, *entityIdx) ) 


// mark an entity as invalid
CDECL void FEM_set_entity_invalid(int mesh, int entityType, int entityIdx){
  FEM_Mesh *m=FEM_Mesh_lookup(mesh,"FEM_Mesh_create_valid_elem");
  FEM_Entity *entity = m->lookup(entityType,"FEM_Mesh_allocate_valid_attr");
  return entity->set_invalid(entityIdx,true);
}
FORTRAN_AS_C(FEM_SET_ENTITY_INVALID,  
             FEM_set_entity_invalid,  
             fem_set_entity_invalid,   
             (int *fem_mesh, int *entity_type, int *entityIdx),  (*fem_mesh, *entity_type, *entityIdx) )  


// Determine if an entity is valid
CDECL int FEM_is_valid(int mesh, int entityType, int entityIdx){
  FEM_Mesh *m=FEM_Mesh_lookup(mesh,"FEM_Mesh_create_valid_elem");
  FEM_Entity *entity = m->lookup(entityType,"FEM_Mesh_allocate_valid_attr");
  return (entity->is_valid(entityIdx) != 0);
}
FORTRAN_AS_C(FEM_IS_VALID,   
             FEM_is_valid,   
             fem_is_valid,    
             (int *fem_mesh, int *entity_type, int *entityIdx),  (*fem_mesh, *entity_type, *entityIdx) )   


// Count number of valid items for this entity type
CDECL unsigned int FEM_count_valid(int mesh, int entityType){
  FEM_Mesh *m=FEM_Mesh_lookup(mesh,"FEM_Mesh_create_valid_elem");
  FEM_Entity *entity = m->lookup(entityType,"FEM_Mesh_allocate_valid_attr");
  return entity->count_valid();
}
FORTRAN_AS_C(FEM_COUNT_VALID,    
             FEM_count_valid,    
             fem_count_valid,     
             (int *fem_mesh, int *entity_type),  (*fem_mesh, *entity_type) )    
 

// Set coordinates for some entity's item number idx 
void FEM_set_entity_coord2(int mesh, int entityType, int idx, double x, double y){
  FEM_Mesh *m=FEM_Mesh_lookup(mesh,"FEM_Mesh_create_valid_elem");
  FEM_Entity *entity = m->lookup(entityType,"FEM_Mesh_allocate_valid_attr");
  entity->set_coord(idx,x,y);
}
void FEM_set_entity_coord3(int mesh, int entityType, int idx, double x, double y, double z){
  FEM_Mesh *m=FEM_Mesh_lookup(mesh,"FEM_Mesh_create_valid_elem");
  FEM_Entity *entity = m->lookup(entityType,"FEM_Mesh_allocate_valid_attr");
  entity->set_coord(idx,x,y,z);
}
