/*
Finite Element Method (FEM) Framework for Charm++
Parallel Programming Lab, Univ. of Illinois 2002
Orion Sky Lawlor, olawlor@acm.org, 12/20/2002

FEM Implementation file: mesh creation and user-data manipulation.
*/
#include "fem.h"
#include "fem_impl.h"
#include "charm-api.h" /*for CDECL, FTN_NAME*/


/*** IDXL Interface ****/
FEM_Comm_Holder::FEM_Comm_Holder(FEM_Comm *sendComm, FEM_Comm *recvComm)
	:comm(sendComm,recvComm)
{
	owner=NULL; 
	idx=-1; 
}
void FEM_Comm_Holder::registerIdx(IDXL_Chunk *c) {
	CkAssert(owner==NULL);
	owner=c;
	if (idx!=-1) // had an old index: try to get it back
		idx=owner->addStatic(&comm,idx);
	else //No old index:
		idx=owner->addStatic(&comm);
}
void FEM_Comm_Holder::pup(PUP::er &p) {
	p|idx;
	// if idx!=-1, better hope a "registerIDXL" call is on the way
}

FEM_Comm_Holder::~FEM_Comm_Holder(void)
{
	if (owner) owner->destroy(idx,"FEM_Comm_Holder::~FEM_Comm_Holder"); 
}

/******* FEM_Mesh API ******/

static void checkIsSet(int fem_mesh,bool wantSet,const char *callingRoutine) {
	if (FEM_Mesh_is_set(fem_mesh)!=wantSet) {
		const char *msg=wantSet?"This mesh (%d) is not a setting mesh":
			"This mesh (%d) is not a getting mesh";
		FEM_Abort(callingRoutine,msg,fem_mesh);
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
CDECL void
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
CDECL void
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
	const char *callingRoutine="FEM_Mesh_data_layout";
	FEM_Mesh_data_layout(fem_mesh,entity,attr,data,firstItem,length,
		getLayouts().get(layout,callingRoutine));
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
	const char *callingRoutine="FEM_Mesh_data_offset";
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

// Accessor API:

CDECL void 
FEM_Mesh_set_length(int fem_mesh,int entity,int newLength) {
	const char *callingRoutine="FEM_Mesh_set_length";
	FEMAPI(callingRoutine);
	checkIsSet(fem_mesh,true,callingRoutine);
	FEM_Entity_lookup(fem_mesh,entity,callingRoutine)->setLength(newLength);
}
FORTRAN_AS_C(FEM_MESH_SET_LENGTH,FEM_Mesh_set_length,fem_mesh_set_length,
	(int *fem_mesh,int *entity,int *newLength),
	(*fem_mesh,*entity,*newLength)
)


CDECL int 
FEM_Mesh_get_length(int fem_mesh,int entity) {
	const char *callingRoutine="FEM_Mesh_get_length";
	FEMAPI(callingRoutine);
	int len=FEM_Entity_lookup(fem_mesh,entity,callingRoutine)->size();
	if (len==-1) return 0; //Special marker value-- shouldn't make it outside...
	return len;
}
FORTRAN_AS_C_RETURN(int,
	FEM_MESH_GET_LENGTH,FEM_Mesh_get_length,fem_mesh_get_length,
	(int *fem_mesh,int *entity),(*fem_mesh,*entity)
)


CDECL void 
FEM_Mesh_set_width(int fem_mesh,int entity,int attr,int newWidth) {
	const char *callingRoutine="FEM_Mesh_set_width";
	FEMAPI(callingRoutine);
	checkIsSet(fem_mesh,true,callingRoutine);
	FEM_Attribute_lookup(fem_mesh,entity,attr,callingRoutine)->setWidth(newWidth,callingRoutine);
}
FORTRAN_AS_C(FEM_MESH_SET_WIDTH,FEM_Mesh_set_width,fem_mesh_set_width,
	(int *fem_mesh,int *entity,int *attr,int *newWidth),
	(*fem_mesh,*entity,*attr,*newWidth)
)

CDECL int 
FEM_Mesh_get_width(int fem_mesh,int entity,int attr) {
	const char *callingRoutine="FEM_Mesh_get_width";
	FEMAPI(callingRoutine);
	return FEM_Attribute_lookup(fem_mesh,entity,attr,callingRoutine)->getWidth();
}
FORTRAN_AS_C_RETURN(int,
	FEM_MESH_GET_WIDTH,FEM_Mesh_get_width,fem_mesh_get_width,
	(int *fem_mesh,int *entity,int *attr),(*fem_mesh,*entity,*attr)
)

CDECL int 
FEM_Mesh_get_datatype(int fem_mesh,int entity,int attr) {
	const char *callingRoutine="FEM_Mesh_get_datatype";
	FEMAPI(callingRoutine);
	return FEM_Attribute_lookup(fem_mesh,entity,attr,callingRoutine)->getDatatype();
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

CDECL IDXL_t 
FEM_Comm_shared(int fem_mesh,int entity) {
	const char *callingRoutine="FEM_Comm_shared";
	FEMAPI(callingRoutine); FEMchunk *fem=FEMchunk::lookup(callingRoutine);
	if (entity!=FEM_NODE) FEM_Abort(callingRoutine,"Only shared nodes supported");
	return fem->meshLookup(fem_mesh,callingRoutine)->node.sharedIDXL.getIndex(fem);
}
FORTRAN_AS_C_RETURN(int,
	FEM_COMM_SHARED,FEM_Comm_shared,fem_comm_shared,
	(int *fem_mesh,int *entity),(*fem_mesh,*entity)
)

CDECL IDXL_t 
FEM_Comm_ghost(int fem_mesh,int entity) {
	const char *callingRoutine="FEM_Comm_ghost";
	FEMAPI(callingRoutine); FEMchunk *fem=FEMchunk::lookup(callingRoutine);
	FEM_Entity *e=fem->meshLookup(fem_mesh,callingRoutine)->
		lookup(entity,callingRoutine);
	if (e->isGhost()) FEM_Abort(callingRoutine,"Can only call FEM_Comm_ghost on real entity type");
	return e->ghostIDXL.getIndex(fem);
}
FORTRAN_AS_C_RETURN(int,
	FEM_COMM_GHOST,FEM_Comm_ghost,fem_comm_ghost,
	(int *fem_mesh,int *entity),(*fem_mesh,*entity)
)


// Internal API:
void FEM_Mesh_data_layout(int fem_mesh,int entity,int attr, 	
  	void *data, int firstItem,int length, const IDXL_Layout &layout) 
{
	if (length==0) return;
	const char *callingRoutine="FEM_Mesh_data";
	FEMAPI(callingRoutine);
	FEM_Mesh *m=FEM_Mesh_lookup(fem_mesh,callingRoutine);
	FEM_Attribute *a=m->lookup(entity,callingRoutine)->
		lookup(attr,callingRoutine);
	
	if (m->isSetting()) 
		a->set(data,firstItem,length,layout,callingRoutine);
	else /* m->isGetting()*/
		a->get(data,firstItem,length,layout,callingRoutine);
}


FEM_Entity *FEM_Entity_lookup(int fem_mesh,int entity,const char *callingRoutine) {
	return FEM_Mesh_lookup(fem_mesh,callingRoutine)->lookup(entity,callingRoutine);
}
FEM_Attribute *FEM_Attribute_lookup(int fem_mesh,int entity,int attr,const char *callingRoutine) {
	return FEM_Entity_lookup(fem_mesh,entity,callingRoutine)->lookup(attr,callingRoutine);
}


/************** FEM_Attribute ****************/

/// Return the human-readable version of this FEM_ATTR code.
///  For example, FEM_attr2name(FEM_CONN)=="FEM_CONN".
const char *FEM_Get_attr_name(int attr,char *storage) 
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
	default: break;
	};
	sprintf(storage,"unknown attribute code (%d)",attr);
	return storage;
}

//Abort with a nice error message saying: 
// Our <field> was previously set to <cur>; it cannot now be <operation> <next>
void FEM_Attribute::bad(const char *field,bool forRead,int cur,int next,const char *callingRoutine) const
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
	
	FEM_Abort(callingRoutine,errBuf);
}


FEM_Attribute::FEM_Attribute(FEM_Entity *e_,int attr_)
		:e(e_),ghost(0),attr(attr_),width(-1),datatype(-1), allocated(false)
{
	tryAllocate();
}
void FEM_Attribute::pup(PUP::er &p) {
	// e, attr, and ghost are always set by the constructor
	p|width;
	p|datatype;
	if (p.isUnpacking()) tryAllocate();
}
FEM_Attribute::~FEM_Attribute() {}

void FEM_Attribute::setLength(int next,const char *callingRoutine) {
	int cur=getLength();
	if (next==cur) return; //Already set--nothing to do 
	if (cur!=-1) bad("length",false,cur,next, callingRoutine);
	e->setLength(next);
	tryAllocate();
}
	
void FEM_Attribute::setWidth(int next,const char *callingRoutine) {
	int cur=getWidth();
	if (next==cur) return; //Already set--nothing to do 
	if (cur!=-1) bad("width",false,cur,next, callingRoutine);
	width=next;
	tryAllocate();
	if (ghost) ghost->setWidth(width,callingRoutine);
}

void FEM_Attribute::setDatatype(int next,const char *callingRoutine) {
	int cur=getDatatype();
	if (next==cur) return; //Already set--nothing to do 
	if (cur!=-1) bad("datatype",false,cur,next, callingRoutine);
	datatype=next;
	tryAllocate();
	if (ghost) ghost->setDatatype(datatype,callingRoutine);
}

void FEM_Attribute::copyShape(const FEM_Attribute &src) {
	setWidth(src.getWidth());
	setDatatype(src.getDatatype()); //Automatically calls tryAllocate
}
void FEM_Attribute::set(const void *src, int firstItem,int length, 
		const IDXL_Layout &layout, const char *callingRoutine) 
{
	if (getLength()==-1) setLength(length);
	else if (length!=1 && length!=getLength()) 
		bad("length",false,getLength(),length, callingRoutine);
	
	int width=layout.width;
	if (getWidth()==-1) setWidth(width);
	else if (width!=getWidth()) 
		bad("width",false,getWidth(),width, callingRoutine);
	
	int datatype=layout.type;
	if (getDatatype()==-1) setDatatype(datatype);
	else if (datatype!=getDatatype()) 
		bad("datatype",false,getDatatype(),datatype, callingRoutine);
	
	/* Assert: our storage should be allocated now.
	   Our subclass will actually copy user data */
}

void FEM_Attribute::get(void *dest, int firstItem,int length, 
		const IDXL_Layout &layout, const char *callingRoutine)  const
{
	if (length==0) return; //Nothing to get
	if (length!=1 && length!=getLength()) 
		bad("length",true,getLength(),length, callingRoutine);
	
	int width=layout.width;
	if (width!=getWidth()) 
		bad("width",true,getWidth(),width, callingRoutine);
	
	int datatype=layout.type;
	if (datatype!=getDatatype()) 
		bad("datatype",true,getDatatype(),datatype, callingRoutine);
	
	/* our subclass will actually copy into user data */
}

//Check if all three of length, width, and datatype are set.
// If so, call allocate.
void FEM_Attribute::tryAllocate(void) {
	if ((!allocated) && getLength()!=-1 && getWidth()!=-1 && getDatatype()!=-1) {
		allocated=true;
		allocate(getLength(),getWidth(),getDatatype());
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
		const IDXL_Layout &layout, const char *callingRoutine)
{
	super::set(u,f,l,layout,callingRoutine);
	switch(getDatatype()) {
	case FEM_BYTE:  setTableData(u,f,l,IDXL_LAYOUT_CALL(layout),char_data); break;
	case FEM_INT: setTableData(u,f,l,IDXL_LAYOUT_CALL(layout),int_data); break;
	case FEM_FLOAT: setTableData(u,f,l,IDXL_LAYOUT_CALL(layout),float_data); break;
	case FEM_DOUBLE: setTableData(u,f,l,IDXL_LAYOUT_CALL(layout),double_data); break;
	}
}
	
void FEM_DataAttribute::get(void *u, int f,int l,
		const IDXL_Layout &layout, const char *callingRoutine) const
{
	super::get(u,f,l,layout,callingRoutine);
	switch(getDatatype()) {
	case FEM_BYTE:  getTableData(u,f,l,IDXL_LAYOUT_CALL(layout),char_data); break;
	case FEM_INT: getTableData(u,f,l,IDXL_LAYOUT_CALL(layout),int_data); break;
	case FEM_FLOAT: getTableData(u,f,l,IDXL_LAYOUT_CALL(layout),float_data); break;
	case FEM_DOUBLE: getTableData(u,f,l,IDXL_LAYOUT_CALL(layout),double_data); break;
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
	case FEM_INT: int_data->setRow(dstEntity,dsrc->int_data->getRow(srcEntity)); break;
	case FEM_FLOAT: float_data->setRow(dstEntity,dsrc->float_data->getRow(srcEntity)); break;
	case FEM_DOUBLE: double_data->setRow(dstEntity,dsrc->double_data->getRow(srcEntity)); break;
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
static int type2base(int base_type,const char *callingRoutine) {
	if (base_type==FEM_INDEX_0) return 0;
	if (base_type==FEM_INDEX_1) return 1;
	FEM_Abort(callingRoutine,"You must use the datatype FEM_INDEX_0 or FEM_INDEX_1 with FEM_CONN, not %d",
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
		const IDXL_Layout &layout,const char *callingRoutine)
{
	IDXL_Layout lo=layout; lo.type=FEM_INT; //Pretend it's always int data, not INDEX
	super::set(src,firstItem,length,lo,callingRoutine);
	
	int indexBase=type2base(layout.type,callingRoutine);
	setIndexTableData(src,firstItem,length,IDXL_LAYOUT_CALL(layout),&idx,indexBase);
	
	if (checker) 
		for (int r=0;r<length;r++)
			checker->check(firstItem+r,idx,callingRoutine);
}

void FEM_IndexAttribute::get(void *dest, int firstItem,int length, 
		const IDXL_Layout &layout,const char *callingRoutine) const
{
	IDXL_Layout lo=layout; lo.type=FEM_INT; //Pretend it's always int data, not INDEX
	super::get(dest,firstItem,length,lo,callingRoutine);
	
	int indexBase=type2base(layout.type,callingRoutine);
	getIndexTableData(dest,firstItem,length,IDXL_LAYOUT_CALL(layout),&idx,indexBase);
}

void FEM_IndexAttribute::copyEntity(int dstEntity,const FEM_Attribute &src,int srcEntity)
{
	const FEM_IndexAttribute *csrc=(const FEM_IndexAttribute *)&src;
	idx.setRow(dstEntity,csrc->idx.getRow(srcEntity));
}

/********************** Entity **************************/

FEM_Entity::FEM_Entity(FEM_Entity *ghost_) //Default constructor
	:length(-1), ghost(ghost_), sym(0), globalno(0), 
	 ghostIDXL(ghost?&ghostSend:NULL, ghost?&ghost->ghostRecv:NULL)
{
	//No attributes initially
} 
void FEM_Entity::pup(PUP::er &p) {
	p|length;
	
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
void FEM_Entity::registerIDXL(IDXL_Chunk *c) {
	ghostIDXL.registerIDXL(c);
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

void FEM_Entity::setLength(int newlen) {
	if (size()!=newlen) {
		length=newlen;
		// Each of our attributes need to be expanded for our new length:
		for (int a=0;a<attributes.size();a++)
			attributes[a]->reallocate();
	}
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
	int dstEntity=size(); if (dstEntity<0) dstEntity=0; //length starts out at -1!
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
FEM_Attribute *FEM_Entity::lookup(int attr,const char *callingRoutine) {
	//Try to find an existing attribute (FIXME: keep attributes in a map, to speed this up)
	for (int a=0;a<attributes.size();a++) {
		if (attributes[a]->getAttr()==attr)
			return attributes[a];
	}
	
	//If we get here, no existing attribute fits the bill: create one
	create(attr,callingRoutine);
	
	// If create did its job, the next lookup should succeed:
	return lookup(attr,callingRoutine);
}

/**
 * Create a new attribute from an FEM_ATTR code.
 * The default implementation handles FEM_DATA tags; entity-specific
 * attributes (like FEM_CONN) need to be overridden and created 
 * by subclasses.
 */
void FEM_Entity::create(int attr,const char *callingRoutine) {
	if (attr<=FEM_ATTRIB_TAG_MAX) 
	{ //It's a valid user data tag
		add(new FEM_DataAttribute(this,attr));
	}
	else if (attr==FEM_SYMMETRIES) {
		allocateSym();
	} else if (attr==FEM_GLOBALNO) {
		allocateGlobalno();
	} else {
	//It's an unrecognized tag: abort
		char attrNameStorage[256], msg[1024];
		sprintf(msg,"Could not locate the attribute %s for entity %s",
			FEM_Get_attr_name(attr,attrNameStorage), getName());
		FEM_Abort(callingRoutine,msg);
	}
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


void FEM_Entity::allocateGlobalno(void) {
	if (globalno) CkAbort("FEM_Entity::allocateGlobalno called, but already allocated");
	globalno=new FEM_IndexAttribute(this,FEM_GLOBALNO,NULL);
	add(globalno); // globalno will be deleted via attributes list now
	globalno->setWidth(1);
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
void FEM_Entity::copyOldGlobalno(const FEM_Entity &e) {
	if ((!hasGlobalno()) && e.hasGlobalno() && size()>=e.size()) {
		for (int i=0;i<size();i++) 
			setGlobalno(i,e.getGlobalno(i));
	}
}

/********************** Node *****************/
FEM_Node::FEM_Node(FEM_Node *ghost_) 
	:FEM_Entity(ghost_), primary(0), sharedIDXL(&shared,&shared)
{}
void FEM_Node::allocatePrimary(void) {
	if (primary) CkAbort("FEM_Node::allocatePrimary called, but already allocated");
	primary=new FEM_DataAttribute(this,FEM_NODE_PRIMARY);
	add(primary); // primary will be deleted by FEM_Entity now
	primary->setWidth(1); //Only 1 flag per node
	primary->setDatatype(FEM_BYTE);
}

void FEM_Node::pup(PUP::er &p) {
	p.comment(" ---------------- Nodes ------------------ ");	
	super::pup(p);
	p.comment(" ---- Shared nodes ----- ");	
	shared.pup(p);
	p.comment(" shared nodes IDXL ");
	sharedIDXL.pup(p);
}
void FEM_Node::registerIDXL(IDXL_Chunk *c) {
	super::registerIDXL(c);
	sharedIDXL.registerIDXL(c);
}
FEM_Node::~FEM_Node() {
}

const char *FEM_Node::getName(void) const {return "FEM_NODE";}

void FEM_Node::create(int attr,const char *callingRoutine) {
	if (attr==FEM_NODE_PRIMARY) {
		allocatePrimary();
	} 
	else  super::create(attr,callingRoutine);
}

/********************** Elem *****************/
/// This checker verifies that FEM_Elem::conn's entries are valid node indices.
class FEM_Elem_Conn_Checker : public FEM_IndexAttribute::Checker {
	const FEM_Entity &sizeSrc;
	const FEM_Entity *sizeSrc2;
public:
	FEM_Elem_Conn_Checker(const FEM_Entity &sizeSrc_,const FEM_Entity *sizeSrc2_) 
		:sizeSrc(sizeSrc_), sizeSrc2(sizeSrc2_) {}
	
	void check(int row,const BasicTable2d<int> &table,const char *callingRoutine) const {
		const int *idx=table.getRow(row);
		int n=table.width();
		int max=sizeSrc.size();
		if (sizeSrc2) max+=sizeSrc2->size();
		for (int i=0;i<n;i++) 
			if ((idx[i]<0) || (idx[i]>=max))
			{ /* This index is out of bounds: */
				if (idx[i]<0)
					FEM_Abort(callingRoutine,"Connectivity entry %d's value, %d, is negative",row,idx[i]);
				else /* (idx[i]>=max) */
					FEM_Abort(callingRoutine,
						"Connectivity entry %d's value, %d, should be less than the number of nodes, %d",
						row,idx[i],max);
			}
	}
};

FEM_Elem::FEM_Elem(const FEM_Mesh &mesh, FEM_Elem *ghost_) 
	:FEM_Entity(ghost_)
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

const char *FEM_Elem::getName(void) const {
	return "FEM_ELEM";
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
	
	void check(int row,const BasicTable2d<int> &table,const char *callingRoutine) const {
		//assert: table.getWidth==2
		const int *elem=table.getRow(row);
		int maxT=mesh.elem.size();
		if ((elem[0]<0) || (elem[1]<0))
			FEM_Abort(callingRoutine,"Sparse element entry %d's values, %d and %d, are negative",
				row,elem[0],elem[1]);
		int t=elem[0];
		if (t>=maxT)
			FEM_Abort(callingRoutine,"Sparse element entry %d's element type, %d, is too big",
				row,elem[0]);
		if (elem[1]>=mesh.elem[t].size())
			FEM_Abort(callingRoutine,"Sparse element entry %d's element index, %d, is too big",
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

void FEM_Sparse::create(int attr,const char *callingRoutine) {
	if (attr==FEM_SPARSE_ELEM) {
		allocateElem();
	}
	else /*super*/ FEM_Entity::create(attr,callingRoutine);
}


/******************* Mesh *********************/
FEM_Mesh::FEM_Mesh() 
	:node(new FEM_Node(NULL)),
	 elem(*this,"FEM_ELEM"),
	 sparse(*this,"FEM_SPARSE")
{
	m_isSetting=true; //Meshes start out setting
}
FEM_Mesh::~FEM_Mesh() {
}

FEM_Entity *FEM_Mesh::lookup(int entity,const char *callingRoutine) {
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
		FEM_Abort(callingRoutine,"Expected an entity type (FEM_NODE, FEM_ELEM, etc.) but got %d",entity);
	return e;
}
const FEM_Entity *FEM_Mesh::lookup(int entity,const char *callingRoutine) const {
	/// FIXME: the const version is quite similar to the above, 
	/// but it should *not* create new Entity types...
	return ((FEM_Mesh *)this)->lookup(entity,callingRoutine);
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
}

void FEM_Mesh::registerIDXL(IDXL_Chunk *c) {
	node.registerIDXL(c);
	elem.registerIDXL(c);
	sparse.registerIDXL(c);
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
#ifndef CMK_OPTIMIZE
	if (t_max<0 || t_max>elem.size()) {
		CkPrintf("FEM> Invalid element type %d used!\n");
		CkAbort("FEM> Invalid element type");
	}
#endif
	int ret=0;
	for (int t=0;t<t_max;t++) 
		if (elem.has(t))
			ret+=elem.get(t).size();
	return ret;
}

int FEM_Mesh::getGlobalElem(int elType,int elNo) const
{
	int base=nElems(elType); //Global number of first element of this type
#ifndef CMK_OPTIMIZE
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
void FEM_Mesh::copyOldGlobalno(const FEM_Mesh &m) {
	node.copyOldGlobalno(m.node);
	for (int e=0;e<m.elem.size();e++)
		if (m.elem.has(e) && e<elem.size() && elem.has(e)) 
			elem[e].copyOldGlobalno(m.elem[e]);
	for (int s=0;s<m.sparse.size();s++)
		if (m.sparse.has(s) && s<sparse.size() && sparse.has(s)) 
			sparse[s].copyOldGlobalno(m.sparse[s]);
}

void FEM_Index_Check(const char *callingRoutine,const char *entityType,int type,int maxType) {
	if (type<0 || type>maxType) {
		char msg[1024];
		sprintf(msg,"%s %d is not a valid entity type (it must be between %d and %d)",
			entityType,type, 0, maxType-1);
		FEM_Abort(callingRoutine,msg);
	}
}
void FEM_Is_NULL(const char *callingRoutine,const char *entityType,int type) {
	char msg[1024];
	sprintf(msg,"%s %d was never set--it cannot now be read",entityType,type);
	FEM_Abort(callingRoutine,msg);
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


static const char *meshFileNames="fem_mesh_vp%d_%d.dat";

FILE *FEM_openMeshFile(int chunkNo,int nchunks,bool forRead)
{
    char fname[256];
    sprintf(fname, meshFileNames, nchunks, chunkNo);
    FILE *fp = fopen(fname, forRead?"r":"w");
    CkPrintf("FEM> %s %s...\n",forRead?"Reading":"Writing",fname);  
    if(fp==0) {
      FEM_Abort(forRead?"FEM: unable to open input file"
      	:"FEM: unable to create output file.\n");
    }
    return fp;
}

FEM_Mesh *FEM_Mesh_read(int chunkNo,int nChunks,const char *dirName)
{
	FEM_Mesh *ret=new FEM_Mesh;
	PUP::fromTextFile p(FEM_openMeshFile(chunkNo,nChunks,true));
	ret->pup(p);
	return ret;
}
void FEM_Mesh_write(FEM_Mesh *m,int chunkNo,int nChunks,const char *dirName)
{
	PUP::toTextFile p(FEM_openMeshFile(chunkNo,nChunks,false));
	m->pup(p);
}



