/**
 * IDXL--Index List communication library.
 * C and Fortran-callable interface routines to library.
 */
#include "charm-api.h"
#include "idxl.h"

/************************* IDXL itself ************************/

/** Create a new, empty index list. Must eventually call IDXL_Destroy on this list. */
CDECL IDXL_t 
IDXL_Create(void) {
	const char *callingRoutine="IDXL_Create";
	IDXLAPI(callingRoutine); IDXL_Chunk *c=IDXL_Chunk::lookup(callingRoutine);
	return c->addDynamic();
}
FORTRAN_AS_C_RETURN(int, IDXL_CREATE,IDXL_Create,idxl_create,
	(void), () );


/** Print the send and recv indices in this communications list: */
void IDXL_Print(IDXL_t l_t)
{
	const char *callingRoutine="IDXL_Print";
	IDXLAPI(callingRoutine); IDXL_Chunk *c=IDXL_Chunk::lookup(callingRoutine);
	const IDXL &l=c->lookup(l_t,callingRoutine);
	if (l.isSingle()) {
		l.getSend().print();
	} else {
		CkPrintf("Send ");
		l.getSend().print();
		CkPrintf("Recv ");
		l.getRecv().print();
	}
}
FORTRAN_AS_C(IDXL_PRINT,IDXL_Print,idxl_print,
	(int *l), (*l) );

/** Copy the indices in src into l. */
CDECL void 
IDXL_Copy(IDXL_t l,IDXL_t src) {
	IDXL_Combine(l,src,0,0);
}
FORTRAN_AS_C(IDXL_COPY,IDXL_Copy,idxl_copy,
	(int *l, int *src), (*l,*src));


//Add all of src's entities into dest, shifting by idxShift
static void shiftSide(IDXL_Side &dest,int idxShift)
{
	int sNo,sMax=dest.size();
	for (sNo=0;sNo<sMax;sNo++) {
		IDXL_List &s=dest.setLocalList(sNo);
		for (int i=0;i<s.size();i++) s[i]+=idxShift;
	}
	dest.flushMap();
}
/** Shift the indices of this list by this amount. */
CDECL void 
IDXL_Shift(IDXL_t l_t,int startSend,int startRecv) {
	const char *callingRoutine="IDXL_Shift";
	IDXLAPI(callingRoutine); IDXL_Chunk *c=IDXL_Chunk::lookup(callingRoutine);
	IDXL &l=c->lookup(l_t,callingRoutine);
	if (l.isSingle()) {
		if (startSend!=startRecv) //FIXME: handle this case, instead of aborting
			IDXL_Abort(callingRoutine,"Cannot independently shift send and recv for this IDXL_t");
		shiftSide(l.getSend(),startSend);
	}
	else /* l has separate send and recv */ {
		shiftSide(l.getSend(),startSend);
		shiftSide(l.getRecv(),startRecv);
	}
}
FORTRAN_AS_C(IDXL_SHIFT,IDXL_Shift,idxl_shift,
	(int *l, int *startSend,int *startRecv), 
	(*l, *startSend-1, *startRecv-1));


//Add all of src's entities into dest, shifting by idxShift
static void combineSide(IDXL_Side &dest,const IDXL_Side &src,
	int idxShift)
{
	int sNo,sMax=src.size();
	for (sNo=0;sNo<sMax;sNo++) {
		const IDXL_List &s=src.getLocalList(sNo);
		IDXL_List &d=dest.addList(s.getDest());
		for (int i=0;i<s.size();i++) {
			d.push_back(s[i]+idxShift);
		}
	}
	dest.flushMap();
}
/** Add these indices into our list.  Any duplicates will get listed twice.
 * @param l the list to add indices to.
 * @param src the list of indices to read from and add.
 * @param startSend value to add to send indices of src.
 * @param startRecv value to add to recv indices of src.
  */
CDECL void 
IDXL_Combine(IDXL_t dest_t,IDXL_t src_t,int startSend,int startRecv)
{
	const char *callingRoutine="IDXL_Combine";
	IDXLAPI(callingRoutine); IDXL_Chunk *c=IDXL_Chunk::lookup(callingRoutine);
	const IDXL &src=c->lookup(src_t,callingRoutine);
	IDXL &dest=c->lookup(dest_t,callingRoutine);
	if (src.isSingle()!=dest.isSingle()) //FIXME: handle this case instead of aborting
		IDXL_Abort(callingRoutine,"Cannot combine IDXL_t's %d and %d,"
			"because one is single and the other is double.",src_t,dest_t);
	if (dest.isSingle()) 
	{ /* A single send and recv list: just copy once */
		if (startSend!=startRecv) //FIXME: handle this case instead of aborting
			IDXL_Abort(callingRoutine,"Cannot independently shift send and recv for this IDXL_t");
		combineSide(dest.getSend(),src.getSend(),startSend);
	}
	else { /* Separate send and recv lists: copy each list */
		combineSide(dest.getSend(),src.getSend(),startSend);
		combineSide(dest.getRecv(),src.getRecv(),startRecv);
	}
}
FORTRAN_AS_C(IDXL_COMBINE,IDXL_Combine,idxl_combine,
	(int *dest, int *src, int *startSend,int *startRecv), 
	(*dest,*src, *startSend-1, *startRecv-1));


/** Sort the indices in this list by these 2D coordinates */
// FIXME: void IDXL_Sort_2d(IDXL_t l,double *coord2d);
/** Sort the indices in this list by these 3D coordinates */
// FIXME: void IDXL_Sort_3d(IDXL_t l,double *coord3d);


//Add a new entity at the given local index.  Copy the communication
// list from the intersection of the communication lists of the given entities.
static void splitEntity(IDXL_Side &c,
	int localIdx,int nBetween,int *between,int idxbase)
{
	//Find the commRecs for the surrounding nodes
	const IDXL_Rec *tween[20];
	int w;
	for (w=0;w<nBetween;w++) {
		tween[w]=c.getRec(between[w]-idxbase);
		if (tween[w]==NULL) 
			return; //An unshared entity! Thus a private-only addition
	}
	//Make a new commRec as the interesection of the surrounding entities--
	// we loop over the first entity's comm. list
	for (int zs=tween[0]->getShared()-1;zs>=0;zs--) {
		int chk=tween[0]->getChk(zs);
		//Make sure this processor shares all our entities
		for (w=0;w<nBetween;w++)
			if (!tween[w]->hasChk(chk))
				break;
		if (w==nBetween) //The new node is shared with chk
			c.addNode(localIdx,chk);
	}
}

static void splitEntity(IDXL &l,
	int localIdx,int nBetween,int *between,int idxbase)
{
	splitEntity(l.getSend(),localIdx,nBetween,between,idxbase);
	if (!l.isSingle()) //Also have to split recv lists:
		splitEntity(l.getRecv(),localIdx,nBetween,between,idxbase);
}

CDECL void IDXL_Add_entity(IDXL_t l_t,int localIdx,int nBetween,int *between)
{
	const char *callingRoutine="IDXL_Add_entity";
	IDXLAPI(callingRoutine); IDXL_Chunk *c=IDXL_Chunk::lookup(callingRoutine);
	IDXL &l=c->lookup(l_t,callingRoutine);
	splitEntity(l,localIdx,nBetween,between,0);
}
FDECL void FTN_NAME(IDXL_ADD_ENTITY,idxl_add_entity)
	(int *l_t,int *localIdx,int *nBetween,int *between)
{
	const char *callingRoutine="IDXL_Add_entity";
	IDXLAPI(callingRoutine); IDXL_Chunk *c=IDXL_Chunk::lookup(callingRoutine);
	IDXL &l=c->lookup(*l_t,callingRoutine);
	splitEntity(l,*localIdx-1,*nBetween,between,1);	
}



/** Throw away this index list */
void IDXL_Destroy(IDXL_t l) {
	const char *callingRoutine="IDXL_Destroy";
	IDXLAPI(callingRoutine); IDXL_Chunk *c=IDXL_Chunk::lookup(callingRoutine);
	c->destroy(l);
}
FORTRAN_AS_C(IDXL_DESTROY,IDXL_Destroy,idxl_destroy,  (int *l), (*l));


/************************* IDXL_Layout ************************/
CDECL IDXL_Layout_t 
IDXL_Layout_create(int type, int width)
{
	const char *callingRoutine="IDXL_Create_simple_data";
	IDXLAPI(callingRoutine);
	return getLayouts().put(IDXL_Layout(type, width));
}
FORTRAN_AS_C_RETURN(int,
	IDXL_LAYOUT_CREATE,IDXL_Layout_create,idxl_layout_create,
	(int *t,int *w), (*t,*w))

CDECL IDXL_Layout_t 
IDXL_Layout_offset(int type, int width, int offsetBytes, int distanceBytes,int skewBytes)
{
	const char *callingRoutine="IDXL_Create_data";
	IDXLAPI(callingRoutine);
	return getLayouts().put(IDXL_Layout(type,width, offsetBytes,distanceBytes,skewBytes));
}
FORTRAN_AS_C_RETURN(int,
	IDXL_LAYOUT_OFFSET,IDXL_Layout_offset,idxl_layout_offset,
	(int *t,int *w,int *o,int *d,int *s), (*t,*w,*o,*d,*s))

#define GET_DATA_DECL(CAPS,Cname,lowercase, field) \
CDECL int Cname(IDXL_Layout_t d) \
{ \
	IDXLAPI(#Cname); \
	return getLayouts().get(d,#Cname).field;\
}\
FORTRAN_AS_C_RETURN(int, IDXL_GET_LAYOUT_##CAPS,Cname,idxl_get_layout_##lowercase, \
	(int *d), (*d))

GET_DATA_DECL(TYPE,IDXL_Get_layout_type,type, type)
GET_DATA_DECL(WIDTH,IDXL_Get_layout_width,width, width)
GET_DATA_DECL(DISTANCE,IDXL_Get_layout_distance,distance, distance)

CDECL void 
IDXL_Layout_destroy(IDXL_Layout_t l) {
	const char *callingRoutine="IDXL_Layout_destroy";
	IDXLAPI(callingRoutine);
	return getLayouts().destroy(l,callingRoutine);
}
FORTRAN_AS_C(IDXL_LAYOUT_DESTROY,IDXL_Layout_destroy,idxl_layout_destroy,
	(int *l), (*l) )

/************************* IDXL_Comm ************************/
/** Comm_begin begins a message exchange.  */
CDECL IDXL_Comm_t 
IDXL_Comm_begin(int tag, int context) {
	const char *callingRoutine="IDXL_Create_data";
	IDXLAPI(callingRoutine); IDXL_Chunk *c=IDXL_Chunk::lookup(callingRoutine);
	return c->addComm(tag,context);
}
FORTRAN_AS_C_RETURN(int,
	IDXL_COMM_BEGIN,IDXL_Comm_begin,idxl_comm_begin,
	(int *tag,int *context), (*tag,*context))

// A bunch of useful locals for use in IDXL_Comm API routines:
#define COMM_BROILERPLATE(routineName) \
	const char *callingRoutine=routineName; \
	IDXLAPI(callingRoutine); IDXL_Chunk *c=IDXL_Chunk::lookup(callingRoutine); \
	const IDXL &list=c->lookup(dest,callingRoutine); \
	const IDXL_Layout *dt=&c->layouts.get(type,callingRoutine);

/** Remote-copy this data on flush/wait. */
CDECL void 
IDXL_Comm_sendrecv(IDXL_Comm_t m,IDXL_t dest, IDXL_Layout_t type, void *data) {
	COMM_BROILERPLATE("IDXL_Comm_sendrecv");
	IDXL_Comm *comm=NULL;
	if (m!=0) 
		comm=c->lookupComm(m,callingRoutine);
	else /* m==0, use a separate temporary comm */
		comm=c->lookupComm(c->addComm(0,0),callingRoutine);
	comm->send(&list.getSend(),dt,data);
	comm->recv(&list.getRecv(),dt,data);
	if (m==0) 
		c->waitComm(comm);
}
FORTRAN_AS_C(IDXL_COMM_SENDRECV,IDXL_Comm_sendrecv,idxl_comm_sendrecv,
	(int *m,int *dest,int *type,void *data), (*m,*dest,*type,data))


/** Sum this data with the remote values during flush and wait. */
void IDXL_Comm_sendsum(IDXL_Comm_t m,IDXL_t dest, IDXL_Layout_t type, void *data) {
	COMM_BROILERPLATE("IDXL_Comm_sendsum");
	IDXL_Comm *comm=NULL;
	if (m!=0) 
		comm=c->lookupComm(m,callingRoutine);
	else /* m==0, use a separate temporary comm */
		comm=c->lookupComm(c->addComm(0,0),callingRoutine);
	comm->send(&list.getSend(),dt,data);
	comm->sum(&list.getRecv(),dt,data);
	if (m==0) 
		c->waitComm(comm);
}
FORTRAN_AS_C(IDXL_COMM_SENDSUM,IDXL_Comm_sendsum,idxl_comm_sendsum,
	(int *m,int *dest,int *type,void *data), (*m,*dest,*type,data))

/** Send this data out when flush is called. Must be paired with a recv or sum call */
void IDXL_Comm_send(IDXL_Comm_t m,IDXL_t dest, IDXL_Layout_t type, const void *srcData){
	COMM_BROILERPLATE("IDXL_Comm_send");
	c->lookupComm(m,callingRoutine)->send(&list.getSend(),dt,srcData);
}
FORTRAN_AS_C(IDXL_COMM_SEND,IDXL_Comm_send,idxl_comm_send,
	(int *m,int *dest,int *type,const void *data), (*m,*dest,*type,data))

/** Copy this data from the remote values when wait is called. */
void IDXL_Comm_recv(IDXL_Comm_t m,IDXL_t dest, IDXL_Layout_t type, void *destData) {
	COMM_BROILERPLATE("IDXL_Comm_recv");
	c->lookupComm(m,callingRoutine)->recv(&list.getRecv(),dt,destData);
}
FORTRAN_AS_C(IDXL_COMM_RECV,IDXL_Comm_recv,idxl_comm_recv,
	(int *m,int *dest,int *type,void *data), (*m,*dest,*type,data))

/** Add this data with the remote values when wait is called. */
void IDXL_Comm_sum(IDXL_Comm_t m,IDXL_t dest, IDXL_Layout_t type, void *sumData) {
	COMM_BROILERPLATE("IDXL_Comm_recv");
	c->lookupComm(m,callingRoutine)->sum(&list.getRecv(),dt,sumData);
}
FORTRAN_AS_C(IDXL_COMM_SUM,IDXL_Comm_sum,idxl_comm_sum,
	(int *m,int *dest,int *type,void *data), (*m,*dest,*type,data))

/** Send all outgoing data. */
void IDXL_Comm_flush(IDXL_Comm_t m) {
	const char *callingRoutine="IDXL_Comm_flush"; 
	IDXLAPI(callingRoutine); IDXL_Chunk *c=IDXL_Chunk::lookup(callingRoutine); 
	c->flushComm(c->lookupComm(m,callingRoutine));
}

/** Block until all communication is complete. This destroys the IDXL_Comm. */
void IDXL_Comm_wait(IDXL_Comm_t m) {
	const char *callingRoutine="IDXL_Comm_flush"; 
	IDXLAPI(callingRoutine); IDXL_Chunk *c=IDXL_Chunk::lookup(callingRoutine); 
	c->waitComm(c->lookupComm(m,callingRoutine));
}





