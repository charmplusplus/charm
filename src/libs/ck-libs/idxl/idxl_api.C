/**
 * IDXL--Index List communication library.
 * C and Fortran-callable interface routines to library.
 */
#include "charm-api.h"
#include "idxl.h"
#include "tcharm.h"

/************************* IDXL itself ************************/

/** Create a new, empty index list. Must eventually call IDXL_Destroy on this list. */
CDECL IDXL_t 
IDXL_Create(void) {
	const char *caller="IDXL_Create";
	IDXLAPI(caller); IDXL_Chunk *c=IDXL_Chunk::get(caller);
	return c->addDynamic();
}
FORTRAN_AS_C_RETURN(int, IDXL_CREATE,IDXL_Create,idxl_create,
	(void), () )


/** Print the send and recv indices in this communications list: */
void IDXL_Print(IDXL_t l_t)
{
	const char *caller="IDXL_Print";
	IDXLAPI(caller); IDXL_Chunk *c=IDXL_Chunk::get(caller);
	const IDXL &l=c->lookup(l_t,caller);
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
	(int *l), (*l) )

/** Copy the indices in src into l. */
CDECL void 
IDXL_Copy(IDXL_t l,IDXL_t src) {
	IDXL_Combine(l,src,0,0);
}
FORTRAN_AS_C(IDXL_COPY,IDXL_Copy,idxl_copy,
	(int *l, int *src), (*l,*src))


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
	const char *caller="IDXL_Shift";
	IDXLAPI(caller); IDXL_Chunk *c=IDXL_Chunk::get(caller);
	IDXL &l=c->lookup(l_t,caller);
	if (l.isSingle()) {
		if (startSend!=startRecv) //FIXME: handle this case, instead of aborting
			IDXL_Abort(caller,"Cannot independently shift send and recv for this IDXL_t");
		shiftSide(l.getSend(),startSend);
	}
	else /* l has separate send and recv */ {
		shiftSide(l.getSend(),startSend);
		shiftSide(l.getRecv(),startRecv);
	}
}
FORTRAN_AS_C(IDXL_SHIFT,IDXL_Shift,idxl_shift,
	(int *l, int *startSend,int *startRecv), 
	(*l, *startSend-1, *startRecv-1))


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
	const char *caller="IDXL_Combine";
	IDXLAPI(caller); IDXL_Chunk *c=IDXL_Chunk::get(caller);
	const IDXL &src=c->lookup(src_t,caller);
	IDXL &dest=c->lookup(dest_t,caller);
	if (src.isSingle()!=dest.isSingle()) //FIXME: handle this case instead of aborting
		IDXL_Abort(caller,"Cannot combine IDXL_t's %d and %d,"
			"because one is single and the other is double.",src_t,dest_t);
	if (dest.isSingle()) 
	{ /* A single send and recv list: just copy once */
		if (startSend!=startRecv) //FIXME: handle this case instead of aborting
			IDXL_Abort(caller,"Cannot independently shift send and recv for this IDXL_t");
		combineSide(dest.getSend(),src.getSend(),startSend);
	}
	else { /* Separate send and recv lists: copy each list */
		combineSide(dest.getSend(),src.getSend(),startSend);
		combineSide(dest.getRecv(),src.getRecv(),startRecv);
	}
}
FORTRAN_AS_C(IDXL_COMBINE,IDXL_Combine,idxl_combine,
	(int *dest, int *src, int *startSend,int *startRecv), 
	(*dest,*src, *startSend-1, *startRecv-1))


/** Sort the indices in this list by these 2D coordinates */
// FIXME: void IDXL_Sort_2d(IDXL_t l,double *coord2d);
CDECL void IDXL_Sort_2d(IDXL_t l_t,double *coord2d){
	const char *caller = "IDXL_Sort_2d";
	IDXLAPI(caller); IDXL_Chunk *c=IDXL_Chunk::get(caller);
	IDXL &l=c->lookup(l_t,caller);
	l.sort2d(coord2d);
}


/** Sort the indices in this list by these 3D coordinates */
// FIXME: void IDXL_Sort_3d(IDXL_t l,double *coord3d);
CDECL void IDXL_Sort_3d(IDXL_t l_t,double *coord3d){
	const char *caller = "IDXL_Sort_3d";
	IDXLAPI(caller); IDXL_Chunk *c=IDXL_Chunk::get(caller);
	IDXL &l=c->lookup(l_t,caller);
	l.sort3d(coord3d);
}



//Add a new entity at the given local index.  Copy the communication
// list from the intersection of the communication lists of the given entities.
void splitEntity(IDXL_Side &c,
	int localIdx,int nBetween,int *between,int idxbase)
{
	//Find the commRecs for the surrounding nodes
	const IDXL_Rec *tween[20];
	int w,w1;
	for (w1=0;w1<nBetween;w1++) {
	  tween[w1]=c.getRec(between[w1]-idxbase);
	  if (tween[w1]==NULL) 
	    return; //An unshared entity! Thus a private-only addition
	}
	
	//Make a new commRec as the interesection of the surrounding entities--
	// we loop over the first entity's comm. list
	for (int zs=tween[0]->getShared()-1;zs>=0;zs--) {
	  for (w1=0;w1<nBetween;w1++) {
	    tween[w1]=c.getRec(between[w1]-idxbase);
	  }
		int chk=tween[0]->getChk(zs);
		//Make sure this processor shares all our entities
		for (w=0;w<nBetween;w++)
			if (!tween[w]->hasChk(chk))
				break;
		if (w==nBetween) {//The new node is shared with chk
			c.addNode(localIdx,chk);
			//break;
		}
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
	const char *caller="IDXL_Add_entity";
	IDXLAPI(caller); IDXL_Chunk *c=IDXL_Chunk::get(caller);
	IDXL &l=c->lookup(l_t,caller);
	splitEntity(l,localIdx,nBetween,between,0);
}
FDECL void FTN_NAME(IDXL_ADD_ENTITY,idxl_add_entity)
	(int *l_t,int *localIdx,int *nBetween,int *between)
{
	const char *caller="IDXL_Add_entity";
	IDXLAPI(caller); IDXL_Chunk *c=IDXL_Chunk::get(caller);
	IDXL &l=c->lookup(*l_t,caller);
	splitEntity(l,*localIdx-1,*nBetween,between,1);	
}


/** Throw away this index list */
CDECL void IDXL_Destroy(IDXL_t l) {
	const char *caller="IDXL_Destroy";
	IDXLAPI(caller); IDXL_Chunk *c=IDXL_Chunk::get(caller);
	c->destroy(l);
}
FORTRAN_AS_C(IDXL_DESTROY,IDXL_Destroy,idxl_destroy,  (int *l), (*l))

/*********************** Lookups **********************
HACK: rather than do some funky dynamic allocation of IDXL_Side_t's,
I'm just setting IDXL_Side_t's to be equal to their source IDXL_t
plus 1 or 2 million.
*/
const IDXL_Side &lookupSide(IDXL_Side_t s,const char *caller) {
	IDXL_Chunk *c=IDXL_Chunk::get(caller);
	if (s-IDXL_SHIFT_SIDE_T_RECV>=IDXL_DYNAMIC_IDXL_T) {
		IDXL_t l=s-IDXL_SHIFT_SIDE_T_RECV;
		return c->lookup(l,caller).getRecv();
	}
	else if (s-IDXL_SHIFT_SIDE_T_SEND>=IDXL_DYNAMIC_IDXL_T) {
		IDXL_t l=s-IDXL_SHIFT_SIDE_T_SEND;
		return c->lookup(l,caller).getSend();
	}
	else /* unknown side_t s */
		IDXL_Abort(caller,"Unrecognized IDXL_Side_t %d\n",s);
	return *new IDXL_Side(); /* LIE: for whining compilers only */
}

CDECL IDXL_Side_t IDXL_Get_send(IDXL_t l) {
	return l+IDXL_SHIFT_SIDE_T_SEND;
}
FORTRAN_AS_C_RETURN(int,IDXL_GET_SEND,IDXL_Get_send,idxl_get_send, (int *l),(*l))

CDECL IDXL_Side_t IDXL_Get_recv(IDXL_t l) {
	return l+IDXL_SHIFT_SIDE_T_RECV;
}
FORTRAN_AS_C_RETURN(int,IDXL_GET_RECV,IDXL_Get_recv,idxl_get_recv, (int *l),(*l))

CDECL void IDXL_Get_end(IDXL_Side_t s) {
	/* Nothing to do: l will be deleted when its IDXL_t is... */
}
FORTRAN_AS_C(IDXL_GET_END,IDXL_Get_end,idxl_get_end, (int *s),(*s))

CDECL int IDXL_Get_partners(IDXL_Side_t s) {
	const char *caller="IDXL_Get_partners"; IDXLAPI(caller); 
	return lookupSide(s,caller).size();
}
FORTRAN_AS_C_RETURN(int,IDXL_GET_PARTNERS,IDXL_Get_partners,idxl_get_partners,
	(int *s),(*s))

CDECL int IDXL_Get_partner(IDXL_Side_t s,int partnerNo) {
	const char *caller="IDXL_Get_partner"; IDXLAPI(caller); 
	return lookupSide(s,caller).getLocalList(partnerNo).getDest();
}
FORTRAN_AS_C_RETURN(int,IDXL_GET_PARTNER,IDXL_Get_partner,idxl_get_partner,
	(int *s,int *p),(*s,*p-1))

CDECL int IDXL_Get_count(IDXL_Side_t s,int partnerNo) {
	const char *caller="IDXL_Get_count"; IDXLAPI(caller); 
	return lookupSide(s,caller).getLocalList(partnerNo).size();
}
FORTRAN_AS_C_RETURN(int,IDXL_GET_COUNT,IDXL_Get_count,idxl_get_count,
	(int *s,int *p),(*s,*p-1))

static void getList(IDXL_Side_t s,int partnerNo,int *list,int idxBase) {
	const char *caller="IDXL_Get_"; IDXLAPI(caller); 
	const IDXL_List &l=lookupSide(s,caller).getLocalList(partnerNo);
	for (int i=0;i<l.size();i++) list[i]=l[i]+idxBase;
}

CDECL void IDXL_Get_list(IDXL_Side_t s,int partnerNo,int *list) {
	getList(s,partnerNo,list,0);
}
FDECL void FTN_NAME(IDXL_GET_LIST,idxl_get_list)
	(int *s,int *p,int *list) {
	getList(*s,*p-1,list,1);
}

CDECL int IDXL_Get_index(IDXL_Side_t s,int partnerNo,int listIndex) {
	const char *caller="IDXL_Get_index"; IDXLAPI(caller); 
	return lookupSide(s,caller).getLocalList(partnerNo)[listIndex];
}
FDECL int FTN_NAME(IDXL_GET_INDEX,idxl_get_index)(int *s,int *p,int *idx)
{
	return IDXL_Get_index(*s,*p-1,*idx-1)+1;
}

CDECL int IDXL_Get_source(IDXL_t l_t,int localNo) {
	const char *caller="IDXL_Get_source"; IDXLAPI(caller); 
	IDXL &l=IDXL_Chunk::get(caller)->lookup(l_t,caller);
	const IDXL_Rec *rec=l.getRecv().getRec(localNo);
	if (rec==NULL) CkAbort("IDXL_Get_source called on non-ghost entity!");
	if (rec->getShared()>1) CkAbort("IDXL_Get_source called on multiply-shared entity!");
	return rec->getChk(0);
}
FDECL int FTN_NAME(IDXL_GET_SOURCE,idxl_get_source)(int *l,int *localNo) {
	return 1+IDXL_Get_source(*l,*localNo-1);
}

/************************* IDXL_Layout ************************/
CDECL IDXL_Layout_t 
IDXL_Layout_create(int type, int width)
{
	const char *caller="IDXL_Create_simple_data";
	IDXLAPI(caller);
	return IDXL_Layout_List::get().put(IDXL_Layout(type, width));
}
FORTRAN_AS_C_RETURN(int,
	IDXL_LAYOUT_CREATE,IDXL_Layout_create,idxl_layout_create,
	(int *t,int *w), (*t,*w))

CDECL IDXL_Layout_t 
IDXL_Layout_offset(int type, int width, int offsetBytes, int distanceBytes,int skewBytes)
{
	const char *caller="IDXL_Create_data";
	IDXLAPI(caller);
	return IDXL_Layout_List::get().put(IDXL_Layout(type,width, offsetBytes,distanceBytes,skewBytes));
}
FORTRAN_AS_C_RETURN(int,
	IDXL_LAYOUT_OFFSET,IDXL_Layout_offset,idxl_layout_offset,
	(int *t,int *w,int *o,int *d,int *s), (*t,*w,*o,*d,*s))

#define GET_DATA_DECL(CAPS,Cname,lowercase, field) \
CDECL int Cname(IDXL_Layout_t d) \
{ \
	IDXLAPI(#Cname); \
	return IDXL_Layout_List::get().get(d,#Cname).field;\
}\
FORTRAN_AS_C_RETURN(int, IDXL_GET_LAYOUT_##CAPS,Cname,idxl_get_layout_##lowercase, \
	(int *d), (*d))

GET_DATA_DECL(TYPE,IDXL_Get_layout_type,type, type)
GET_DATA_DECL(WIDTH,IDXL_Get_layout_width,width, width)
GET_DATA_DECL(DISTANCE,IDXL_Get_layout_distance,distance, distance)

CDECL void 
IDXL_Layout_destroy(IDXL_Layout_t l) {
	const char *caller="IDXL_Layout_destroy";
	IDXLAPI(caller);
	IDXL_Layout_List::get().destroy(l,caller);
}
FORTRAN_AS_C(IDXL_LAYOUT_DESTROY,IDXL_Layout_destroy,idxl_layout_destroy,
	(int *l), (*l) )

/************************* IDXL_Comm ************************/
/** Comm_begin begins a message exchange.  */
CDECL IDXL_Comm_t 
IDXL_Comm_begin(int tag, int context) {
	const char *caller="IDXL_Create_data";
	IDXLAPI(caller); IDXL_Chunk *c=IDXL_Chunk::get(caller);
	return c->addComm(tag,context);
}
FORTRAN_AS_C_RETURN(int,
	IDXL_COMM_BEGIN,IDXL_Comm_begin,idxl_comm_begin,
	(int *tag,int *context), (*tag,*context))

// A bunch of useful locals for use in IDXL_Comm API routines:
#define COMM_BROILERPLATE(routineName) \
	const char *caller=routineName; \
	IDXLAPI(caller); IDXL_Chunk *c=IDXL_Chunk::get(caller); \
	const IDXL &list=c->lookup(dest,caller); \
	const IDXL_Layout *dt=&c->layouts.get(type,caller);

/** Remote-copy this data on flush/wait. */
CDECL void 
IDXL_Comm_sendrecv(IDXL_Comm_t m,IDXL_t dest, IDXL_Layout_t type, void *data) {
	COMM_BROILERPLATE("IDXL_Comm_sendrecv");
	IDXL_Comm *comm=NULL;
	if (m!=0) 
		comm=c->lookupComm(m,caller);
	else /* m==0, use a separate temporary comm */
		comm=c->lookupComm(c->addComm(0,0),caller);
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
		comm=c->lookupComm(m,caller);
	else /* m==0, use a separate temporary comm */
		comm=c->lookupComm(c->addComm(0,0),caller);
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
	c->lookupComm(m,caller)->send(&list.getSend(),dt,srcData);
}
FORTRAN_AS_C(IDXL_COMM_SEND,IDXL_Comm_send,idxl_comm_send,
	(int *m,int *dest,int *type,const void *data), (*m,*dest,*type,data))

/** Copy this data from the remote values when wait is called. */
void IDXL_Comm_recv(IDXL_Comm_t m,IDXL_t dest, IDXL_Layout_t type, void *destData) {
	COMM_BROILERPLATE("IDXL_Comm_recv");
	c->lookupComm(m,caller)->recv(&list.getRecv(),dt,destData);
}
FORTRAN_AS_C(IDXL_COMM_RECV,IDXL_Comm_recv,idxl_comm_recv,
	(int *m,int *dest,int *type,void *data), (*m,*dest,*type,data))

/** Add this data with the remote values when wait is called. */
void IDXL_Comm_sum(IDXL_Comm_t m,IDXL_t dest, IDXL_Layout_t type, void *sumData) {
	COMM_BROILERPLATE("IDXL_Comm_recv");
	c->lookupComm(m,caller)->sum(&list.getRecv(),dt,sumData);
}
FORTRAN_AS_C(IDXL_COMM_SUM,IDXL_Comm_sum,idxl_comm_sum,
	(int *m,int *dest,int *type,void *data), (*m,*dest,*type,data))

/** Send all outgoing data. */
void IDXL_Comm_flush(IDXL_Comm_t m) {
	const char *caller="IDXL_Comm_flush"; 
	IDXLAPI(caller); IDXL_Chunk *c=IDXL_Chunk::get(caller); 
	c->lookupComm(m,caller)->post();
}

/** Block until all communication is complete. This destroys the IDXL_Comm. */
void IDXL_Comm_wait(IDXL_Comm_t m) {
	const char *caller="IDXL_Comm_flush"; 
	IDXLAPI(caller); IDXL_Chunk *c=IDXL_Chunk::get(caller); 
	c->waitComm(c->lookupComm(m,caller));
}





