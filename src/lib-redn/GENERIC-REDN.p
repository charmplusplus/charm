#include HEAD
#include "pg.int"

#define ReductionOperation(a,b,c,d) PrivateCall(GENERIC_REDOP_NAME(a,b,c,d))

module GENERIC_MODULE_NAME {

message {
    int gid ;
    ChareNumType pgBoc ;
} REDUCE_INIT ;
    
message {
    varSize GENERIC_DATATYPE data[] ;
} REDN_MSG ;

message {
    int size ;
    varSize GENERIC_DATATYPE data[] ;
} REDN_MSG_INTERNAL ;


typedef struct ReductionRefInstance {
    int refnum ;
    int r_type ;
    int send_result_flag ;
    int leftToCollect ;
    int is_first ;
    GENERIC_DATATYPE *y ;
    GENERIC_DATATYPE *r_z ;
    int numBuffered ;
    GENERIC_DATATYPE *buffered[4] ; /* This should be the maximum number of children */
    int numEls ;            /*      that a node can have */
    void (* r_function)() ;
    EntryPointType r_ep ;
    ChareIDType r_cid ;
    ChareNumType r_bocnum ;
    struct ReductionRefInstance *next ;
} ReductionRefInstance ;

#define R_BY_MESSAGE  0
#define R_BY_FUNCTION 1

BranchOffice reduce {
    int me, gid, myParent, numChildren, numToCollect ;
    ChareNumType pgBoc ;
    ReductionRefInstance *refList ;

    entry init : (message REDUCE_INIT *msg)
    {
	gid = msg->gid ;
	pgBoc = msg->pgBoc ;
	
	me = CkMyPe() ;
	if(gid == 0) {
	    myParent = CkSpanTreeParent(me) ;
	    numChildren = CkNumSpanTreeChildren(me) ;
	}
	else {
	    myParent = PG::PgMySpanTreeParent(pgBoc, gid) ;
	    numChildren = PG::PgMyNumSpanTreeChildren(pgBoc, gid) ;
	}
	numToCollect = numChildren+1 ; /* I send a message to myself */
	refList = NULL ;
	CkFreeMsg(msg);
    }





    entry collect : (message REDN_MSG_INTERNAL *cMsg)
    {
	int i, refnum, gotAll ;
	GENERIC_DATATYPE *x ;
	private void GENERIC_REDOP_NAME() ;
	private ReductionRefInstance *FindRef() ;
	private void AddRef() ;
	ReductionRefInstance *ref ;

	refnum = GetRefNumber(cMsg) ;

	ref = (ReductionRefInstance*)PrivateCall(FindRef(refnum)) ;

	/* ref may be null if I get messages from my children before
	 * I join up.  So create a base structure and go on */
	if(ref==NULL) {
	    ref=(ReductionRefInstance*)CkAlloc(sizeof(ReductionRefInstance)) ;
	    ref->numEls = cMsg->size ;
	    ref->refnum = refnum ;
	    ref->leftToCollect = numToCollect ;
	    ref->is_first = 1 ;
	    ref->y = (GENERIC_DATATYPE *)CkAlloc(sizeof(GENERIC_DATATYPE)*ref->numEls) ;
	    ref->next = NULL ;
	    PrivateCall(AddRef(ref,refnum)) ;
	}
	
        x = (GENERIC_DATATYPE *)cMsg->data;
	ReductionOperation(x,ref->y,ref->numEls,ref->is_first) ;

	ref->is_first = 0 ;
	ref->leftToCollect-- ;

	if (ref->leftToCollect == 0) {
	    SetRefNumber(cMsg, refnum) ;
            for(i=0;i<ref->numEls;i++)
                    cMsg->data[i] = (GENERIC_DATATYPE) ref->y[i] ;
	    /* If I'm the root, then reduction is done - else keep sending
	     * it up the tree */
	    if (myParent == -1) {
		if(gid == 0) 
		    BroadcastMsgBranch(distribute, cMsg, MyBocNum()) ;
		else 
  	            PG::Multicast(pgBoc, gid, cMsg, distribute, MyBocNum()) ;
	    }
	    else {
		SendMsgBranch(collect, cMsg, myParent) ;
	    }
	}
	else { 
	    CkFreeMsg(cMsg);  /* Only free it if we're not reusing it */
	}
    }




	entry distribute : (message REDN_MSG_INTERNAL *dMsg)
	{
	int refnum, i, size ;
	ReductionRefInstance *ref, *temp, *skip ;
	private ReductionRefInstance *FindRef() ;
	private void DeleteRef() ;
	void (*fptr)() ;
	REDN_MSG *msg ;

	refnum = GetRefNumber(dMsg) ;
	/* Look up the reply mechanism for this branch */
	ref = (ReductionRefInstance*)PrivateCall(FindRef(refnum)) ;

	/* Check for error */
	if (!ref) {
	    CkPrintf("Error in %s.  Lost reference number %d\n",
		     "GENERIC_MODULE_NAME", refnum) ;
	    ChareExit() ;
	    return ;
	}
		    
	/* Figure out what/where to send it */
	if (ref->send_result_flag) {
	    if (ref->r_type == R_BY_FUNCTION) {
		for(i=0;i<ref->numEls;i++) 
		    ref->r_z[i] = (GENERIC_DATATYPE)dMsg->data[i] ;
		fptr = ref->r_function ;
		BranchCall(ref->r_bocnum, fptr(MyBocNum())) ;
		}
	    else {
		size = ref->numEls ;
		msg = (REDN_MSG*)CkAllocMsg(REDN_MSG,&size) ;
		for(i=0;i<ref->numEls;i++) 
		    msg->data[i] = dMsg->data[i] ;
		SetRefNumber(msg,refnum) ;
		SendMsg(ref->r_ep, msg, &(ref->r_cid)) ;
	    }
	}
        CkFreeMsg(dMsg) ;

	/* Delete this refnum info from the list */
	PrivateCall(DeleteRef(refnum)) ;
    }
	

    /***********************************************************************/

    public f(x,z,size,refnum,fptr,id)
    GENERIC_DATATYPE x[],*z ;
    int size, refnum ;
    void (*fptr)() ;
    void *id ;
    {
	ReductionRefInstance *ref ;
	private void AddRef() ;
	private ReductionRefInstance *FindRef() ;
	REDN_MSG_INTERNAL *cMsg ;
	int i, varSizes[1] ;

	ref = (ReductionRefInstance*)PrivateCall(FindRef(refnum)) ;

	if(ref==NULL) {
	    ref=(ReductionRefInstance*)CkAlloc(sizeof(ReductionRefInstance)) ;
	    ref->numEls = size ;
	    ref->leftToCollect = numToCollect ;
	    ref->refnum = refnum ;
	    ref->is_first = 1 ;
	    ref->y = (GENERIC_DATATYPE *)CkAlloc(sizeof(GENERIC_DATATYPE)*ref->numEls) ;
	    ref->next = NULL ;
	    PrivateCall(AddRef(ref,refnum)) ;
	}

	/* Fill in the return information */
	ref->r_type = R_BY_FUNCTION ;
	if (id == NULL) 
	    ref->send_result_flag = 0 ;
	else {
	    ref->r_z = z ;
	    ref->send_result_flag = 1 ;
	    ref->r_function = fptr ;
	    ref->r_bocnum = *((ChareNumType *) id) ;
	}

	
	varSizes[0]=ref->numEls ;
	cMsg = (REDN_MSG_INTERNAL *)CkAllocMsg(REDN_MSG_INTERNAL, varSizes) ;
	cMsg->size = ref->numEls ;
	for(i=0;i<ref->numEls;i++){
	    /* ref->y[i] = x[i] ; this  is wrong tooo */
	    cMsg->data[i] = (GENERIC_DATATYPE) x[i] ;
	}
	SetRefNumber(cMsg, refnum) ;

	/* If I'm a leaf, then start by sending up the tree.  Otherwise,
	 * send the portion to myself */
	if(numToCollect == 0)
	    SendMsgBranch(collect,cMsg,myParent) ;
	else {
	    SendMsgBranch(collect,cMsg,me) ;
	}
    }

    /***********************************************************************/
	
    /* return by message */
    public f_msg(x,size,refnum,ep,id)
    GENERIC_DATATYPE      x[] ;
    int           size ;
    int           refnum ;
    EntryNumType  ep;
    ChareIDType   *id;
    {
	private void AddRef() ;
	private ReductionRefInstance *FindRef() ;
	GENERIC_DATATYPE y ;
	int i, varSizes[1] ;
	ReductionRefInstance *ref, *temp ;
	REDN_MSG_INTERNAL *cMsg ;

	ref = (ReductionRefInstance*)PrivateCall(FindRef(refnum)) ;

	if(ref==NULL) {
	    ref=(ReductionRefInstance*)CkAlloc(sizeof(ReductionRefInstance)) ;
	    ref->numEls = size ;
	    ref->leftToCollect = numToCollect ;
	    ref->refnum = refnum ;
	    ref->is_first = 1 ;
	    ref->y = (GENERIC_DATATYPE *)CkAlloc(sizeof(GENERIC_DATATYPE)*ref->numEls) ;
	    ref->next = NULL ;
	    PrivateCall(AddRef(ref,refnum)) ;
	}

	/* Fill in the return information */

	ref->r_type = R_BY_MESSAGE ;
	/* ref->is_first = 1 ; this is an error */
	if (id == NULL) 
	    ref->send_result_flag = 0 ;
	else {
	    ref->send_result_flag = 1 ;
	    ref->r_ep = ep ;
	    ref->r_cid = *id ;
	}

	varSizes[0]=ref->numEls ;
	cMsg = (REDN_MSG_INTERNAL *)CkAllocMsg(REDN_MSG_INTERNAL, varSizes) ;
	cMsg->size = ref->numEls ;
	for(i=0;i<ref->numEls;i++){
	    /* ref->y[i] = x[i] ; this is wrong */
	    cMsg->data[i] = (GENERIC_DATATYPE) x[i] ;
	}
	SetRefNumber(cMsg, refnum) ;

	/* If I'm a leaf, then start by sending up the tree.  Otherwise,
	 * send the local portion to myself */
	if(numToCollect == 0) 
	    SendMsgBranch(collect,cMsg,myParent) ;
	else 
	    SendMsgBranch(collect,cMsg,me) ;
    }

	/* Reduction operations */
	
	private void GENERIC_REDOP_NAME(x,y,n,first)
	GENERIC_DATATYPE x[], y[] ;
	int n,first ;
        {
	    int i ;
	    if(first) 
		for(i=0 ; i<n ; i++)
		    GENERIC_OPERATOR1
	    else 
		for(i=0 ; i<n ; i++)
		    GENERIC_OPERATOR2
	}

	/********************************************************/
	/* Data structure access functions **********************/
	/********************************************************/
	
	private void AddRef(newRef, refnum)
	ReductionRefInstance *newRef ;
        int refnum ;
        {
	    ReductionRefInstance *temp ;

	    temp = refList ;
	    if (temp == NULL)
		refList = newRef ;
	    else {
		while ( (temp->next) && (temp->refnum != refnum) )
		    temp=temp->next ;
		if (temp->refnum == refnum) 
		    CkPrintf("ERROR in GENERIC_MODULE_NAME.  Reference number (%d) used in overlapping reductions\n",refnum) ;
		else 
		    temp->next = newRef ;
	    }
	}
	
	private ReductionRefInstance *FindRef(refNum)
        int refNum ;
	{
	    ReductionRefInstance *ref ;
	    
	    ref = refList ;
	    while( (ref != NULL) && (ref->refnum != refNum) ) ref = ref->next ;
	    return ref ;
	}

	private void DeleteRef(refNum)
	int refNum ;
	{
	    ReductionRefInstance *temp, *skip ;
	    
	    temp = refList ;
	    if (refList->refnum == refNum) {
		refList = refList->next ;
                if (temp->y) CkFree(temp->y);
		CkFree(temp) ;
	    }
	    else {
		while (temp->next->refnum != refNum) temp=temp->next ;
		if(temp->next == NULL)
		    skip = NULL ;
		else
		    skip = temp->next->next ;
                if (temp->next->y) CkFree(temp->next->y);
		CkFree(temp->next) ;
		temp->next = skip ;
	    }
	}
    } /* end BOC */

    /********************************************************/
    /* Functions accessable from the outside are below here */
    /********************************************************/
    
    Create()
    {
        REDUCE_INIT *msg;
        msg = (REDUCE_INIT *) CkAllocMsg(REDUCE_INIT);
	msg->gid = 0 ;
        return CreateBoc(GENERIC_MODULE_NAME::reduce,GENERIC_MODULE_NAME::reduce@init,msg);
    }
    
    CreateOverGroup(gid, pgBoc)
	int gid ;
    ChareNumType pgBoc ;
    {
	int        boc;
	REDUCE_INIT *msg;
	
        msg = (REDUCE_INIT *) CkAllocMsg(REDUCE_INIT);
	msg->gid = gid ;
	msg->pgBoc = pgBoc ;
        boc=CreateBoc(GENERIC_MODULE_NAME::reduce,GENERIC_MODULE_NAME::reduce@init,msg);
        return boc;
    }


    DepositData(boc,x,z,nelements,ref,fptr,id)
    ChareNumType      boc;
    GENERIC_DATATYPE          x[],z[];
    int               nelements ;
    int               ref ;
    void              (*fptr)() ;
    void              *id;
    {
	BranchCall(boc,GENERIC_MODULE_NAME::reduce@f(x,z,nelements,ref,fptr,id));
    }


    
    DepositDataMsg(boc,x,nelements,ref,ep,id)
    ChareNumType      boc;
    GENERIC_DATATYPE          x[];
    int               nelements ;
    int               ref ;
    EntryNumType      ep;
    ChareIDType       *id;
    {
	BranchCall(boc,GENERIC_MODULE_NAME::reduce@f_msg(x,nelements,ref,ep,id));
    }


    /* One of the fuctions below will be preprocessed into a "LocalReduction"
     * call which will be visible from outside. */
    
    void LocalReduction(x,y,n)
    GENERIC_DATATYPE x[],*y ;
    int n ;
    {
	int i ;
	*y = x[0] ;
	for(i=1;i<n;i++) 
	    GENERIC_OPERATOR
    }

} /* end module GENERIC_MODULE_NAME */

