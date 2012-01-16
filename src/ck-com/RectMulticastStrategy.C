/**
   @addtogroup ComlibCharmStrategy
*/
/*@{*/

/** @file */



//#define COMLIB_RECT_DEBUG
//#define LOCAL_MULTI_OFF
/********************************************************
        Section multicast strategy sgetRectGeometryuite. DirectMulticast and its
        derivatives, multicast messages to a section of array elements
        created on the fly. The section is invoked by calling a
        section proxy. These strategies can also multicast to a subset
        of processors for groups.

        These strategies are non-bracketed. When the first request is
        made a route is dynamically built on the section. The route
        information is stored in

 - Sameer Kumar

**********************************************/

/** EJB
 *
 * RectMulticastStrategy is designed to use the Rectangular broadcast
 * feature of the bgml library to accelerate multicasts.
 * 
 * It will use the AsyncRectBcast feature which has ways to define rectangles.
 * 
 *
 * The interaction with the rectangle layer works thusly:
 *  initialize the rectangle for the section using 
 *     bgl_machine_RectBcastinit
 *     to create the communicator
 *
 *  That returns a request object which must be kept and used in each
 *  rectangular send.
 *
 *  send using a single invocation of bgl_machine_RectBcast
 *   If the root is inside the rectangle do this on the root
 *   else forward to (preferably a corner) something in the rectangle.
 *
 *   This latter case is the only use of a forwarding comlib
 *   object in this strategy.  Remote delivery is otherwise handled by
 *   the converse machine level and BG/L.  There is no other use of
 *   comlib forwarding schemes.
 *
 *   The all cases the rectangle sender will not receive a copy from the
 *   interface so it will need to do its own local multicast.
 * 
 *  On receipt at the intermediate/destination (all intermediates are
 *  destinations) processor level we deliver using local multicast if
 *  our list of multicast subscribing chares is not null.  If null,
 *  ignore the message (machine broadcast may be a bit overzealous).
 *
 *   This is a 2 pass process.
 *
 *    When create on src is triggered.  send geometry to entire
 *    rectangle (including processors which have no chares in the
 *    section list) so every element can initialize.  Use a remote
 *    listsend on every element in the rectangle.
 *
 *    Create on intermediate should be the receiver of that 1st pass process.
 *    Each intermediate should initialize the rectangle, but not send.
 *    
 *    Subsequent sends can then use the rectangle send.
 *
 *
 */


    // rectangle communicator (arrid| srcpe |section id)  -> request


#include "RectMulticastStrategy.h"
CkpvDeclare(comRectHashType *, com_rect_ptr); 
#ifdef CMK_RECT_API
#include "bgltorus.h"
#include "bgml.h"
extern "C" void    *   bgl_machine_RectBcastInit  (unsigned               commID, const BGTsRC_Geometry_t* geometry);

extern "C"  void bgl_machine_RectBcast (unsigned                 commid,			    const char             * sndbuf,			    unsigned                 sndlen); 


extern "C" void  bgl_machine_RectBcastConfigure (void *(*Fn)(int));

extern "C" void isSane( void *, unsigned);

CkpvExtern(CkGroupID, cmgrID);

void *sourceOffRectstrategyHandler(void *msg) {
    CmiMsgHeaderExt *conv_header = (CmiMsgHeaderExt *) msg;
    int instid = conv_header->stratid;
    ComlibPrintf("sourceOff handler called on %d\n",CkMyPe(), instid);

    RectMulticastStrategy *strat = (RectMulticastStrategy *)    ConvComlibGetStrategy(instid);
    strat->handleMessageForward(msg);
    return NULL;
}

void * rectRequest (int comm) {
  //  fprintf(stderr,"[%d] rectRequest for %d gives %p\n",CkMyPe(),comm,CkpvAccess(com_rect_ptr)->get(comm));
#ifdef COMLIB_RECT_DEBUG
  CkAssert(CkpvAccess(com_rect_ptr)->get(comm)!=NULL);
  isSane(CkpvAccess(com_rect_ptr)->get(comm),comm);
#endif
  return CkpvAccess(com_rect_ptr)->get(comm);
}

RectMulticastStrategy::RectMulticastStrategy(CkArrayID aid)
    : Strategy(), CharmStrategy() {

    ainfo.setDestinationArray(aid);
    setType(ARRAY_STRATEGY);
}

//Destroy all old built routes
RectMulticastStrategy::~RectMulticastStrategy() {
    
    ComlibPrintf("Calling Destructor\n");

    if(getLearner() != NULL)
        delete getLearner();
        
    CkHashtableIterator *ht_iterator = sec_ht.iterator();
    ht_iterator->seekStart();
    while(ht_iterator->hasNext()){
        void **data;
        data = (void **)ht_iterator->next();        
        ComlibRectSectionHashObject *obj = (ComlibRectSectionHashObject *) (* data);
        if(obj != NULL)
            delete obj;
    }
}

void RectMulticastStrategy::insertMessage(CharmMessageHolder *cmsg){
  cmsg->checkme();

  
  //    ComlibPrintf("[%d] Comlib Rect Section Multicast: insertMessage \n", 


    if(cmsg->dest_proc == IS_SECTION_MULTICAST && cmsg->sec_id != NULL) { 
        CkSectionID *sid = cmsg->sec_id;
        int cur_sec_id = sid->getSectionID();
	ComlibPrintf("[%d] Comlib Rect Section Multicast: insertMessage section id %d\n", CkMyPe(), cur_sec_id);           
        if(cur_sec_id > 0) {        
            sinfo.processOldSectionMessage(cmsg);            

	    //	    ComlibPrintf("[%d] insertMessage old sectionid %d \n",CkMyPe(),cur_sec_id);
	    ComlibPrintf("[%d] insertMessage old sectionid %d \n",CkMyPe(),cur_sec_id);
            ComlibSectionHashKey 
                key(CkMyPe(), sid->_cookie.sInfo.cInfo.id);        
            ComlibRectSectionHashObject *obj = sec_ht.get(key);

            if(obj == NULL)
                CkAbort("Cannot Find Section\n");

            envelope *env = UsrToEnv(cmsg->getCharmMessage());
#ifndef LOCAL_MULTI_OFF
	    localMulticast(env, obj);
#endif
	    if(obj->sourceInRectangle) 
	      {
		remoteMulticast(env, obj);
	      }
	    else // forward
	      {
		forwardMulticast(env, obj);
	      }
        }
        else {
	  ComlibPrintf("[%d] insertMessage new section id %d\n", CkMyPe(), cur_sec_id);           
	  //New sec id, so send it along with the message
	  void *newmsg = sinfo.getNewMulticastMessage(cmsg, needSorting());
	  insertSectionID(sid);

	  ComlibSectionHashKey 
	    key(CkMyPe(), sid->_cookie.sInfo.cInfo.id);        
	  ComlibPrintf("[%d] insertMessage new sectionid %d \n",CkMyPe(),sid->_cookie.sInfo.cInfo.id);
	  ComlibRectSectionHashObject *obj = sec_ht.get(key);

	  if(obj == NULL)
	    CkAbort("Cannot Find Section\n");

	  /*
	    ComlibPrintf("%u: Src = %d dest:", key.hash(), CkMyPe());
	    for (int i=0; i<obj->npes; ++i)
	    ComlibPrintf(" %d",obj->pelist[i]);
	    ComlibPrintf(", map:");
	    ComlibMulticastMsg *lll = (ComlibMulticastMsg*)newmsg;
	    envelope *ppp = UsrToEnv(newmsg);
	    CkUnpackMessage(&ppp);
	    int ttt=0;
	    for (int i=0; i<lll->nPes; ++i) {
	    ComlibPrintf(" %d (",lll->indicesCount[i].pe);
	    for (int j=0; j<lll->indicesCount[i].count; ++j) {
	    ComlibPrintf(" %d",((int*)&(lll->indices[ttt]))[1]);
	    ttt++;
	    }
	    ComlibPrintf(" )");
	    }
	    CkPackMessage(&ppp);
	    ComlibPrintf("\n");
	  */
	  // our object needs indices, npes, pelist
	  sinfo.getRemotePelist(sid->_nElems, sid->_elems, obj->npes, obj->pelist);
	  sinfo.getLocalIndices(sid->_nElems, sid->_elems, obj->indices);
	  char *msg = cmsg->getCharmMessage();
	  /*
	    ComlibMulticastMsg *lll = (ComlibMulticastMsg*)newmsg;
	    envelope *ppp = UsrToEnv(newmsg);
	    CkUnpackMessage(&ppp);
	    int ttt=0;
	    int uuu=0;
	    for (int i=0; i<lll->nPes; ++i) {
	    //	      ComlibPrintf(" %d (",lll->indicesCount[i].pe);
	    uuu++;
	    for (int j=0; j<lll->indicesCount[i].count; ++j) {
	    //		ComlibPrintf(" %d",((int*)&(lll->indices[ttt]))[1]);
	    ttt++;
	    }
	    //	      ComlibPrintf(" )");
	    }


	    ComlibPrintf("[%d] newmsg for sendRectDest has %d indices %d pes\n",CkMyPe(),ttt, uuu);
	    CkAssert(uuu>0);
	    CkAssert(ttt>0);
	  */
	  envelope *ppp = UsrToEnv(newmsg);
	  CkPackMessage(&ppp);	    
#ifndef LOCAL_MULTI_OFF
	  localMulticast(UsrToEnv(msg), obj);
#endif

	  sendRectDest(obj ,CkMyPe(), UsrToEnv(newmsg));
	  //            CkFreeMsg(msg);	    // need this?

        }        
    }
    else 
        CkAbort("Section multicast cannot be used without a section proxy");

    delete cmsg;       
}

void RectMulticastStrategy::insertSectionID(CkSectionID *sid) {

  //    ComlibPrintf("[%d] insertSectionId \n", CkMyPe());   
    ComlibPrintf("[%d] insertSectionId \n", CkMyPe());   

    
    ComlibSectionHashKey 
        key(CkMyPe(), sid->_cookie.sInfo.cInfo.id);

    ComlibRectSectionHashObject *obj = NULL;    
    obj = sec_ht.get(key);
    
    if(obj != NULL)
        delete obj;
    
    obj = createObjectOnSrcPe(sid->_nElems, sid->_elems, sid->_cookie.sInfo.cInfo.id);
    sec_ht.put(key) = obj;

}


ComlibRectSectionHashObject *
RectMulticastStrategy::createObjectOnSrcPe(int nindices, CkArrayIndex *idxlist, unsigned int thisSectionID) {

    ComlibPrintf("[%d] createObjectOnSrcPe \n", CkMyPe());   
    ComlibPrintf("[%d] Rect createObjectOnSrcPe \n", CkMyPe());   
    ComlibRectSectionHashObject *obj = new ComlibRectSectionHashObject();
    
    sinfo.getRemotePelist(nindices, idxlist, obj->npes, obj->pelist);
    sinfo.getLocalIndices(nindices, idxlist, obj->indices);
    // TODO: how do we get the root pe here?
    // cheap hack, assume section built from root
    int rootpe=CkMyPe();
    BGLTorusManager *bgltm= BGLTorusManager::getObject();
    int x,y,z;
    bgltm->getCoordinatesByRank(rootpe,x, y, z);
    obj->aid=sinfo.getDestArrayID();
    unsigned int comm=computeKey( thisSectionID, rootpe, sinfo.getDestArrayID());
    CkAssert(obj->npes>0);
    BGTsRC_Geometry_t *geometry=getRectGeometry(obj ,rootpe);
    // handle the root not in rectangle case
    if( x >= geometry->x0 && x  < geometry->x0+ geometry->xs &&
	y >= geometry->y0 && y  < geometry->y0+ geometry->ys &&
	z >= geometry->z0 && z  < geometry->z0+ geometry->zs)
      { // ok
	ComlibPrintf("[%d] create root %d %d %d %d in rectangle %d %d %d, %d %d %d comm id = %d geom root %d %d %d geom t0 %d tr %d ts %d\n", CkMyPe(), rootpe, x, y, z, 
		     geometry->x0, geometry->y0, geometry->z0,
		     geometry->xs, geometry->ys, geometry->zs, comm,
		     geometry->xr, geometry->yr, geometry->zr,
		     geometry->t0, geometry->tr, geometry->ts
		     );   
	obj->sourceInRectangle=true;
	obj->cornerRoot=rootpe;
	void *request = CkpvAccess(com_rect_ptr)->get(comm);
	if(request==NULL)
	  {
	    request=bgl_machine_RectBcastInit(comm, geometry);
	    ComlibPrintf("[%d] csrc init comm %d section %d srcpe %d request %p\n",CkMyPe(), comm, thisSectionID, rootpe, request);
	    CkpvAccess(com_rect_ptr)->put(comm)= request;
#ifdef COMLIB_RECT_DEBUG
	    isSane(request,comm);
#endif
	  }
	else{
	  ComlibPrintf("[%d] csrc already init comm %d section %d srcpe %d\n",CkMyPe(), comm, thisSectionID, rootpe);
	}
#ifdef COMLIB_RECT_DEBUG
	void *getrequest =     CkpvAccess(com_rect_ptr)->get(comm);
	CkAssert(*((char *) request)==*((char *)getrequest));
	isSane(getrequest,comm);
#endif
      }
    else // sender not inside rectangle.
      {  // we cannot initialize or use the rectangle from here
	ComlibPrintf("[%d] root %d %d %d %d NOT in rectangle %d %d %d, %d %d %d  \n", CkMyPe(), rootpe, x, y, z, 
		     geometry->x0, geometry->y0, geometry->z0,
		     geometry->xs, geometry->ys, geometry->zs
		     );   

	obj->sourceInRectangle=false;
	obj->cornerRoot=assignCornerRoot(geometry, rootpe);
	/*  if we were to actually use the geom from here.
	  obj->sourceInRectangle=false;
	  geometry.xr=geometry.x0;
	  geometry.yr=geometry.y0;
	  geometry.zr=geometry.z0;
	*/

      }
    //    delete geometry;
    return obj;
}


void RectMulticastStrategy::sendRectDest(ComlibRectSectionHashObject *obj, int srcpe, envelope *env) {
  ComlibPrintf("[%d] sendRectDest \n", CkMyPe());   

  BGTsRC_Geometry_t *geometry=getRectGeometry(obj,srcpe);
  // now make a list of PEs based on the prism formed by the corners
  BGLTorusManager *bgltm= BGLTorusManager::getObject();
  int npes=geometry->xs* geometry->ys *geometry->zs;
  ComlibPrintf("[%d] sendRectDest has %d * %d * %d = %d pes\n", CkMyPe(), geometry->xs, geometry->ys, geometry->zs, npes);
  int *pelist= new int[npes];
  int destpe=0;
  for(int x=geometry->x0;x<geometry->xs+geometry->x0;x++)
    for(int y=geometry->y0;y<geometry->ys+geometry->y0;y++)
      for(int z=geometry->z0;z<geometry->zs+geometry->z0;z++)
	{
	  int pe=bgltm->coords2rank(x,y,z);  
	  if(pe!=srcpe) //don't send this to ourselves
	    pelist[destpe++]=pe;
	}
  //  for(int i=0;i<destpe;i++)
  //    ComlibPrintf("rect pe %d is %d\n",i,pelist[i]);
  delete geometry;
  //now we have the list, fire off the message
  if(destpe == 0) {
    CmiFree(env);
    return;    
  }
    
  //CmiSetHandler(env, handlerId);
  CmiSetHandler(env, CkpvAccess(strategy_handlerid));

  ((CmiMsgHeaderExt *) env)->stratid = getInstance();

  //Collect Multicast Statistics
  RECORD_SENDM_STATS(getInstance(), env->getTotalsize(), pelist, destpe);
    
  CkPackMessage(&env);
  //Sending a remote multicast
  CmiSyncListSendAndFree(destpe, pelist, env->getTotalsize(), (char*)env);
  //CmiSyncBroadcastAndFree(env->getTotalsize(), (char*)env);

} 


// based on the outside rectangle root coords, pick a root
int RectMulticastStrategy::assignCornerRoot(BGTsRC_Geometry_t *geometry, int srcpe)
{
  fprintf(stderr,"[%d] in assign corner for not in rect root %d\n",CkMyPe(), srcpe);
  ComlibPrintf("[%d] in assign corner for not in rect root %d\n",CkMyPe(), srcpe);
  // just find the shortest hopcount
  BGLTorusManager *bgltm= BGLTorusManager::getObject();

  int pelist[8];
  // make a list of the corners
  int destpe=0;
  pelist[destpe++]=bgltm->coords2rank(geometry->x0,geometry->y0,geometry->z0);
  pelist[destpe++]=bgltm->coords2rank(geometry->x0+ geometry->xs-1, 
				      geometry->y0,geometry->z0);
  pelist[destpe++]=bgltm->coords2rank(geometry->x0,geometry->y0+geometry->ys-1,
				      geometry->z0);
  pelist[destpe++]=bgltm->coords2rank(geometry->x0,geometry->y0,
				      geometry->z0 + geometry->zs-1);
  pelist[destpe++]=bgltm->coords2rank(geometry->x0+ geometry->xs-1, 
				      geometry->y0 +geometry->ys-1,
				      geometry->z0);
  pelist[destpe++]=bgltm->coords2rank(geometry->x0+ geometry->xs-1, 
				      geometry->y0,
				      geometry->z0+geometry->zs-1);
  pelist[destpe++]=bgltm->coords2rank(geometry->x0,
				      geometry->y0+ geometry->ys-1,
				      geometry->z0+ geometry->zs-1);
  pelist[destpe++]=bgltm->coords2rank(geometry->x0+ geometry->xs-1, 
				      geometry->y0+ geometry->ys-1,
				      geometry->z0+ geometry->zs-1);

  int newrootidx=bgltm->pickClosestRank(srcpe, pelist, destpe);
  int newroot=pelist[newrootidx];
  int x, y,z;
  bgltm->getCoordinatesByRank(newroot, x,y ,z);
  geometry->xr=x;
  geometry->yr=y;
  geometry->zr=z;
  ComlibPrintf("[%d] choosing proc %d at %d,%d,%d as corner root\n",CkMyPe(), newroot, x,y ,z);
  return(newroot);
}

BGTsRC_Geometry_t *RectMulticastStrategy::getRectGeometry(ComlibRectSectionHashObject *obj, int srcpe)
{
    // define geometry/
    ComlibPrintf("[%d] getRectGeometry \n", CkMyPe());   
    BGTsRC_Geometry_t *geometry = new BGTsRC_Geometry_t;
    // find the bounding box for the pelist
    // get a torus manager pointer
    BGLTorusManager *bgltm= BGLTorusManager::getObject();
    // initialize to max
    geometry->x0=bgltm->getXNodeSize()*2+1;
    geometry->y0=bgltm->getYNodeSize()*2+1;
    geometry->z0=bgltm->getZNodeSize()*2+1;
    int xmax=0;
    int ymax=0;
    int zmax=0;
    for(int i=0;i<obj->npes;i++)
      { 
	int x=0; int y=0; int z=0;
	bgltm->getCoordinatesByRank(obj->pelist[i], x,y,z);
	geometry->x0 = (x < geometry->x0) ? x : geometry->x0;
	geometry->y0 = (y < geometry->y0) ? y : geometry->y0;
	geometry->z0 = (z < geometry->z0) ? z : geometry->z0;
	xmax = (x > xmax) ? x : xmax;
	ymax = (y > ymax) ? y : ymax;
	zmax = (z > zmax) ? z : zmax;
      }
    geometry->xs = xmax + 1 - geometry->x0;
    geometry->ys = ymax + 1 - geometry->y0;
    geometry->zs = zmax + 1 - geometry->z0;
    geometry->t0=0; //non VN mode values
    geometry->ts=1;
    geometry->tr=0;

    int x,y,z;
    bgltm->getCoordinatesByRank(srcpe,x, y, z);
    geometry->xr=x;
    geometry->yr=y;
    geometry->zr=z;
    if(bgltm->isVnodeMode())
      {
	ComlibPrintf("VN mode Untested for rectbcast!\n");
	// torus manager screws with us on this a bit
	// if VN determine whether we are TXYZ or XYZT
	// then use lowest/highest/root x or z and some division for t0 ts tr
	// also need to halve the VN doubled dimension in [xyz]s [xyz]0 [xyz]r

	// Assume TXYZ
	geometry->ts=2;
	geometry->t0 = geometry->x0 % 2;
	geometry->ts = geometry->xs % 2;
	geometry->tr = geometry->xr % 2;
	geometry->x0 /= 2;
	geometry->xs /= 2;
	geometry->xr /= 2;
      }
    return geometry;
}

ComlibRectSectionHashObject *
RectMulticastStrategy::createObjectOnIntermediatePe(int nindices,
						      CkArrayIndex *idxlist,
						      int npes,
						      ComlibMulticastIndexCount *counts,
						      int srcpe, int thisSectionID) {

    ComlibPrintf("[%d] createObjectOnIntermediatePe \n", CkMyPe());   
    ComlibRectSectionHashObject *obj = new ComlibRectSectionHashObject();
    CkAssert(npes>0);
    sinfo.getRemotePelist(nindices, idxlist, obj->npes, obj->pelist);
    sinfo.getLocalIndices(nindices, idxlist, obj->indices);
    obj->aid=sinfo.getDestArrayID();
    //obj->indices.resize(0);
    //for (int i=0; i<nindices; ++i) obj->indices.insertAtEnd(idxlist[i]);
    //sinfo.getLocalIndices(nindices, idxlist, obj->indices);
    BGLTorusManager *bgltm= BGLTorusManager::getObject();
    int x,y,z;
    bgltm->getCoordinatesByRank(srcpe,x, y, z);

    unsigned int comm = computeKey( thisSectionID, srcpe, sinfo.getDestArrayID());

    if(obj->npes<=0)
      {
	ComlibPrintf("[%d] nindices %d, npes %d, obj->npes %d\n", CkMyPe(), nindices, npes, obj->npes);
      }
    CkAssert(obj->npes>0);
    BGTsRC_Geometry_t *geometry=getRectGeometry(obj,srcpe);
    // handle the root not in rectangle case
    if( x >= geometry->x0 && x  < geometry->x0+ geometry->xs &&
	y >= geometry->y0 && y  < geometry->y0+ geometry->ys &&
	z >= geometry->z0 && z  < geometry->z0+ geometry->zs)
      { // ok
	ComlibPrintf("[%d] create intermediate %d %d %d %d in rectangle %d %d %d, %d %d %d comm = %d \n", CkMyPe(), srcpe, x, y, z, 
		     geometry->x0, geometry->y0, geometry->z0,
		     geometry->xs, geometry->ys, geometry->zs, comm
		     );   

	obj->sourceInRectangle=true;
	obj->cornerRoot=srcpe;
      }
    else
      {
	ComlibPrintf("[%d] root %d %d %d %d NOT in rectangle %d %d %d, %d %d %d  \n", CkMyPe(), srcpe, x, y, z, 
		     geometry->x0, geometry->y0, geometry->z0,
		     geometry->xs, geometry->ys, geometry->zs
		     );   


	obj->sourceInRectangle=false;
	// fix the root to a corner
	obj->cornerRoot=assignCornerRoot(geometry,srcpe);
      }

    void *request = CkpvAccess(com_rect_ptr)->get(comm);
    if(request==NULL)
      {
	request=bgl_machine_RectBcastInit(comm, geometry);
	ComlibPrintf("[%d] cinter init comm %d section %d srcpe %d\n",CkMyPe(), comm, thisSectionID, srcpe);	
      }
    else{
      ComlibPrintf("[%d] cinter already init comm %d section %d srcpe %d\n",CkMyPe(), comm, thisSectionID, srcpe);
    }
    // we don't actually need the request on a typical intermediate
    // only the forwarding case cares.
    CkpvAccess(com_rect_ptr)->put(comm)= request;
#ifdef COMLIB_RECT_DEBUG
    void *getrequest =     CkpvAccess(com_rect_ptr)->get(comm);
    CkAssert(*((char *) request)==*((char *)getrequest));
    isSane(getrequest,comm);
#endif
    //    delete geometry;
    return obj;
}


void RectMulticastStrategy::doneInserting(){
    //Do nothing! Its a bracketed strategy
}

extern void CmiReference(void *);

//Send the multicast message the local array elements. The message is 
//copied and sent if elements exist. 
void RectMulticastStrategy::localMulticast(envelope *env, 
					   ComlibRectSectionHashObject *obj) {
    int nIndices = obj->indices.size();
    
    if(obj->msg != NULL) {
        CmiFree(obj->msg);
	obj->msg = NULL;
    } 
    
    if(nIndices > 0) {
	void *msg = EnvToUsr(env);
	void *msg1 = msg;
        
        msg1 = CkCopyMsg(&msg);
	
        ComlibArrayInfo::localMulticast(&(obj->indices), UsrToEnv(msg1));
    }    
}


//Calls default multicast scheme to send the messages. It could 
//also call a converse lower level strategy to do the muiticast.
//For example pipelined multicast
void RectMulticastStrategy::remoteMulticast(envelope *env, 
					    ComlibRectSectionHashObject *obj) {

    ComlibPrintf("[%d] remoteMulticast \n", CkMyPe());       
    int npes = obj->npes;
    int *pelist = obj->pelist;
    
    //    if(npes == 0) {
    //        CmiFree(env);
    //        return;    
    //    }
    
    //CmiSetHandler(env, handlerId);
    CmiSetHandler(env, CkpvAccess(strategy_handlerid));

    ((CmiMsgHeaderExt *) env)->stratid = getInstance();

    //Collect Multicast Statistics
    RECORD_SENDM_STATS(getInstance(), env->getTotalsize(), pelist, npes);
    int srcpe=env->getSrcPe();    
    CkPackMessage(&env);


    //    int destid=env->getGroupNum().idx;    
    
    //Sending a remote multicast
    ComlibMulticastMsg *cbmsg = (ComlibMulticastMsg *)EnvToUsr(env);
    int sectionID=cbmsg->_cookie.sInfo.cInfo.id;
    CkArrayID destid=obj->aid;
    // bgl_machineRectBcast should only be called once globally
    // per multicast
    int comm=computeKey(sectionID, srcpe, destid);
    ComlibPrintf("[%d] rectbcast using comm %d section %d srcpe %d request %p\n",CkMyPe(), comm, sectionID, srcpe, CkpvAccess(com_rect_ptr)->get(comm));
#ifdef COMLIB_RECT_DEBUG
    isSane(CkpvAccess(com_rect_ptr)->get(comm),comm);
#endif
    bgl_machine_RectBcast(comm  , (char*)env, env->getTotalsize());
    //CmiSyncListSendAndFree(npes, pelist, env->getTotalsize(), (char*)env);
    //CmiSyncBroadcastAndFree(env->getTotalsize(), (char*)env);
}

// For source not in rectangle case, forward to corner of rectangle
//
void RectMulticastStrategy::forwardMulticast(envelope *env, 
                                              ComlibRectSectionHashObject *obj) {

    ComlibPrintf("[%d] forwardMulticast \n", CkMyPe());       
    int *pelist = obj->pelist;
    int npes    = obj->npes;
    if(npes == 0) {
        CmiFree(env);
        return;    
    }
    // handler is changed to special root handler
    CmiSetHandler(env, handlerId);

    ((CmiMsgHeaderExt *) env)->stratid = getInstance();

    //Collect Multicast Statistics
    RECORD_SENDM_STATS(getInstance(), env->getTotalsize(), pelist, npes);
    
    CkPackMessage(&env);
    //Sending a remote multicast
    ComlibMulticastMsg *cbmsg = (ComlibMulticastMsg *)EnvToUsr(env);
    int sectionID=cbmsg->_cookie.sInfo.cInfo.id;

    //   CmiSyncListSendAndFree(npes, pelist, env->getTotalsize(), (char*)env);
    CmiSyncSendAndFree(obj->cornerRoot, env->getTotalsize(), (char*)env);
    //CmiSyncBroadcastAndFree(env->getTotalsize(), (char*)env);
}


void RectMulticastStrategy::pup(PUP::er &p){

    CharmStrategy::pup(p);
}

void RectMulticastStrategy::beginProcessing(int numElements){
    
    //handlerId = CkRegisterHandler((CmiHandler)DMHandler);    
    handlerId = CkRegisterHandler((CmiHandler)sourceOffRectstrategyHandler);    
    bgl_machine_RectBcastConfigure (rectRequest);
    CkArrayID dest;
    int nidx;
    CkArrayIndex *idx_list;

    ainfo.getDestinationArray(dest, idx_list, nidx);
    sinfo = ComlibSectionInfo(dest, myInstanceID);

    ComlibLearner *learner = new ComlibLearner();
    //setLearner(learner);
}

void RectMulticastStrategy::handleMessage(void *msg){
    envelope *env = (envelope *)msg;
    RECORD_RECV_STATS(getInstance(), env->getTotalsize(), env->getSrcPe());

    //Section multicast base message
    CkMcastBaseMsg *cbmsg = (CkMcastBaseMsg *)EnvToUsr(env);
    
    int status = cbmsg->_cookie.sInfo.cInfo.status;
    ComlibPrintf("[%d] In handleMessage %d\n", CkMyPe(), status);
    
    if(status == COMLIB_MULTICAST_NEW_SECTION)
        handleNewMulticastMessage(env);
    else {
        //status == COMLIB_MULTICAST_OLD_SECTION, use the cached section id
        ComlibSectionHashKey key(cbmsg->_cookie.pe, 
                                 cbmsg->_cookie.sInfo.cInfo.id);    
        
        ComlibRectSectionHashObject *obj;
        obj = sec_ht.get(key);
        
        if(obj == NULL)
            CkAbort("Destination indices is NULL\n");
#ifndef LOCAL_MULTI_OFF        
        localMulticast(env, obj);
#endif
	// remote is handled by the rectangle

    }
}

void RectMulticastStrategy::handleMessageForward(void *msg){
  
    envelope *env = (envelope *)msg;
    RECORD_RECV_STATS(getInstance(), env->getTotalsize(), env->getSrcPe());

    //Section multicast base message
    CkMcastBaseMsg *cbmsg = (CkMcastBaseMsg *)EnvToUsr(env);
    
    int status = cbmsg->_cookie.sInfo.cInfo.status;
    ComlibPrintf("[%d] In handleMessageForward %d\n", CkMyPe(), status);
    
    if(status == COMLIB_MULTICAST_NEW_SECTION)
        handleNewMulticastMessage(env);
    else {
        //status == COMLIB_MULTICAST_OLD_SECTION, use the cached section id
        ComlibSectionHashKey key(cbmsg->_cookie.pe, 
                                 cbmsg->_cookie.sInfo.cInfo.id);    
        
        ComlibRectSectionHashObject *obj;
        obj = sec_ht.get(key);
        
        if(obj == NULL)
            CkAbort("Destination indices is NULL\n");
#ifndef LOCAL_MULTI_OFF                
        localMulticast(env, obj);
#endif
        remoteMulticast(env, obj);
    }
}

#include <string>

void RectMulticastStrategy::handleNewMulticastMessage(envelope *env) {
    
    ComlibPrintf("[%d] : In handleNewMulticastMessage\n", CkMyPe());

    CkUnpackMessage(&env);    
    int sender=env->getSrcPe();
    int localElems;
    envelope *newenv;
    CkArrayIndex *local_idx_list;    
    
    sinfo.unpack(env, localElems, local_idx_list, newenv);

    ComlibMulticastMsg *cbmsg = (ComlibMulticastMsg *)EnvToUsr(env);
    ComlibSectionHashKey key(cbmsg->_cookie.pe, 
                             cbmsg->_cookie.sInfo.cInfo.id);
    
    ComlibRectSectionHashObject *old_obj = NULL;

    old_obj = sec_ht.get(key);

    if(old_obj != NULL) {
        delete old_obj;
    }

    /*
    CkArrayIndex *idx_list_array = new CkArrayIndex[idx_list.size()];
    for(int count = 0; count < idx_list.size(); count++)
        idx_list_array[count] = idx_list[count];
    */

    int cur_sec_id=cbmsg->_cookie.sInfo.cInfo.id;

    // need everyPe for rectangle not just local_idx_list

    ComlibMulticastMsg *lll = cbmsg;
    envelope *ppp = UsrToEnv(cbmsg);
    CkUnpackMessage(&ppp);
    int ttt=0;
    int uuu=0;
    for (int i=0; i<lll->nPes; ++i) {
      //	      ComlibPrintf(" %d (",lll->indicesCount[i].pe);
      uuu++;
      for (int j=0; j<lll->indicesCount[i].count; ++j) {
	//		ComlibPrintf(" %d",((int*)&(lll->indices[ttt]))[1]);
	ttt++;
      }
      //	      ComlibPrintf(" )");
    }
    ComlibPrintf("[%d] cbmsg for intermediate has %d indices %d pes\n",CkMyPe(),ttt, uuu);
    CkAssert(uuu>0);
    CkAssert(ttt>0);

    ComlibRectSectionHashObject *new_obj = createObjectOnIntermediatePe(ttt, cbmsg->indices, cbmsg->nPes, cbmsg->indicesCount, sender, cur_sec_id );
    // now revise obj for local use only
    sinfo.getRemotePelist(localElems, local_idx_list, new_obj->npes, new_obj->pelist);
    sinfo.getLocalIndices(localElems, local_idx_list, new_obj->indices);

    sec_ht.put(key) = new_obj;
    CkPackMessage(&newenv);   
#ifndef LOCAL_MULTI_OFF        

    localMulticast(newenv, new_obj); //local multicast always copies
#endif
    CmiFree(newenv);  //NEED this
}
#endif


/*@}*/
