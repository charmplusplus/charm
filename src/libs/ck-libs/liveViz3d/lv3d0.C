/*
  Implementation of the simplest server portion 
  of the liveViz3d library.
  
  Orion Sky Lawlor, olawlor@acm.org, 2003/9/13
*/
#include "lv3d0.h"

#ifdef CMK_LIVEVIZ3D_CLIENT
#  include "ogl/main.h"
#  include "ogl/util.h"
#endif

/// Private class that stores object data for the universe client side.
class LV3D_Universe_Table {
	CkHashtableT<CkViewableID,CkView *> table;
public:
	void add(CkView *v) {
		delete table.get(v->id);
		table.put(v->id)=v;
	}
	
	CkView *lookup(const CkViewableID &src) {
		return table.get(src);
	}
	
	void render(const CkViewpoint &vp) {
#ifdef CMK_LIVEVIZ3D_CLIENT
		CkHashtableIterator *it=table.iterator();
		void *obj;
		while (NULL!=(obj=it->next())) {
			CkView *v=*(CkView **)obj;
			v->render(1.0,0);
		}
		delete it;
#endif
	}
};

LV3D_Universe::~LV3D_Universe() {
	if (object_table) delete object_table;
}
void LV3D_Universe::pup(PUP::er &p) {
	PUP::able::pup(p);
	if (object_table) CkAbort("Cannot migrate a LV3D_Universe with objects!\n");
}

#ifdef CMK_LIVEVIZ3D_CLIENT
/**
  Set up this client.  Should set any needed drawing options.
  Default does nothing, which gives you no lighting,
  no depth, with alpha, and clear blue background color.
*/
void LV3D_Universe::setupClient(oglOptions &i)
{
	/* nothing */
}
void LV3D_Universe::setupGL(void) {}

/**
  Return the client GUI controller.  Default is a trackball,
  centered on the middle of the unit cube.
*/
oglController *LV3D_Universe::makeController(void)
{
	return new oglTrackballController(3.0,40.0, CkVector3d(0.5,0.5,0.5));
}

/**
  Add this view to the universe.  This is called
  once per incoming network view on the client.
  This call transfers ownership of the view.
*/
void LV3D_Universe::viewResponse(CkView *v) {
	if (!object_table) {
		object_table=new LV3D_Universe_Table;
	}
	v->ref();
	object_table->add(v);
	oglRepaint(10);
}

CkView *LV3D_Universe::lookup(const CkViewableID &src) {
	if (object_table) 
		return object_table->lookup(src);
	else
		return NULL;
}

/**
  Draw the world to this camera using OpenGL calls.  
  This routine is called once per frame on the client.
*/
void LV3D_Universe::render(const CkViewpoint &vp) {
	if (object_table) object_table->render(vp);
}


#endif


PUPable_def(LV3D_Universe)

