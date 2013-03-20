/** 

    A system for exposing application and runtime "control points" 
    to the dynamic optimization framework.

*/
#ifndef __ARRAYREDISTRIBUTOR_H__
#define __ARRAYREDISTRIBUTOR_H__

#include <vector>
#include <list>
#include <map>
#include <cmath>
//#include "ControlPoints.decl.h"

#include<pup_stl.h>


#if CMK_WITH_CONTROLPOINT


/**
 * \addtogroup ControlPointFramework
 *   @{
 */


/// A message containing a chunk of a data array used when redistributing to a different set of active chares 
class redistributor2DMsg : public CMessage_redistributor2DMsg { 
 public: 
  int top;         
  int left; 
  int height;      
  int width; 
  int new_chare_cols; 
  int  new_chare_rows; 
  int which_array; 
  double *data;
}; 
 


/// Integer Maximum
static int maxi(int a, int b){
  if(a>b)
    return a;
  else
    return b;
}

/// Integer Minimum
static int mini(int a, int b){
  if(a<b)
    return a;
  else
    return b;
}


/// A chare group that can redistribute user data arrays. It is used by binding it to a user's Chare Array
class redistributor2D: public CBase_redistributor2D {
 public:

  std::map<int,double*> data_arrays;
  std::map<int,int> data_arrays_sizes;

  /// The array associated with this data redistribution
  CProxyElement_ArrayElement associatedArray;
  
  int incoming_count;
  std::map<int,double*> data_arrays_incoming;
  std::map<int,int> data_arrays_incoming_sizes;

  /// Is this array element active
  bool thisElemActive;

  bool resizeGranulesHasBeenCalled;

  CkVec<redistributor2DMsg *> bufferedMsgs;

 private:


  void *fakeMemoryUsage;


  CkCallback dataRedistributedCallback;

  int x_chares; // number of active chares in x dimension
  int y_chares; // number of active chares in y dimension

  int data_width;  // The width of the global array, not the local piece
  int data_height; // The height of the global array, not the local piece

  int data_x_ghost; // The padding in the x dimension on each side of the data
  int data_y_ghost; // The padding in the y dimension on each side of the data


 public:

  void pup(PUP::er &p) {
    p | data_arrays_sizes;
    p | data_arrays_incoming_sizes;
    p | incoming_count;
    p | associatedArray;
    p | thisElemActive;

    p | dataRedistributedCallback;

    p | resizeGranulesHasBeenCalled;

    p | x_chares;
    p | y_chares;
    p | data_width;
    p | data_height;
    p | data_x_ghost;
    p | data_y_ghost;

    if(p.isPacking() && fakeMemoryUsage!=NULL)
      free(fakeMemoryUsage);

    fakeMemoryUsage = NULL;

    ////////////////////////////////
    // when packing, iterate through data_arrays
    // when unpacking

    {
      std::map<int,int>::iterator iter;
      for(iter = data_arrays_sizes.begin(); iter != data_arrays_sizes.end(); iter++){
	int whichArray = iter->first;
	int arraySize = iter->second;

	//	CkPrintf("Pupping data array %d\n",whichArray);
	p | whichArray;

	if(p.isUnpacking())
	  data_arrays[whichArray] = new double[arraySize];

	PUParray(p,data_arrays[whichArray] ,arraySize);
	
	if(p.isPacking())
	  delete[] data_arrays[whichArray];
	
      }
    }


    ///////////////////////////////
    {
      std::map<int,int>::iterator iter;
      for(iter = data_arrays_incoming_sizes.begin(); iter != data_arrays_incoming_sizes.end(); iter++){
	int whichArray = iter->first;
	int arraySize = iter->second;

	//	CkPrintf("Pupping incoming array %d\n",whichArray);
	p | whichArray;

	if(p.isUnpacking() && data_arrays_incoming_sizes[whichArray] > 0)
	  data_arrays_incoming[whichArray] = new double[arraySize];
	
	PUParray(p,data_arrays_incoming[whichArray],arraySize);
	
	if(p.isPacking())
	  delete[] data_arrays_incoming[whichArray];
	
      }
    }

    //    CkPrintf("pup redistributor2D\n");
  } 


  void ckJustMigrated(){
    //   CkPrintf("redistributor element %02d %02d migrated to %d", thisIndex.x, thisIndex.y, CkMyPe());
  }


  // ------------ Some routines for computing the array bounds for this chare  ------------ 

  // The index in the global array for my top row  
  int top_data_idx();
 
  int bottom_data_idx();
 
  int left_data_idx();
 
  int right_data_idx();
 
  int top_neighbor();
   
  int bottom_neighbor();
   
  int left_neighbor();
 
  int right_neighbor();
  
  
  /// the width of the non-ghost part of the local partition 
  int mywidth();
    
    
  // the height of the non-ghost part of the local partition 
  int myheight();
    


  // ------------ Some routines for computing the array bounds for arbitrary chares  ------------ 

  int top_data_idx(int y, int y_total){
    return (data_height * y) / y_total;
  }

  int bottom_data_idx(int y, int y_total){
    return ((data_height * (y+1)) / y_total) - 1;
  }

  int left_data_idx(int x, int x_total){
    return (data_width * x) / x_total;
  }

  int right_data_idx(int x, int x_total){
    return ((data_width * (x+1)) / x_total) - 1;
  }


  int top_data_idx(int y){
    return (data_height * y) / y_chares;
  }

  int bottom_data_idx(int y){
    return ((data_height * (y+1)) / y_chares) - 1;
  }

  int left_data_idx(int x){
    return (data_width * x) / x_chares;
  }

  int right_data_idx(int x){
    return ((data_width * (x+1)) / x_chares) - 1;
  }

  /// Return which chare array element(x index) owns the global data item i
  int who_owns_idx_x(int i){
    int w=0;
    while(1){
      if( i >= left_data_idx(w) && i <= right_data_idx(w) ){
	return w;
      }
      w++;
    }
  }
  
  /// Return which chare array element(y index) owns the global data item i
  int who_owns_idx_y(int i){
    int w=0;
    while(1){
      if( i >= top_data_idx(w) && i <= bottom_data_idx(w) ){
	return w;
      }
      w++;
    }
  }
  



  
  // Convert a local column,row id (0 to mywidth()-1, 0 to myheight()-1) to the index in the padded array
  int local_to_padded(int x, int y){
    CkAssert(thisElemActive);
    CkAssert(x < (mywidth()+data_x_ghost) && x >= (0-data_x_ghost) && y < (myheight()+data_y_ghost) && y >= (0-data_y_ghost) );
    return (mywidth()+2*data_x_ghost)*(y+data_y_ghost)+x+data_x_ghost;
  }

  // get a data value
  double data_local(int which, int x, int y){
    CkAssert(local_to_padded(x,y) < data_arrays_sizes[which]);
    return data_arrays[which][local_to_padded(x,y)];
  }


  // Convert a local column id (0 to mywidth-1) to the global column id (0 to data_width-1)
  int local_to_global_x(int x){
    return left_data_idx() + x;
  }

  // Convert a local row id (0 to myheight-1) to the global row id (0 to data_height-1)
  int local_to_global_y(int y){
    return top_data_idx() + y;
  }

  int global_array_width(){
    return data_width;
  }

  int global_array_height(){
    return data_height;
  }

  int global_array_size(){
    return global_array_width() * global_array_height();
  }

  int my_array_width(){
    return mywidth()+2*data_x_ghost;
  }

  int my_array_height(){
    return myheight()+2*data_y_ghost;
  }

  // Total size of arrays including ghost layers
  int my_array_size(){
    return my_array_width() * my_array_height();
  }

  /// Create an array. If multiple arrays are needed, each should have its own index
  template <typename t> t* createDataArray(int which=0) {
    t* data = new t[my_array_size()];
    data_arrays[which] = data;
    data_arrays_sizes[which] = my_array_size();

    if(thisIndex.x==0 && thisIndex.y==0)  
      CkPrintf("data_arrays_sizes[which] set to %d\n", data_arrays_sizes[which] );  


    CkAssert(data_arrays[which] != NULL);
#if DEBUG > 2
    CkPrintf("Allocated array of size %d at %p\n", my_array_size(), data_arrays[which] );
#endif
    return data;
  }
  
  template <typename t> t* getDataArray(int which=0) {
    return data_arrays[which]; 
  }

  /// Constructor takes in the dimensions of the array, including any desired ghost layers
  /// The local part of the arrays will have (mywidth+x_ghosts*2)*(myheight+y_ghosts*2) elements 
  void setInitialDimensions(int width, int height, int x_chares_, int y_chares_, int x_ghosts=0, int y_ghosts=0){
    data_width = width;      // These values cannot change after this method is called.
    data_height = height;
    data_x_ghost = x_ghosts;
    data_y_ghost = y_ghosts;
    
    setDimensions(x_chares_, y_chares_);

  }
  

  void setDimensions( int x_chares_, int y_chares_){
    x_chares = x_chares_;
    y_chares = y_chares_;
    
    
    if( thisIndex.x < x_chares && thisIndex.y < y_chares ){
      thisElemActive = true;
    } else {
      thisElemActive = false;
    }
    
  }


  redistributor2D(){
    incoming_count = 0;
    fakeMemoryUsage = NULL;
    CkAssert(bufferedMsgs.size() == 0);
  }


  redistributor2D(CkMigrateMessage*){
    CkAssert(bufferedMsgs.size() == 0);
  }


  void startup(){
#if DEBUG > 3 
   CkPrintf("redistributor 2D startup %03d,%03d\n", thisIndex.x, thisIndex.y);
#endif

    contribute();
  }
  

  void printArrays(){
#if DEBUG > 2
    CkAssert(data_arrays.size()==2);
    for(std::map<int,double*>::iterator diter = data_arrays.begin(); diter != data_arrays.end(); diter++){
      int which_array = diter->first;
      double *data = diter->second;
      CkPrintf("%d,%d data_arrays[%d] = %p\n", thisIndex.x, thisIndex.y, which_array, data);
    }
#endif
  }

  
  // Called on all elements involved with the new granularity or containing part of the old data
  void resizeGranules(int new_active_chare_cols, int new_active_chare_rows){
#if DEBUG>1
    CkPrintf("Resize Granules called for elem %d,%d\n", thisIndex.x, thisIndex.y);  	
#endif

    resizeGranulesHasBeenCalled = true;

    const bool previouslyActive = thisElemActive;
    const int old_top = top_data_idx();
    const int old_left = left_data_idx();
    const int old_bottom = top_data_idx()+myheight()-1;
    const int old_right = left_data_idx()+mywidth()-1;
    const int old_myheight = myheight();
    const int old_mywidth = mywidth();

    setDimensions(new_active_chare_cols, new_active_chare_rows); // update dimensions & thisElemActive
    
    const int new_mywidth = mywidth();
    const int new_myheight = myheight();

    // Transpose Data
    // Assume only one new owner of my data

    if(previouslyActive){
     
      // Send all my data to any blocks that will need it

      int newOwnerXmin = who_owns_idx_x(old_left);
      int newOwnerXmax = who_owns_idx_x(old_right);
      int newOwnerYmin = who_owns_idx_y(old_top);
      int newOwnerYmax = who_owns_idx_y(old_bottom);

      for(int newx=newOwnerXmin; newx<=newOwnerXmax; newx++){
	for(int newy=newOwnerYmin; newy<=newOwnerYmax; newy++){
	  
	  // Determine overlapping region between my data and this destination
#if DEBUG > 2
	  CkPrintf("newy(%d)*new_myheight(%d)=%d, old_top=%d\n",newy,new_myheight,newy*new_myheight,old_top);
#endif
	  // global range for overlapping area
	  int global_top = maxi(top_data_idx(newy),old_top);
	  int global_left = maxi(left_data_idx(newx),old_left);
	  int global_bottom = mini(bottom_data_idx(newy),old_bottom);
	  int global_right = mini(right_data_idx(newx),old_right);
	  int w = global_right-global_left+1;
	  int h = global_bottom-global_top+1;
	 
	  CkAssert(w*h>0);

	  int x_offset = global_left - old_left;
	  int y_offset = global_top - old_top;

#if DEBUG > 2	  
	  CkPrintf("w=%d h=%d x_offset=%d y_offset=%d\n", w, h, x_offset, y_offset);
#endif
	  
	  std::map<int,double*>::iterator diter;
	  for(diter =data_arrays.begin(); diter != data_arrays.end(); diter++){
	    
	    redistributor2DMsg* msg = new(w*h) redistributor2DMsg;  
	    //	    CkPrintf("Created message msg %p\n", msg);  
	    
	    int which_array = diter->first;
	    double *t = diter->second;
	    int s = data_arrays_sizes[which_array];
	    
	    for(int j=0; j<h; j++){
	      for(int i=0; i<w; i++){		
		CkAssert(j*w+i < w*h);
		CkAssert((data_x_ghost*2+old_mywidth)*(j+y_offset+data_y_ghost)+(i+ x_offset+data_x_ghost) < s);
		msg->data[j*w+i] = t[(data_x_ghost*2+old_mywidth)*(j+y_offset+data_y_ghost)+(i+ x_offset+data_x_ghost)];
	      }
	    }
	    
	    msg->top = global_top;
	    msg->left = global_left;
	    msg->height = h;
	    msg->width = w;
	    msg->new_chare_cols = new_active_chare_cols;
	    msg->new_chare_rows = new_active_chare_rows; 
	    msg->which_array = which_array;

	    //	    CkPrintf("Sending message msg %p\n", msg); 	    
	    thisProxy(newx, newy).receiveTransposeData(msg);
	    
	  }
	  
	}
	
	
      }
    } 
    
    if(!thisElemActive){
#if DEBUG > 2
      CkPrintf("Element %d,%d is no longer active\n", thisIndex.x, thisIndex.y);
#endif

      // Free my arrays
      for(std::map<int,double*>::iterator diter = data_arrays.begin(); diter != data_arrays.end(); diter++){
	int which_array = diter->first;
	delete data_arrays[which_array]; 
	data_arrays[which_array] = NULL;
	data_arrays_sizes[which_array] = 0;
      }
      continueToNextStep();
      
    }


    // Call receiveTransposeData for any buffered messages.
    int size = bufferedMsgs.size();
    for(int i=0;i<size;i++){
      redistributor2DMsg *msg = bufferedMsgs[i];
      //     CkPrintf("Delivering buffered receiveTransposeData(msg=%p) i=%d\n", msg, i);
      receiveTransposeData(msg); // this will delete the message
    }
    bufferedMsgs.removeAll();

    int newPe = (thisIndex.y * new_active_chare_cols + thisIndex.x) % CkNumPes();
    if(newPe == CkMyPe()){
      //      CkPrintf("Keeping %02d , %02d on PE %d\n", thisIndex.x, thisIndex.y, newPe);
    }
    else{
      // CkPrintf("Migrating %02d , %02d to PE %d\n", thisIndex.x, thisIndex.y, newPe);
      migrateMe(newPe);
    }
    // CANNOT CALL ANYTHING AFTER MIGRATE ME
  }
  
  
  void continueToNextStep(){
#if DEBUG > 2
    CkPrintf("Elem %d,%d is ready to continue\n", thisIndex.x, thisIndex.y);
#endif

    resizeGranulesHasBeenCalled = false;

    for(std::map<int,double*>::iterator diter =data_arrays.begin(); diter != data_arrays.end(); diter++){
      int which_array = diter->first;
      double *data = diter->second;
      if( ! ((data==NULL && !thisElemActive) || (data!=NULL && thisElemActive) )){
	CkPrintf("[%d] ERROR: ! ((data==NULL && !thisElemActive) || (data!=NULL && thisElemActive) )",CkMyPe());
	CkPrintf("[%d] ERROR: data=%p thisElemActive=%d  (perhaps continueToNextStep was called too soon)\n",CkMyPe(), data, (int)thisElemActive );

	CkAbort("ERROR");	
      }
    }
    
    
#if USE_EXTRAMEMORY
#error NO USE_EXTRAMEMORY ALLOWED YET
    if(thisElemActive){

      long totalArtificialMemory = controlPoint("Artificial Memory Usage", 100, 500);
      long artificialMemoryPerChare = totalArtificialMemory *1024*1024 / x_chares / y_chares;
      
      CkPrintf("Allocating fake memory of %d MB (of the total %d MB) (xchares=%d y_chares=%d)\n", artificialMemoryPerChare/1024/1024, totalArtificialMemory, x_chares, y_chares);
      free(fakeMemoryUsage);
      fakeMemoryUsage = malloc(artificialMemoryPerChare);
      CkAssert(fakeMemoryUsage != NULL);
    } else {
      free(fakeMemoryUsage);
      fakeMemoryUsage = NULL;
    }
#endif



    incoming_count = 0; // prepare for future granularity change 
    contribute();
  }
  
  




  
  void receiveTransposeData(redistributor2DMsg *msg){
    
    // buffer this message until resizeGranules Has Been Called
    if(!resizeGranulesHasBeenCalled){
      bufferedMsgs.push_back(msg);
      //      CkPrintf("Buffering receiveTransposeData(msg=%p)\n", msg);
      return;
    }
    
    CkAssert(resizeGranulesHasBeenCalled);
    
    int top_new = top_data_idx(thisIndex.y, msg->new_chare_rows);
    int bottom_new = bottom_data_idx(thisIndex.y, msg->new_chare_rows);
    int left_new = left_data_idx(thisIndex.x, msg->new_chare_cols);
    int right_new = right_data_idx(thisIndex.x, msg->new_chare_cols);    

    int new_height = bottom_new - top_new + 1;
    int new_width = right_new - left_new + 1;

    if(incoming_count == 0){
      // Allocate new arrays 
      std::map<int,double*>::iterator diter;
      for(diter =data_arrays.begin(); diter != data_arrays.end(); diter++){
	int w = diter->first;
	data_arrays_incoming[w] = new double[(new_width+2*data_x_ghost)*(new_height+2*data_y_ghost)];
	data_arrays_incoming_sizes[w] = (new_width+2*data_x_ghost)*(new_height+2*data_y_ghost);

	//	CkPrintf("data_arrays_incoming_sizes[%d] set to %d\n", w, data_arrays_incoming_sizes[w] );  

      }
    }
    
    
    // Copy values from the incoming array to the appropriate place in data_arrays_incoming
    // Current top left of my new array


    double *localData = data_arrays_incoming[msg->which_array];
    int s = data_arrays_incoming_sizes[msg->which_array];

    //    CkPrintf("%d,%d data_arrays_incoming.size() = %d\n", thisIndex.x, thisIndex.y, data_arrays_incoming.size() );
    //    CkPrintf("msg->which_array=%d   localData=%p   s=%d\n", msg->which_array, localData, s);
    CkAssert(localData != NULL);

    for(int j=0; j<msg->height; j++){
      for(int i=0; i<msg->width; i++){

	if( (msg->top+j >= top_new) && (msg->top+j <= bottom_new) && (msg->left+i >= left_new) && (msg->left+i <= right_new) ) {
	  CkAssert(j*msg->width+i<msg->height*msg->width);
	  CkAssert((msg->top+j-top_new)*new_width+(msg->left+i-left_new) < new_width*new_height);
	  CkAssert((msg->top+j-top_new)*new_width+(msg->left+i-left_new) >= 0);
	  
	  CkAssert((msg->top+j-top_new+data_y_ghost)*(new_width+2*data_x_ghost)+(msg->left+i-left_new+data_x_ghost) < s);
	  localData[(msg->top+j-top_new+data_y_ghost)*(new_width+2*data_x_ghost)+(msg->left+i-left_new+data_x_ghost)] = msg->data[j*msg->width+i];
	  incoming_count++;
	  
	}
	
      }
    }
    
    //    CkPrintf("Deleting message msg %p\n", msg); 
    delete msg;


    if(incoming_count == new_height*new_width*data_arrays.size()){

      std::map<int,double*>::iterator diter;
      for(diter =data_arrays.begin(); diter != data_arrays.end(); diter++){
	int w = diter->first;
	delete[] data_arrays[w];
	data_arrays[w] = data_arrays_incoming[w];
	data_arrays_sizes[w] = data_arrays_incoming_sizes[w];
	data_arrays_incoming[w] = NULL;
	data_arrays_incoming_sizes[w] = 0;

	//        if(thisIndex.x==0 && thisIndex.y==0)   
	  //          CkPrintf("data_arrays_incoming_sizes[%d] set to %d\n",w, data_arrays_incoming_sizes[w] );   

	  //        if(thisIndex.x==0 && thisIndex.y==0) 
	  //  CkPrintf("data_arrays_sizes[%d] set to %d\n",w, data_arrays_sizes[w] ); 

      }

      continueToNextStep();
    }
    
  }
};

/** @} */
#endif
#endif
