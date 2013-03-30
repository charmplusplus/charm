#include "jacobi2d.decl.h"

/*
Jacobi iteratoin with a 2D Array.

Demonstrates 2d dense char array creation and use.  This code computes
the steady state heat distribution of a 2d plate using the jacobi
iteration:

while new_temp != temp
  //next temperature is average of surrounding temperatures 
  new_temp(x,y) = (temp(x-1,y)+temp(x+1,y)+temp(x,y-1)+temp(x,y+1))/4
  temp(x,y) = new_temp(x,y)
end

Every temp(x,y) is a chare in this sample application.  The main chare
sends a notice to all the nodes in our simulation telling them to
exchange information.  After each node is done updating it's
temperature, it sends a message back the error as the difference
between temp and new_temp.  Main quits driving the iterations when the
error has been minimized below some threshold.

*/

CProxy_Main mainProxy;
int num_rows;
int num_cols;

//allowed variation between temp and new_temp
float epsilon=1./1000;

//temperatures on the various boundries
float left = 1;
float top = 1;
float bottom = 0;
float right = 0;


class Main : public CBase_Main
{
public:
    int receive_count;
    CProxy_Jacobi array;
    int num_chares;

    Main(CkArgMsg* m) {
	mainProxy = thisProxy;

	//Allow the user to adjust the size of the grid at runtime
	num_cols = 5;
	if (m->argc == 2) {
	    num_cols = atoi(m->argv[1]);
	}
	delete m;
	
	num_rows = num_cols;

	CkPrintf("Running Jacobi on %d processors with (%d,%d) elements\n",
		 CkNumPes(), num_rows, num_cols);

	array = CProxy_Jacobi::ckNew(num_rows, num_cols);

	//save the total number of worker chares we have in this simulation
	num_chares = num_rows*num_cols;

	//Start the computation
	receive_count = num_chares;
	array.begin_iteration();
    }

    float max_error;

    void report_error(int row, int col, float error) {
      //CkPrintf("[main] (%d, %d) error=%g\n", row, col, error);

	if ((receive_count == num_chares)
	    ||
	    (fabs(error) > max_error)
	    ) {
	    max_error = fabs(error);
	}

	receive_count--;
	if (0 == receive_count) {
	    if (max_error < epsilon) {
		CkPrintf("All done\n");
		CkExit();
	    } else {
		CkPrintf("[main] error = %g, starting new iteration.\n", max_error);
		receive_count=num_chares;
		array.begin_iteration();
	    }
	}
    }
};
    
class Jacobi: public CBase_Jacobi {
public:
    float temperature;
    float update;
    int messages_due;

    Jacobi() {
	temperature = 0;
	messages_due = 4;
    }

    Jacobi(CkMigrateMessage* m) {}

    //added for array migration
    void pup(PUP::er &p){
	p | temperature;
	p | update;
	p | messages_due;
    }

    void begin_iteration(void) {
	/* messages_due should not be assigned here because it is not guaranteed
	 * that all elements' function "begin_iteration" are called before the
	 * function "receive_neighbor". Otherwise, the "messages_due" is messed up.
	 * For example, assuming the 2D array is of size 2*2. Then the array element(0,1)
	 * would send its data to element(1,1) and element(0,0) according to the sequence
	 * of calling "receive_neighbor" in this function. If element(1,1)'s "receive_neighbor"
	 * is called before its "begin_iteration" call and the "message_due" is reset
	 * in the "begin_iteration" function, then the "messages_due" of element(1,1) would be 
	 * messed up. Because one of 4 messages that element(1,1) needs to receive is already
	 * posted, the "messages_due" should not be reset, otherwise element(1,1) would think
	 * there's still one msg to receive and woudl not progress to the next iteration.
	 * The solution is to initialize the value in Jacobi's constructor and reset it
	 * before doing the next iteration (in the check_done_iteration function) --Chao Mei
	 */
//	messages_due = 4;
	update = 0;

	//enforce the boundary conditions.  Nodes on an edge shouldn't
	//send messages to non-existant chares.
	if (thisIndex.x == 0) {
	    update += top;
	    messages_due--;
	} else {
	    thisProxy(thisIndex.x-1, thisIndex.y).receive_neighbor(temperature);
	}

	if (thisIndex.x == num_rows-1) {
	    update += bottom;
	    messages_due--;
	} else {
	    thisProxy(thisIndex.x+1, thisIndex.y).receive_neighbor(temperature);
	}

	if (thisIndex.y == 0) {
	    update += left;
	    messages_due--;
	} else {
	    thisProxy(thisIndex.x, thisIndex.y-1).receive_neighbor(temperature);
	}

	if (thisIndex.y == num_cols-1) {
	    update += right;
	    messages_due--;
	} else {
	    thisProxy(thisIndex.x, thisIndex.y+1).receive_neighbor(temperature);
	}

	//check to see if we still need messages.
	check_done_iteration();
    }

    void receive_neighbor(float new_temperature) {
	update += new_temperature;
	messages_due--;
	check_done_iteration();
    }

    void check_done_iteration() {
	if (messages_due == 0) {
	    update /= 4;
	    float error = update-temperature;
	    temperature = update;
	    messages_due = 4;
	    mainProxy.report_error(thisIndex.x, thisIndex.y, error);
	}
    }
 
};

#include "jacobi2d.def.h"
