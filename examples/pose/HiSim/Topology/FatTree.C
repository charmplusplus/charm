#include "FatTree.h"
#include "InitNetwork.h"
#include "math.h"

FatTree::FatTree() {
}

void FatTree::getNeighbours(int switchId,int numP) {
        int numNodes = config.numNodes,i,fanout=numP/2;
        int numLevelSwitches = numNodes/fanout;
        int numDimSwitches,curDimOffset,curDim,upperDim,lowerDim;
        level = switchId/numLevelSwitches;
        curLevelId = switchId - (level * numLevelSwitches) ;

	next = new int[2*fanout];

        int tmp = numNodes,numLevels=-1; 
        while(tmp) { tmp/= fanout; numLevels++; }

        if( level != 0) {

	//numDimSwitches*fanout nodes can communicate with each other with numDimSwitches at the top
	// Basically we can think of switch at current level as blocks of switches with (numDimSwitches*fanout) links

         numDimSwitches = (int)(pow((float)fanout,level)); 
         curDimOffset = curLevelId % numDimSwitches;
         curDim    = curLevelId / numDimSwitches ;
         upperDim   = (curLevelId % (numDimSwitches * fanout)) /  numDimSwitches ;   // 0 < curDimOffset < numDimSwitches
         lowerDim   = curDimOffset / (numDimSwitches/fanout) ;

        // Use a cool fact that the ports at next level have all same Id when connecting to current port

        }
        // Upper bound should be nodeRangeEnd-1
        if(level == 0) {
                nodeRangeStart = curLevelId * fanout;
                nodeRangeEnd = (curLevelId+1) * fanout;
                outNextPortUp = curLevelId % fanout;
        } else {
                nodeRangeStart = curDim * numDimSwitches * fanout ;
                nodeRangeEnd   = (curDim+1) * numDimSwitches * fanout ;
                outNextPortUp = upperDim;
                outNextPortDown = lowerDim;
        }
        int nextLevel = level - 1;

        if( level != 0) {
                for(i = 0;i < fanout; i++)  {
		// The current dimension can be thought of as interconnecting four smaller dimensions at the level below
		// Each port will signify the sub-dimension in the level immediately below. One addnl bit of info is that
		// my id within my sub-dimension in level (i+1) will connect to same id with "fanout" sub-divisions in level (i)

                        next[i]  = config.switchStart + nextLevel * numLevelSwitches  + curDim * numDimSwitches
                        +i * (numDimSwitches/fanout) + curDimOffset % (numDimSwitches/fanout) ;
                }
        } else {
                for(i = 0;i < fanout; i++)  {
                        next[i] = curLevelId*fanout+i+config.nicStart;
                }
        }
        nextLevel = level + 1;

        if(level == 0) {
                for(int i= fanout; i< 2*fanout; i++)
                        next[i] = switchId + config.switchStart + numLevelSwitches - (switchId % fanout)+ (i-fanout);
        }
        else if(level != numLevels-1    ) {
                for(int i = fanout;i < 2*fanout; i++)  {

		// First step is to figure out, where my parent dimension starts ? Basically  I have to increase the dimension
		// size by fanout and select the dimension to which I belong. Once the start of dimension in nextLevel is found,
		// I will connect to 4 sub-dimensions in the next level. Each sub-dimension has the same size as mine. I will
		// preserve my offset in the next level sub-dimensions too. As usual sub-dimensions are selected based on port
	
                        next[i]  = config.switchStart + nextLevel * numLevelSwitches
                        + (curLevelId / (numDimSwitches * fanout)) * (numDimSwitches * fanout)
                        +(i-fanout) * numDimSwitches + curDimOffset ;
                }
        } else {
                for(int i = fanout;i < 2*fanout; i++)
                        next[i] = -1;
        }
}

int FatTree::getNext(int portid,int nodeid,int numP) {
	if(next[portid] < 0) {
		CkPrintf("nodeid %d portid %d \n",nodeid,portid);
	}
        return(next[portid]);
}

int FatTree::getNextChannel(int portid,int switchid,int numP) {
	// Offset for nodes to connect to switches ( if channels are added)
	int offset = 0,startChan; 
	startChan = config.numP * (switchid-config.switchStart) + portid;
	return(config.ChannelStart + startChan + offset );
}	

int FatTree::getStartSwitch(int nodeid) {
	return(nodeid/(config.numP/2));
}

int FatTree::getStartPort(int nodeid,int nump) {
	int fanout;
	fanout = (config.numP/2);
	return((nodeid%fanout)+fanout);
}

int FatTree::getStartVc() {
	// No flow/buffering control with the net interface
	return 0;
}

int FatTree::getStartNode() {
	return nodeRangeStart;
}

int FatTree::getEndNode() {
	return nodeRangeEnd;
}
