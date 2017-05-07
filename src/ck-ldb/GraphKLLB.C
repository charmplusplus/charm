#include "GraphKLLB.h"

#include <algorithm>
#include<limits>

#include "cklists.h"
#define THRESHOLD 50.0
#define DEBUG(x) /*CkPrintf x*/ 

using namespace std;

CreateLBFunc_Def(GraphKLLB, "Greedy loadbalancer for multiple constraints.")

GraphKLLB::GraphKLLB(const CkLBOptions &opt): CBase_GraphKLLB(opt) {
  lbname = "GraphKLLB";

  /* Centralized load balancer */
  if (CkMyPe()==0)
    CkPrintf("[%d] GraphKLLB created\n",CkMyPe());

  srand(17);
}

bool GraphKLLB::QueryBalanceNow(int _step) {
  return true;
}

template <class T, class Compare>
void BubbleDown(int index, std::vector<T>& data, Compare& comp, std::vector<int>& pos) {
    int length = data.size();
    if(length == 0)
  	    return;
    if(index < 0)
        return;
    int left_child = 2*index + 1;
    int right_child = 2*index + 2;

    if(left_child >= length)
        return; /*index is a leaf*/

    int min_index = index;

    if(comp(data[index], data[left_child])) {
        min_index = left_child;
    }

    if((right_child < length) && comp(data[min_index], data[right_child])) {
        min_index = right_child;
    }

    if(min_index != index) {
    /*need to swap*/
        T temp = data[index];
        data[index] = data[min_index];
        /* Update the position*/				
        data[min_index] = temp;
        pos[data[min_index]]=min_index;
        pos[data[index]]=index;
        BubbleDown(min_index, data, comp, pos);
    }
}

template <class T, class Compare>
void BubbleUp(int index, std::vector<T>& data, Compare& comp, std::vector<int>& pos) {
    int length = data.size();
    if(length==0 )
        return;
    if(index <= 0)
        return;

    int parentIndex = (index-1)/2;

    /* The parent is supposed to have smaller value, so swap*/
    if(comp(data[parentIndex], data[index])) {
        T temp = data[parentIndex];
        data[parentIndex] = data[index];
    
        data[index] = temp;
        /* Update the position*/
        pos[data[parentIndex]] = parentIndex;
        pos[data[index]] = index;
        BubbleUp(parentIndex, data, comp, pos);
    }
}

template <class T, class Compare>
void heap_update(int pe_id, std::vector<T>& data, Compare comp,
    std::vector<int>& pos) {

    int idx = pos[pe_id];
    if (idx < 0) {
        return;
    }
    int parent_index = (idx-1)/2;
    if (idx > 0 && comp(data[parent_index], data[idx])) {
        BubbleUp(idx, data, comp, pos);
    } else {
        BubbleDown(idx, data, comp, pos);
    }
}

template <class T>
void CheckHeapPos(std::vector<T>& data, std::vector<int>& pos) {
    for (int i = 0; i < data.size(); i++) {
        if (pos[data[i]] != i) {
            CkPrintf("position of data %d is not %d as expected but is %d\n", data[i], i, pos[data[i]]);
        }
    }
}


template <class T, class Compare>
T heap_pop(std::vector<T>& data, Compare comp, std::vector<int>& pos) {
    int length = data.size();
    if(length==0)
        return -1;
    T ret_val  = data[0];
    if (length != 0) {
        data[0] = data[length-1];
        data.pop_back();
        pos[data[0]]=0; 
        BubbleDown(0, data, comp, pos);
    }
    pos[ret_val] = -1;
    return ret_val;
}

template <class T, class Compare>
void heap_insert(std::vector<T>& data, T val, Compare comp, std::vector<int>& pos) {
    int length = data.size();
    data.push_back(val);
    pos[val] = length;
    BubbleUp(length, data, comp, pos);
}

template <class T, class Compare>
void heapify(std::vector<T>& data, Compare comp, std::vector<int>& pos) {
    for (int i = data.size()-1; i >= 0; --i) {
        BubbleDown(i, data, comp, pos); 
    }
}

class ObjCompareOperator {
 public:
    ObjCompareOperator(std::vector<Vertex>* obj, int* gain_val): objs(obj),
      gains(gain_val) {}

    bool operator()(int v1, int v2) {
        return (gains[v1] > gains[v2]);
        /*return (v1 > v2);*/
    }
 private:
    std::vector<Vertex>* objs;
    int* gains;
};

void GraphKLLB::InitializeObjHeap(LDStats *stats, int* obj_arr,int size,
    int* gain_val) {
    for(int obj = 0; obj <size; obj++) {
        obj_heap[obj]=obj_arr[obj];
        heap_pos[obj_arr[obj]]=obj;
    }
    heapify(obj_heap, ObjCompareOperator(&objs, gain_val), heap_pos);
}

void GraphKLLB::InitializeObjs(LDStats *stats, int* obj_arr, int* gain_val) {
  total_vload = 0.0;
  objs.reserve(n_objs);

    for(int obj = 0; obj < n_objs; obj++) {
        LDObjData &oData = stats->objData[obj];
        objs.push_back(Vertex(obj, oData.wallTime, stats->objData[obj].migratable, stats->from_proc[obj]));
        total_vload += oData.wallTime;
        obj_arr[obj] = obj;

        gain_val[obj] = 0;
        
        /* Go over the edges of this vertex and add the total bytes to be the gain
        Usually gain is sum of edge weights with same partition - edge weight
        outside this partition. Initially we assume every node is in one partition.*/
        for(int i = 0; i < ogr->vertices[obj].sendToList.size(); i++) {
            gain_val[obj] += ogr->vertices[obj].sendToList[i].getNumBytes();
        }
        for(int i = 0; i < ogr->vertices[obj].recvFromList.size(); i++) {
            gain_val[obj] += ogr->vertices[obj].recvFromList[i].getNumBytes();
        }
    }
}

void GraphKLLB::RefineGraphKLLB_setgains(int* obj_arr,unordered_map<int,int>& Ra,unordered_map<int,int>& Rb,
  int idx,int size)
{
    /*Updating the gain values*/
    for(int ib=0;ib<idx;ib++)  {
        for(int i = 0; i < ogr->vertices[obj_arr[ib]].sendToList.size(); i++)   {
            int nbr = ogr->vertices[obj_arr[ib]].sendToList[i].getNeighborId();
		    unordered_map<int,int>::const_iterator got = Rb.find (nbr);
			unordered_map<int,int>::const_iterator got_2 = Ra.find (nbr);
		    if(got!=Rb.end())  {
			    Ra[obj_arr[ib]]+=ogr->vertices[obj_arr[ib]].sendToList[i].getNumBytes();
			    Rb[nbr]+=ogr->vertices[obj_arr[ib]].sendToList[i].getNumBytes();
			}
			else if(got_2!=Ra.end())  {
			    Ra[obj_arr[ib]]-=ogr->vertices[obj_arr[ib]].sendToList[i].getNumBytes();
			    Ra[nbr]-=ogr->vertices[obj_arr[ib]].sendToList[i].getNumBytes();
			}	
        }
    }
  
    /*gain values of Rb*/
    for(int ib=idx;ib<size;ib++)  {
        for(int i = 0; i < ogr->vertices[obj_arr[ib]].sendToList.size(); i++)  {
        int nbr = ogr->vertices[obj_arr[ib]].sendToList[i].getNeighborId();
			unordered_map<int,int>::const_iterator got = Ra.find (nbr);
			unordered_map<int,int>::const_iterator got_2 = Rb.find (nbr);
			if(got!=Ra.end())  {
		        Rb[obj_arr[ib]]+=ogr->vertices[obj_arr[ib]].sendToList[i].getNumBytes();
				Ra[nbr]+=ogr->vertices[obj_arr[ib]].sendToList[i].getNumBytes();
			}
			else if(got_2!=Rb.end())  {
			    Rb[obj_arr[ib]]-=ogr->vertices[obj_arr[ib]].sendToList[i].getNumBytes();
				Rb[nbr]-=ogr->vertices[obj_arr[ib]].sendToList[i].getNumBytes();
			}	
  	    }
    }
}

void GraphKLLB::RefineGraphKLLB(int* obj_arr, unordered_map<int,int>& Ra, unordered_map<int,int>& Rb,
  int start, int idx, int size, float part_load, float total_load, int new_part_sz, int part_size, int* gain_val)
{
    if(size<=1 || part_size==1)
        return;
  
    double fraction_load = total_load * (new_part_sz) / double(part_size);
    double threshold_load=total_load/THRESHOLD;
  
    /*Setting the gain values*/
    //RefineGraphKLLB_setgains(obj_arr, Ra, Rb, idx, size);
  
    /*While load is unbalanced*/    
    while((part_load > (fraction_load + threshold_load)) ||  
      ((total_load - part_load) > (total_load - fraction_load + threshold_load))) {  
        int max_gain=0;
        bool notSet = true;
        int Ra_node=-1,Rb_node=-1,Ra_ib=-1;
        int strt=0,end=idx;
        if(start!=-1) {
            strt=idx;
            end=size;
        }
        
        for(int ib = strt; ib < end; ib++)  {
            unordered_map<int,int>::const_iterator exists = Ra.find(obj_arr[ib]);
  	        if(exists != Ra.end())  {
    	        for(int i = 0; i < ogr->vertices[obj_arr[ib]].sendToList.size(); i++) {
                    int nbr = ogr->vertices[obj_arr[ib]].sendToList[i].getNeighborId();
			        unordered_map<int,int>::const_iterator got = Rb.find (nbr);
			        if(got != Rb.end())  {
			        /*updated load if at all the nodes are swapped*/
			            int updated_load = part_load-objs[obj_arr[ib]].getVertexLoad()+objs[nbr].getVertexLoad();
				        if(updated_load >= fraction_load && updated_load < part_load) {
					        int edge_wt=ogr->vertices[obj_arr[ib]].sendToList[i].getNumBytes();
					        int gain = -gain_val[obj_arr[ib]] - gain_val[nbr] - 2*edge_wt;
					        if(notSet || gain > max_gain)  {
                                notSet = false;
						        Ra_node=obj_arr[ib];
						        Rb_node=nbr;
						        Ra_ib=ib;
					        }
				        }
			        }		
                }
                for(int i = 0; i < ogr->vertices[obj_arr[ib]].recvFromList.size(); i++) 
    	        {
        	        int nbr = ogr->vertices[obj_arr[ib]].recvFromList[i].getNeighborId();
			        unordered_map<int,int>::const_iterator got = Rb.find (nbr);
			        if(got!=Rb.end())  {
			            /* updated load if at all the nodes are swapped*/
				        int updated_load=part_load-objs[obj_arr[ib]].getVertexLoad()+objs[nbr].getVertexLoad();
				        if(updated_load >= fraction_load && updated_load < part_load) {
					        int edge_wt=ogr->vertices[obj_arr[ib]].recvFromList[i].getNumBytes();
					        int gain = -gain_val[obj_arr[ib]] - gain_val[nbr] - 2*edge_wt;
					        if(notSet || gain > max_gain)  {
                                notSet = false;
						        Ra_node=obj_arr[ib];
						        Rb_node=nbr;
						        Ra_ib=ib;
					        }
				        }
			        }   		
                }
            }
        }
        if(Ra_node == -1 || Rb_node == -1 || Ra_ib == -1)
            break;
  
        CkPrintf("Refinement happened : init part_load %f part load 2 %f\n ", part_load, total_load - part_load);
        /*Swap Ra_node and Rb_node as gain is max*/
        part_load = part_load-objs[Ra_node].getVertexLoad() + objs[Rb_node].getVertexLoad();
	    
        CkPrintf("Refinement happened : final part_load %f part load 2 %f\n ", part_load, total_load - part_load);
        /*Update the neighbors gain values*/
	    Update_NeighborsGain(Ra_node, Rb_node, Ra, Rb, gain_val);
	    /*So that they are not considered again later	delte from map*/
	    Ra.erase(Ra_node);
	    Rb.erase(Rb_node);  					
	    /*Update the obj_arr*/
	    int j=0;
	    for(j=0;j<size;j++)  {
		    if(obj_arr[j]==Rb_node)
			    break;
	    }
    	obj_arr[j]=obj_arr[Ra_ib];
	    obj_arr[Ra_ib] = Rb_node;
    }
}

void GraphKLLB::setup_RefineGraphKLLB(int* obj_arr,unordered_map<int,int>& Ra,
  unordered_map<int,int>& Rb,int idx,int size,float part_load,float total_load,int part_size, int* gain_val)
{
    int new_part_sz=part_size/2;
    double fraction_load = total_load * (new_part_sz) / double(part_size);
    double threshold_load=total_load/THRESHOLD;
                                          
    /*Ra is unbalanced*/
    if((part_load > (fraction_load + threshold_load)))
        RefineGraphKLLB(obj_arr, Ra, Rb, -1, idx, size, part_load, total_load, new_part_sz, part_size, gain_val);
  
    /*Rb is unbalanced*/
    else if((total_load - part_load) > (total_load - fraction_load + threshold_load))
        RefineGraphKLLB(obj_arr, Rb, Ra, 1, idx, size, total_load-part_load, total_load,
        (part_size-new_part_sz), part_size, gain_val);
}

void GraphKLLB::Update_NeighborsGain(int source,int dest,unordered_map<int,int>& Ra,
  unordered_map<int,int>& Rb, int* gain_val)
{
	/*Updating the neighbors the swapping vertices of refine KLLB*/
	int nbr;
    for(int i = 0; i < ogr->vertices[source].sendToList.size(); i++) {
        nbr = ogr->vertices[source].sendToList[i].getNeighborId();
        unordered_map<int,int>::const_iterator got = Rb.find (nbr);
        unordered_map<int,int>::const_iterator got_2 = Ra.find (nbr);
		if(got!=Rb.end()) 
			gain_val[nbr] += 2*(ogr->vertices[source].sendToList[i].getNumBytes());	
		else if(got_2 !=Ra.end())
			gain_val[nbr] -= 2*(ogr->vertices[source].sendToList[i].getNumBytes());	
    }
  	
    for(int i = 0; i < ogr->vertices[source].recvFromList.size(); i++) {
        nbr = ogr->vertices[source].recvFromList[i].getNeighborId();
        unordered_map<int,int>::const_iterator got = Rb.find (nbr);
        unordered_map<int,int>::const_iterator got_2 = Ra.find (nbr);
        if(got!=Rb.end())
			gain_val[nbr] += 2*(ogr->vertices[source].sendToList[i].getNumBytes());	
		else if(got_2 !=Ra.end())
			gain_val[nbr] -= 2*(ogr->vertices[source].sendToList[i].getNumBytes());	
    }
  	
    for(int i = 0; i < ogr->vertices[dest].sendToList.size(); i++) {
        nbr = ogr->vertices[dest].sendToList[i].getNeighborId();
        unordered_map<int,int>::const_iterator got = Ra.find (nbr);
        unordered_map<int,int>::const_iterator got_2 = Rb.find (nbr);
		if(got!=Ra.end())
            gain_val[nbr] += 2*(ogr->vertices[source].sendToList[i].getNumBytes());	
        else if(got_2 !=Rb.end())
            gain_val[nbr] -= 2*(ogr->vertices[source].sendToList[i].getNumBytes());	
    }
   	
    for(int i = 0; i < ogr->vertices[dest].recvFromList.size(); i++) {
        nbr = ogr->vertices[source].recvFromList[i].getNeighborId();
        unordered_map<int,int>::const_iterator got = Rb.find (nbr);
        unordered_map<int,int>::const_iterator got_2 = Rb.find (nbr);
		if(got!=Ra.end())
            gain_val[nbr] += 2*(ogr->vertices[source].sendToList[i].getNumBytes());	
		else if(got_2 !=Rb.end())
            gain_val[nbr] -= 2*(ogr->vertices[source].sendToList[i].getNumBytes());	
    }	
}

void GraphKLLB::UpdateVertexNeighbor(int v_idx, int* gain_val,int* gain_update,
  unordered_map<int,int>& Rb) {
    int nbr;
    for(int i = 0; i < ogr->vertices[v_idx].sendToList.size(); i++) {
        nbr = ogr->vertices[v_idx].sendToList[i].getNeighborId();
        /* Update the gains of the vertices*/
        unordered_map<int,int>::const_iterator got = Rb.find (nbr);
        if(got != Rb.end())  {
            gain_val[nbr] -= 2*(ogr->vertices[v_idx].sendToList[i].getNumBytes());
            heap_update(nbr, obj_heap, ObjCompareOperator(&objs, gain_val), heap_pos);
        }
        /* No need to update heap because the objects in the other partition are 
        already removed from heap*/
        else {
            gain_val[nbr] += 2*(ogr->vertices[v_idx].sendToList[i].getNumBytes());
        }
        gain_update[nbr] = gain_update[nbr] + ogr->vertices[v_idx].sendToList[i].getNumBytes();  
    }

    for(int i = 0; i < ogr->vertices[v_idx].recvFromList.size(); i++) {
        nbr = ogr->vertices[v_idx].recvFromList[i].getNeighborId();
        unordered_map<int,int>::const_iterator got = Rb.find (nbr);
        
        if(got != Rb.end())  {
            gain_val[nbr] -= 2*(ogr->vertices[v_idx].sendToList[i].getNumBytes());
            heap_update(nbr, obj_heap, ObjCompareOperator(&objs, gain_val), heap_pos);
        }
        else {
            gain_val[nbr] += 2*(ogr->vertices[v_idx].sendToList[i].getNumBytes());
        }
        gain_update[nbr] = gain_update[nbr] + ogr->vertices[v_idx].sendToList[i].getNumBytes();
    }
    gain_val[v_idx]= -gain_val[v_idx];
}

void GraphKLLB::Gain_Update(int* gain_update,int* gain_val)   
{
    for(int i=0;i<n_objs;i++)  {
	    if(gain_update[i] != -1)  { 
	        gain_val[i]+=gain_update[i];
	    }
    }
}

void GraphKLLB::RecursiveBisection(LDStats* stats, int* obj_arr, int size, int start,
    double total_load, int* gain_val, int* partition, int part_size)
{
    DEBUG(("Size is: %d  start is: %d total_load is: %f part_size is: %d \n",size,start, total_load,part_size));

    if(size<=1 || part_size == 1)
        return;
        
    obj_heap.clear();
    heap_pos.clear();

    /*Create heap for nodes in the current partition*/
    obj_heap.resize(size);
    heap_pos.resize(n_objs); 
      
    /*Ra and Rb are the two partitions generated by this function*/
    unordered_map<int,int> Ra,Rb;  
    int gain_update[n_objs]; 
    /* TODO: memset(gain_update, -1, n_objs);*/
    for(int i=0;i<n_objs;i++)
        gain_update[i]=-1;
     
    /*Initially everyone is in the same partition assume it to be Rb*/
    float load_b=0.0;
    for(int i=0;i<size;i++)
    {
        load_b+=objs[obj_arr[i]].getVertexLoad();
        Rb[obj_arr[i]]=0;
    }
    
    /* Create a heap of objs based on their gain function which is the sum of edge
    weight with vertices in the current partition - sum of edge weights with
    the vertices in the other partition*/
    InitializeObjHeap(stats, obj_arr,size, gain_val);    

    int new_part_sz = part_size/2;

    int idx = 0;
    double part_load = 0.0;
    double fraction_load = total_load * (new_part_sz) / double(part_size);
   
    // Fraction of total load which is acceptable error. 
    double threshold_load=total_load /THRESHOLD ; 
    int v_id;

    /* while !balanced*/
    while (part_load < fraction_load ) {
        
        /* Pop vertex v from the heap and assign it to partition A, increment idx*/
        v_id = heap_pop(obj_heap, ObjCompareOperator(&objs, gain_val), heap_pos);
        
        /*If the heap becomes empty*/
        if(v_id==-1)          
            break;
        
        /* This object need not be considered again, hence not adding back to the heap.*/
        if(part_load + objs[v_id].getVertexLoad() >= (fraction_load + threshold_load))
        {
            DEBUG(("Couldnt shift %d object of weight %f as it crossed threshold \n",v_id,objs[v_id].getVertexLoad()));
            continue;
        }
        
        /*Changing the partition of the minimum gain node*/
        unordered_map<int,int>::iterator itr=Rb.find(v_id);
        if(itr!=Rb.end())
        {
            Rb.erase(itr);
            Ra[v_id]=0;
        }
        
        /*Updating obj_arr*/
        int curr_vert=obj_arr[idx];
        int j=0;
        for(j=idx;j<size;j++)				
        {
            if(obj_arr[j]==v_id)
                break;
        }
        if(j==size) {
            DEBUG(("ERROR heap pop not working as element is considered again"));
        }
        obj_arr[j]=curr_vert;
        obj_arr[idx] = v_id;
        idx++;
        DEBUG(("Shifting object %d with id: %d of load : %f with: %d  with id %d size became: %d \n ",v_id,j,objs[v_id].getVertexLoad(),curr_vert,idx-1,obj_heap.size()));
    
        /* Update the partitions weight*/
        part_load += objs[v_id].getVertexLoad();
        load_b=load_b-objs[v_id].getVertexLoad();

        /* Update the gain and also heap for v's neighbors*/
        UpdateVertexNeighbor(v_id, gain_val,gain_update,Rb);  
    }
  
    /* partition array corresponds to the splits and will contain the index in the
    obj_arr. The size of the parition array is n_pes.*/
    partition[new_part_sz] = start + idx;		
    
    /*Refinement:*/
    setup_RefineGraphKLLB(obj_arr,Ra,Rb,idx,size,part_load,total_load,part_size, gain_val);
    
    /* Call recursiveBisection on each of the partition */
    RecursiveBisection(stats, obj_arr, idx, start, part_load, gain_val, partition, new_part_sz);
    RecursiveBisection(stats, (obj_arr+idx), (size - idx), start+idx, (total_load -
      part_load), gain_val, (partition + new_part_sz), (part_size - new_part_sz));
}


void GraphKLLB::work(LDStats* stats) {
    n_pes = stats->nprocs();
    n_objs = stats->n_objs;
    int* obj_arr = new int[n_objs];
    int* partition = new int[n_pes];
      
    //memset(partition, -1, n_pes);
    for(int i = 0; i < n_pes; i++)
        partition[i]=-1;
    
    int* gain_val = new int[n_objs];
    ogr = new ObjGraph(stats);		// Object Graph 
    InitializeObjs(stats, obj_arr, gain_val);
    RecursiveBisection(stats, obj_arr, n_objs, 0, total_vload, gain_val, partition,n_pes);
    int part_pe = 1; 
    int part_idx = partition[part_pe];
        
    for (int i = 0; i < n_objs; i++) {
        stats->to_proc[obj_arr[i]] = part_pe-1;	
        if (i+1 == part_idx) {
            part_pe++;
            if(part_pe<n_pes)
                part_idx = partition[part_pe];
            else
                part_idx=n_objs+1; 
        }
    }

    /* Print stats.
    if (_lb_args.debug() > 0) {
      double weight_procs[n_pes];
      double comm_procs[n_pes];
      for(int i=0; i<n_pes; i++) {
        comm_procs[i] = 0.0;
        weight_procs[i]=0.0;
      }
      memset(weight_procs, 0.0, n_pes);
      double max_load = 0;
      double avg_load = 0;
      for(int i=0;i<n_objs;i++)  {
        int proc=stats->to_proc[i];
        weight_procs[proc]+=objs[i].getVertexLoad();
        avg_load+=objs[i].getVertexLoad();
        if(max_load< weight_procs[proc])
          max_load=weight_procs[proc]; 
        
        for(int j = 0; j < ogr->vertices[i].sendToList.size(); j++) {
            int nbr = ogr->vertices[i].sendToList[j].getNeighborId();
            int rProc = stats->to_proc[nbr];
            if(rProc != proc) {
                comm_procs[proc] -= ogr->vertices[i].sendToList[j].getNumBytes();
                comm_procs[rProc] -= ogr->vertices[i].sendToList[j].getNumBytes();
            }
            else if(rProc == proc)
                comm_procs[proc] += ogr->vertices[i].sendToList[j].getNumBytes();
        }
        for(int j = 0; j < ogr->vertices[i].recvFromList.size(); j++) {
            int nbr = ogr->vertices[i].recvFromList[j].getNeighborId();
            int rProc = stats->to_proc[nbr];
            if(rProc != proc) {
                comm_procs[proc] -= ogr->vertices[i].recvFromList[j].getNumBytes();
                comm_procs[rProc] -= ogr->vertices[i].recvFromList[j].getNumBytes();
            }
            else if(rProc == proc)
                comm_procs[proc] += ogr->vertices[i].recvFromList[j].getNumBytes();
        }
      }
      
      double max_gain = comm_procs[0];
      double min_gain = comm_procs[0];
      for(int i = 1; i < n_pes; i++) {
        if(max_gain < comm_procs[i])
            max_gain = comm_procs[i];
        if(min_gain > comm_procs[i])
            min_gain = comm_procs[i];
      }
      if (_lb_args.debug() > 1)
      {
        CkPrintf("Weight of each processor\n");
        for(int i=0;i<n_pes;i++)
          CkPrintf("proc %d load: %f \n",i,weight_procs[i]);
      }
      if (_lb_args.debug() > 0)
      CkPrintf(" Max load: %f average load: %f max_gain %f min gain %f\n",max_load,avg_load/n_pes, max_gain, min_gain);

    } */
      
    ClearDatastructure();
    delete[] obj_arr;
    delete[] partition;
    delete[] gain_val;
    delete ogr;
}

void GraphKLLB::ClearDatastructure() {
  objs.clear();
  obj_heap.clear();
  heap_pos.clear();
}

#include "GraphKLLB.def.h"
