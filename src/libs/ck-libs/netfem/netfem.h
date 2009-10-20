/*Charm++ Network FEM: C interface file
*/
#ifndef __CHARM_NETFEM_H
#define __CHARM_NETFEM_H

#ifdef __cplusplus
extern "C" {
#endif

/*Opaque type to represent one discrete data update*/
typedef void *NetFEM;

/*We can either point to your arrays (very cheap, but you must leave 
data allocated), write out the passed data (expensive, but permanent), 
or keep the last K copies of the data in memory (balance).

You can also switch at each step-- one reasonable strategy might be:
POINTAT every timestep, COPY_5 every hundred timesteps, and WRITE 
every 500 timesteps.  Then the latest responses are always available, 
copies of recent data are online, and the complete history is also 
available.
*/
#define NetFEM_POINTAT 1 /*Just keep the given pointer in case asked*/
#define NetFEM_WRITE 2 /*Write out the data to disk.*/
#define NetFEM_COPY 10 /*Keep the last i versions in memory*/
#define NetFEM_COPY_1  (NetFEM_COPY+1) /*Keep only the last version*/
#define NetFEM_COPY_2  (NetFEM_COPY+2) /*Keep the last 2 versions*/
#define NetFEM_COPY_5  (NetFEM_COPY+5) /*Keep the last 5 versions*/
#define NetFEM_COPY_10 (NetFEM_COPY+10) /*Keep the last 10 versions*/

/* Extract an initial offset and distance from this struct/field pair: */
#define NetFEM_Field(myStruct,myValue) offsetof(myStruct,myValue),sizeof(myStruct)

/*----------------------------------------------
All NetFEM calls must be between a Begin and End pair:*/
NetFEM NetFEM_Begin(
	int source,/*Integer ID for the source of this data (need not be sequential)*/
	int timestep,/*Integer ID for this instant (need not be sequential)*/
	int dim,/*Number of spatial dimensions (2 or 3)*/
	int flavor /*What to do with data (point at, write, or copy)*/
);
void NetFEM_End(NetFEM n); /*Publish these updates*/

/*---- Register the locations of the nodes.  (Exactly once, required)
   In 2D, node i has location (loc[2*i+0],loc[2*i+1])
   In 3D, node i has location (loc[3*i+0],loc[3*i+1],loc[3*i+2])
*/
void NetFEM_Nodes(NetFEM n,int nNodes,const double *loc,const char *name);
void NetFEM_Nodes_field(NetFEM n,int nNodes,
	int init_offset,int bytesPerNode,const void *loc,const char *name);

/*----- Register the connectivity of the elements. 
   Element i is adjacent to nodes conn[nodePerEl*i+{0,1,...,nodePerEl-1}]
*/
void NetFEM_Elements(NetFEM n,int nEl,int nodePerEl,const int *conn,const char *name);
void NetFEM_Elements_field(NetFEM n,int nEl,int nodePerEl,
	int init_offset,int bytesPerEl,int indexBase,
	const void *conn,const char *name);

/*--------------------------------------------------
Associate a spatial vector (e.g., displacement, velocity, accelleration)
with each of the previous objects (nodes or elements).
*/
void NetFEM_Vector_field(NetFEM n,const void *start,
	int init_offset,int distance,
	const char *name);

/*Simpler version of the above if your data is packed as
data[item*3+{0,1,2}].
*/
void NetFEM_Vector(NetFEM n,const double *data,const char *name);

/*--------------------------------------------------
Associate a scalar (e.g., stress, temperature, pressure, damage)
with each of the previous objects (nodes or elements).
*/
void NetFEM_Scalar_field(NetFEM n,const void *start,
	int vec_len,int init_offset,int distance,
	const char *name);

/*Simpler version of above for contiguous double-precision data*/
void NetFEM_Scalar(NetFEM n,const double *start,int doublePer,
	const char *name);

#ifdef __cplusplus
}
#endif

#endif
