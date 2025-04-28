# collectiveSim

A library for implementing collectives commonly used in machine learning tasks including allGather and allReduce.

## allGather

allGather lets you gather data distributed accross different chare array elements. The library provides 3 algorithms for doing the allGather operations, namely ring, hypercube and flooding.

### How to use

You can build the library using 
```bash
make allGather
```

After that you need to declare allGather as an extern module in your `.ci` file and include the `allGather.hh` header file in your `cc/hh` file. Then create an AllGather group object.

```C++
AllGather = CProxy_AllGather::ckNew(k, (int)allGatherType::ALL_GATHER_RING, seed);
```

Here, k refers to the number of data elements present in each chare array element(assuming the gather is happening among chare array elements, it can also be done among group elements) and the second parameter lets you choose the algorithm you want to run. The algorithms are:

```C++
enum allGatherType {
  ALL_GATHER_RING,
  ALL_GATHER_HYPERCUBE,
  ALL_GATHER_FLOODING
};
```
The third argument is a random seed which is set randomly if zero is passed otherwise all the random seeds in the library are set equal to the passed value.

You must also declare a callback to a function which the library can return to after its done and it must only take a pointer to `allGatherMsg` as its argument. To start operations, each chare array element must make a local pointer to the library chare group element on the same PE as it.

```C++
AllGather *libptr = AllGatherGroup.ckLocalBranch();
libptr->init(result, data, thisIndex, cb);
```
Here, result is a pointer to where the user wants the result of allGather operation to be stored(note that it must be of size n * k, where n is the number of `PEs` or the number of participants in allGather), data refers to per chare array element contributed data and cb refers to the callback. Both of these are expected to be RdmaBuffers allocated using `CkRdmaAlloc`. Also, we must pass `thisIndex` to `init` to place the gathered data in the order of chare array indexes. Note that you can modify this to change the order of the gather.
For example,`libptr->init(result, data,CkNumPes() - thisIndex, cb);`, will gather the data in the order opposite to the array indicex.

Once the library is done, it will send an empty message (a kick if you will) telling the user that the result is now available in the destination that the user specified earlier.

#### Notes on Implementation
Each group element is a representative of one of the participants of the allGather. We use zero copy api so we do not tranfer the data as is. We first send zero copy buffers and the one's getting them can decide whether to get some data or not. This is signifincant mostly in the `ALL_GATHER_FLOODING` algorithm

In `ALL_GATHER_RING` the data for all the groups, starting from the originating group elements, gets forwarded in a ring(each element getting the data from lower `PE` group element and passing it to higher `PE` group element).

In `ALL_GATHER_HYPERCUBE`, when `n` is a power of 2, the group elements assume a hypercube connectivity and use the standard [hypercube communication pattern](https://en.wikipedia.org/wiki/Hypercube_(communication_pattern)) .`ALL_GATHER_HYPERCUBE` switches to recursive doubling when n is not a power of 2 as described in [this paper](https://ieeexplore.ieee.org/abstract/document/342126?casa_token=vuF8Rhhm2f4AAAAA:TBigoTv8ge_lz8Bqt7wF0jWnyVrEXfPBL7cQGsWgnsXVZqEx3pFgtputZ8lvNma9pHjKAnR_pck5).

In `ALL_GATHER_FLOODING`, we make a sparse graph over the group elements to specify the connectivity. The communication is done by each group element getting data from it's neighbours, keeping it and forwarding/flooding it to all it's neighbours () in case it has not already seen it and discard it otherwise.

### Notes
- Currently only gathering equal sized data is supported.
- The number of PEs needs to be the same as `n`.
- The program still has an unresolved bug, where very rarely(2 in a 100 runs), the program reports a segfault after all the data has been correctly gathered.
