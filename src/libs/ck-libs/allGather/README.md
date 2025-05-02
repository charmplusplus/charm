# collectiveSim

A library for implementing collectives commonly used in machine learning tasks including allGather and allReduce.

## allGather

allGather lets you gather data distributed accross different chare array/group elements. The library provides 3 algorithms for doing the allGather operations, namely ring, hypercube and flooding.

### How to use

The library is available in a default charm++ build and to use allGather, you simply have to include the header `allGather.h` and link against the library using the flag `-lmoduleallGather`.

After that you need to declare allGather as an extern module in your `.ci` file and create an AllGather group object.

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
The third argument is a random seed which is set to pid if zero is passed otherwise the passed value is used. Note that this is used only when `flooding` algorithm is run. 

You must also declare a callback to a function which the library can return to after its done and it must only take a pointer to `allGatherMsg` as its argument. To start operations, each chare array element must make a local pointer to the library chare group element on the same PE as it.

```C++
AllGather *libptr = AllGatherGroup.ckLocalBranch();
libptr->init(result, data, thisIndex, cb);
```
The parameters for the `init` function are:

- `result`: A pointer (void *) to where the allGather operation results will be stored. This must be allocated with enough space to hold n * k elements, where n is the (number of participants in allGather).
- `data`: The per-chare array element data pointer (void *) that will be contributed to the allGather operation.
- `thisIndex`: The index value used to determine the order of the gathered data.
- `cb`: The callback that will be invoked when the allGather operation completes.

You can customize the gathering order by modifying the `thisIndex` parameter. For example:
```C++
libptr->init(result, data, n - thisIndex, cb);
```
This would gather the data in the reverse order of array indices.

Once the library is done, it will send an empty message (a kick if you will) telling the user that the result is now available in the destination that the user specified earlier.

#### Notes on Implementation
Each group element is a representative of one of the participants of the allGather. We use zero copy api so we do not tranfer the data as it is. We first send zero copy buffers and the one's getting them can decide whether to get some data or not. This is signifincant in the `ALL_GATHER_FLOODING` algorithm.

In `ALL_GATHER_RING` the data for all the groups, starting from the originating group elements, gets forwarded in a ring(each element getting the data from lower `PE` group element and passing it to higher `PE` group element).

In `ALL_GATHER_HYPERCUBE`, when `n` is a power of 2, the group elements assume a hypercube connectivity and use the standard [hypercube communication pattern](https://en.wikipedia.org/wiki/Hypercube_(communication_pattern)) .`ALL_GATHER_HYPERCUBE` switches to recursive doubling when n is not a power of 2 as described in [this paper](https://ieeexplore.ieee.org/abstract/document/342126?casa_token=vuF8Rhhm2f4AAAAA:TBigoTv8ge_lz8Bqt7wF0jWnyVrEXfPBL7cQGsWgnsXVZqEx3pFgtputZ8lvNma9pHjKAnR_pck5).

In `ALL_GATHER_FLOODING`, we make a sparse graph over the group elements to specify the connectivity. The communication is done by each group element getting data from it's neighbours, keeping it and forwarding/flooding it to all it's neighbours () in case it has not already seen it and discard it otherwise.

### Notes
- Currently only gathering equal sized data is supported.
- The number of PEs needs to be the same as `n`(the participants in all gather).
