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
AllGather = CProxy_AllGather::ckNew(k, n, (int)allGatherType::ALL_GATHER_RING);
```

Here n refers to the size of the chare array, k refers to the number of data elements present in each chare array element and the third parameter lets you choose the algorithm you want to run. The algorithms are:

```C++
enum allGatherType {
  ALL_GATHER_RING,
  ALL_GATHER_HYPERCUBE,
  ALL_GATHER_FLOODING
};
```

##### [NOTE]: `ALL_GATHER_HYPERCUBE` switches to recursive doubling when n is not a power of 2 as described in [this paper](https://ieeexplore.ieee.org/abstract/document/342126?casa_token=vuF8Rhhm2f4AAAAA:TBigoTv8ge_lz8Bqt7wF0jWnyVrEXfPBL7cQGsWgnsXVZqEx3pFgtputZ8lvNma9pHjKAnR_pck5).

You must also declare a callback to a function which the library can return to after its done and it must only take a pointer to `allGatherMsg` as its argument. To start operations, each chare array element must make a local pointer to the library chare group element on the same PE as it.

```C++
AllGather *libptr = AllGatherGroup.ckLocalBranch();
libptr->init(result, data, thisIndex, cb);
```
Here, result is a pointer to where the user wants the result of allGather operation to be stored(note that it must be of size n * k), data refers to per chare array element contributed data and cb refers to the callback. ALso, we must pass the `thisIndex` to `init` to place the gathered data in the right order. 

Once the library is done, it will send an empty message (a kick if you will) telling the user that the result is now available in the destination that the user specified earlier.

### Notes
- Currently only gathering equal sized data is supported.
- The number of PEs needs to be the same as `n`. Please refer to the makefile for a concrete command.
- The program still has an unresolved bug, where very rarely(2 in a 100 runs), the program reports a segfault after all the data has been correctly gathered.
