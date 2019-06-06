
PENNANT Mini-App
Instructions and Run Rules for APEX Benchmarking

The PENNANT-specific rules given here are a supplement to the general
run rules given in the APEX RFP.


Run Instructions

PENNANT run instructions are provided in the regular documentation.
(See the doc/ directory in the PENNANT distribution.)


Problem Definitions

The RFP defines problem sizes to be used in benchmarking.
The PENNANT test problems corresponding to this definition are:

"Small" problem:   leblancbig
"Medium" problem:  leblancx4
"Large" problem:   leblancx16

Test problems are in the test/ directory in the PENNANT distribution.

In addition, the nohpoly problem will be used for verifying that the
baseline code functionality has been preserved (see below).  Note
for reference that nohpoly would also be considered a "small" problem
by the RFP definition.


Base Results

The baseline PENNANT implementation allows both MPI-only and MPI+OpenMP
build variants.  For purposes of generating the "base results" required
by the RFP, either of these variants may be used.


PENNANT-Specific Run Rules 

The purpose of the PENNANT benchmark is to demonstrate performance
on an application using general unstructured meshes, that is, meshes
which contain arbitrary polygons with arbitrary connectivity between
them.  Any code changes made by the Offeror must preserve this
capability.  That is, although the leblanc* benchmark problems use
meshes with an underlying Cartesian structure, the code may not be
modified in any way that takes advantage of this structure.
As a proof of this, any modified version must be able to correctly
run the nohpoly test problem, in which the mesh is truly unstructured.


