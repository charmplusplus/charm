/****
**  reductionTesting3D
**
** 	Code written by Rahul Jain.
** 	If you have any questions regarding this code, please email me at jain10@illinois.edu
**
****/

This program tests the section reduction functionality for array sections in Charm++ and smp.


/*******
**
** How to run this program
**
*******/
The program takes in three parameters, in the following order:
arrayDimesionX, arrayDimensionY and vectorSize respectively.

arrayDimesionX -- x dimension of the 3D chare array.
arrayDimesionY -- y dimension of the 3D chare array.
arrayDimesionZ -- z dimension of the 3D chare array.
vectorSize -- Each element of the 3D chare array, has a vector of doubles over which reduction is performed.
			  This variable defines the size of the vector. Each vector element is initialized to its index value.
pgm  -- executable

So, for example, if you want to create a 30x20x10 chare array with a vector size of 5, you could do the following:
	%./charmrun +n4 +ppn8 ./pgm 30 20 10 5





********
???????/
???
??? What this program does ?
???
??????/
********

Here's a control flow of this program:

Input from user(arrayDimensions X Y and Z, vectorSize)
				|
				|
				V
Create an 3D chare array (Test3D) of the given dimension.
Each chare element has a vector of size vectorSize, each element of which is 
initialized to it's (the element's) index.
				|
				|
				V
Create an array of Section Proxies. 
//Random things that I decided, in order create the section proxies.
Number of array section proxies that will be created = max(arrayDimesionX, arrayDimensionY) = N.
				|
				|
				V
Each section proxy does a Multicast (calls Test3D::compute(msg)).
A message is sent to the compute entry, which essentially helps to identify which sectoion Proxy 
are we dealing with, on the receiving end. 
				|
				|
				V
The compute entry, essentially reduces the vector elements and callbacks to Main::reportSum.
