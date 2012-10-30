
chare_array [1D] Chunk {
    Array<double, 2> A, B;
    double myMax;
    int myxdim, myydim, total, counter;

    entry Chunk(int t, int x, int y) {
        int xdim,ydim;
        xdim = x;
        ydim = y;
        total = t; 
        myMax = 99999.999;

        myxdim = (int)(xdim/total);
        if(thisIndex == total-1) myxdim = xdim - myxdim*(total-1);    
        myydim = ydim;
        counter = 0;

        if (thisIndex != 0 && thisIndex != total-1) {
            xdim = myxdim+2;
        } else {
            xdim = myxdim+1;
        }
        A = new Array<double, 2>([xdim, myydim]);
        B = new Array<double, 2>([xdim, myydim]);
        A.fill(0);
        B.fill(0);
    }

    void sendStrips() {
        Array<double> strip = new Array<double>(myydim);

        if(thisIndex > 0) {
            for(int i=0;i<myydim;i++) strip[i] = A[1,i];
            thisProxy[thisIndex-1]@getStripFromRight(strip);
        } else {
            //Send dummy if thisIndex==0 for dagger to work
            thisProxy[total-1]@getStripFromRight(strip);
        }

        if(thisIndex < total-1) {
            for(int i=0;i<myydim;i++) strip[i] = A[myxdim,i];
            thisProxy[thisIndex+1]@getStripFromLeft(strip);
        } else {
            //Send dummy if thisIndex==total-1:For dagger to work
            thisProxy[0+0]@getStripFromLeft(strip);
        }
    }

    void doStencil() {
        double maxChange = 0.0;
        resetBoundary();

        if((thisIndex !=0)&&(thisIndex != total-1))
            for (int i=1; i<myxdim+1; i++)
                for (int j=1; j<myydim-1; j++) {
                    B[i,j] = (0.2)*(A[i,j] + A[i,j+1] + A[i,j-1] + A[i+1,j] + A[i-1,j]);
                    double change = B[i,j] - A[i,j];
                    if (change < 0) change = - change;
                    if (change > maxChange) maxChange = change;
                }

        if(thisIndex == 0)
            for (int i=1; i<myxdim; i++)
                for (int j=1; j<myydim-1; j++) {
                    B[i,j] = (0.2)*(A[i,j] + A[i,j+1] + A[i,j-1] + A[i+1,j] + A[i-1,j]);
                    double change = B[i,j] - A[i,j];
                    if (change < 0) change = - change;
                    if (change > maxChange) maxChange = change;
                }

        if(thisIndex == total-1) {
            for (int i=1; i<myxdim; i++)
                for (int j=1; j<myydim-1; j++) {
                    B[i,j] = (0.2)*(A[i,j] + A[i,j+1] + A[i,j-1] + A[i+1,j] + A[i-1,j]);
                    double change =  B[i,j] - A[i,j];
                    if (change < 0) change = - change;
                    if (change > maxChange) maxChange = change;
                }
        }

        Array<double, 2> tmp = A;
        A = B;
        B = tmp;  
    }

    void resetBoundary() {
         if (thisIndex != 0)
            if (thisIndex < (int)(total/2))
                for (int i=1; i<myxdim+1; i++)
                    A[i,0] = 1.0;

        if (thisIndex ==0) {
            for (int i=0; i<myxdim; i++)
                A[i,0] = 1.0;

            for (int i=0; 2*i<myydim; i++) 
                A[0,i] = 1.0;
        }
    }

    entry void processStripFromLeft(Array<double> s) {
        if (thisIndex !=0) {
            for (int i=0; i<myydim; i++)
                A[0,i] = s[i];
        }
    }

    entry void processStripFromRight(Array<double> s) {
        if (thisIndex != total - 1) {
            if (thisIndex != 0)
                for (int i=0; i<myydim; i++)
                    A[myxdim+1,i] = s[i];
            else
                for (int i=0; i<myydim; i++)
                    A[myxdim,i] = s[i];
        }
    }
    
    entry void jacobi() {
        for (int i=0; i<3; ++i) {
            sendStrips();
            overlap {
                when getStripFromLeft(Array<double> s) {
                    processStripFromLeft(s);
                }
                when getStripFromRight(Array<double> s) {
                    processStripFromRight(s);
                }
            }
            doStencil();
            if (CkMyPe() == 0) {
                CkPrintf("Numfin=%d, total=%d, Pes = %d\n",i,total,CkNumPes());
            }
        }
        CkExit();
    }
}
