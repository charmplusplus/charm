
class sendStrips {
    void sendStrips() {
        Array<double> dummy = new Array<double>(0);
        int left = thisIndex-1;
        int right = thisIndex+1;
        int max = total-1;

        if(thisIndex > 0) {
            chunks[left]@getStripfromright(A[1, 0:myydim]);
        } else {
            chunks[max]@getStripfromright(dummy);
        }

        if(thisIndex < total-1) {
            chunks[right]@getStripfromleft(A[myxdim, 0:myydim]);
        } else {
            chunks[0]@getStripfromleft(dummy);
        }
    }
}

