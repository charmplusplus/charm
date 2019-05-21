#include <stdio.h>
#include <string.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    if(argc != 2) {
        fprintf(stderr, "Usage:\n\tck-convcore [lcov trace file]\n");
        exit(1);
    }

    FILE *tracefile = fopen(argv[1], "r");
    if(tracefile == NULL) {
        fprintf(stderr, "Error: couldn't open tracefile %s\n", argv[1]);
        exit(1);
    }

    int inCK = 0, inCONV = 0, finishedCK = 0, finishedCONV = 0;
    int FNFck = 0, FNHck = 0, LHck = 0, LFck = 0;
    int FNFconv = 0, FNHconv = 0, LHconv = 0, LFconv = 0;
    char *lineptr = NULL;
    size_t n = 0;
    while(getline(&lineptr, &n, tracefile) != -1) {
        if(inCK) {
            if(lineptr[0] == 'F' && lineptr[1] == 'N' && lineptr[2] == 'F')
                FNFck = atoi(lineptr + 4);
            else if(lineptr[0] == 'F' && lineptr[1] == 'N' && lineptr[2] == 'H')
                FNHck = atoi(lineptr + 4);
            else if(lineptr[0] == 'L' && lineptr[1] == 'F')
                LFck = atoi(lineptr + 3);
            else if(lineptr[0] == 'L' && lineptr[1] == 'H') {
                LHck = atoi(lineptr + 3);
                inCK = 0;
                finishedCK = 1;
                if(finishedCK && finishedCONV)
                    break;
            }
        } else if(inCONV) {
            if(lineptr[0] == 'F' && lineptr[1] == 'N' && lineptr[2] == 'F')
                FNFconv = atoi(lineptr + 4);
            else if(lineptr[0] == 'F' && lineptr[1] == 'N' && lineptr[2] == 'H')
                FNHconv = atoi(lineptr + 4);
            else if(lineptr[0] == 'L' && lineptr[1] == 'F')
                LFconv = atoi(lineptr + 3);
            else if(lineptr[0] == 'L' && lineptr[1] == 'H') {
                LHconv = atoi(lineptr + 3);
                inCONV = 0;
                finishedCONV = 1;
                if(finishedCK && finishedCONV)
                    break;
            }
        } else if(lineptr[0] == 'S' && lineptr[1] == 'F') {
            if(strstr(lineptr, "tmp/ck.C") != NULL)
                inCK = 1;
            else if(strstr(lineptr, "tmp/convcore.c") != NULL)
                inCONV = 1;
	    else if(strstr(lineptr, "tmp/convcore.C") != NULL)
                inCONV = 1;

        }

        free(lineptr);
        lineptr = NULL;
        n = 0;
    }

    free(lineptr);
    lineptr = NULL;
    n = 0;

    printf("Summary ck.C in %s:\n", argv[1]);
    printf("\tlines......: %.1f%% (%d of %d lines)\n", ((double)(100 * LHck) / LFck), LHck, LFck);
    printf("\tfunctions..: %.1f%% (%d of %d functions)\n", ((double)(100 * FNHck) / FNFck), FNHck, FNFck);

    printf("Summary convcore.C in %s:\n", argv[1]);
    printf("\tlines......: %.1f%% (%d of %d lines)\n", ((double)(100 * LHconv) / LFconv), LHconv, LFconv);
    printf("\tfunctions..: %.1f%% (%d of %d functions)\n", ((double)(100 * FNHconv) / FNFconv), FNHconv, FNFconv);
}
