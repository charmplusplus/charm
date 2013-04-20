#include <unistd.h>

int gethugepagesize()
{
    return getpagesize();
}
