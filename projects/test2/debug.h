#include <stdio.h>
#include <time.h>

/**
* Add a prefix to indicate the time of the message
*/
void printTimingMsg()
{    
    time_t now = time(NULL);
    struct tm *t = localtime(&now);
    printf("%02d:%02d:%02d | ", t->tm_hour, t->tm_min, t->tm_sec);
}
