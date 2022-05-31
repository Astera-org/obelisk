#include "shmutil.h"
#include <cstring>

int main()
{
    void *v = open_shmem("shm_name", 256);
    printf("v is: %p\n", v);
    char buff[256];
    memcpy(buff, v, 256);
    printf("read '%s' from shared mem\n", buff);
    return 0;
}
