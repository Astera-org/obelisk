#include "shmutil.h"
#include <cstring>
#include <unistd.h>

int main()
{
    void *v = create_shmem("shm_name", 256);
    char *data = "hello world";
    memcpy(v, data, strlen(data));
    printf("wrote '%s' to shared mem\n", data);

    sleep(10);
    return 0;
}
