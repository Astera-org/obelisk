#ifndef SHMUTIL_H
#define SHMUTIL_H

#include <string>

void* create_shmem(std::string name, int size);
void* open_shmem(std::string name, int size);

#endif
