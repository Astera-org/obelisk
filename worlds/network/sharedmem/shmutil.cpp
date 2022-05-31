#include "shmutil.h"
#include <string.h>
#include <errno.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stdio.h>
#include <unistd.h>

int _create(const char* name, int size, int flag) {
    mode_t mode = S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP;
    int fd = shm_open(name, flag, mode);
    if (fd < 0) {
	printf("shm_open failed errno: %s\n", strerror(errno));
	return -1;
    }
    if (ftruncate(fd, size) != 0) {
	printf("shm_open ftruncate failed errno: %s\n", strerror(errno));
	close(fd);
	return -2;
    }
    return fd;
}

int create(const char* name, int size) {
    int flag = O_RDWR | O_CREAT;
    return _create(name, size, flag);
}

int _open(const char* name, int size) {
    int flag = O_RDWR;
    return _create(name, size, flag);
}

void* map(int fd, int size) {
    void* p = mmap(NULL, size,
		   PROT_READ | PROT_WRITE,
		   MAP_SHARED, fd, 0);
    if (p == MAP_FAILED) {
	return NULL;
    }
    return p;
}

void _close(int fd, void* p, int size) {
    if (p != NULL) {
	munmap(p, size);
    }
    if (fd != 0) {
	close(fd);
    }
}

void delete_shmem(const char* name) {
    shm_unlink(name);
}

void* create_shmem(std::string name, int size) {
    int fd = create(name.c_str(), size);
    if (fd < 0) {
	return NULL;
    }
    return map(fd, size);
}

void* open_shmem(std::string name, int size) {
    int fd = _open(name.c_str(), size);
    if (fd < 0) {
	return NULL;
    }
    return map(fd, size);
}
