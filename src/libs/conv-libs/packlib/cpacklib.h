#ifndef CPACKLIB_H
#define CPACKLIB_H


void* new_packer();
void delete_packer(void* me);
void pack_char(void* me, char i);
void pack_uchar(void* me, unsigned char i);
void pack_int(void* me, int i);
void pack_uint(void* me, unsigned int i);
void pack_long(void* me, long i);
void pack_ulong(void* me, unsigned long i);
void pack_float(void* me, float i);
void pack_double(void* me, double i);
void pack_chars(void* me, char* i, int count);
void pack_uchars(void* me, unsigned char* i, int count);
void pack_ints(void* me, int* i, int count);
void pack_uints(void* me, unsigned int* i, int count);
void pack_longs(void* me, long* i, int count);
void pack_ulongs(void* me, unsigned long* i, int count);
void pack_floats(void* me, float* i, int count);
void pack_doubles(void* me, double* i, int count);
int pack_buffer_size(void* me);
int pack_fill_buffer(void* me, void* buffer, int bytes);

void* new_unpacker(void* buffer);
void delete_unpacker(void* me);
int unpack_char(void* me, char* i);
int unpack_uchar(void* me, unsigned char* i);
int unpack_int(void* me, int* i);
int unpack_uint(void* me, unsigned int* i);
int unpack_long(void* me, long* i);
int unpack_ulong(void* me, unsigned long* i);
int unpack_float(void* me, float* i);
int unpack_double(void* me, double* i);
int unpack_chars(void* me, char* i, int count);
int unpack_uchars(void* me, unsigned char* i, int count);
int unpack_ints(void* me, int* i, int count);
int unpack_uints(void* me, unsigned int* i, int count);
int unpack_longs(void* me, long* i, int count);
int unpack_ulongs(void* me, unsigned long* i, int count);
int unpack_floats(void* me, float* i, int count);
int unpack_doubles(void* me, double* i, int count);

#endif /* CPACKLIB_H */
