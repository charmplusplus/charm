#include "PackLib.h"

extern "C" void* new_packer()
{
  return (void*)(new Packer);
}

extern "C" void delete_packer(void* me)
{
  delete (Packer*)(me);
}

extern "C" void pack_char(void* me, char i)
{
  ((Packer*)me)->pack(i);
}

extern "C" void pack_uchar(void* me, unsigned char i)
{
  ((Packer*)me)->pack(i);
}

extern "C" void pack_int(void* me, int i)
{
  ((Packer*)me)->pack(i);
}

extern "C" void pack_uint(void* me, unsigned int i)
{
  ((Packer*)me)->pack(i);
}

extern "C" void pack_long(void* me, long i)
{
  ((Packer*)me)->pack(i);
}

extern "C" void pack_ulong(void* me, unsigned long i)
{
  ((Packer*)me)->pack(i);
}

extern "C" void pack_float(void* me, float i)
{
  ((Packer*)me)->pack(i);
}

extern "C" void pack_double(void* me, double i)
{
  ((Packer*)me)->pack(i);
}

extern "C" void pack_chars(void* me, char* i, int count)
{
  ((Packer*)me)->pack(i,count);
}

extern "C" void pack_uchars(void* me, unsigned char* i, int count)
{
  ((Packer*)me)->pack(i,count);
}

extern "C" void pack_ints(void* me, int* i, int count)
{
  ((Packer*)me)->pack(i,count);
}

extern "C" void pack_uints(void* me, unsigned int* i, int count)
{
  ((Packer*)me)->pack(i,count);
}

extern "C" void pack_longs(void* me, long* i, int count)
{
  ((Packer*)me)->pack(i,count);
}

extern "C" void pack_ulongs(void* me, unsigned long* i, int count)
{
  ((Packer*)me)->pack(i,count);
}

extern "C" void pack_floats(void* me, float* i, int count)
{
  ((Packer*)me)->pack(i,count);
}

extern "C" void pack_doubles(void* me, double* i, int count)
{
  ((Packer*)me)->pack(i,count);
}

extern "C" int pack_buffer_size(void* me)
{
  return ((Packer*)me)->buffer_size();
}

extern "C" int pack_fill_buffer(void* me, void* buffer, int bytes)
{
  return ((Packer*)me)->fill_buffer(buffer,bytes);
}
  
extern "C" void* new_unpacker(void* buffer)
{
  return (void*)(new Unpacker(buffer));
}

extern "C" void delete_unpacker(void* me)
{
  delete ((Unpacker*)me);
}

extern "C" int unpack_char(void* me, char* i)
{
  return ((Unpacker*)me)->unpack(i);
}

extern "C" int unpack_uchar(void* me, unsigned char* i)
{
  return ((Unpacker*)me)->unpack(i);
}

extern "C" int unpack_int(void* me, int* i)
{
  return ((Unpacker*)me)->unpack(i);
}

extern "C" int unpack_uint(void* me, unsigned int* i)
{
  return ((Unpacker*)me)->unpack(i);
}

extern "C" int unpack_long(void* me, long* i)
{
  return ((Unpacker*)me)->unpack(i);
}

extern "C" int unpack_ulong(void* me, unsigned long* i)
{
  return ((Unpacker*)me)->unpack(i);
}

extern "C" int unpack_float(void* me, float* i)
{
  return ((Unpacker*)me)->unpack(i);
}

extern "C" int unpack_double(void* me, double* i)
{
  return ((Unpacker*)me)->unpack(i);
}

extern "C" int unpack_chars(void* me, char* i, int count)
{
  return ((Unpacker*)me)->unpack(i,count);
}

extern "C" int unpack_uchars(void* me, unsigned char* i, int count)
{
  return ((Unpacker*)me)->unpack(i,count);
}

extern "C" int unpack_ints(void* me, int* i, int count)
{
  return ((Unpacker*)me)->unpack(i,count);
}

extern "C" int unpack_uints(void* me, unsigned int* i, int count)
{
  return ((Unpacker*)me)->unpack(i,count);
}

extern "C" int unpack_longs(void* me, long* i, int count)
{
  return ((Unpacker*)me)->unpack(i,count);
}

extern "C" int unpack_ulongs(void* me, unsigned long* i, int count)
{
  return ((Unpacker*)me)->unpack(i,count);
}

extern "C" int unpack_floats(void* me, float* i, int count)
{
  return ((Unpacker*)me)->unpack(i,count);
}

extern "C" int unpack_doubles(void* me, double* i, int count)
{
  return ((Unpacker*)me)->unpack(i,count);
}

