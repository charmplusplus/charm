#include "PackLib.h"

extern "C" void new_packer_(long* me)
{
  *me = reinterpret_cast<long>(new Packer);
}

extern "C" void delete_packer_(long* me)
{
  delete reinterpret_cast<Packer*>(*me);
}

extern "C" void pack_char_(long* me, char* i)
{
  reinterpret_cast<Packer*>(*me)->pack(*i);
}

extern "C" void pack_int_(long* me, int* i)
{
  reinterpret_cast<Packer*>(*me)->pack(*i);
}

extern "C" void pack_long_(long* me, long* i)
{
  reinterpret_cast<Packer*>(*me)->pack(*i);
}

extern "C" void pack_float_(long* me, float* i)
{
  reinterpret_cast<Packer*>(*me)->pack(*i);
}

extern "C" void pack_double_(long* me, double* i)
{
  reinterpret_cast<Packer*>(*me)->pack(*i);
}

extern "C" void pack_chars_(long* me, char* i, int* count)
{
  reinterpret_cast<Packer*>(*me)->pack(i,*count);
}

extern "C" void pack_ints_(long* me, int* i, int* count)
{
  reinterpret_cast<Packer*>(*me)->pack(i,*count);
}

extern "C" void pack_longs_(long* me, long* i, int* count)
{
  reinterpret_cast<Packer*>(*me)->pack(i,*count);
}

extern "C" void pack_floats_(long* me, float* i, int* count)
{
  reinterpret_cast<Packer*>(*me)->pack(i,*count);
}

extern "C" void pack_doubles_(long* me, double* i, int* count)
{
  reinterpret_cast<Packer*>(*me)->pack(i,*count);
}

extern "C" int pack_buffer_size(long* me)
{
  return reinterpret_cast<Packer*>(*me)->buffer_size();
}

extern "C" void pack_fill_buffer(long* me, char* buffer, int* bytes)
{
  reinterpret_cast<Packer*>(*me)->fill_buffer(buffer,*bytes);
}
  
extern "C" void new_unpacker_(long* me, char* buffer)
{
  *me = reinterpret_cast<long>(new Unpacker(static_cast<void*>(buffer)));
}

extern "C" void delete_unpacker_(long* me)
{
  delete reinterpret_cast<Unpacker*>(*me);
}

extern "C" void unpack_char_(long* me, char* i)
{
  reinterpret_cast<Unpacker*>(*me)->unpack(i);
}

extern "C" void unpack_int_(long* me, int* i)
{
  reinterpret_cast<Unpacker*>(*me)->unpack(i);
}

extern "C" void unpack_long_(long* me, long* i)
{
  reinterpret_cast<Unpacker*>(*me)->unpack(i);
}

extern "C" void unpack_float_(long* me, float* i)
{
  reinterpret_cast<Unpacker*>(*me)->unpack(i);
}

extern "C" void unpack_double_(long* me, double* i)
{
  reinterpret_cast<Unpacker*>(*me)->unpack(i);
}

extern "C" void unpack_chars_(long* me, char* i, int *count)
{
  reinterpret_cast<Unpacker*>(*me)->unpack(i,*count);
}

extern "C" void unpack_ints_(long* me, int* i, int *count)
{
  reinterpret_cast<Unpacker*>(*me)->unpack(i,*count);
}

extern "C" void unpack_longs_(long* me, long* i, int *count)
{
  reinterpret_cast<Unpacker*>(*me)->unpack(i,*count);
}

extern "C" void unpack_floats_(long* me, float* i, int *count)
{
  reinterpret_cast<Unpacker*>(*me)->unpack(i,*count);
}

extern "C" void unpack_doubles_(long* me, double* i, int *count)
{
  reinterpret_cast<Unpacker*>(*me)->unpack(i,*count);
}

