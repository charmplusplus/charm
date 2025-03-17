#include <random>
#include "converse.h"

using Distribution = std::uniform_real_distribution<double>;
using Generator = std::minstd_rand;

CpvStaticDeclare(Generator*, _defaultStream);
CpvStaticDeclare(Distribution*, distribution);

void CrnInit(void)
{
  CpvInitialize(Generator*, _defaultStream);
  CpvInitialize(Distribution*, distribution);
  CpvAccess(distribution) = new Distribution(0.0, 1.0);
  CpvAccess(_defaultStream) = new Generator(0); // This should probably be seeded with random_device
}

void CrnSrand(unsigned int seed)
{
  CpvAccess(_defaultStream)->seed(seed);
}

int CrnRand(void)
{
  return (int)(CrnDrand()*0x80000000U);
}

int CrnRandRange(int min, int max)
{
  return std::uniform_int_distribution<int>(min, max)(*(CpvAccess(_defaultStream)));
}

double CrnDrand(void)
{
  return (*CpvAccess(distribution))(*(CpvAccess(_defaultStream)));
}

double CrnDrandRange(double min, double max)
{
  return std::uniform_real_distribution<double>(min, max)(*(CpvAccess(_defaultStream)));
}
