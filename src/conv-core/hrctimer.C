#include <chrono>
#include <type_traits>
#include "hrctimer.h"
#include <ratio>

using clock_type=typename std::conditional<
  std::chrono::high_resolution_clock::is_steady,
  std::chrono::high_resolution_clock,
  std::chrono::steady_clock>::type;
static std::chrono::time_point<clock_type> epoch;

double inithrc() { //defines our HRC epoch
  epoch = clock_type::now();
  return std::chrono::duration<double>(epoch.time_since_epoch()).count();
}
double gethrctime() {
  auto timepoint = clock_type::now(); //gets HRC timepoint
  auto time_since_epoch = timepoint-epoch; //gets the elapsed time since the start of HRC clock
  double seconds = std::chrono::duration<double>(time_since_epoch).count(); //converts that time into seconds(double)
  return seconds;
}
uint64_t gethrctime_micro() {
  auto timepoint = clock_type::now(); //gets HRC timepoint
  auto time_since_epoch = timepoint-epoch; //gets the elapsed time since the start of HRC clock
  auto microseconds = std::chrono::duration_cast<std::chrono::microseconds>(time_since_epoch).count(); //converts that time into microseconds(integer)
  return microseconds;
}
