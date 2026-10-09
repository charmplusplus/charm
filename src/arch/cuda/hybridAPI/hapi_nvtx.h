#ifndef __HAPI_NVTX_H_
#define __HAPI_NVTX_H_

// NVTX ranges inside HAPI (callbacks, enqueue). Runtime-gated: hapiNvtxOn is set by the +nvtx
// command-line flag (ck-core/cknvtx.C); without it every NVTXTracer is a no-op.
#include "nvtx3/nvToolsExt.h"
#include <cstring>
#include <type_traits>

extern bool hapiNvtxOn;

enum class NVTXColor : uint32_t {
  Turquoise    = 0x1abc9c, Emerald   = 0x2ecc71,
  PeterRiver   = 0x3498db, Amethyst  = 0x9b59b6,
  WetAsphalt   = 0x34495e, SunFlower = 0xf1c40f,
  Carrot       = 0xe67e22, Alizarin  = 0xe74c3c,
  Clouds       = 0xecf0f1, Concrete  = 0x95a5a6,
  GreenSea     = 0x16a085, Nephritis = 0x27ae60,
  BelizeHole   = 0x2980b9, Wisteria  = 0x8e44ad,
  MidnightBlue = 0x2c3e50, Orange    = 0xf39c12,
  Pumpkin      = 0xd35400, Silver    = 0xbdc3c7,
  Pomegranate  = 0xc0392b, Asbestos  = 0x7f8c8d
};

class NVTXTracer {
  bool pushed = false;

  public:
  NVTXTracer(const char* name, NVTXColor color) {
    if (!hapiNvtxOn) return;
    nvtxEventAttributes_t eventAttrib;
    memset(&eventAttrib, 0, sizeof eventAttrib);
    eventAttrib.version = NVTX_VERSION;
    eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    eventAttrib.colorType = NVTX_COLOR_ARGB;
    using et = typename std::underlying_type<NVTXColor>::type;
    eventAttrib.color = 0xff000000u | static_cast<et>(color);
    eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
    eventAttrib.message.ascii = name;
    nvtxRangePushEx(&eventAttrib);
    pushed = true;
  }

  ~NVTXTracer() {
    if (pushed) nvtxRangePop();
  }
};

#endif // __HAPI_NVTX_H_
