#ifndef __HAPI_NVTX_H_
#define __HAPI_NVTX_H_

#include "nvtx3/nvToolsExt.h"
#include <type_traits>
#include <string>

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
  std::string name;

  void init(char const* name, NVTXColor color) {
    nvtxEventAttributes_t eventAttrib;
    eventAttrib.version = NVTX_VERSION;
    eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    eventAttrib.colorType = NVTX_COLOR_ARGB;
    using et = typename std::underlying_type<NVTXColor>::type;
    eventAttrib.color = static_cast<et>(color);
    eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
    eventAttrib.message.ascii = name;
    nvtxRangePushEx(&eventAttrib);
  }

  public:
  NVTXTracer(std::string name, NVTXColor color) : name{std::move(name)} {
    init(this->name.c_str(), color);
  }

  NVTXTracer(const char* name, NVTXColor color) {
    init(name, color);
  }

  ~NVTXTracer() {
    nvtxRangePop();
  }
};

#endif // __HAPI_NVTX_H_
