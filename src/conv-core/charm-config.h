#ifndef CHARM_CONFIG_ALIAS_H
#define CHARM_CONFIG_ALIAS_H
/* Classic builds: charm-config.h is an alias for the conv-config chain, so
 * boundary headers shared with reconverse builds can include "charm-config.h"
 * unconditionally. Reconverse builds use reconverse's own charm-config.h
 * (installed by cmake/fetch_reconverse); this file is excluded from install
 * there and must never shadow it. */
#include "conv-config.h"
#endif
