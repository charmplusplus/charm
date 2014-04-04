/** \file ckBIconfig.h
 *  Author: Eric Bohm
 *  Date created: Feb 5th, 2013
 *  
 */

#ifndef CK_BI_CONFIG
#define CK_BI_CONFIG
#ifdef CMK_BALANCED_INJECTION_API
#include <stdint.h>
#include <gni_pub.h>

static inline uint16_t ck_get_GNI_BIConfig()
{
  gni_bi_desc_t gni_bi_desc;
  uint32_t gni_device_id = 0;
  gni_return_t gni_rc = GNI_GetBIConfig(gni_device_id, &gni_bi_desc);
  if (gni_rc != GNI_RC_SUCCESS && CkMyPe() == 0) {
    CmiPrintf("Error, unable to retrieve BI config, rc=%d\n",gni_rc);
  }
  return(gni_bi_desc.current_bw);
}

static inline void ck_set_GNI_BIConfig(uint16_t biValue)
{
  uint16_t modes = GNI_BI_FLAG_APPLY_NOW;
  uint32_t gni_device_id = 0;
  uint16_t gni_rc = GNI_SetBIConfig(gni_device_id, biValue, 0, modes);
  if (gni_rc != GNI_RC_SUCCESS && CkMyPe() == 0) {
    CmiPrintf("Error, unable to set BI config, rc=%d\n",gni_rc);
  }
}

#endif
#endif
