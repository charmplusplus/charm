#include "fs_parameters.h"
#include "charm.h"

#if CMK_HAS_LUSTREFS
#include <lustre/lustreapi.h>
#include <lustre/lustre_user.h>
#include <errno.h>
#include <libgen.h>
#include <string.h>

static inline int maxInt(int a, int b) {
  return a > b ? a : b;
}

static void* alloc_lum() {
  int v1, v3;
  v1 = sizeof(struct lov_user_md_v1) +
    LOV_MAX_STRIPE_COUNT * sizeof(struct lov_user_ost_data_v1);
  v3 = sizeof(struct lov_user_md_v3) +
    LOV_MAX_STRIPE_COUNT * sizeof(struct lov_user_ost_data_v1);

  return malloc(maxInt(v1, v3));
}

size_t CkGetFileStripeSize(const char *filename) {
  struct lov_user_md *lump = NULL;
  lump = alloc_lum();

  if (lump == NULL) {
    CkAbort("[CkIO] Cannot allocate memory to extract lustre file stripe size\n");
  }

  int rc = llapi_file_get_stripe(filename, lump);

  if (rc != 0 && errno == ENOENT) {
    // If errno == ENOENT, may be trying to write a file that doesn't exist yet,
    // so try reading the properties of the path's parent instead.
    // This won't work on Windows (could implement with _splitpath_s).
    char* filenameCopy = strdup(filename);
    char* directory = dirname(filenameCopy);
    rc = llapi_file_get_stripe(directory, lump);
    free(filenameCopy);
  }

  if (rc != 0) {
    CkPrintf("[CkIO] Cannot extract lustre file stripe size for %s, using default of 4 MB\n", filename);
    return 4 * 1024 * 1024;
  }

  return lump->lmm_stripe_size;
}

#else

size_t CkGetFileStripeSize(const char *filename) {
  return 4 * 1024 * 1024;
}

#endif
