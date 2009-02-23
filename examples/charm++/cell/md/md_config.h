#ifndef __MD_CONFIG_H__
#define __MD_CONFIG_H__


#define DEFAULT_NUM_PARTICLES_PER_PATCH  (128)

#define DEFAULT_NUM_PATCHES_X              (2)
#define DEFAULT_NUM_PATCHES_Y              (2)
#define DEFAULT_NUM_PATCHES_Z              (2)

#define DEFAULT_NUM_STEPS                  (2)
#define STEPS_PER_PRINT                    (4)

#define PATCH_XYZ_TO_I(x,y,z)  (((z)*numPatchesX*numPatchesY)+((y)*numPatchesX)+(x))
#define PATCH_I_TO_X(i)        ((i)%numPatchesX)
#define PATCH_I_TO_Y(i)        (((i)/numPatchesX)%numPatchesY)
#define PATCH_I_TO_Z(i)        ((i)/(numPatchesX*numPatchesY))


#define TIME_PER_STEP       (1.0e-15f)           // Unit: s
#define SIM_BOX_SIDE_LEN    (1.0e-7f)            // Unit: m (NOTE: 1 nm = 10A)
#define COULOMBS_CONSTANT   (8.987551787e-9f)    // Unit: N*(m^2)*(C^-2)
#define ELECTRON_CHARGE     (-1.602176487e-19f)  // Unit: C
#define ELECTRON_MASS       (9.109382154e-31f)   // Unit: kg


#endif //__MD_CONFIG_H__
