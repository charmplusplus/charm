#ifndef __MD_CONFIG_H__
#define __MD_CONFIG_H__


////////////////////////////////////////////////////////////////////////////////
// Default Simulation Parameters

#define DEFAULT_NUM_PARTICLES_PER_PATCH  (128)

#define DEFAULT_NUM_PATCHES_X              (2)
#define DEFAULT_NUM_PATCHES_Y              (2)
#define DEFAULT_NUM_PATCHES_Z              (2)

#define DEFAULT_NUM_STEPS                 (16)
#define STEPS_PER_PRINT                   (16)

#define USE_ARRAY_SECTIONS                 (1)

#define USE_PROXY_PATCHES                  (1)


////////////////////////////////////////////////////////////////////////////////
// Physics Constants

#define TIME_PER_STEP       (1.0e-15f)           // Unit: s
#define SIM_BOX_SIDE_LEN    (1.0e-7f)            // Unit: m (NOTE: 1 nm = 10A)
#define COULOMBS_CONSTANT   (8.987551787e-9f)    // Unit: N*(m^2)*(C^-2)
#define ELECTRON_CHARGE     (-1.602176487e-19f)  // Unit: C
#define ELECTRON_MASS       (9.109382154e-31f)   // Unit: kg


////////////////////////////////////////////////////////////////////////////////
// Misc. Helper Macros

#define PATCH_XYZ_TO_I(x,y,z)  (((z)*numPatchesX*numPatchesY)+((y)*numPatchesX)+(x))
#define PATCH_I_TO_X(i)        ((i)%numPatchesX)
#define PATCH_I_TO_Y(i)        (((i)/numPatchesX)%numPatchesY)
#define PATCH_I_TO_Z(i)        ((i)/(numPatchesX*numPatchesY))


////////////////////////////////////////////////////////////////////////////////
// Misc. Macros for Performance Testing

// DMK - DEBUG
#define ENABLE_USER_EVENTS               (1)
#define PROJ_USER_EVENT_PATCH_FORCECHECKIN_CALLBACK  (1120)
#define PROJ_USER_EVENT_PATCH_INTEGRATE_CALLBACK     (1121)
#define PROJ_USER_EVENT_SELFCOMPUTE_DOCALC_CALLBACK  (1130)
#define PROJ_USER_EVENT_PAIRCOMPUTE_DOCALC_CALLBACK  (1140)
#define PROJ_USER_EVENT_MACHINEPROGRESS              (1150)

// DMK - DEBUG
#define ENABLE_NETWORK_PROGRESS          (0)
#if ENABLE_NETWORK_PROGRESS != 0
  #if ENABLE_USER_EVENTS != 0
    #define NetworkProgress  \
      {  \
        double __start_time__ = CmiWallTimer();  \
        CmiMachineProgressImpl();  \
        traceUserBracketEvent(PROJ_USER_EVENT_MACHINEPROGRESS, __start_time__, CmiWallTimer());  \
      }
  #else
    #define NetworkProgress  CmiMachineProgressImpl();
  #endif
#else
  #define NetworkProgress
#endif


#endif //__MD_CONFIG_H__
