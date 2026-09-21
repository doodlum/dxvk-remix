#pragma once
#include "rtx/utility/shader_types.h"
#include "rtx/pass/common_binding_indices.h"

#define NATIVE_REFRACTION_DEPTH (COMMON_NUM_BINDINGS + 0)
#define NATIVE_REFRACTION_MASK (COMMON_NUM_BINDINGS + 1)
#define NATIVE_REFRACTION_COLOR (COMMON_NUM_BINDINGS + 2)
#define NATIVE_REFRACTION_OUTPUT (COMMON_NUM_BINDINGS + 3)
#define NATIVE_REFRACTION_SAMPLER (COMMON_NUM_BINDINGS + 4)

struct NativeRefractionArgs {
  uvec2 imageSize;
  uint debugMask;
  uint padding;
};

// NativeEffectStorage::flags; low nine bits retain their effect-shader meanings.
static const uint kNativeRefraction = 512;
static const uint kNativeRefractionFalloff = 1024;
static const uint kNativeRefractionClamp = 2048;
