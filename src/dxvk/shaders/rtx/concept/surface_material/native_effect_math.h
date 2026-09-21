#pragma once

#ifdef __cplusplus
#include <cmath>
#define NATIVE_EFFECT_INLINE inline
#else
#define NATIVE_EFFECT_INLINE
#endif

// Native non-LL radiance uses Color::SkyrimGammaToLinear, including HDR values.
NATIVE_EFFECT_INLINE float nativeEffectRadianceToLinear(float color) {
#ifdef __cplusplus
  return std::pow(std::abs(color), 1.6f);
#else
  return pow(abs(color), 1.6f);
#endif
}

// Native Effect VS angular opacity, evaluated before attribute interpolation.
NATIVE_EFFECT_INLINE float nativeEffectFalloff(float cosine, float startAngle, float stopAngle,
                                               float startOpacity, float stopOpacity) {
  const float range = stopAngle - startAngle;
  if (range > -1e-6f && range < 1e-6f) {
    return 1.0f;
  }
  const float magnitude = cosine < 0.0f ? -cosine : cosine;
  const float ratio = (magnitude - startAngle) / range;
  const float t = ratio < 0.0f ? 0.0f : ratio > 1.0f ? 1.0f : ratio;
  return startOpacity + (stopOpacity - startOpacity) * (t * t * (3.0f - 2.0f * t));
}

// The native depth buffer comparison is in view-depth units, not ray length.
NATIVE_EFFECT_INLINE float nativeEffectSoftIntersection(float rayGap, float viewDepthScale, float softDepth) {
  const float gap = rayGap * viewDepthScale;
  if (!(gap > 0.0f)) {
    return 0.0f;
  }
  if (!(softDepth > 0.0f)) {
    return 1.0f;
  }
  const float ratio = gap / softDepth;
  return ratio < 1.0f ? ratio : 1.0f;
}

#undef NATIVE_EFFECT_INLINE
