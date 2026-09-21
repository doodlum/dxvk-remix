#pragma once

// Skyrim Utility/ISRefraction equations. Scalar helpers are shared with CPU tests.
#ifdef __cplusplus
#define REFRACTION_INLINE inline
#else
#define REFRACTION_INLINE
#endif

REFRACTION_INLINE float nativeRefractionClamp(float value)
{
  return value < -0.1f ? -0.1f : value > 0.1f ? 0.1f : value;
}

REFRACTION_INLINE float nativeRefractionDistance(float clipZ)
{
  const float distance = 0.0013333333f * clipZ + 0.8f;
  return distance > 1.0f ? distance : 1.0f;
}

REFRACTION_INLINE float nativeRefractionNormal(float sample, float viewNormal, float distance, bool clampNormal)
{
  float normal = sample * 2.0f - 1.0f;
  if (clampNormal) {
    normal = nativeRefractionClamp(normal);
  }
  return ((normal * 0.9f + viewNormal) / distance) * 0.5f + 0.5f;
}

REFRACTION_INLINE float nativeRefractionCoordinate(float uv, float normal, float strength, float direction)
{
  const float displaced = uv + direction * 0.1f * strength * (normal - 0.5f);
  const float edge = displaced > 0.85f ? 0.85f : displaced < 0.15f ? 0.15f : displaced;
  const float adjusted = edge + (displaced - edge) * 0.78f;
  return displaced + (adjusted - displaced) * strength;
}

#undef REFRACTION_INLINE
