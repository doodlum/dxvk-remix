#pragma once

#include "rtx/utility/shader_types.h"

// NV-DXVK start: Shared native grass albedo equation for CPU and shader verification.
#ifdef __cplusplus
#include <cmath>
#define NATIVE_FOLIAGE_INLINE inline
#else
#define NATIVE_FOLIAGE_INLINE
#endif

NATIVE_FOLIAGE_INLINE float nativeFoliageSaturate(float value)
{
  return value < 0.0f ? 0.0f : (value > 1.0f ? 1.0f : value);
}

// Lighting.hlsl and RunGrass.hlsl GetSoftLightMultiplier, including both hemispheres.
NATIVE_FOLIAGE_INLINE float nativeSoftLightMultiplier(float angle, float rolloff)
{
  const float soft = nativeFoliageSaturate((rolloff + angle) / (1.0f + rolloff));
  const float direct = nativeFoliageSaturate(angle);
  return nativeFoliageSaturate(soft * soft * (3.0f - 2.0f * soft) -
    direct * direct * (3.0f - 2.0f * direct));
}

NATIVE_FOLIAGE_INLINE float nativeFoliageResponseChannel(float softColor, float backColor, float angle, float rolloff)
{
  return softColor * nativeSoftLightMultiplier(angle, rolloff) + backColor * nativeFoliageSaturate(-angle);
}

NATIVE_FOLIAGE_INLINE float nativeGrassNormalSign(bool sphereNormal, bool frontFace)
{
  return sphereNormal || frontFace ? 1.0f : -1.0f;
}

NATIVE_FOLIAGE_INLINE vec3 nativeGrassCross(vec3 a, vec3 b)
{
  return vec3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

NATIVE_FOLIAGE_INLINE float nativeGrassDot(vec3 a, vec3 b)
{
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

// RunGrass VS outputs R*N without size variation, World or normalization.
NATIVE_FOLIAGE_INLINE vec3 nativeGrassTransformNormal(vec3 row0, vec3 row1, vec3 row2, vec3 normal)
{
  return vec3(nativeGrassDot(row0, normal), nativeGrassDot(row1, normal), nativeGrassDot(row2, normal));
}

NATIVE_FOLIAGE_INLINE float nativeGrassInverseSqrt(float value)
{
#ifdef __cplusplus
  return 1.0f / std::sqrt(value);
#else
  return rsqrt(value);
#endif
}

// CS CalculateTBN uses negative world position and a common T/B scale.
// Triangle UV derivatives replace screen derivatives; their Jacobian's magnitude
// cancels under that scale, while its orientation is retained below.
NATIVE_FOLIAGE_INLINE vec3 nativeGrassMapNormal(vec3 normal, vec3 rawTangent,
  vec3 rawBitangent, vec3 viewDirection, vec3 tangentNormal)
{
  const float orientation = nativeGrassDot(nativeGrassCross(rawTangent, rawBitangent), viewDirection) >= 0.0f ? 1.0f : -1.0f;
  const vec3 tangent = nativeGrassCross(rawBitangent, normal) * orientation;
  const vec3 bitangent = nativeGrassCross(normal, rawTangent) * orientation;
  const float t2 = nativeGrassDot(tangent, tangent);
  const float b2 = nativeGrassDot(bitangent, bitangent);
  const float maxLengthSquared = t2 > b2 ? t2 : b2;
  if (!(maxLengthSquared > 1e-20f && maxLengthSquared < 1e20f)) {
    return normal;
  }
  const float scale = nativeGrassInverseSqrt(maxLengthSquared);
  const vec3 mapped = tangent * (tangentNormal.x * scale) +
    bitangent * (tangentNormal.y * scale) + normal * tangentNormal.z;
  const float lengthSquared = nativeGrassDot(mapped, mapped);
  return lengthSquared > 1e-20f && lengthSquared < 1e20f ?
    mapped * nativeGrassInverseSqrt(lengthSquared) : normal;
}

NATIVE_FOLIAGE_INLINE float nativeGrassAlbedoChannel(float textureColor, float linearVertexColor,
  float vertexAO, float colorGamma, float diffuseScale, float brightness,
  bool complexGrass, bool overrideComplex)
{
#ifdef __cplusplus
  const float diffuse = std::pow(std::abs(textureColor), colorGamma) * diffuseScale;
#else
  const float diffuse = pow(abs(textureColor), colorGamma) * diffuseScale;
#endif
  const float denominator = vertexAO > 1e-6f ? vertexAO : 1e-6f;
  const float brightnessScale = !complexGrass || overrideComplex ? brightness : 1.0f;
  return diffuse * (linearVertexColor / denominator) * brightnessScale;
}

// Converts a CS diffuse colour into the RTX linear working space.
NATIVE_FOLIAGE_INLINE float nativeFoliageLinearChannel(float color, bool gammaSpace)
{
  if (!gammaSpace) {
    return color;
  }
#ifdef __cplusplus
  return std::pow(std::abs(color), 2.2f);
#else
  return pow(abs(color), 2.2f);
#endif
}

// Lighting.hlsl converts BaseColor; its soft/back texture factors remain linear.
NATIVE_FOLIAGE_INLINE float nativeFoliageLightingChannel(float baseColor, float lightingWeight, bool gammaSpace)
{
  return nativeFoliageLinearChannel(baseColor, gammaSpace) * lightingWeight;
}

#undef NATIVE_FOLIAGE_INLINE
// NV-DXVK end
