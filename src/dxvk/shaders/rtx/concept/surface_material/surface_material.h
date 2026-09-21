/*
* Copyright (c) 2022-2024, NVIDIA CORPORATION. All rights reserved.
*
* Permission is hereby granted, free of charge, to any person obtaining a
* copy of this software and associated documentation files (the "Software"),
* to deal in the Software without restriction, including without limitation
* the rights to use, copy, modify, merge, publish, distribute, sublicense,
* and/or sell copies of the Software, and to permit persons to whom the
* Software is furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
* THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
* FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
* DEALINGS IN THE SOFTWARE.
*/
#pragma once
#include "rtx/utility/shader_types.h"

#include "../../utility/shared_constants.h"

static const uint8_t opaqueLobeTypeDiffuseReflection = uint8_t(0u);
static const uint8_t opaqueLobeTypeSpecularReflection = uint8_t(1u);
static const uint8_t opaqueLobeTypeOpacityTransmission = uint8_t(2u);
static const uint8_t opaqueLobeTypeDiffuseTransmission = uint8_t(3u);

static const uint8_t translucentLobeTypeSpecularReflection = uint8_t(0u);
static const uint8_t translucentLobeTypeSpecularTransmission = uint8_t(1u);

struct MemoryPolymorphicSurfaceMaterial
{
  // NV-DXVK start: Shared host/shader stride with 16-byte aligned vector loads.
  uvec4 data[SURFACE_MATERIAL_GPU_SIZE / 16];
  // NV-DXVK end

  bool isOpaque()
  {
    // Note: First two bits of data are reserved for common polymorphic type
    return (data[0].x & surfaceMaterialTypeMask) == surfaceMaterialTypeOpaque;
  }

  bool hasValidDisplacement() {
    return isOpaque() && (data[0].x & OPAQUE_SURFACE_MATERIAL_FLAG_HAS_DISPLACEMENT);
  }
};

struct OpaqueSurfaceMaterial
{
  // 0 - 3
  // bitmask of OPAQUE_SURFACE_MATERIAL_FLAG_* bits
  uint16_t flags;
  uint16_t samplerIndex;
  uint16_t albedoOpacityTextureIndex;
  uint16_t secondaryTextureIndex;

  // 4-7
  f16vec4 albedoOpacityConstant;

  // 8-11
  float16_t displaceIn;
  float16_t displaceOut;
  uint16_t heightTextureIndex;
  // note: thinFilmThicknessConstant should be between 0 and 1 
  float16_t thinFilmThicknessConstant;

  // For performance, we want to keep fields used in the visibility check in the first 32 bytes.
  // The fields below here are overridden to constant values in that code, so should be left at the end
  // If we add a new field that is used for visibility, it should go above this.
  // If it isn't used for visibility, it should go below and be overridden in opaqueSurfaceMaterialCreate().

  // 12-15
  uint16_t emissiveColorTextureIndex;
  uint16_t roughnessTextureIndex;
  uint16_t metallicTextureIndex;
  uint16_t normalTextureIndex;

  // 16-19
  f16vec3 emissiveColorConstant;
  float16_t emissiveIntensity;

  // 20-23
  float16_t roughnessConstant;
  float16_t metallicConstant;
  float16_t anisotropy;
  uint16_t tangentTextureIndex;

  // 24-25
  uint subsurfaceMaterialIndex;

  // 26
  uint16_t samplerFeedbackStamp;

  // 27-36: the five extra albedo then five extra normal layers a host's terrain
  // blends between. Layer zero is the material's own albedo and normal, so only
  // the additional five of each are carried here.
  uint16_t nativeLandscapeTextureIndices[10];

  // 37: bit per layer saying it was sampled in gamma space, with the top bit
  // set to mark the material as terrain.
  uint16_t nativeLandscapeFlags;

  // Todo: Fixed function blend state info here in the future (Actually this should go on a Legacy Material, or some sort of non-PBR Legacy Surface)

  // padding (to keep size matching with MemoryPolymorphicSurfaceMaterial).
  // This has to span the whole 112 bytes: a host's effect recovers its block
  // from the tail with reinterpret<NativeEffectStorage>, and a struct that
  // stops short leaves every effect field, including the UV scale, reading zero.
  // NV-DXVK start: Pad to the shared material record stride.
  uint16_t data[(SURFACE_MATERIAL_GPU_SIZE - 76) / 2];
  // NV-DXVK end

  bool hasValidDisplacement() {
    return flags & OPAQUE_SURFACE_MATERIAL_FLAG_HAS_DISPLACEMENT;
  }
};

// A host's animated effect, recovered from the tail of an opaque material by
// reinterpret<>. It occupies the same bytes as the terrain layer indices; the
// two are mutually exclusive and OPAQUE_SURFACE_MATERIAL_FLAG_NATIVE_EFFECT
// says which is present. The prefix is the 64 bytes it does not claim.
struct NativeEffectStorage {
  uint4 prefix[4];
  float2 uvOffset;
  float2 uvScale;
  float4 falloff;
  float softDepth;
  float propertyAlpha;
  float16_t lightingInfluence;
  uint16_t paletteIndex;
  uint16_t paletteSampler;
  uint16_t flags;
};

// NV-DXVK start: Mutually exclusive with effect/landscape material storage.
struct NativeFoliageStorage
{
  uint4 prefix[4];
  uint16_t softLightTexture;
  uint16_t backLightTexture;
  uint flags;
  float rolloff;
  float brightness;
  float complexThreshold;
  float colorGamma;
  float diffuseScale;
  float scatteringAmount;
  uint4 padding;
};
// NV-DXVK end

struct TranslucentSurfaceMaterial
{
  // bitmask of TRANSLUCENT_SURFACE_MATERIAL_FLAG_* bits
  uint16_t flags;
  float16_t baseReflectivity;
  f16vec3 transmittanceColor;
  uint16_t samplerIndex;
  uint16_t transmittanceOrDiffuseTextureIndex;
  // encodes either the thin-walled thickness or the transmittance measurement distance
  // thin-walled thickness is represented as a negative number
  float16_t thicknessOrMeasurementDistance;
  uint16_t normalTextureIndex;
  uint16_t emissiveColorTextureIndex;
  float16_t emissiveIntensity;
  float16_t refractiveIndex;

  // 12-13: Source surface material index (21 bits).  Placed here (byte offset 24)
  // so the uint32_t is naturally 4-byte aligned — avoids Slang inserting padding
  // that would break reinterpret<> from the 64-byte MemoryPolymorphicSurfaceMaterial.
  uint32_t sourceSurfaceMaterialIndex;

  // 14-16
  f16vec3 emissiveColorConstant;

  // 17-53: a host's water. The three scrolling normal layers are world space,
  // not tangent space, and the flow fields are sampled through the projected
  // UV basis in nativeWaterU/V. Meaningful only under
  // TRANSLUCENT_SURFACE_MATERIAL_FLAG_NATIVE_WATER.
  uint16_t nativeWaterNormal2;
  uint16_t nativeWaterNormal3;
  f16vec3 nativeWaterScale;
  f16vec3 nativeWaterAmplitude;
  f16vec2 nativeWaterScroll1;
  f16vec2 nativeWaterScroll2;
  f16vec2 nativeWaterScroll3;
  uint16_t nativeWaterWading;
  // Byte 64 onwards, so these land 4-byte aligned.
  float3 nativeWaterU;
  float3 nativeWaterV;
  f16vec4 nativeWaterCell;
  float16_t nativeWaterDimension; // negative means native BLEND_NORMALS
  uint16_t nativeWaterFlowAtlas;
  uint16_t nativeWaterFlowNormal;
  uint16_t nativeWaterFlowSampler;
  float nativeWaterTime;

  // 54
  uint16_t samplerFeedbackStamp;

  // padding (to keep size matching with MemoryPolymorphicSurfaceMaterial)
  // NV-DXVK start: Pad to the shared material record stride.
  uint16_t data[(SURFACE_MATERIAL_GPU_SIZE - 110) / 2];
  // NV-DXVK end
};

struct RayPortalSurfaceMaterial
{
  uint16_t flags;
  uint16_t rayPortalIndex;

  uint16_t maskTextureIndex;
  uint16_t maskTextureIndex2;

  float16_t rotationSpeed;
  float16_t emissiveIntensity;

  uint16_t samplerIndex;
  uint16_t samplerIndex2;

  // padding (to keep size matching with MemoryPolymorphicSurfaceMaterial)
  // NV-DXVK start: Pad to the shared material record stride.
  uint16_t data[(SURFACE_MATERIAL_GPU_SIZE - 16) / 2];
  // NV-DXVK end

};

struct SubsurfaceMaterial
{
  uint16_t flags;

  uint16_t subsurfaceTransmittanceTextureIndex;
  uint16_t subsurfaceThicknessTextureIndex;
  uint16_t subsurfaceSingleScatteringAlbedoTextureIndex;

  float16_t volumetricAnisotropy;
  f16vec3 volumetricAttenuationCoefficient;
  float16_t measurementDistance;
  f16vec3 singleScatteringAlbedo;

  float16_t maxSampleRadius;
  
  // padding (to keep size matching with MemoryPolymorphicSurfaceMaterial)
  // NV-DXVK start: Pad to the shared material record stride.
  uint16_t data[(SURFACE_MATERIAL_GPU_SIZE - 26) / 2];
  // NV-DXVK end
};

struct SubsurfaceMaterialInteraction
{
  uint32_t packedTransmittanceColor;
  float16_t measurementDistance;
  uint32_t packedSingleScatteringAlbedo;
  uint8_t volumetricAnisotropy;
  uint8_t maxSampleRadius;
};

struct OpaqueSurfaceMaterialInteraction
{
  vec3 shadingNormal;
  float opacity;
  vec3 albedo;
  float normalDetail; // 1.0 - dot(shadingNormal, interpolatedNormal)
  vec3 baseReflectivity;
  // Note: These roughness values are non-perceptual roughness.
  float isotropicRoughness;
  vec2 anisotropicRoughness;
  // Note: fp16 may not be sufficient here for high radiance values, potentially change if clamping
  vec3 emissiveRadiance;
  SubsurfaceMaterialInteraction subsurfaceMaterialInteraction;
  // Note: A value of 0 in the thin film thickness indicates the thin film is disabled.
  float thinFilmThickness;
  uint8_t flags;

  // An instance-up diffuse lighting proxy, not an oriented blade surface.
};

struct DecalMaterialInteraction
{
  vec3 shadingNormal;
  vec3 albedo;
  vec3 baseReflectivity;
  vec3 emissiveRadiance;
  float opacity;
  float roughness;
  float anisotropy;
};

struct MemoryDecalMaterialInteraction
{
  uint4 packed;
  f16vec3 emissiveRadiance;
};

struct TranslucentSurfaceMaterialInteraction
{
  vec3 shadingNormal;
  float normalDetail; // 1.0 - dot(shadingNormal, interpolatedNormal)

  float baseReflectivity;
  float refractiveIndex;
  vec3 transmittanceColor;
  vec3 emissiveRadiance;

  // diffuse layer parameters, only valid if TRANSLUCENT_SURFACE_MATERIAL_FLAG_USE_ALBEDO_LAYER is set in flags
  vec3 diffuseColor;
  float diffuseOpacity;

  // Note: Source values only used for serialization purposes.
  // Note: Used as much of a translucent material is constant and typically reading from the material even
  // if it requires an indirection should be better than reading/writing more data to per-pixel buffers. Additionally the
  // lack of the original values such as the transmittance measurement distance and color make it hard to send this compactly
  // without otherwise having to upload those to the Translucent Surface Material (could be done if needed though).
  uint sourceSurfaceMaterialIndex;
  // Note: Raw (gamma encoded) emissive color packed in R5G6B5 needed for more tight packing, not ideal as this carries live state
  // across other code but this is an easy way to get the required info.
  uint16_t sourcePackedGammaEmissiveColor;

  // encodes either the thin-walled thickness or the transmittance measurement distance
  // thin-walled thickness is represented as a negative number
  float thicknessOrMeasurementDistance;

  uint8_t flags;

  bool isAnimatedWater;
};

struct RayPortalSurfaceMaterialInteraction
{
  vec4 mask;

  uint8_t rayPortalIndex;
  uint8_t isInsidePortal;
};

// Note: GBuffer-specific serialization data, not as tightly packed as it could be but done in this manner to share
// data with other passes (NRD, RTXDI, etc) and reuse that data for deserialization to not duplicate information. Some
// of these values may not be populated depending on the material and will instead be set to the desired special output
// value to indicate non-presence to subsequent passes. Additionally some of this data is assumed to be packed later
// by the gbuffer helper functions just to avoid code duplication (not ideal but we probably need a better way to do this).
struct GBufferMemoryPolymorphicSurfaceMaterialInteraction
{
  f16vec3 worldShadingNormal;
  float16_t perceptualRoughness;
  f16vec3 albedo;
  f16vec3 baseReflectivity;

  uint data0;
  uint data1;
};

struct PolymorphicSurfaceMaterialInteraction
{
  vec3 shadingNormal;
  vec3 emissiveRadiance;
  vec3 vdata0;
  vec3 vdata1;

  float fdata0;
  float fdata1;
  float fdata2;
  float fdata3;
  float fdata4;
  float fdata5;

  uint idata0;
  uint16_t idata1;

  uint32_t i32data0;
  uint32_t i32data1;

  uint8_t bdata0;
  uint8_t bdata1;
  uint8_t bdata2;

  uint8_t type;

  uint8_t isAnimatedWater;
};
