#include "rtx/pass/post_fx/native_refraction.h"
#include "rtx/pass/post_fx/native_refraction_math.h"
#include "rtx/pass/common_bindings.slangh"
#include "rtx/concept/ray/ray.slangh"

[[vk::push_constant]] NativeRefractionArgs args;
layout(binding = NATIVE_REFRACTION_DEPTH) Texture2D<float> SceneDepth;
layout(binding = NATIVE_REFRACTION_MASK, rgba16f) RWTexture2D<float4> RefractionMask;

// Evaluate native VS varyings before barycentric interpolation. In particular,
// the normal is not normalized or face-forwarded at the hit.
float4 readRefraction(Surface surface, RayInteraction hit, Ray ray, OpaqueSurfaceMaterial material)
{
  const NativeEffectStorage effect = reinterpret<NativeEffectStorage>(material);
  const bool clampNormal = (effect.flags & kNativeRefractionClamp) != 0;
  const float3 bary = uintToBarycentrics(hit.barycentricCoordinates);
  float2 viewNormal = 0;
  float distance = 0;
  float falloff = 0;
  for (uint i = 0; i < 3; ++i) {
    const uint index = surface.firstIndex + hit.primitiveIndex * 3 + i;
    const uint vertex = surface.indexBufferIndex == BINDING_INDEX_INVALID ? index :
      surface.indexStride == 4 ? BUFFER_ARRAY(indices32, surface.indexBufferIndex, index) :
      uint(BUFFER_ARRAY(indices, surface.indexBufferIndex, index));
    const uint p = (vertex * uint(surface.positionStride) + surface.positionOffset) / 4;
    const float3 position = float3(BUFFER_ARRAY(geometries, surface.positionBufferIndex, p),
      BUFFER_ARRAY(geometries, surface.positionBufferIndex, p + 1),
      BUFFER_ARRAY(geometries, surface.positionBufferIndex, p + 2));
    const float3 worldPosition = mul(surface.objectToWorld, float4(position, 1));
    const float4 viewPosition = mul(cb.camera.worldToView, float4(worldPosition, 1));
    const float clipZ = mul(cb.camera.viewToProjection, viewPosition).z;
    distance += bary[i] * nativeRefractionDistance(clipZ);
    float3 normal = float3(0, 0, 1);
    if (surface.normalBufferIndex != BINDING_INDEX_INVALID) {
      const uint n = (vertex * uint(surface.normalStride) + surface.normalOffset) / 4;
      if (surface.normalsEncoded) {
        const uint packed = floatBitsToUint(BUFFER_ARRAY(geometries, surface.normalBufferIndex, n));
        normal = unsignedOctahedralToSphereDirection(float2(
          unorm16ToF32(uint16_t(packed)), unorm16ToF32(uint16_t(packed >> 16))));
      } else {
        normal = float3(BUFFER_ARRAY(geometries, surface.normalBufferIndex, n),
          BUFFER_ARRAY(geometries, surface.normalBufferIndex, n + 1),
          BUFFER_ARRAY(geometries, surface.normalBufferIndex, n + 2));
      }
    }
    const float3 worldNormal = mul(surface.objectToWorld, float4(normal, 0));
    float2 nativeNormal = mul(cb.camera.worldToView, float4(worldNormal, 0)).xy;
    if (clampNormal) {
      nativeNormal = float2(nativeRefractionClamp(nativeNormal.x), nativeRefractionClamp(nativeNormal.y));
    }
    viewNormal += bary[i] * nativeNormal;
    // Skyrim instances have uniform scale; removing it recovers the authored
    // object-space dot product for static geometry and the skinned world dot.
    const float scale = length(mul(surface.objectToWorld, float4(1, 0, 0, 0)));
    const float angular = (effect.flags & kNativeRefractionFalloff) != 0 ?
      dot(worldNormal / max(scale, 1e-8f), normalize(ray.origin - worldPosition)) : 1.0f;
    falloff += bary[i] * angular;
  }
  const SurfaceInteraction interaction = surfaceInteractionCreate<SurfaceIgnoreTangents>(surface, hit, ray);
  const float2 uv = interaction.textureCoordinates * effect.uvScale + effect.uvOffset;
  const float4 sample = textures[nonuniformEXT(uint(material.albedoOpacityTextureIndex))].SampleGrad(
    samplers[nonuniformEXT(uint(material.samplerIndex))], uv,
    interaction.textureGradientX * effect.uvScale, interaction.textureGradientY * effect.uvScale);
  const float alpha = (effect.flags & 32) != 0 ? float(interaction.vertexColor.a) : 1.0f;
  return float4(nativeRefractionNormal(sample.x, viewNormal.x, distance, clampNormal),
    nativeRefractionNormal(sample.y, viewNormal.y, distance, clampNormal),
    alpha * effect.falloff.x * effect.propertyAlpha * falloff, 1.0f);
}

[shader("compute")]
[numthreads(16, 8, 1)]
void main(uint2 pixel : SV_DispatchThreadID)
{
  if (any(pixel >= args.imageSize)) {
    return;
  }
  const float2 uv = (float2(pixel) + 0.5f) / float2(args.imageSize);
  const Ray ray = rayCreatePrimary(cb.camera, uv, cb.screenSpacePixelSpreadHalfAngle, false);
  const uint2 depthPixel = min(uint2(uv * float2(cb.camera.resolution)), cb.camera.resolution - 1);
  const float viewZ = abs(SceneDepth.Load(int3(depthPixel, 0)));
  const float rayViewZ = abs(mul(cb.camera.worldToView, float4(ray.direction, 0)).z);
  float nearest = viewZ / max(rayViewZ, 1e-8f);
  float4 mask = float4(0.5f, 0.5f, 0, 0);
  RayDesc desc;
  desc.Origin = ray.origin;
  desc.Direction = ray.direction;
  desc.TMin = cb.camera.nearPlane / max(rayViewZ, 1e-8f);
  desc.TMax = max(nearest, desc.TMin);
  RayQuery<RAY_FLAG_FORCE_NON_OPAQUE | RAY_FLAG_CULL_BACK_FACING_TRIANGLES> query;
  query.TraceRayInline(unorderedTLAS, RAY_FLAG_NONE, 0xff, desc);
  while (query.Proceed()) {
    if (query.CandidateType() != CANDIDATE_NON_OPAQUE_TRIANGLE || query.CandidateTriangleRayT() >= nearest) {
      continue;
    }
    RayHitInfo info;
    RAY_HIT_INFO_CREATE_RAY_QUERY_CANDIDATE(info, query);
    info.hasHit = true;
    const RayInteraction hit = rayInteractionCreate(ray, info);
    const OpaqueSurfaceMaterial material = reinterpret<OpaqueSurfaceMaterial>(surfaceMaterials[hit.surfaceIndex]);
    if ((material.flags & OPAQUE_SURFACE_MATERIAL_FLAG_NATIVE_EFFECT) == 0) {
      continue;
    }
    const NativeEffectStorage effect = reinterpret<NativeEffectStorage>(material);
    if ((effect.flags & kNativeRefraction) == 0 || material.albedoOpacityTextureIndex == BINDING_INDEX_INVALID) {
      continue;
    }
    mask = readRefraction(surfaces[hit.surfaceIndex], hit, ray, material);
    nearest = hit.hitDistance;
  }
  RefractionMask[pixel] = mask;
}
