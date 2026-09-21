#include "rtx/pass/post_fx/native_refraction.h"
#include "rtx/pass/post_fx/native_refraction_math.h"

[[vk::push_constant]] NativeRefractionArgs args;
layout(binding = NATIVE_REFRACTION_MASK) Texture2D<float4> RefractionMask;
layout(binding = NATIVE_REFRACTION_COLOR) Texture2D<float4> SceneColor;
layout(binding = NATIVE_REFRACTION_OUTPUT, rgba16f) RWTexture2D<float4> Output;
layout(binding = NATIVE_REFRACTION_SAMPLER) SamplerState LinearSampler;

[shader("compute")]
[numthreads(16, 8, 1)]
void main(uint2 pixel : SV_DispatchThreadID)
{
  if (any(pixel >= args.imageSize)) {
    return;
  }
  const float2 uv = (float2(pixel) + 0.5f) / float2(args.imageSize);
  const float4 mask = RefractionMask.Load(int3(pixel, 0));
  const float2 refracted = float2(
    nativeRefractionCoordinate(uv.x, mask.x, mask.z, -1.0f),
    nativeRefractionCoordinate(uv.y, mask.y, mask.z, 1.0f));
  const float coverage = RefractionMask.SampleLevel(LinearSampler, refracted, 0).a;
  const float4 original = SceneColor.Load(int3(pixel, 0));
  const float4 distorted = SceneColor.SampleLevel(LinearSampler, refracted, 0);
  Output[pixel] = args.debugMask ? mask : lerp(original, distorted, coverage);
}
