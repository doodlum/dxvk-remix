// Skyrim native grass deformation. Immutable source mesh and placement records
// are uploaded once. Only this GPU output changes as native wind/time changes.
#include "native_grass.h"
#include "rtx/concept/surface_material/native_foliage_math.h"

layout(binding = NATIVE_GRASS_VERTICES_INPUT) StructuredBuffer<uint> sourceVertices;
layout(binding = NATIVE_GRASS_INDICES_INPUT) StructuredBuffer<uint> sourceIndices;
layout(binding = NATIVE_GRASS_PLACEMENTS_INPUT) StructuredBuffer<uint> placements;
layout(binding = NATIVE_GRASS_VERTICES_OUTPUT) RWStructuredBuffer<uint> vertices;
layout(binding = NATIVE_GRASS_INDICES_OUTPUT) RWStructuredBuffer<uint> indices;
layout(push_constant) ConstantBuffer<NativeGrassArgs> cb;

[shader("compute")]
[numthreads(128, 1, 1)]
void main(uint idx : SV_DispatchThreadID) {
  if (cb.writeIndices != 0 && idx < cb.indexCount * cb.instanceCount) {
    const uint instance = idx / cb.indexCount;
    indices[idx] = sourceIndices[idx % cb.indexCount] + instance * cb.vertexCount;
  }
  if (idx >= cb.vertexCount * cb.instanceCount) return;
  const uint instance = idx / cb.vertexCount;
  const uint src = (idx % cb.vertexCount) * 16; // Public HardcodedVertex includes seven padding words.
  const uint dst = idx * 9;
  const uint record = instance * 32; // NativeGrassRecord: 128 bytes, position columns then normal rows.
  const float3 c0 = asfloat(uint3(placements[record], placements[record+1], placements[record+2]));
  const float3 c1 = asfloat(uint3(placements[record+4], placements[record+5], placements[record+6]));
  const float3 c2 = asfloat(uint3(placements[record+8], placements[record+9], placements[record+10]));
  const float3 translation = asfloat(uint3(placements[record+12], placements[record+13], placements[record+14]));
  const float3 source = asfloat(uint3(sourceVertices[src], sourceVertices[src+1], sourceVertices[src+2]));
  const uint color = sourceVertices[src+8];
  const float weight = float(color >> 24) / 255.0;
  const float phase = asfloat(placements[record+18]); // original instance-local (x+y)*-1/128
  const float angle = 0.4 * (phase + cb.timer);
  const float s = sin(angle);
  const float c = cos(angle);
  const float wave = 0.3 * (sin(3.141592653589793 * s) + sin(6.283185307179586 * s)) + 0.2 * cos(3.141592653589793 * c);
  const float3 position = c0 * source.x + c1 * source.y + c2 * source.z + translation +
    float3(cb.windX, cb.windY, cb.windZ) * (wave * (weight * weight));
  const float3 sourceNormal = asfloat(uint3(sourceVertices[src+3], sourceVertices[src+4], sourceVertices[src+5]));
  const float3 normalRow0 = asfloat(uint3(placements[record+20], placements[record+21], placements[record+22]));
  const float3 normalRow1 = asfloat(uint3(placements[record+24], placements[record+25], placements[record+26]));
  const float3 normalRow2 = asfloat(uint3(placements[record+28], placements[record+29], placements[record+30]));
  float3 normal = nativeGrassTransformNormal(normalRow0, normalRow1, normalRow2, sourceNormal);
  if (!placements[record+19]) {
    normal = normalize(normal);
  }
  vertices[dst] = asuint(position.x);
  vertices[dst+1] = asuint(position.y);
  vertices[dst+2] = asuint(position.z);
  vertices[dst+3] = asuint(normal.x);
  vertices[dst+4] = asuint(normal.y);
  vertices[dst+5] = asuint(normal.z);
  vertices[dst+6] = sourceVertices[src+6];
  vertices[dst+7] = sourceVertices[src+7];
  const float brightness = float(placements[record+16] & 255) / 255.0;
  const uint r = uint(round(float((color >> 16) & 255) * brightness));
  const uint g = uint(round(float((color >> 8) & 255) * brightness));
  const uint b = uint(round(float(color & 255) * brightness));
  // NV-DXVK start: Alpha is wind/soft-light rolloff, not cutout opacity (texture alpha only).
  vertices[dst+8] = (color & 0xff000000) | (r << 16) | (g << 8) | b;
  // NV-DXVK end
}
