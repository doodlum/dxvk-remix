#pragma once

#define NATIVE_GRASS_VERTICES_INPUT 0
#define NATIVE_GRASS_INDICES_INPUT 1
#define NATIVE_GRASS_PLACEMENTS_INPUT 2
#define NATIVE_GRASS_VERTICES_OUTPUT 3
#define NATIVE_GRASS_INDICES_OUTPUT 4

struct NativeGrassArgs {
  uint32_t vertexCount;
  uint32_t indexCount;
  uint32_t instanceCount;
  uint32_t writeIndices;
  float windX;
  float windY;
  float windZ;
  float timer;
};
