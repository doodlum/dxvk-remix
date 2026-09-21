#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>

#include <remix/remix_c.h>

namespace dxvk {
  struct NativeBasisVertex {
    float position[3];
    // Identity pose basis for MSN; authored N/T/B for tangent-space maps.
    float basis[9];
    float texcoord[2];
    uint32_t color;
    // Landscape decoder reads these at color + 4/+8 bytes.
    uint32_t landscapeWeights[2];

    static constexpr uint32_t stride(bool landscape) {
      return landscape ? sizeof(NativeBasisVertex) : offsetof(NativeBasisVertex, landscapeWeights);
    }
  };

  static_assert(sizeof(NativeBasisVertex) == 68);
  static_assert(NativeBasisVertex::stride(false) == 60);
  static_assert(offsetof(NativeBasisVertex, landscapeWeights) == offsetof(NativeBasisVertex, color) + 4);

  inline void writeNativeBasisVertex(uint8_t* pDestination, const remixapi_HardcodedVertex& source,
      const float* pTangentFrame, bool landscape) {
    NativeBasisVertex vertex {};
    std::memcpy(vertex.position, source.position, sizeof(vertex.position));
    vertex.basis[0] = vertex.basis[4] = vertex.basis[8] = 1.f;
    if (pTangentFrame) {
      std::memcpy(vertex.basis, source.normal, 3 * sizeof(float));
      std::memcpy(vertex.basis + 3, pTangentFrame, 6 * sizeof(float));
    }
    std::memcpy(vertex.texcoord, source.texcoord, sizeof(vertex.texcoord));
    vertex.color = source.color;
    vertex.landscapeWeights[0] = source._pad0;
    vertex.landscapeWeights[1] = source._pad1;
    std::memcpy(pDestination, &vertex, NativeBasisVertex::stride(landscape));
  }
}
