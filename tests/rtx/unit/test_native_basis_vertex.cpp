#include <array>
#include <cstring>
#include <iostream>
#include <stdexcept>

#include "../../../src/dxvk/rtx_render/rtx_native_basis_vertex.h"

namespace {
  void require(bool condition, const char* message) {
    if (!condition) {
      throw std::runtime_error(message);
    }
  }

  void testLayout(bool landscape, bool authored) {
    constexpr size_t kGuard = 16;
    constexpr uint8_t kSentinel = 0xcd;
    const auto stride = dxvk::NativeBasisVertex::stride(landscape);
    std::array<uint8_t, 2 * sizeof(dxvk::NativeBasisVertex) + 2 * kGuard> bytes;
    bytes.fill(kSentinel);
    std::array<remixapi_HardcodedVertex, 2> sources {};
    const float frames[2][6] = {{-0.5f, 0.2f, 0.7f, 0.1f, -1.f, 0.3f}, {0.8f, -0.4f, 0.1f, -0.2f, 0.5f, -0.7f}};
    for (uint32_t i = 0; i < 2; ++i) {
      auto& source = sources[i];
      source.position[0] = float(17 + i);
      source.position[1] = -47.f;
      source.position[2] = 0.25f;
      source.normal[0] = 0.13f;
      source.normal[1] = -0.27f;
      source.normal[2] = 0.94f;
      source.texcoord[0] = 0.75f;
      source.texcoord[1] = float(i) - 1.5f;
      source.color = 0xff123456u + i;
      // First six bytes are weights. Sum exceeds 255; seventh/eighth must survive too.
      source._pad0 = 0x806040ffu - i;
      source._pad1 = 0xdead3050u + i;
      dxvk::writeNativeBasisVertex(bytes.data() + kGuard + i * stride, source, authored ? frames[i] : nullptr, landscape);
    }
    for (uint32_t i = 0; i < 2; ++i) {
      dxvk::NativeBasisVertex decoded {};
      std::memcpy(&decoded, bytes.data() + kGuard + i * stride, stride);
      const auto& source = sources[i];
      require(std::memcmp(decoded.position, source.position, sizeof(source.position)) == 0, "Position changed");
      require(std::memcmp(decoded.texcoord, source.texcoord, sizeof(source.texcoord)) == 0, "UV changed");
      require(decoded.color == source.color, "Vertex color changed");
      if (authored) {
        require(std::memcmp(decoded.basis, source.normal, sizeof(source.normal)) == 0, "Authored normal changed");
        require(std::memcmp(decoded.basis + 3, frames[i], sizeof(frames[i])) == 0, "Authored T/B changed");
      } else {
        for (uint32_t component = 0; component < 9; ++component) {
          require(decoded.basis[component] == (component % 4 == 0 ? 1.f : 0.f), "MSN identity basis changed");
        }
      }
      if (landscape) {
        require(decoded.landscapeWeights[0] == source._pad0 && decoded.landscapeWeights[1] == source._pad1,
          "Packed terrain weights changed");
        uint32_t shaderWords[3];
        std::memcpy(shaderWords, bytes.data() + kGuard + i * stride + offsetof(dxvk::NativeBasisVertex, color), sizeof(shaderWords));
        require(shaderWords[1] == source._pad0 && shaderWords[2] == source._pad1, "Color-relative shader offsets changed");
      }
    }
    for (size_t i = 0; i < kGuard; ++i) {
      require(bytes[i] == kSentinel, "Leading guard overwritten");
    }
    for (size_t i = kGuard + 2 * stride; i < bytes.size(); ++i) {
      require(bytes[i] == kSentinel, "Trailing guard overwritten");
    }
  }
}

int main() {
  try {
    testLayout(false, false);
    testLayout(false, true);
    testLayout(true, true);
    std::cout << "Native basis packing: 60/68-byte strides, MSN, authored N/T/B, color and raw terrain weights passed.\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return -1;
  }
}
