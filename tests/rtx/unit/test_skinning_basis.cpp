#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>

#include "rtx/pass/gpu_skinning_binding_indices.h"
#include "rtx/pass/skinning.h"

static_assert(offsetof(SkinningArgs, modelSpaceNormals) == 256 * 64 + 64);
static_assert(sizeof(SkinningArgs) == 256 * 64 + 80);

namespace dxvk {
Logger Logger::s_instance("test_skinning_basis.log");
}

namespace {
void expectNear(float actual, float expected) {
  if (!std::isfinite(actual) || std::abs(actual - expected) > 0.0001f) {
    throw std::runtime_error("skinning result mismatch: " + std::to_string(actual) + " != " + std::to_string(expected));
  }
}

void testBasis(bool indexed, float weight) {
  SkinningArgs args{};
  const dxvk::Matrix4 rotation(dxvk::Vector4(0, 1, 0, 0), dxvk::Vector4(0, 0, 1, 0),
    dxvk::Vector4(1, 0, 0, 0), dxvk::Vector4(100, 200, 300, 1));
  args.bones[indexed ? 7 : 0] = rotation;
  args.bones[indexed ? 3 : 1] = dxvk::Matrix4();
  args.numBones = 2;
  args.useIndices = indexed;
  args.modelSpaceNormals = 1;
  args.srcPositionStride = args.dstPositionStride = 64;
  args.srcNormalStride = args.dstNormalStride = 64;
  args.srcNormalOffset = args.dstNormalOffset = 16;
  args.blendWeightStride = 4;
  args.blendIndicesStride = 4;
  std::array<float, 32> input{}, output;
  output.fill(-999.f);
  const float weights[2] = {weight, weight};
  const uint8_t indices[8] = {7, 3, 0, 0, 7, 3, 0, 0};
  for (uint32_t vertex = 0; vertex < 2; ++vertex) {
    const uint32_t base = vertex * 16;
    input[base] = 2.f + vertex;
    input[base + 1] = 3.f;
    input[base + 2] = 4.f;
    for (uint32_t axis = 0; axis < 3; ++axis) {
      input[base + 4 + axis * 3 + (axis + vertex) % 3] = 1.f;
    }
    dxvk::skinning(vertex, output.data(), output.data(), input.data(), weights, indices, input.data(), args);
    expectNear(output[base], (1 - weight) * input[base] + weight * 4 + weight * 100);
    expectNear(output[base + 1], (1 - weight) * 3 + weight * input[base] + weight * 200);
    expectNear(output[base + 2], (1 - weight) * 4 + weight * 3 + weight * 300);
    for (uint32_t axis = 0; axis < 3; ++axis) {
      const uint32_t original = (axis + vertex) % 3;
      const float norm = std::sqrt(weight * weight + (1 - weight) * (1 - weight));
      for (uint32_t component = 0; component < 3; ++component) {
        const float expected = ((component == original ? 1 - weight : 0) +
          (component == (original + 1) % 3 ? weight : 0)) / norm;
        expectNear(output[base + 4 + axis * 3 + component], expected);
      }
    }
    for (uint32_t guard : {3u, 13u, 14u, 15u}) {
      expectNear(output[base + guard], -999.f);
    }
  }
}

void testSingleNormal(bool encoded) {
  SkinningArgs args{};
  args.bones[0] = dxvk::Matrix4();
  args.numBones = 1;
  args.useOctahedralNormals = encoded;
  float position[3] = {2, 3, 4}, outputPosition[3]{};
  float normal[3] = {0, 0, 1}, output[9];
  std::fill(std::begin(output), std::end(output), -999.f);
  if (encoded) { normal[0] = dxvk::asfloat(dxvk::encodeNormal(dxvk::Vector3(0, 0, 1))); }
  dxvk::skinning(0, outputPosition, output, position, nullptr, nullptr, normal, args);
  const auto decoded = encoded ? dxvk::decodeNormal(dxvk::asuint(output[0])) : dxvk::Vector3(output[0], output[1], output[2]);
  expectNear(decoded.x, 0);
  expectNear(decoded.y, 0);
  expectNear(decoded.z, 1);
  for (uint32_t i = encoded ? 1 : 3; i < 9; ++i) { expectNear(output[i], -999.f); }
}

void testAuthoredFrame() {
  SkinningArgs args{};
  args.bones[0] = dxvk::Matrix4(dxvk::Vector4(0, 1, 0, 0), dxvk::Vector4(-1, 0, 0, 0),
    dxvk::Vector4(0, 0, 1, 0), dxvk::Vector4(20, 30, 40, 1));
  args.bones[1] = dxvk::Matrix4();
  args.numBones = 2;
  args.modelSpaceNormals = 1;
  // Packed N/T/B can be nonorthogonal and mirrored; skin all supplied vectors.
  const float frame[9] = {0.1f, 0.2f, 0.9f, -0.7f, 0.4f, 0.1f, 0.3f, 0.8f, -0.2f};
  const float position[3] = {1, 2, 3}, weights[1] = {0.3f};
  float outPosition[3]{}, output[9]{};
  dxvk::skinning(0, outPosition, output, position, weights, nullptr, frame, args);
  for (uint32_t axis = 0; axis < 3; ++axis) {
    const float x = frame[axis * 3], y = frame[axis * 3 + 1], z = frame[axis * 3 + 2];
    const float expected[3] = {0.7f * x - 0.3f * y, 0.7f * y + 0.3f * x, z};
    const float norm = std::sqrt(expected[0] * expected[0] + expected[1] * expected[1] + expected[2] * expected[2]);
    for (uint32_t component = 0; component < 3; ++component) {
      expectNear(output[axis * 3 + component], expected[component] / norm);
    }
  }
}
}

int main() {
  try {
    for (bool indexed : {false, true}) {
      for (float weight : {0.f, 0.25f, 1.f}) { testBasis(indexed, weight); }
    }
    testSingleNormal(false);
    testSingleNormal(true);
    testAuthoredFrame();
    std::cout << "Skinning basis: indexed/unindexed, blended/full/zero weight, offsets/strides, guards, RGB/octahedral passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return -1;
  }
}
