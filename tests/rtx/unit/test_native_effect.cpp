#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

#include "rtx/concept/surface_material/native_effect_math.h"

void expectNear(float actual, float expected) {
  if (!std::isfinite(actual) || std::abs(actual - expected) > 0.000001f) {
    throw std::runtime_error(std::to_string(actual) + " != " + std::to_string(expected));
  }
}

int main() {
  try {
    expectNear(nativeEffectRadianceToLinear(0.f), 0.f);
    expectNear(nativeEffectRadianceToLinear(1.f), 1.f);
    expectNear(nativeEffectRadianceToLinear(0.5f), 0.329876978f);
    expectNear(nativeEffectRadianceToLinear(-0.5f), 0.329876978f);
    expectNear(nativeEffectRadianceToLinear(3.f), 5.799546135f);
    expectNear(nativeEffectFalloff(0.f, 0.5f, 1.f, 0.4f, 0.f), 0.4f);
    expectNear(nativeEffectFalloff(0.5f, 0.5f, 1.f, 0.4f, 0.f), 0.4f);
    expectNear(nativeEffectFalloff(0.75f, 0.5f, 1.f, 0.4f, 0.f), 0.2f);
    expectNear(nativeEffectFalloff(-0.75f, 0.5f, 1.f, 0.4f, 0.f), 0.2f);
    expectNear(nativeEffectFalloff(1.f, 0.5f, 1.f, 0.4f, 0.f), 0.f);
    expectNear(nativeEffectFalloff(2.f, 0.5f, 1.f, 0.4f, 0.f), 0.f);
    expectNear(nativeEffectFalloff(0.75f, 1.f, 0.5f, 0.f, 0.4f), 0.2f);
    expectNear(nativeEffectFalloff(0.f, 0.5f, 0.5f, 0.4f, 0.f), 1.f);
    const float a = nativeEffectFalloff(0.5f, 0.5f, 1.f, 1.f, 0.f);
    const float b = nativeEffectFalloff(1.f, 0.5f, 1.f, 1.f, 0.f);
    // Interpolate evaluated varyings, not a cosine before the nonlinear curve.
    expectNear(0.75f * a + 0.25f * b, 0.75f);
    expectNear(nativeEffectFalloff(0.625f, 0.5f, 1.f, 1.f, 0.f), 0.84375f);
    expectNear(nativeEffectSoftIntersection(0.f, 1.f, 100.f), 0.f);
    expectNear(nativeEffectSoftIntersection(-1.f, 1.f, 100.f), 0.f);
    expectNear(nativeEffectSoftIntersection(25.f, 1.f, 100.f), 0.25f);
    expectNear(nativeEffectSoftIntersection(50.f, 0.5f, 100.f), 0.25f);
    expectNear(nativeEffectSoftIntersection(100.f, 1.f, 100.f), 1.f);
    expectNear(nativeEffectSoftIntersection(10000.f, 1.f, 100.f), 1.f);
    expectNear(nativeEffectSoftIntersection(10.f, 0.f, 100.f), 0.f);
    expectNear(nativeEffectSoftIntersection(10.f, 1.f, 0.f), 1.f);
    expectNear(nativeEffectSoftIntersection(10.f, 1.f, -1.f), 1.f);
    // Translating both depths preserves the authored intersection fade.
    expectNear(nativeEffectSoftIntersection(125.f - 100.f, 1.f, 100.f),
      nativeEffectSoftIntersection(1025.f - 1000.f, 1.f, 100.f));
    std::cout << "Native effect radiance, angular and depth opacity equations passed (CPU; not GPU parity).\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return -1;
  }
}
