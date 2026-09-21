#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include "rtx/pass/post_fx/native_refraction_math.h"

void expectNear(float actual, float expected) {
  if (!std::isfinite(actual) || std::abs(actual - expected) > 0.000001f) {
    throw std::runtime_error(std::to_string(actual) + " != " + std::to_string(expected));
  }
}

int main() {
  try {
    expectNear(nativeRefractionDistance(-100), 1);
    expectNear(nativeRefractionDistance(0), 1);
    expectNear(nativeRefractionDistance(150), 1);
    expectNear(nativeRefractionDistance(900), 2);
    expectNear(nativeRefractionClamp(-2), -0.1f);
    expectNear(nativeRefractionClamp(2), 0.1f);
    expectNear(nativeRefractionClamp(0.05f), 0.05f);
    expectNear(nativeRefractionNormal(0.5f, 0, 1, false), 0.5f);
    expectNear(nativeRefractionNormal(1, 0.2f, 2, false), 0.775f);
    expectNear(nativeRefractionNormal(1, 0.05f, 2, true), 0.535f);
    expectNear(nativeRefractionNormal(0, -0.05f, 2, true), 0.465f);
    expectNear(nativeRefractionCoordinate(0.5f, 1, 0, -1), 0.5f);
    expectNear(nativeRefractionCoordinate(0.5f, 1, 0.1f, -1), 0.495f);
    expectNear(nativeRefractionCoordinate(0.5f, 1, 0.1f, 1), 0.505f);
    expectNear(nativeRefractionCoordinate(0.9f, 0.5f, 0.5f, 1), 0.8945f);
    expectNear(nativeRefractionCoordinate(0.1f, 0.5f, 0.5f, -1), 0.1055f);
    expectNear(nativeRefractionCoordinate(0.15f, 0.5f, 1, 1), 0.15f);
    expectNear(nativeRefractionCoordinate(0.85f, 0.5f, 1, 1), 0.85f);
    // Negative angular falloff is preserved, not saturated like ordinary opacity.
    expectNear(nativeRefractionCoordinate(0.5f, 1, -0.1f, -1), 0.505f);
    std::cout << "Native refraction scalar equations passed (CPU; not GPU image parity).\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return -1;
  }
}
