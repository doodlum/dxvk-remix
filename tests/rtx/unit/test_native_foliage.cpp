#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

#include "rtx/concept/surface_material/native_foliage_math.h"

namespace {
  void expectNear(float actual, float expected) {
    if (!std::isfinite(actual) || std::abs(actual - expected) > 1e-6f) {
      throw std::runtime_error("Native grass equation mismatch");
    }
  }

  vec3 unit(vec3 value) {
    return value * (1.f / std::sqrt(nativeGrassDot(value, value)));
  }

  void expectVector(vec3 actual, vec3 expected) {
    expectNear(actual.x, expected.x);
    expectNear(actual.y, expected.y);
    expectNear(actual.z, expected.z);
  }

  void testGrassFrame(bool front, bool sphere, bool mirrored) {
    const vec3 edge0(2.f, 0.3f, 0.2f), edge1(-0.4f, 1.5f, -0.1f);
    const vec2 uv0(mirrored ? -0.9f : 0.9f, 0.25f), uv1(mirrored ? 0.1f : -0.1f, 0.7f);
    const float determinant = uv0.x * uv1.y - uv0.y * uv1.x;
    const vec3 rawT = (edge0 * uv1.y + edge1 * -uv0.y) * (1.f / determinant);
    const vec3 rawB = (edge1 * uv0.x + edge0 * -uv1.x) * (1.f / determinant);
    const vec3 view = unit(nativeGrassCross(edge0, edge1)) * (front ? 1.f : -1.f);
    const vec3 normal = unit(vec3(0.25f, -0.17f, 1.f)) * nativeGrassNormalSign(sphere, front);
    const vec3 mapped(0.45f, -0.3f, 0.7f);
    // Independent CS screen-derivative construction. Screen Y points down,
    // so this Jacobian reverses orientation for the front-facing triangle.
    const float a = 1.7f, b = 0.4f, c = front ? -0.3f : 0.3f, d = front ? -1.2f : 1.2f;
    const vec3 dpdx = (edge0 * a + edge1 * b) * -1.f;
    const vec3 dpdy = (edge0 * c + edge1 * d) * -1.f;
    const vec2 duvdx = uv0 * a + uv1 * b, duvdy = uv0 * c + uv1 * d;
    const vec3 perp2 = nativeGrassCross(dpdy, normal), perp1 = nativeGrassCross(normal, dpdx);
    const vec3 t = perp2 * duvdx.x + perp1 * duvdy.x;
    const vec3 bitangent = perp2 * duvdx.y + perp1 * duvdy.y;
    const float scale = 1.f / std::sqrt(std::max(nativeGrassDot(t, t), nativeGrassDot(bitangent, bitangent)));
    const vec3 reference = unit(t * (mapped.x * scale) + bitangent * (mapped.y * scale) + normal * mapped.z);
    expectVector(nativeGrassMapNormal(normal, rawT, rawB, view, mapped), reference);
    expectVector(nativeGrassMapNormal(normal, rawT * 0.017f, rawB * 0.017f, view, mapped), reference);
    expectVector(nativeGrassMapNormal(normal, rawT, rawB, view, vec3(0.f, 0.f, 1.f)), normal);
  }
}

int main() {
  try {
    // Independent native HLSL response over both hemispheres and authored rolloffs.
    for (float rolloff : {0.f, 0.1f, 1.f, 2.f, 8.f}) {
      for (int i = -100; i <= 100; ++i) {
        const float angle = i / 100.f;
        const float soft = std::clamp((rolloff + angle) / (1.f + rolloff), 0.f, 1.f);
        const float direct = std::clamp(angle, 0.f, 1.f);
        const float expected = std::clamp(soft * soft * (3.f - 2.f * soft) -
          direct * direct * (3.f - 2.f * direct), 0.f, 1.f);
        expectNear(nativeSoftLightMultiplier(angle, rolloff), expected);
        expectNear(nativeFoliageResponseChannel(0.3f, 0.7f, angle, rolloff),
          0.3f * expected + 0.7f * std::clamp(-angle, 0.f, 1.f));
      }
    }
    expectNear(nativeSoftLightMultiplier(0.f, 1.f), 0.5f);
    expectNear(nativeSoftLightMultiplier(-1.f, 1.f), 0.f);
    expectNear(nativeFoliageResponseChannel(0.f, 0.7f, -1.f, 0.f), 0.7f);
    expectNear(nativeFoliageResponseChannel(0.3f, 0.f, -1.f, 0.f), 0.f);
    // Native rows include authored tilt/scale but not position size variation.
    const vec3 row0(0.f, -2.f, 0.25f), row1(1.5f, 0.f, 0.1f), row2(0.2f, 0.3f, 0.75f);
    const vec3 n0(0.2f, 0.8f, 0.6f), n1(-0.7f, 0.1f, 0.3f);
    const vec3 t0 = nativeGrassTransformNormal(row0, row1, row2, n0);
    const vec3 t1 = nativeGrassTransformNormal(row0, row1, row2, n1);
    expectVector(t0, vec3(-1.45f, 0.36f, 0.73f));
    expectVector(t1, vec3(-0.125f, -1.02f, 0.115f));
    // Interpolate raw VS outputs, then normalize once in the pixel/hit shader.
    expectVector(unit(t0 * 0.3f + t1 * 0.7f),
      unit(nativeGrassTransformNormal(row0, row1, row2, n0 * 0.3f + n1 * 0.7f)));
    const vec3 premature = unit(unit(t0) * 0.3f + unit(t1) * 0.7f);
    const vec3 correct = unit(t0 * 0.3f + t1 * 0.7f);
    if (nativeGrassDot(premature, correct) > 0.999f) {
      throw std::runtime_error("Normal fixture does not detect premature normalization");
    }
    expectVector(nativeGrassTransformNormal(row0, row1, row2, vec3(0.f, 0.f, 0.f)), vec3(0.f, 0.f, 0.f));
    expectNear(nativeGrassNormalSign(false, true), 1.f);
    expectNear(nativeGrassNormalSign(false, false), -1.f);
    expectNear(nativeGrassNormalSign(true, true), 1.f);
    expectNear(nativeGrassNormalSign(true, false), 1.f);
    for (bool front : {false, true}) {
      for (bool sphere : {false, true}) {
        for (bool mirrored : {false, true}) {
          testGrassFrame(front, sphere, mirrored);
        }
      }
    }
    const vec3 up(0.f, 0.f, 1.f), zero(0.f, 0.f, 0.f);
    expectVector(nativeGrassMapNormal(up, zero, zero, up, up), up);
    expectVector(nativeGrassMapNormal(up, vec3(1.f, 0.f, 0.f), vec3(0.f, 1.f, 0.f), up, zero), up);
    const float invalid = std::numeric_limits<float>::quiet_NaN();
    expectVector(nativeGrassMapNormal(up, vec3(invalid, invalid, invalid), zero, up, up), up);
    // RunGrass Color::Diffuse and normalized ColorToLinear(vertex), LL disabled.
    expectNear(nativeGrassAlbedoChannel(0.5f, 0.1f, 0.2f, 1.f, 1.f, 1.f, false, false), 0.25f);
    // Uniform instance/baked AO cannot change the normalized albedo hue.
    expectNear(nativeGrassAlbedoChannel(0.5f, 0.25f, 0.5f, 1.f, 1.f, 1.f, false, false), 0.25f);
    expectNear(nativeGrassAlbedoChannel(0.5f, 0.5f, 1.f, 1.f, 1.f, 1.f, false, false), 0.25f);
    // Active LL gamma and diffuse multiplier apply once, not generic gamma again.
    expectNear(nativeGrassAlbedoChannel(0.5f, 0.0625f, 0.25f, 2.f, 2.f, 1.f, false, false), 0.125f);
    expectNear(nativeGrassAlbedoChannel(0.5f, 1.f, 1.f, 1.f, 1.f, 2.f, false, false), 1.f);
    // Complex atlas opts out of BasicGrassBrightness unless explicitly overridden.
    expectNear(nativeGrassAlbedoChannel(0.5f, 1.f, 1.f, 1.f, 1.f, 2.f, true, false), 0.5f);
    expectNear(nativeGrassAlbedoChannel(0.5f, 1.f, 1.f, 1.f, 1.f, 2.f, true, true), 1.f);
    expectNear(nativeGrassAlbedoChannel(0.5f, 0.f, 0.f, 1.f, 1.f, 1.f, false, false), 0.f);
    expectNear(nativeGrassAlbedoChannel(0.5f, 1e-8f, 1e-8f, 1.f, 1.f, 1.f, false, false), 0.005f);
    // DisableTerrainVertexColors supplies white before this shared equation.
    expectNear(nativeGrassAlbedoChannel(0.5f, 1.f, 1.f, 2.f, 1.f, 1.f, false, false), 0.25f);
    // CS LL-off material colours need one decode at the RTX linear boundary.
    expectNear(nativeFoliageLinearChannel(0.5f, true), std::pow(0.5f, 2.2f));
    expectNear(nativeFoliageLinearChannel(0.5f, false), 0.5f);
    expectNear(nativeFoliageLinearChannel(0.f, true), 0.f);
    expectNear(nativeFoliageLinearChannel(1.f, true), 1.f);
    // Lighting.hlsl samples soft/back maps without ColorToLinear, including LL-on.
    for (float weight : {0.f, 0.1f, 0.2f, 0.5f, 1.f}) {
      const float baseLinear = std::pow(0.3f, 2.2f);
      expectNear(nativeFoliageLightingChannel(0.3f, weight, true), baseLinear * weight);
      expectNear(nativeFoliageLightingChannel(baseLinear, weight, false), baseLinear * weight);
      expectNear(nativeFoliageLightingChannel(1.f, weight, true), weight);
      expectNear(nativeFoliageLightingChannel(0.f, weight, true), 0.f);
    }
    // Grass SSS uses derived albedo squared; angular weight remains linear.
    const float reflected = nativeFoliageLinearChannel(0.4f, true);
    expectNear(nativeFoliageLinearChannel(0.4f * 0.4f, true), reflected * reflected);
    expectNear(nativeFoliageLinearChannel(0.4f * 0.4f, false), 0.16f);
    // LL-on and LL-off both produce the same RTX reflectance for a neutral
    // vertex colour and the same authored sample (no repeated gamma).
    const float off = nativeGrassAlbedoChannel(0.5f, 1.f, 1.f, 1.f, 1.f, 1.f, false, false);
    const float on = nativeGrassAlbedoChannel(0.5f, 1.f, 1.f, 2.2f, 1.f, 1.f, false, false);
    expectNear(nativeFoliageLinearChannel(off, true), nativeFoliageLinearChannel(on, false));
    std::cout << "Native grass colour, facing and complex cotangent-frame tests passed (CPU, not raster parity).\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return -1;
  }
}
