#include <array>
#include <cstring>
#include <stdexcept>

#include "../../test_utils.h"
#include "rtx_render/rtx_materials.h"
#include "rtx_render/rtx_remix_api_convert.h"

namespace dxvk {
  Logger Logger::s_instance("test_material_layout.log");
}

namespace {
  void require(bool condition, const char* message) {
    if (!condition) {
      throw std::runtime_error(message);
    }
  }

  uint16_t readWord(const unsigned char* bytes, size_t offset) {
    uint16_t value;
    std::memcpy(&value, bytes + offset, sizeof(value));
    return value;
  }

  template<typename Material>
  void checkRecords(const Material& first, const Material& second,
                    size_t payloadSize, size_t identityOffset,
                    uint16_t firstIdentity, uint16_t secondIdentity) {
    constexpr size_t kGuard = 16;
    constexpr auto kStride = dxvk::kSurfaceMaterialGPUSize;
    std::array<unsigned char, 2 * kStride + 2 * kGuard> bytes;
    bytes.fill(0xcd);
    size_t offset = kGuard;
    dxvk::RtSurfaceMaterial(first).writeGPUData(bytes.data(), offset, 7);
    require(offset == kGuard + kStride, "First material does not advance one GPU record");
    dxvk::RtSurfaceMaterial(second).writeGPUData(bytes.data(), offset, 8);
    require(offset == kGuard + 2 * kStride, "Second material does not advance one GPU record");
    require(readWord(bytes.data(), kGuard + identityOffset) == firstIdentity, "First record field mismatch");
    require(readWord(bytes.data(), kGuard + kStride + identityOffset) == secondIdentity, "Second record field mismatch");
    for (size_t i = 0; i < kGuard; ++i) {
      require(bytes[i] == 0xcd && bytes[kGuard + 2 * kStride + i] == 0xcd, "Material write crossed guard");
    }
    for (size_t record = 0; record < 2; ++record) {
      for (size_t i = payloadSize; i < kStride; ++i) {
#ifdef NDEBUG
        constexpr unsigned char kPadding = 0xcd; // Release skips padding; caller owns initialization.
#else
        constexpr unsigned char kPadding = 0xff; // Debug deliberately poisons padding.
#endif
        require(bytes[kGuard + record * kStride + i] == kPadding, "Material wrote fields into padding");
      }
    }
  }
}

int main() {
  try {
    // native_grass.comp reads 32-word records: phase18, flag19, rows20/24/28.
    dxvk::NativeGrassRecord grass {};
    grass.placement.padding[1] = 1;
    for (uint32_t row = 0; row < 3; ++row) {
      for (uint32_t col = 0; col < 4; ++col) {
        grass.normalRows[row][col] = float(row * 4 + col + 1);
      }
    }
    std::array<uint32_t, 32> grassWords {};
    std::memcpy(grassWords.data(), &grass, sizeof(grass));
    require(grassWords[19] == 1, "Grass native-normal flag offset mismatch");
    for (uint32_t row = 0; row < 3; ++row) {
      for (uint32_t col = 0; col < 4; ++col) {
        float value;
        std::memcpy(&value, &grassWords[20 + row * 4 + col], sizeof(value));
        require(value == float(row * 4 + col + 1), "Grass normal-row GPU offset mismatch");
      }
    }
    using namespace dxvk;
    remixapi_MaterialInfoOpaqueSubsurfaceEXT foliage {};
    foliage.sType = REMIXAPI_STRUCT_TYPE_MATERIAL_INFO_OPAQUE_SUBSURFACE_EXT;
    foliage.subsurfaceTransmittanceColor = { 0.2f, 0.6f, 0.1f };
    foliage.subsurfaceMeasurementDistance = 1.0f;
    foliage.subsurfaceSingleScatteringAlbedo = { 0.5f, 0.5f, 0.5f };
    remixapi_MaterialInfoOpaqueEXT sourceOpaque {};
    sourceOpaque.sType = REMIXAPI_STRUCT_TYPE_MATERIAL_INFO_OPAQUE_EXT;
    sourceOpaque.pNext = &foliage;
    remixapi_MaterialInfo source {};
    source.sType = REMIXAPI_STRUCT_TYPE_MATERIAL_INFO;
    source.pNext = &sourceOpaque;
    auto imported = convert::toRtMaterialWithoutTexturePreload(source);
    require(imported.getOpaqueMaterialData().getSubsurfaceMeasurementDistance() == 1.f, "Thin material lost thickness in API conversion");
    require(!imported.getOpaqueMaterialData().getSubsurfaceDiffusionProfile(), "Leaf became a diffusion-profile material");
    require(imported.getOpaqueMaterialData().getSubsurfaceTransmittanceColor().y == 0.6f, "Thin material lost transmission color");
    foliage.subsurfaceDiffusionProfile = true;
    foliage.subsurfaceRadius = { 0.5f, 0.25f, 0.125f };
    foliage.subsurfaceRadiusScale = 1.f;
    foliage.subsurfaceMaxSampleRadius = 16.f;
    imported = convert::toRtMaterialWithoutTexturePreload(source);
    const auto& importedSkin = imported.getOpaqueMaterialData();
    require(importedSkin.getSubsurfaceDiffusionProfile(), "Skin lost diffusion-profile routing");
    require(importedSkin.getSubsurfaceRadius().x == 0.5f &&
            importedSkin.getSubsurfaceRadius().y == 0.25f &&
            importedSkin.getSubsurfaceRadius().z == 0.125f, "Skin lost per-channel radii");
    require(importedSkin.getSubsurfaceRadiusScale() == 1.f &&
            importedSkin.getSubsurfaceMaxSampleRadius() == 16.f, "Skin lost radius scales");
    sourceOpaque.pNext = nullptr;
    imported = convert::toRtMaterialWithoutTexturePreload(source);
    require(imported.getOpaqueMaterialData().getSubsurfaceMeasurementDistance() == 0.f, "Ordinary opaque material gained thin scattering");
    const RtSubsurfaceMaterial skin(11, 12, 13, Vector3(0.7f), 1.f, Vector3(0.8f), 0.1f, 2.f, 4.f);
    const RtSubsurfaceMaterial leaf(21, 22, 23, Vector3(0.3f), 0.5f, Vector3(0.9f), 0.2f, -1.f, 8.f);
    checkRecords(skin, leaf, 26, 2, 11, 21);
    const RtRayPortalSurfaceMaterial portal1(31, 32, 1, 0.25f, false, 1.f, 2, 3);
    const RtRayPortalSurfaceMaterial portal2(41, 42, 2, 0.5f, true, 2.f, 4, 5);
    checkRecords(portal1, portal2, 16, 4, 31, 41);
    const RtTranslucentSurfaceMaterial glass1(51, 52, 53, 1.5f, 4.f, Vector3(0.8f),
      false, 0.f, Vector3(0.f), false, 1.f, false, 1);
    const RtTranslucentSurfaceMaterial glass2(61, 62, 63, 1.3f, 8.f, Vector3(0.5f),
      false, 0.f, Vector3(0.f), true, 0.5f, false, 2);
    checkRecords(glass1, glass2, 110, 16, 51, 61);
    RtOpaqueSurfaceMaterial opaque1(71, 72, 73, 74, 75, 76, 77, 0.f, 0.f,
      Vector4(0.5f), 0.4f, 0.f, Vector3(0.f), false, false, false, false, 0.f,
      1, 0.f, 0.f, SURFACE_INDEX_INVALID, false, false, SAMPLER_FEEDBACK_INVALID);
    RtOpaqueSurfaceMaterial opaque2(81, 82, 83, 84, 85, 86, 87, 0.f, 0.f,
      Vector4(0.8f), 0.6f, 0.f, Vector3(0.f), false, false, false, false, 0.f,
      2, 0.f, 0.f, SURFACE_INDEX_INVALID, false, false, SAMPLER_FEEDBACK_INVALID);
    checkRecords(opaque1, opaque2, 76, 4, 71, 81);
    auto nativeColour = opaque1;
    nativeColour.setNativeSrgbAlbedo(true);
    std::array<unsigned char, kSurfaceMaterialGPUSize> nativeColourBytes {};
    size_t nativeColourOffset = 0;
    RtSurfaceMaterial(nativeColour).writeGPUData(nativeColourBytes.data(), nativeColourOffset, 0);
    require((readWord(nativeColourBytes.data(), 0) & OPAQUE_SURFACE_MATERIAL_FLAG_NATIVE_SRGB_ALBEDO) != 0,
      "Native albedo flag lost in 16-bit GPU field");
    require(readWord(nativeColourBytes.data(), 2) == 1, "Native albedo flag corrupted sampler index");
    nativeColour.setNativeSrgbAlbedo(false);
    nativeColourOffset = 0;
    RtSurfaceMaterial(nativeColour).writeGPUData(nativeColourBytes.data(), nativeColourOffset, 0);
    require((readWord(nativeColourBytes.data(), 0) & OPAQUE_SURFACE_MATERIAL_FLAG_NATIVE_SRGB_ALBEDO) == 0,
      "Native albedo flag could not be cleared");
    auto foliageOpaque1 = opaque1;
    auto foliageOpaque2 = opaque2;
    const auto ordinaryHash = foliageOpaque1.getHash();
    auto nativeFoliage1 = std::make_shared<NativeFoliageMaterialData>();
    auto nativeFoliage2 = std::make_shared<NativeFoliageMaterialData>();
    nativeFoliage1->flags = NATIVE_FOLIAGE_GRASS | NATIVE_FOLIAGE_GAMMA_COLOR | NATIVE_FOLIAGE_SPHERE_NORMAL;
    nativeFoliage2->flags = NATIVE_FOLIAGE_SOFT | NATIVE_FOLIAGE_BACK;
    nativeFoliage1->parameters = { 2.f, 1.2f, 0.03f, 1.8f, 1.f, 0.75f };
    nativeFoliage2->parameters = { 3.f, 1.f, 0.04f, 1.f, 1.5f, 1.f };
    foliageOpaque1.setNativeFoliage(nativeFoliage1, 201, 202);
    foliageOpaque2.setNativeFoliage(nativeFoliage2, 211, 212);
    require(foliageOpaque1.getHash() != ordinaryHash, "Foliage metadata did not affect material identity");
    uint32_t softVisits = 0, backVisits = 0;
    foliageOpaque1.forEachTextureIndex([&](uint32_t index) {
      softVisits += index == 201;
      backVisits += index == 202;
    });
    require(softVisits == 1 && backVisits == 1, "Foliage textures missing from retain/release traversal");
    checkRecords(foliageOpaque1, foliageOpaque2, 96, 64, 201, 211);
    checkRecords(foliageOpaque1, foliageOpaque2, 96, 66, 202, 212);
    std::array<unsigned char, kSurfaceMaterialGPUSize> foliageBytes {};
    size_t foliageOffset = 0;
    RtSurfaceMaterial(foliageOpaque1).writeGPUData(foliageBytes.data(), foliageOffset, 0);
    require((readWord(foliageBytes.data(), 0) & OPAQUE_SURFACE_MATERIAL_FLAG_NATIVE_FOLIAGE) != 0, "Foliage flag lost in 16-bit GPU field");
    require(readWord(foliageBytes.data(), 68) == nativeFoliage1->flags, "Foliage flags offset mismatch");
    std::array<float, 6> foliageParameters {};
    std::memcpy(foliageParameters.data(), foliageBytes.data() + 72, sizeof(foliageParameters));
    require(foliageParameters == nativeFoliage1->parameters, "Native foliage shader parameters offset mismatch");
    auto differentFoliage = std::make_shared<NativeFoliageMaterialData>(*nativeFoliage1);
    differentFoliage->flags &= ~NATIVE_FOLIAGE_GAMMA_COLOR;
    require(differentFoliage->hash() != nativeFoliage1->hash(), "Colour domain absent from material identity");
    differentFoliage->flags = nativeFoliage1->flags & ~NATIVE_FOLIAGE_SPHERE_NORMAL;
    require(differentFoliage->hash() != nativeFoliage1->hash(), "Sphere-normal rule absent from material identity");
    differentFoliage->flags = nativeFoliage1->flags;
    differentFoliage->parameters[5] = 0.f;
    require(differentFoliage->hash() != nativeFoliage1->hash(), "SSS strength absent from material identity");
    auto inheritedFoliage = convert::toRtMaterialWithoutTexturePreload(source).getOpaqueMaterialData();
    inheritedFoliage.setNativeFoliage(nativeFoliage1);
    auto mergedFoliage = convert::toRtMaterialWithoutTexturePreload(source).getOpaqueMaterialData();
    mergedFoliage.merge(inheritedFoliage);
    require(mergedFoliage.getNativeFoliage() == nativeFoliage1, "Native foliage lost during material merge");
    std::array<uint16_t, 10> layers { 91, 92, 93, 94, 95, 96, 97, 98, 99, 100 };
    opaque1.setNativeLandscape(layers, 0x3f);
    opaque2.setNativeLandscape(layers, 0x1f);
    checkRecords(opaque1, opaque2, 76, 54, 91, 91);
    auto effect1 = std::make_shared<NativeEffectMaterialData>();
    auto effect2 = std::make_shared<NativeEffectMaterialData>();
    effect1->identity = 1;
    effect2->identity = 2;
    opaque1.setNativeEffect(effect1, 101, 102);
    opaque2.setNativeEffect(effect2, 111, 112);
    checkRecords(opaque1, opaque2, 112, 106, 101, 111);
    auto water1 = glass1;
    auto water2 = glass2;
    auto nativeWater1 = std::make_shared<NativeWaterMaterialData>();
    auto nativeWater2 = std::make_shared<NativeWaterMaterialData>();
    nativeWater1->identity = 3;
    nativeWater2->identity = 4;
    water1.setNativeWater(nativeWater1, { 121, 122, 123, 124 }, 125);
    water2.setNativeWater(nativeWater2, { 131, 132, 133, 134 }, 135);
    checkRecords(water1, water2, 110, 34, 121, 131);
  } catch (const std::exception& error) {
    std::cerr << "TEST FAILED: " << error.what() << std::endl;
    return -1;
  }
  std::cout << "Material record stride, field offsets, padding boundaries and guards passed" << std::endl;
  return 0;
}
