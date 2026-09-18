/*
* Copyright (c) 2021-2026, NVIDIA CORPORATION. All rights reserved.
*
* Permission is hereby granted, free of charge, to any person obtaining a
* copy of this software and associated documentation files (the "Software"),
* to deal in the Software without restriction, including without limitation
* the rights to use, copy, modify, merge, publish, distribute, sublicense,
* and/or sell copies of the Software, and to permit persons to whom the
* Software is furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
* THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
* FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
* DEALINGS IN THE SOFTWARE.
*/

#define RTX_REMIX_PNEXT_CHECK_STRUCTS

#pragma once

// Conversion from the public Remix API's descriptions into the renderer's own
// types, shared by every frontend that can host the API.
//
// This is header-only on purpose. The D3D9 host compiles into the shared dxvk
// library and the D3D11 host into its own DLL; if the conversion lived in a
// translation unit of either, linking it from the other would drag that host's
// frontend symbols along with it.

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <remix/remix_c.h>

#include "rtx_asset_data_manager.h"
#include "rtx_asset_replacer.h"
#include "rtx_globals.h"
#include "rtx_light_manager.h"
#include "rtx_option.h"
#include "rtx_options.h"
#include "rtx_remix_pnext.h"
#include "rtx_scene_manager.h"
#include "rtx_texture_manager.h"

#include "../dxvk_device.h"
#include "../dxvk_image.h"

#include "../../util/util_math.h"
#include "../../util/util_string.h"
#include "../../util/util_vector.h"

namespace dxvk {
  // Because DrawCallState/LegacyMaterialData hide needed fields as private
  struct RemixAPIPrivateAccessor {
    static std::unique_ptr<ExternalDrawState> toRtDrawState(const remixapi_InstanceInfo& info);
  };

  // The API version the host declared. Older hosts do not set the newer fields,
  // so the conversion below falls back where this says it must. Each frontend
  // has its own copy, set when that frontend's host registers itself.
  inline uint64_t g_remixApiVersion = 0;
}

  namespace convert {
    using namespace dxvk;

    inline std::string tostr(const remixapi_MaterialHandle& h) {
      static_assert(sizeof h == sizeof uint64_t);
      return std::to_string(reinterpret_cast<uint64_t>(h));
    }

    inline Matrix4 tomat4(const remixapi_Transform& transform) {
      const auto& m = transform.matrix;
      return Matrix4 {
        m[0][0], m[1][0], m[2][0], 0.f,
        m[0][1], m[1][1], m[2][1], 0.f,
        m[0][2], m[1][2], m[2][2], 0.f,
        m[0][3], m[1][3], m[2][3], 1.f
      };
    }

    inline Vector2 tovec2(const remixapi_Float2D& v) {
      return Vector2{ v.x, v.y };
    }

    inline Vector3 tovec3(const remixapi_Float3D& v) {
      return Vector3 { v.x, v.y, v.z };
    }

    inline Vector4 tovec4(const remixapi_Float4D& v) {
      return Vector4 { v.x, v.y, v.z, v.w };
    }

    inline Vector3d tovec3d(const remixapi_Float3D& v) {
      return Vector3d{ v.x, v.y, v.z };
    }

    inline std::vector<float> toAnimatedFloat1D(const remixapi_AnimatedFloat1D& animated) {
      std::vector<float> result;
      if (animated.pData && animated.numberElements > 0) {
        result.reserve(animated.numberElements);
        for (uint32_t i = 0; i < animated.numberElements; ++i) {
          result.push_back(animated.pData[i]);
        }
      }
      return result;
    }

    inline std::vector<vec2> toAnimatedFloat2D(const remixapi_AnimatedFloat2D& animated) {
      std::vector<vec2> result;
      if (animated.pData && animated.numberElements > 0) {
        result.reserve(animated.numberElements);
        for (uint32_t i = 0; i < animated.numberElements; ++i) {
          result.push_back(vec2(animated.pData[i].x, animated.pData[i].y));
        }
      }
      return result;
    }

    inline std::vector<vec3> toAnimatedFloat3D(const remixapi_AnimatedFloat3D& animated) {
      std::vector<vec3> result;
      if (animated.pData && animated.numberElements > 0) {
        result.reserve(animated.numberElements);
        for (uint32_t i = 0; i < animated.numberElements; ++i) {
          result.push_back(vec3(animated.pData[i].x, animated.pData[i].y, animated.pData[i].z));
        }
      }
      return result;
    }

    inline std::vector<vec4> toAnimatedFloat4D(const remixapi_AnimatedFloat4D& animated) {
      std::vector<vec4> result;
      if (animated.pData && animated.numberElements > 0) {
        result.reserve(animated.numberElements);
        for (uint32_t i = 0; i < animated.numberElements; ++i) {
          result.push_back(vec4(animated.pData[i].x, animated.pData[i].y, animated.pData[i].z, animated.pData[i].w));
        }
      }
      return result;
    }

    inline constexpr bool tobool(remixapi_Bool b) {
      return !!b;
    }

    inline std::filesystem::path topath(remixapi_Path p) {
      if (!p) {
        return {};
      }
      return p;
    }

    // --

    struct PreloadSource {
      std::filesystem::path albedoTexture;
      std::filesystem::path normalTexture;
      std::filesystem::path tangentTexture;
      std::filesystem::path emissiveTexture;
      std::filesystem::path transmittanceTexture;
      std::filesystem::path roughnessTexture;
      std::filesystem::path metallicTexture;
      std::filesystem::path heightTexture;
      std::filesystem::path subsurfaceTransmittanceTexture;
      std::filesystem::path subsurfaceThicknessTexture;
      std::filesystem::path subsurfaceSingleScatteringAlbedoTexture;
      std::filesystem::path subsurfaceRadiusTexture;
    };

    inline PreloadSource makePreloadSource(const remixapi_MaterialInfo& info) {
      // TODO: C++20 designated initializers
      if (auto extOpaque = pnext::find<remixapi_MaterialInfoOpaqueEXT>(&info)) {
        auto extSubsurface = pnext::find<remixapi_MaterialInfoOpaqueSubsurfaceEXT>(&info);
        return PreloadSource {
          topath(info.albedoTexture),   // albedoTexture;
          topath(info.normalTexture),   // normalTexture;
          topath(info.tangentTexture),  // tangentTexture;
          topath(info.emissiveTexture), // emissiveTexture;
          {},                           // transmittanceTexture;
          topath(extOpaque->roughnessTexture),  // roughnessTexture;
          topath(extOpaque->metallicTexture),   // metallicTexture;
          topath(extOpaque->heightTexture),     // heightTexture;
          topath(extSubsurface ? extSubsurface->subsurfaceTransmittanceTexture          : nullptr), // subsurfaceTransmittanceTexture;
          topath(extSubsurface ? extSubsurface->subsurfaceThicknessTexture              : nullptr), // subsurfaceThicknessTexture;
          topath(extSubsurface ? extSubsurface->subsurfaceSingleScatteringAlbedoTexture : nullptr), // subsurfaceSingleScatteringAlbedoTexture;
          topath(dxvk::g_remixApiVersion >= REMIXAPI_VERSION_MAKE(0, 5, 1) && extSubsurface ? extSubsurface->subsurfaceRadiusTexture : nullptr), // subsurfaceRadiusTexture;
        };
      }
      if (auto extTranslucent = pnext::find<remixapi_MaterialInfoTranslucentEXT>(&info)) {
        return PreloadSource {
          topath(info.albedoTexture),   // albedoTexture;
          topath(info.normalTexture),   // normalTexture;
          topath(info.tangentTexture),  // tangentTexture;
          topath(info.emissiveTexture), // emissiveTexture;
          topath(extTranslucent->transmittanceTexture), // transmittanceTexture;
          {}, // roughnessTexture;
          {}, // metallicTexture;
          {}, // heightTexture;
          {}, // subsurfaceTransmittanceTexture;
          {}, // subsurfaceThicknessTexture;
          {}, // subsurfaceSingleScatteringAlbedoTexture;
          {}, // subsurfaceRadiusTexture;
        };
      }
      if (auto extPortal = pnext::find<remixapi_MaterialInfoPortalEXT>(&info)) {
        return PreloadSource {
          topath(info.albedoTexture),   // albedoTexture;
          topath(info.normalTexture),   // normalTexture;
          topath(info.tangentTexture),  // tangentTexture;
          topath(info.emissiveTexture), // emissiveTexture;
          {}, // transmittanceTexture;
          {}, // roughnessTexture;
          {}, // metallicTexture;
          {}, // heightTexture;
          {}, // subsurfaceTransmittanceTexture;
          {}, // subsurfaceThicknessTexture;
          {}, // subsurfaceSingleScatteringAlbedoTexture;
          {}, // subsurfaceRadiusTexture;
        };
      }
      return {};
    }

    inline MaterialData toRtMaterialFinalized(dxvk::DxvkContext& ctx, const MaterialData& materialWithoutPreload, const PreloadSource& preload) {
      auto preloadTexture = [&ctx](const std::filesystem::path& path)->TextureRef {
        if (path.empty()) {
          return {};
        }
        auto assetData = AssetDataManager::get().findAsset(path.string());
        if (assetData == nullptr) {
          return {};
        }
        auto uploadedTexture = ctx.getCommonObjects()->getTextureManager()
          .preloadTextureAsset(assetData, dxvk::ColorSpace::AUTO, false);
        return TextureRef { uploadedTexture };
      };

      switch (materialWithoutPreload.getType()) {
      case MaterialDataType::Opaque:
      {
        const auto& src = materialWithoutPreload.getOpaqueMaterialData();
        return MaterialData { OpaqueMaterialData{
          preloadTexture(preload.albedoTexture),
          preloadTexture(preload.normalTexture),
          preloadTexture(preload.tangentTexture),
          preloadTexture(preload.heightTexture),
          preloadTexture(preload.roughnessTexture),
          preloadTexture(preload.metallicTexture),
          preloadTexture(preload.emissiveTexture),
          preloadTexture(preload.subsurfaceTransmittanceTexture),
          preloadTexture(preload.subsurfaceThicknessTexture),
          preloadTexture(preload.subsurfaceSingleScatteringAlbedoTexture),
          preloadTexture(preload.subsurfaceRadiusTexture),
          TextureRef(),
          src.getAnisotropyConstant(),
          src.getEmissiveIntensity(),
          src.getAlbedoConstant(),
          src.getOpacityConstant(),
          src.getRoughnessConstant(),
          src.getMetallicConstant(),
          src.getEmissiveColorConstant(),
          src.getEnableEmission(),
          src.getSpriteSheetRows(),
          src.getSpriteSheetCols(),
          src.getSpriteSheetFPS(),
          src.getEnableThinFilm(),
          src.getAlphaIsThinFilmThickness(),
          src.getThinFilmThicknessConstant(),
          src.getUseLegacyAlphaState(),
          src.getBlendEnabled(),
          src.getBlendType(),
          src.getInvertedBlend(),
          src.getAlphaTestType(),
          src.getAlphaTestReferenceValue(),
          src.getDisplaceIn(),
          src.getDisplaceOut(),
          src.getSubsurfaceTransmittanceColor(),
          src.getSubsurfaceMeasurementDistance(),
          src.getSubsurfaceSingleScatteringAlbedo(),
          src.getSubsurfaceVolumetricAnisotropy(),
          src.getSubsurfaceDiffusionProfile(),
          src.getSubsurfaceRadius(),
          src.getSubsurfaceRadiusScale(),
          src.getSubsurfaceMaxSampleRadius(),
          src.getFilterMode(),
          src.getWrapModeU(),
          src.getWrapModeV()
        } };
      }
      case MaterialDataType::Translucent: 
      {
        const auto& src = materialWithoutPreload.getTranslucentMaterialData();
        return MaterialData { TranslucentMaterialData {
          preloadTexture(preload.normalTexture),
          preloadTexture(preload.transmittanceTexture),
          preloadTexture(preload.emissiveTexture),
          src.getRefractiveIndex(),
          src.getTransmittanceColor(),
          src.getTransmittanceMeasurementDistance(),
          src.getEnableEmission(),
          src.getEmissiveIntensity(),
          src.getEmissiveColorConstant(),
          src.getSpriteSheetRows(),
          src.getSpriteSheetCols(),
          src.getSpriteSheetFPS(),
          src.getEnableThinWalled(),
          src.getThinWallThickness(),
          src.getEnableDiffuseLayer(),
          src.getFilterMode(),
          src.getWrapModeU(),
          src.getWrapModeV()
        } };
      }
      case MaterialDataType::RayPortal:
      {
        const auto& src = materialWithoutPreload.getRayPortalMaterialData();
        return MaterialData { RayPortalMaterialData {
          preloadTexture(preload.emissiveTexture),
          {}, // unused
          src.getRayPortalIndex(),
          src.getSpriteSheetRows(),
          src.getSpriteSheetCols(),
          src.getSpriteSheetFPS(),
          src.getRotationSpeed(),
          src.getEnableEmission(),
          src.getEmissiveIntensity(),
          src.getFilterMode(),
          src.getWrapModeU(),
          src.getWrapModeV()
        } };
      }
      default: assert(0); return materialWithoutPreload;
      }
    }

    inline MaterialData toRtMaterialWithoutTexturePreload(const remixapi_MaterialInfo& info) {
      if (auto extOpaque = pnext::find<remixapi_MaterialInfoOpaqueEXT>(&info)) {
        auto extSubsurface = pnext::find<remixapi_MaterialInfoOpaqueSubsurfaceEXT>(&info);
        return MaterialData { OpaqueMaterialData {
          {},
          {},
          {},
          {},
          {},
          {},
          {},
          {},
          {},
          {},
          {},
          {},
          extOpaque->anisotropy,
          info.emissiveIntensity,
          tovec3(extOpaque->albedoConstant),
          extOpaque->opacityConstant,
          extOpaque->roughnessConstant,
          extOpaque->metallicConstant,
          tovec3(info.emissiveColorConstant),
          info.emissiveIntensity > 0.f,
          info.spriteSheetRow,
          info.spriteSheetCol,
          info.spriteSheetFps,
          tobool(extOpaque->thinFilmThickness_hasvalue),
          tobool(extOpaque->alphaIsThinFilmThickness),
          extOpaque->thinFilmThickness_hasvalue ? extOpaque->thinFilmThickness_value : 200.f, // default OpaqueMaterial::ThinFilmThicknessConstant
          tobool(extOpaque->useDrawCallAlphaState), // OpaqueMaterial::UseLegacyAlphaState
          tobool(extOpaque->blendType_hasvalue),
          extOpaque->blendType_hasvalue ? static_cast<BlendType>(extOpaque->blendType_value) : BlendType::kAlpha,  // default OpaqueMaterial::BlendType
          tobool(extOpaque->invertedBlend),
          static_cast<AlphaTestType>(extOpaque->alphaTestType),
          extOpaque->alphaReferenceValue,
          extOpaque->displaceIn,
          (dxvk::g_remixApiVersion >= REMIXAPI_VERSION_MAKE(0, 4, 2)) ? extOpaque->displaceOut : 0.f,
          extSubsurface ? tovec3(extSubsurface->subsurfaceTransmittanceColor) : Vector3{ 0.5f, 0.5f, 0.5f },
          extSubsurface ? extSubsurface->subsurfaceMeasurementDistance : 0.f,
          extSubsurface ? tovec3(extSubsurface->subsurfaceSingleScatteringAlbedo) : Vector3{ 0.5f, 0.5f, 0.5f },
          extSubsurface ? extSubsurface->subsurfaceVolumetricAnisotropy : 0.f,
          extSubsurface ? static_cast<bool>(extSubsurface->subsurfaceDiffusionProfile) : false,
          extSubsurface ? tovec3(extSubsurface->subsurfaceRadius) : Vector3{ 0.5f, 0.5f, 0.5f },
          extSubsurface ? extSubsurface->subsurfaceRadiusScale : 0.f,
          extSubsurface ? extSubsurface->subsurfaceMaxSampleRadius : 0.f,
          info.filterMode,
          info.wrapModeU,
          info.wrapModeV,
        } };
      }
      if (auto extTranslucent = pnext::find<remixapi_MaterialInfoTranslucentEXT>(&info)) {
        return MaterialData { TranslucentMaterialData {
          {},
          {},
          {},
          extTranslucent->refractiveIndex,
          tovec3(extTranslucent->transmittanceColor),
          extTranslucent->transmittanceMeasurementDistance,
          info.emissiveIntensity > 0.f,
          info.emissiveIntensity,
          tovec3(info.emissiveColorConstant),
          info.spriteSheetRow,
          info.spriteSheetCol,
          info.spriteSheetFps,
          tobool(extTranslucent->thinWallThickness_hasvalue),
          extTranslucent->thinWallThickness_hasvalue ? extTranslucent->thinWallThickness_value : 0.001f, // default TranslucentMaterial::ThinWallThickness
          tobool(extTranslucent->useDiffuseLayer),
          info.filterMode,
          info.wrapModeU,
          info.wrapModeV,
        } };
      }
      if (auto extPortal = pnext::find<remixapi_MaterialInfoPortalEXT>(&info)) {
        return MaterialData { RayPortalMaterialData {
          {},
          {}, // unused
          extPortal->rayPortalIndex,
          info.spriteSheetRow,
          info.spriteSheetCol,
          info.spriteSheetFps,
          extPortal->rotationSpeed,
          info.emissiveIntensity > 0.f,
          info.emissiveIntensity,
          info.filterMode,
          info.wrapModeU,
          info.wrapModeV,
        } };
      }

      assert(0);
      return MaterialData { OpaqueMaterialData {} };
    }

    // --
    inline CameraType::Enum toRtCameraType(remixapi_CameraType from) {
      switch (from) {
      case REMIXAPI_CAMERA_TYPE_WORLD: return CameraType::Main;
      case REMIXAPI_CAMERA_TYPE_VIEW_MODEL: return CameraType::ViewModel;
      case REMIXAPI_CAMERA_TYPE_SKY: return CameraType::Sky;
      default: assert(0); return CameraType::Main;
      }
    }

    struct ExternalCameraInfo {
      CameraType::Enum type {};
      Matrix4 worldToView {};
      Matrix4 viewToProjection {};
    };

    inline ExternalCameraInfo toRtCamera(const remixapi_CameraInfo& info) {
      if (auto params = pnext::find<remixapi_CameraInfoParameterizedEXT>(&info)) {
        auto result = ExternalCameraInfo {
          toRtCameraType(info.type),
        };
        {
          const auto newViewToWorld = Matrix4d {
           Vector4d{ normalize(tovec3d(params->right)), 0.0 },
           Vector4d{ normalize(tovec3d(params->up)), 0.0 },
           Vector4d{ normalize(tovec3d(params->forward)), 0.0 },
           Vector4d{ tovec3d(params->position), 1.0 },
          };
          result.worldToView = inverse(newViewToWorld);
        }
        {
          constexpr bool isLhs = true;
          auto proj = float4x4 {};
          proj.SetupByHalfFovy(
            DegToRad(params->fovYInDegrees) / 2,
            params->aspect,
            params->nearPlane,
            params->farPlane,
            isLhs ? PROJ_LEFT_HANDED : 0);
          static_assert(sizeof result.viewToProjection == sizeof proj);
          memcpy(&result.viewToProjection, &proj, sizeof float4x4);
        }
        return result;
      }
      return ExternalCameraInfo {
        toRtCameraType(info.type),
        Matrix4 { info.view },
        Matrix4 { info.projection },
      };
    }

    // --

    inline std::optional<RtLightShaping> toRtLightShaping(const remixapi_LightInfoLightShaping* info) {
      if (info) {
        return RtLightShaping::tryCreate(
          true,
          tovec3(info->direction),
          std::cos(DegToRad(info->coneAngleDegrees)),
          info->coneSoftness,
          info->focusExponent
        );
      }

      // Note: Default constructed Light Shaping returned when no info is provided to have a valid but disabled
      // Light Shaping object (different from returning an empty optional here, which means creation of a Light
      // Shaping failed).
      return RtLightShaping{};
    }

    inline std::optional<RtLight> toRtLight(const remixapi_LightInfo& info) {
      if (auto src = pnext::find<remixapi_LightInfoUSDEXT>(&info)) {
        if (auto lightData = LightData::tryCreate(*src)) {
          return lightData->toRtLight();
        }
        return {};
      }
      if (auto src = pnext::find<remixapi_LightInfoSphereEXT>(&info)) {
        const auto shaping = toRtLightShaping(src->shaping_hasvalue ? &src->shaping_value : nullptr);

        if (!shaping.has_value()) {
          return {};
        }

        return RtSphereLight::tryCreate(
          tovec3(src->position),
          tovec3(info.radiance),
          src->radius,
          *shaping,
          src->volumetricRadianceScale
        );
      }
      if (auto src = pnext::find<remixapi_LightInfoRectEXT>(&info)) {
        const auto shaping = toRtLightShaping(src->shaping_hasvalue ? &src->shaping_value : nullptr);

        if (!shaping.has_value()) {
          return {};
        }

        return RtRectLight::tryCreate(
          tovec3(src->position),
          Vector2{src->xSize, src->ySize},
          tovec3(src->xAxis),
          tovec3(src->yAxis),
          tovec3(src->direction),
          tovec3(info.radiance),
          *shaping,
          src->volumetricRadianceScale
        );
      }
      if (auto src = pnext::find<remixapi_LightInfoDiskEXT>(&info)) {
        const auto shaping = toRtLightShaping(src->shaping_hasvalue ? &src->shaping_value : nullptr);

        if (!shaping.has_value()) {
          return {};
        }

        return RtDiskLight::tryCreate(
          tovec3(src->position),
          Vector2{src->xRadius, src->yRadius},
          tovec3(src->xAxis),
          tovec3(src->yAxis),
          tovec3(src->direction),
          tovec3(info.radiance),
          *shaping,
          src->volumetricRadianceScale
        );
      }
      if (auto src = pnext::find<remixapi_LightInfoCylinderEXT>(&info)) {
        return RtCylinderLight::tryCreate(
          tovec3(src->position),
          src->radius,
          tovec3(src->axis),
          src->axisLength,
          tovec3(info.radiance),
          src->volumetricRadianceScale
        );
      }
      if (auto src = pnext::find<remixapi_LightInfoDistantEXT>(&info)) {
        return RtDistantLight::tryCreate(
          tovec3(src->direction),
          DegToRad(src->angularDiameterDegrees * 0.5f),
          tovec3(info.radiance),
          src->volumetricRadianceScale
        );
      }

      // Note: Return an empty optional if the LightInfo struct does not contain a supported
      // LightInfo extension struct.
      return {};
    }

    // --

    inline CameraType::Enum categoryToCameraType(remixapi_InstanceCategoryFlags flags) {
      if (flags & REMIXAPI_INSTANCE_CATEGORY_BIT_VIEW_MODEL) {
        return CameraType::ViewModel;
      }
      if (flags & REMIXAPI_INSTANCE_CATEGORY_BIT_SKY) {
        return CameraType::Sky;
      }
      return CameraType::Main;
    }

    inline CategoryFlags toRtCategories(remixapi_InstanceCategoryFlags flags) {
      CategoryFlags result { 0 };
      if (flags & REMIXAPI_INSTANCE_CATEGORY_BIT_WORLD_UI                 ){ result.set(InstanceCategories::WorldUI               ); }
      if (flags & REMIXAPI_INSTANCE_CATEGORY_BIT_WORLD_MATTE              ){ result.set(InstanceCategories::WorldMatte            ); }
      if (flags & REMIXAPI_INSTANCE_CATEGORY_BIT_SKY                      ){ result.set(InstanceCategories::Sky                   ); }
      if (flags & REMIXAPI_INSTANCE_CATEGORY_BIT_IGNORE                   ){ result.set(InstanceCategories::Ignore                ); }
      if (flags & REMIXAPI_INSTANCE_CATEGORY_BIT_IGNORE_LIGHTS            ){ result.set(InstanceCategories::IgnoreLights          ); }
      if (flags & REMIXAPI_INSTANCE_CATEGORY_BIT_IGNORE_ANTI_CULLING      ){ result.set(InstanceCategories::IgnoreAntiCulling     ); }
      if (flags & REMIXAPI_INSTANCE_CATEGORY_BIT_IGNORE_MOTION_BLUR       ){ result.set(InstanceCategories::IgnoreMotionBlur      ); }
      if (flags & REMIXAPI_INSTANCE_CATEGORY_BIT_IGNORE_OPACITY_MICROMAP  ){ result.set(InstanceCategories::IgnoreOpacityMicromap ); }
      if (flags & REMIXAPI_INSTANCE_CATEGORY_BIT_IGNORE_ALPHA_CHANNEL     ){ result.set(InstanceCategories::IgnoreAlphaChannel    ); }
      if (flags & REMIXAPI_INSTANCE_CATEGORY_BIT_HIDDEN                   ){ result.set(InstanceCategories::Hidden                ); }
      if (flags & REMIXAPI_INSTANCE_CATEGORY_BIT_PARTICLE                 ){ result.set(InstanceCategories::Particle              ); }
      if (flags & REMIXAPI_INSTANCE_CATEGORY_BIT_BEAM                     ){ result.set(InstanceCategories::Beam                  ); }
      if (flags & REMIXAPI_INSTANCE_CATEGORY_BIT_DECAL_STATIC             ){ result.set(InstanceCategories::DecalStatic           ); }
      if (flags & REMIXAPI_INSTANCE_CATEGORY_BIT_DECAL_DYNAMIC            ){ result.set(InstanceCategories::DecalDynamic          ); }
      if (flags & REMIXAPI_INSTANCE_CATEGORY_BIT_DECAL_SINGLE_OFFSET      ){ result.set(InstanceCategories::DecalSingleOffset     ); }
      if (flags & REMIXAPI_INSTANCE_CATEGORY_BIT_DECAL_NO_OFFSET          ){ result.set(InstanceCategories::DecalNoOffset         ); }
      if (flags & REMIXAPI_INSTANCE_CATEGORY_BIT_ALPHA_BLEND_TO_CUTOUT    ){ result.set(InstanceCategories::AlphaBlendToCutout    ); }
      if (flags & REMIXAPI_INSTANCE_CATEGORY_BIT_TERRAIN                  ){ result.set(InstanceCategories::Terrain               ); }
      if (flags & REMIXAPI_INSTANCE_CATEGORY_BIT_ANIMATED_WATER           ){ result.set(InstanceCategories::AnimatedWater         ); }
      if (flags & REMIXAPI_INSTANCE_CATEGORY_BIT_THIRD_PERSON_PLAYER_MODEL){ result.set(InstanceCategories::ThirdPersonPlayerModel); }
      if (flags & REMIXAPI_INSTANCE_CATEGORY_BIT_THIRD_PERSON_PLAYER_BODY ){ result.set(InstanceCategories::ThirdPersonPlayerBody ); }
      if (flags & REMIXAPI_INSTANCE_CATEGORY_BIT_IGNORE_BAKED_LIGHTING    ){ result.set(InstanceCategories::IgnoreBakedLighting   ); }
      if (flags & REMIXAPI_INSTANCE_CATEGORY_BIT_PARTICLE_EMITTER)         { result.set(InstanceCategories::ParticleEmitter); }
      if (flags & REMIXAPI_INSTANCE_CATEGORY_BIT_SMOOTH_NORMALS)            { result.set(InstanceCategories::SmoothNormals); }
      if (flags & REMIXAPI_INSTANCE_CATEGORY_BIT_HAIR_CARDS)                { result.set(InstanceCategories::HairCards); }
      
      static_assert((int)InstanceCategories::Count == 25, "Instance categories changed, please update Remix SDK");
      return result;
    }

    inline RtxParticleSystemDesc toRtParticleDesc(const remixapi_InstanceInfoParticleSystemLegacyEXT& info) {
      RtxParticleSystemDesc desc {};

      // Lifetimes
      desc.minTimeToLive = info.minTimeToLive;
      desc.maxTimeToLive = info.maxTimeToLive;

      // Initial 
      desc.spawnRatePerSecond = info.spawnRatePerSecond;
      desc.initialVelocityFromMotion = info.initialVelocityFromMotion;
      desc.initialVelocityFromNormal = info.initialVelocityFromNormal;
      desc.initialVelocityConeAngleDegrees = info.initialVelocityConeAngleDegrees;
      desc.gravityForce = info.gravityForce;
      desc.motionTrailMultiplier = info.motionTrailMultiplier;

      // Turbulence
      desc.turbulenceFrequency = info.turbulenceFrequency;
      desc.turbulenceForce = info.turbulenceForce;

      // Convert spawn/target pairs to 2-element animated vectors
      // Color: spawn at index 0, target at index 1
      desc.minColor = { vec4(info.minSpawnColor.x, info.minSpawnColor.y, info.minSpawnColor.z, info.minSpawnColor.w), 
                        vec4(info.minTargetColor.x, info.minTargetColor.y, info.minTargetColor.z, info.minTargetColor.w) };
      desc.maxColor = { vec4(info.maxSpawnColor.x, info.maxSpawnColor.y, info.maxSpawnColor.z, info.maxSpawnColor.w), 
                        vec4(info.maxTargetColor.x, info.maxTargetColor.y, info.maxTargetColor.z, info.maxTargetColor.w) };

      // Size: spawn at index 0, target at index 1
      desc.minSize = { vec2(info.minSpawnSize, info.minSpawnSize), vec2(info.minTargetSize, info.minTargetSize) };
      desc.maxSize = { vec2(info.maxSpawnSize, info.maxSpawnSize), vec2(info.maxTargetSize, info.maxTargetSize) };

      // Rotation speed: spawn at index 0, target at index 1
      desc.minRotationSpeed = { info.minSpawnRotationSpeed, info.minTargetRotationSpeed };
      desc.maxRotationSpeed = { info.maxSpawnRotationSpeed, info.maxTargetRotationSpeed };

      // Velocity: spawn at index 0, target at index 1
      desc.maxVelocity = { vec3(info.maxSpeed, info.maxSpeed, info.maxSpeed), vec3(info.maxSpeed, info.maxSpeed, info.maxSpeed) };

      // Collision
      desc.collisionThickness = info.collisionThickness;
      desc.collisionRestitution = info.collisionRestitution;
      desc.collisionMode = ParticleCollisionMode::Bounce;

      // Counts/flags
      desc.maxNumParticles = info.maxNumParticles;
      desc.useTurbulence = static_cast<uint8_t>(info.useTurbulence);
      desc.alignParticlesToVelocity = static_cast<uint8_t>(info.alignParticlesToVelocity);
      desc.useSpawnTexcoords = static_cast<uint8_t>(info.useSpawnTexcoords);
      desc.enableCollisionDetection = static_cast<uint8_t>(info.enableCollisionDetection);
      desc.enableMotionTrail = static_cast<uint8_t>(info.enableMotionTrail);
      desc.hideEmitter = static_cast<uint8_t>(info.hideEmitter);

      // Types/modes
      desc.billboardType = static_cast<ParticleBillboardType>(info.billboardType);
      desc.spriteSheetMode = ParticleSpriteSheetMode::UseMaterialSpriteSheet;
      desc.randomFlipAxis = ParticleRandomFlipAxis::None;

      return desc;
    }

    inline RtxParticleSystemDesc toRtParticleDesc(const remixapi_InstanceInfoParticleSystemEXT& info) {
      RtxParticleSystemDesc desc;
      desc.maxNumParticles = info.maxNumParticles;

      // Animated float conversions
      desc.minColor = toAnimatedFloat4D(info.minColor);
      desc.maxColor = toAnimatedFloat4D(info.maxColor);
      desc.minRotationSpeed = toAnimatedFloat1D(info.minRotationSpeed);
      desc.maxRotationSpeed = toAnimatedFloat1D(info.maxRotationSpeed);
      desc.minSize = toAnimatedFloat2D(info.minSize);
      desc.maxSize = toAnimatedFloat2D(info.maxSize);
      desc.maxVelocity = toAnimatedFloat3D(info.maxVelocity);

      desc.minTimeToLive = info.minTimeToLive;
      desc.maxTimeToLive = info.maxTimeToLive;
      desc.initialVelocityFromNormal = info.initialVelocityFromNormal;
      desc.initialVelocityConeAngleDegrees = info.initialVelocityConeAngleDegrees;
      desc.initialVelocityFromMotion = info.initialVelocityFromMotion;
      desc.turbulenceFrequency = info.turbulenceFrequency;
      desc.turbulenceForce = info.turbulenceForce;
      desc.motionTrailMultiplier = info.motionTrailMultiplier;
      desc.spawnRatePerSecond = info.spawnRatePerSecond;
      desc.collisionThickness = info.collisionThickness;
      desc.collisionRestitution = info.collisionRestitution;
      desc.gravityForce = info.gravityForce;
      desc.billboardType = static_cast<ParticleBillboardType>(info.billboardType);
      desc.hideEmitter = static_cast<uint8_t>(info.hideEmitter);
      desc.enableMotionTrail = static_cast<uint8_t>(info.enableMotionTrail);
      desc.useTurbulence = static_cast<uint8_t>(info.useTurbulence);
      desc.alignParticlesToVelocity = static_cast<uint8_t>(info.alignParticlesToVelocity);
      desc.useSpawnTexcoords = static_cast<uint8_t>(info.useSpawnTexcoords);
      desc.enableCollisionDetection = static_cast<uint8_t>(info.enableCollisionDetection);
      desc.dragCoefficient = info.dragCoefficient;
      desc.initialRotationDeviationDegrees = info.initialRotationDeviationDegrees;
      desc.spawnBurstDuration = info.spawnBurstDuration;
      desc.attractorRadius = info.attractorRadius;
      desc.attractorPosition = tovec3(info.attractorPosition);
      desc.attractorForce = info.attractorForce;
      desc.spriteSheetMode = static_cast<ParticleSpriteSheetMode>(info.spriteSheetMode);
      desc.collisionMode = static_cast<ParticleCollisionMode>(info.collisionMode);
      desc.randomFlipAxis = static_cast<ParticleRandomFlipAxis>(info.randomFlipAxis);
      desc.restrictVelocityX = static_cast<uint8_t>(info.restrictVelocityX);
      desc.restrictVelocityY = static_cast<uint8_t>(info.restrictVelocityY);
      desc.restrictVelocityZ = static_cast<uint8_t>(info.restrictVelocityZ);

      return desc;
    }

    inline std::unique_ptr<dxvk::ExternalDrawState> toRtDrawState(const remixapi_InstanceInfo& info) {
      return RemixAPIPrivateAccessor::toRtDrawState(info);
    }
  }

namespace dxvk {
  inline
std::unique_ptr<ExternalDrawState> RemixAPIPrivateAccessor::toRtDrawState(const remixapi_InstanceInfo& info)
{
  auto state = std::make_unique<dxvk::ExternalDrawState>();
  const CameraType::Enum cameraType = convert::categoryToCameraType(info.categoryFlags);

  auto& prototype = state->drawCall;
  {
    prototype.cameraType = cameraType;
    prototype.transformData.objectToWorld = convert::tomat4(info.transform);
    prototype.transformData.texgenMode = TexGenMode::None;
    prototype.categories = convert::toRtCategories(info.categoryFlags);
  }

  if (auto objectPicking = pnext::find<remixapi_InstanceInfoObjectPickingEXT>(&info)) {
    prototype.drawCallID = objectPicking->objectPickingValue;
  }

  if (auto extBones = pnext::find<remixapi_InstanceInfoBoneTransformsEXT>(&info)) {
    const uint32_t boneCount =
      extBones->boneTransforms_count < REMIXAPI_INSTANCE_INFO_MAX_BONES_COUNT ?
      extBones->boneTransforms_count : REMIXAPI_INSTANCE_INFO_MAX_BONES_COUNT;
    prototype.skinningData.minBoneIndex = 0;
    prototype.skinningData.numBones = boneCount;
    prototype.skinningData.numBonesPerVertex = prototype.getGeometryData().numBonesPerVertex;
    prototype.skinningData.pBoneMatrices.resize(boneCount);
    for (uint32_t boneIdx = 0; boneIdx < boneCount; boneIdx++) {
      prototype.skinningData.pBoneMatrices[boneIdx] = convert::tomat4(extBones->boneTransforms_values[boneIdx]);
    }
    prototype.skinningData.computeHash();
  }

  if (auto extBlend = pnext::find<remixapi_InstanceInfoBlendEXT>(&info)) {
    prototype.materialData.alphaTestEnabled = extBlend->alphaTestEnabled;
    prototype.materialData.alphaTestReferenceValue = extBlend->alphaTestReferenceValue;
    prototype.materialData.alphaTestCompareOp = (VkCompareOp) extBlend->alphaTestCompareOp;
    prototype.materialData.blendMode.enableBlending = extBlend->alphaBlendEnabled;
    prototype.materialData.textureColorOperation = (DxvkRtTextureOperation) extBlend->textureColorOperation;
    prototype.materialData.textureColorArg1Source = (RtTextureArgSource) extBlend->textureColorArg1Source;
    prototype.materialData.textureColorArg2Source = (RtTextureArgSource) extBlend->textureColorArg2Source;
    prototype.materialData.textureAlphaOperation = (DxvkRtTextureOperation) extBlend->textureAlphaOperation;
    prototype.materialData.textureAlphaArg1Source = (RtTextureArgSource) extBlend->textureAlphaArg1Source;
    prototype.materialData.textureAlphaArg2Source = (RtTextureArgSource) extBlend->textureAlphaArg2Source;
    prototype.materialData.tFactor = extBlend->tFactor;
    prototype.materialData.isTextureFactorBlend = extBlend->isTextureFactorBlend;
    prototype.materialData.isVertexColorBakedLighting = (dxvk::g_remixApiVersion >= REMIXAPI_VERSION_MAKE(0, 5, 2)) ? extBlend->isVertexColorBakedLighting : RtxOptions::vertexColorIsBakedLighting();
    prototype.materialData.blendMode.colorSrcFactor = (VkBlendFactor) extBlend->srcColorBlendFactor;
    prototype.materialData.blendMode.colorDstFactor = (VkBlendFactor) extBlend->dstColorBlendFactor;
    prototype.materialData.blendMode.colorBlendOp = (VkBlendOp) extBlend->colorBlendOp;
    prototype.materialData.blendMode.alphaSrcFactor = (VkBlendFactor) extBlend->srcAlphaBlendFactor;
    prototype.materialData.blendMode.alphaDstFactor = (VkBlendFactor) extBlend->dstAlphaBlendFactor;
    prototype.materialData.blendMode.alphaBlendOp = (VkBlendOp) extBlend->alphaBlendOp;
    prototype.materialData.blendMode.writeMask = (VkColorComponentFlags) extBlend->writeMask;
  }

  if (auto extParticles = pnext::find<remixapi_InstanceInfoParticleSystemEXT>(&info)) {
    state->optionalParticleDesc.emplace(convert::toRtParticleDesc(*extParticles));
  }
  if (auto extParticles = pnext::find<remixapi_InstanceInfoParticleSystemLegacyEXT>(&info)) {
    state->optionalParticleDesc.emplace(convert::toRtParticleDesc(*extParticles));
  }

  if (auto extInstancing = pnext::find<remixapi_InstanceInfoGpuInstancingEXT>(&info)) {
    if (extInstancing->instanceTransforms_count > 0 && extInstancing->instanceTransforms_values) {
      auto& gpuInstancingTransforms = state->gpuInstancingTransforms;
      gpuInstancingTransforms.reserve(extInstancing->instanceTransforms_count);
      for (uint32_t i = 0; i < extInstancing->instanceTransforms_count; ++i) {
        gpuInstancingTransforms.push_back(convert::tomat4(extInstancing->instanceTransforms_values[i]));
      }
    }
  }

  state->mesh = info.mesh;
  state->cameraType = cameraType;
  state->categories = convert::toRtCategories(info.categoryFlags);
  state->doubleSided = convert::tobool(info.doubleSided);

  return state;
}
}
