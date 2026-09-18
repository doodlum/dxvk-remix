/*
* Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
* THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
* FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
* DEALINGS IN THE SOFTWARE.
*/

// Remix API entry points for a host that drives D3D11 itself.
//
// The public API in rtx_remix_api.cpp is hosted by the D3D9 device and compiles
// into the shared dxvk library, which is linked into both frontends -- so it
// cannot name a D3D11 symbol without breaking the D3D9 link. Everything here
// lives in the D3D11 DLL instead.
//
// Such a host keeps its own swap chain and asks for the ray-traced image with
// csRemixRender rather than presenting through Remix. It also registers its
// scene once and replays it, instead of re-describing every instance every
// frame; see csRemixDrawRetainedInstances.

#include <atomic>
#include <cmath>
#include <memory>
#include <mutex>
#include <unordered_map>

#include <remix/remix_c.h>

#include "d3d11_device.h"
#include "d3d11_context_imm.h"
#include "d3d11_texture.h"
#include "d3d11_view_srv.h"

#include "../dxvk/rtx_render/rtx_context.h"
#include "../dxvk/rtx_render/rtx_scene_manager.h"
#include "../dxvk/rtx_render/rtx_remix_api_convert.h"
#include "../dxvk/rtx_render/rtx_options.h"

#include "../util/util_math.h"

namespace dxvk {
  // D3D11DeviceContext::EmitCs and RestoreState are protected; this accessor is
  // a friend of it.
  struct D3D11RemixApiAccess {
    template<typename Cmd>
    static void EmitCs(D3D11ImmediateContext* ctx, Cmd&& command) {
      ctx->EmitCs(std::forward<Cmd>(command));
    }

    static void RestoreState(D3D11ImmediateContext* ctx) {
      ctx->RestoreState();
    }
  };
}

namespace {
  dxvk::D3D11Device*           s_device  { nullptr };
  dxvk::D3D11ImmediateContext* s_context { nullptr };
  dxvk::mutex                  s_mutex {};

  dxvk::D3D11ImmediateContext* tryGetContext() {
    return s_context;
  }

  // An API-submitted mesh has no draw call to hash, so the geometry hashes are
  // synthesised. See the comments in UsdMod::Impl::processMesh, rtx_mod_usd.cpp.
  XXH64_hash_t nextGeometryHash() {
    static uint64_t id = UINT64_MAX;
    --id;
    return XXH64(&id, sizeof(id), 0);
  }

  template<typename T>
  size_t sizeInBytes(const T* values, size_t count) {
    return sizeof(T) * count;
  }

  // The vertex a host submits when its normal maps are authored in model space.
  // The nine floats replacing the normal are the columns of the bind-pose to
  // current-pose orientation, which skinning rewrites; for rigid geometry they
  // stay the identity and the sampled normal is already in model space.
  struct ModelNormalVertex {
    float position[3];
    float basis[9];
    float texcoord[2];
    uint32_t color;
  };

  dxvk::Rc<dxvk::DxvkBuffer> allocMeshBuffer(size_t sizeInBytes) {
    if (sizeInBytes == 0) {
      return {};
    }
    auto bufferInfo = dxvk::DxvkBufferCreateInfo {};
    bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT
                     | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
                     | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;
    bufferInfo.stages = VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT
                      | VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR;
    bufferInfo.access = VK_ACCESS_TRANSFER_WRITE_BIT;
    bufferInfo.size = dxvk::align(sizeInBytes, dxvk::CACHE_LINE_SIZE);
    // Device local rather than host visible: every ray hit reads these
    // attributes and indices again, so they belong in video memory. The
    // contents are staged across on the command stream.
    return s_device->GetDXVKDevice()->createBuffer(
      bufferInfo, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
      dxvk::DxvkMemoryStats::Category::RTXBuffer, "Remix API mesh buffer");
  }

  struct MeshUpload {
    dxvk::Rc<dxvk::DxvkBuffer> buffer;
    VkDeviceSize offset;
    std::vector<uint8_t> bytes;
  };

  bool isFinite(const remixapi_Transform& transform) {
    for (uint32_t row = 0; row < 3; ++row) {
      for (uint32_t col = 0; col < 4; ++col) {
        if (!std::isfinite(transform.matrix[row][col])) {
          return false;
        }
      }
    }
    return true;
  }

  remixapi_ErrorCode createMeshInternal(
    const remixapi_MeshInfo* info, remixapi_MeshHandle* outHandle,
    bool modelSpaceNormals, bool preserveVertexNormals = false,
    bool nativeGrass = false, bool nativeLandscape = false) {
    auto* context = tryGetContext();
    if (!context) {
      return REMIXAPI_ERROR_CODE_REMIX_DEVICE_WAS_NOT_REGISTERED;
    }
    if (!outHandle || !info || info->sType != REMIXAPI_STRUCT_TYPE_MESH_INFO) {
      return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
    }
    static_assert(sizeof(remixapi_MeshHandle) == sizeof(info->hash));
    auto handle = reinterpret_cast<remixapi_MeshHandle>(info->hash);
    if (!handle) {
      return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
    }

    std::vector<dxvk::RasterGeometry> allocatedSurfaces;
    std::vector<MeshUpload> uploads;
    auto queueUpload = [&uploads](const dxvk::Rc<dxvk::DxvkBuffer>& buffer, VkDeviceSize offset,
                                  const void* data, size_t size) {
      if (!size) {
        return;
      }
      MeshUpload upload { buffer, offset, std::vector<uint8_t>(size) };
      std::memcpy(upload.bytes.data(), data, size);
      uploads.push_back(std::move(upload));
    };

    for (size_t i = 0; i < info->surfaces_count; i++) {
      const remixapi_MeshInfoSurfaceTriangles& src = info->surfaces_values[i];

      std::vector<ModelNormalVertex> modelVertices;
      if (modelSpaceNormals) {
        modelVertices.resize(src.vertices_count);
        for (size_t v = 0; v < src.vertices_count; ++v) {
          auto& vertex = modelVertices[v];
          std::memcpy(vertex.position, src.vertices_values[v].position, sizeof(vertex.position));
          std::memset(vertex.basis, 0, sizeof(vertex.basis));
          vertex.basis[0] = vertex.basis[4] = vertex.basis[8] = 1.f;
          std::memcpy(vertex.texcoord, src.vertices_values[v].texcoord, sizeof(vertex.texcoord));
          vertex.color = src.vertices_values[v].color;
        }
      }
      const size_t vertexDataSize = modelSpaceNormals
        ? modelVertices.size() * sizeof(ModelNormalVertex)
        : sizeInBytes(src.vertices_values, src.vertices_count);
      const size_t indexDataSize = sizeInBytes(src.indices_values, src.indices_count);

      dxvk::Rc<dxvk::DxvkBuffer> vertexBuffer = allocMeshBuffer(vertexDataSize);
      dxvk::Rc<dxvk::DxvkBuffer> indexBuffer = allocMeshBuffer(indexDataSize);
      dxvk::Rc<dxvk::DxvkBuffer> skinningBuffer = nullptr;

      auto vertexSlice = dxvk::DxvkBufferSlice { vertexBuffer };
      queueUpload(vertexBuffer, 0,
        modelSpaceNormals ? static_cast<const void*>(modelVertices.data())
                          : static_cast<const void*>(src.vertices_values), vertexDataSize);

      auto indexSlice = dxvk::DxvkBufferSlice {};
      if (indexDataSize > 0) {
        indexSlice = dxvk::DxvkBufferSlice { indexBuffer };
        queueUpload(indexBuffer, 0, src.indices_values, indexDataSize);
      }

      auto blendWeightsSlice = dxvk::DxvkBufferSlice {};
      auto blendIndicesSlice = dxvk::DxvkBufferSlice {};
      if (src.skinning_hasvalue) {
        const size_t wordsPerCompressedTuple = dxvk::divCeil(src.skinning_value.bonesPerVertex, 4u);
        const size_t weightsBytes = sizeInBytes(src.skinning_value.blendWeights_values,
                                                src.skinning_value.blendWeights_count);
        const size_t indicesBytes = src.vertices_count * wordsPerCompressedTuple * sizeof(uint32_t);

        skinningBuffer = allocMeshBuffer(weightsBytes + indicesBytes);

        std::vector<uint32_t> compressedBlendIndices(src.vertices_count * wordsPerCompressedTuple);
        for (size_t vert = 0; vert < src.vertices_count; vert++) {
          uint32_t* dstCompressed = &compressedBlendIndices[vert * wordsPerCompressedTuple];
          const uint32_t* blendIndicesStorage =
            &src.skinning_value.blendIndices_values[vert * src.skinning_value.bonesPerVertex];

          for (int j = 0; j < src.skinning_value.bonesPerVertex; j += 4) {
            uint32_t vertIndices = 0;
            for (int k = 0; k < 4 && j + k < src.skinning_value.bonesPerVertex; ++k) {
              vertIndices |= blendIndicesStorage[j + k] << 8 * k;
            }
            dstCompressed[j / 4] = vertIndices;
          }
        }

        blendWeightsSlice = dxvk::DxvkBufferSlice { skinningBuffer, 0, weightsBytes };
        blendIndicesSlice = dxvk::DxvkBufferSlice { skinningBuffer, weightsBytes, indicesBytes };

        queueUpload(skinningBuffer, 0, src.skinning_value.blendWeights_values, weightsBytes);
        queueUpload(skinningBuffer, weightsBytes, compressedBlendIndices.data(), indicesBytes);
      }

      auto dst = dxvk::RasterGeometry {};
      dst.externalMaterial = src.material;
      dst.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
      dst.cullMode = VK_CULL_MODE_NONE; // overwritten by the instance info at draw time
      dst.frontFace = VK_FRONT_FACE_CLOCKWISE;
      dst.vertexCount = src.vertices_count;
      assert(src.vertices_count < std::numeric_limits<uint32_t>::max());

      // A merged bottom-level structure needs the extents, and nothing else
      // computes them for a mesh that never went through a draw call.
      for (size_t vertex = 0; vertex < src.vertices_count; ++vertex) {
        for (uint32_t axis = 0; axis < 3; ++axis) {
          const float coordinate = src.vertices_values[vertex].position[axis];
          dst.boundingBox.minPos[axis] = std::min(dst.boundingBox.minPos[axis], coordinate);
          dst.boundingBox.maxPos[axis] = std::max(dst.boundingBox.maxPos[axis], coordinate);
        }
      }

      if (modelSpaceNormals) {
        dst.positionBuffer = dxvk::RasterBuffer { vertexSlice, offsetof(ModelNormalVertex, position), sizeof(ModelNormalVertex), VK_FORMAT_R32G32B32_SFLOAT };
        dst.normalBuffer   = dxvk::RasterBuffer { vertexSlice, offsetof(ModelNormalVertex, basis),    sizeof(ModelNormalVertex), VK_FORMAT_R32G32B32_SFLOAT };
        dst.texcoordBuffer = dxvk::RasterBuffer { vertexSlice, offsetof(ModelNormalVertex, texcoord), sizeof(ModelNormalVertex), VK_FORMAT_R32G32_SFLOAT };
        dst.color0Buffer   = dxvk::RasterBuffer { vertexSlice, offsetof(ModelNormalVertex, color),    sizeof(ModelNormalVertex), VK_FORMAT_B8G8R8A8_UNORM };
      } else {
        dst.positionBuffer = dxvk::RasterBuffer { vertexSlice, offsetof(remixapi_HardcodedVertex, position), sizeof(remixapi_HardcodedVertex), VK_FORMAT_R32G32B32_SFLOAT };
        dst.normalBuffer   = dxvk::RasterBuffer { vertexSlice, offsetof(remixapi_HardcodedVertex, normal),   sizeof(remixapi_HardcodedVertex), VK_FORMAT_R32G32B32_SFLOAT };
        dst.texcoordBuffer = dxvk::RasterBuffer { vertexSlice, offsetof(remixapi_HardcodedVertex, texcoord), sizeof(remixapi_HardcodedVertex), VK_FORMAT_R32G32_SFLOAT };
        dst.color0Buffer   = dxvk::RasterBuffer { vertexSlice, offsetof(remixapi_HardcodedVertex, color),    sizeof(remixapi_HardcodedVertex), VK_FORMAT_B8G8R8A8_UNORM };
      }

      dst.modelSpaceNormals = modelSpaceNormals;
      dst.preserveVertexNormals = preserveVertexNormals;
      dst.nativeGrass = nativeGrass;
      dst.nativeLandscape = nativeLandscape;
      if (nativeGrass) {
        // The host puts a per-vertex wind weight in the vertex colour's alpha.
        for (size_t vertex = 0; vertex < src.vertices_count; ++vertex) {
          dst.nativeGrassHasWind |= (src.vertices_values[vertex].color >> 24) != 0;
        }
      }

      if (src.skinning_hasvalue) {
        dst.numBonesPerVertex = src.skinning_value.bonesPerVertex;
        // One weight per bone per vertex, so the stride spans the whole tuple.
        dst.blendWeightBuffer = dxvk::RasterBuffer { blendWeightsSlice, 0, uint32_t(sizeof(float)) * src.skinning_value.bonesPerVertex, VK_FORMAT_R32_SFLOAT };
        dst.blendIndicesBuffer = dxvk::RasterBuffer { blendIndicesSlice, 0, sizeof(uint32_t), VK_FORMAT_R8G8B8A8_USCALED };
      }

      dst.indexCount = src.indices_count;
      static_assert(sizeof(src.indices_values[0]) == 4);
      dst.indexBuffer = dxvk::RasterBuffer { indexSlice, 0, sizeof(uint32_t), VK_INDEX_TYPE_UINT32 };
      dst.hashes[dxvk::HashComponents::Indices] = dst.hashes[dxvk::HashComponents::VertexPosition] = nextGeometryHash();
      dst.hashes[dxvk::HashComponents::VertexTexcoord] = nextGeometryHash();
      dst.hashes[dxvk::HashComponents::GeometryDescriptor] = nextGeometryHash();
      dst.hashes[dxvk::HashComponents::VertexLayout] = nextGeometryHash();
      dst.hashes.precombine();

      allocatedSurfaces.push_back(std::move(dst));
    }

    std::lock_guard lock { s_mutex };
    dxvk::D3D11RemixApiAccess::EmitCs(context,
      [cHandle = handle, cSurfaces = std::move(allocatedSurfaces), cUploads = std::move(uploads)](dxvk::DxvkContext* ctx) mutable {
        // Staged here so the transfers are ordered before any structure build
        // or shader read of the same buffers.
        for (const auto& upload : cUploads) {
          ctx->writeToBuffer(upload.buffer, upload.offset, upload.bytes.size(), upload.bytes.data());
        }
        auto& assets = ctx->getCommonObjects()->getSceneManager().getAssetReplacer();
        assets->registerExternalMesh(cHandle, std::move(cSurfaces));
      });

    *outHandle = handle;
    return REMIXAPI_ERROR_CODE_SUCCESS;
  }

  // A texture the host already has on this device becomes a material without a
  // readback or a file: the view names a Vulkan image, and TextureRef holds it
  // for the material's lifetime.
  //
  // An API-only host does not run the legacy texture-hashing path, so an
  // imported image can arrive with no hash. MaterialData hashes its textures by
  // image hash, and leaving it zero would alias unrelated native textures in
  // the surface-material cache, so one is derived from the image handle.
  dxvk::Rc<dxvk::DxvkImageView> importedView(ID3D11ShaderResourceView* view) {
    if (!view) {
      return {};
    }
    auto image = static_cast<dxvk::D3D11ShaderResourceView*>(view)->GetImageView();
    if (image == nullptr) {
      return {};
    }
    if (image->image()->getHash() == 0) {
      const auto nativeHandle = image->image()->handle();
      image->image()->setHash(XXH64(&nativeHandle, sizeof(nativeHandle), 0x435352454d4958ull));
    }
    return image;
  }

  // Placement snapshots the host publishes once and then names by handle. A
  // destroyed handle cannot be submitted again, but a draw already queued keeps
  // its snapshot alive through the shared pointer it captured.
  std::unordered_map<uint64_t, std::shared_ptr<dxvk::NativeInstanceSet>> s_instanceSets;
  uint64_t s_nextInstanceSet = 0;

  remixapi_ErrorCode createInstanceSet(
    const remixapi_Transform* transforms, const float* brightness, const float* phases,
    uint32_t count, uint64_t* outHandle) {
    if (!outHandle) {
      return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
    }
    *outHandle = 0;
    if (!tryGetContext()) {
      return REMIXAPI_ERROR_CODE_REMIX_DEVICE_WAS_NOT_REGISTERED;
    }
    // An upper bound so a corrupt count cannot ask for an unbounded allocation.
    if (!transforms || !count || count > 1000000) {
      return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
    }

    auto set = std::make_shared<dxvk::NativeInstanceSet>();
    set->hasGrassPhases = phases != nullptr;
    set->transforms.reserve(count);
    set->records.reserve(count);

    for (uint32_t i = 0; i < count; ++i) {
      if (!isFinite(transforms[i])) {
        return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
      }
      dxvk::NativeInstanceRecord record {};
      record.transform = convert::tomat4(transforms[i]);
      if (phases) {
        if (!std::isfinite(phases[i])) {
          return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
        }
        std::memcpy(&record.padding[0], phases + i, sizeof(float));
      }
      if (brightness) {
        if (!std::isfinite(brightness[i]) || brightness[i] < 0.f || brightness[i] > 1.f) {
          return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
        }
        const uint32_t channel = uint32_t(std::lround(brightness[i] * 255.f));
        record.tFactor = 0xff000000u | channel * 0x010101u;
        record.applyColor = 1;
      }
      set->transforms.push_back(record.transform);
      set->records.push_back(record);
      // Order matters: it is the expanded topology and the frame-to-frame
      // correspondence, not just the contents.
      set->contentHash = XXH64(&record, sizeof(record), set->contentHash ^ 0x9E3779B97F4A7C15ull);
    }

    std::lock_guard lock { s_mutex };
    const uint64_t handle = ++s_nextInstanceSet;
    s_instanceSets.emplace(handle, std::move(set));
    *outHandle = handle;
    return REMIXAPI_ERROR_CODE_SUCCESS;
  }

  // Points a draw state at a published snapshot. The placements become the
  // draw's instancing transforms, which is the same path a USD point instancer
  // takes, and the snapshot rides along so the surface can read per-placement
  // colour out of it.
  bool attachInstanceSet(dxvk::ExternalDrawState& state, const std::shared_ptr<dxvk::NativeInstanceSet>& set) {
    if (!set) {
      return false;
    }
    state.gpuInstancingTransforms = set->transforms;
    state.drawCall.modifyTransformData().nativeInstanceSet = set;
    return true;
  }

  std::atomic<uint64_t> s_nextRetainedInstance { 0 };

  remixapi_ErrorCode setRetainedInstance(const remixapi_InstanceInfo* info, uint64_t handle, uint64_t setHandle) {
    auto* context = tryGetContext();
    if (!context) {
      return REMIXAPI_ERROR_CODE_REMIX_DEVICE_WAS_NOT_REGISTERED;
    }
    if (!info || info->sType != REMIXAPI_STRUCT_TYPE_INSTANCE_INFO || !info->mesh || !handle) {
      return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
    }
    // Converted outside the lock: this is the expensive half, and it does not
    // touch anything shared.
    auto state = dxvk::RemixAPIPrivateAccessor::toRtDrawState(*info);
    if (!state) {
      return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
    }

    std::lock_guard lock { s_mutex };
    if (setHandle) {
      const auto found = s_instanceSets.find(setHandle);
      if (found == s_instanceSets.end() || !attachInstanceSet(*state, found->second)) {
        return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
      }
    }
    dxvk::D3D11RemixApiAccess::EmitCs(context,
      [cHandle = handle, cState = std::move(state)](dxvk::DxvkContext* dxvkCtx) mutable {
        static_cast<dxvk::RtxContext*>(dxvkCtx)->setRetainedExternalGeometry(cHandle, std::move(*cState));
      });
    return REMIXAPI_ERROR_CODE_SUCCESS;
  }

  // The interface the host receives from remixapi_InitializeLibrary. These are
  // the D3D9 host's entry points with its device swapped for the D3D11
  // immediate context; everything they actually convert is shared with it
  // through rtx_remix_api_convert.h.

  remixapi_ErrorCode REMIXAPI_CALL d3d11_CreateMesh(const remixapi_MeshInfo* info, remixapi_MeshHandle* outHandle) {
    return createMeshInternal(info, outHandle, false);
  }

  remixapi_ErrorCode REMIXAPI_CALL d3d11_DestroyMesh(remixapi_MeshHandle handle) {
    auto* context = tryGetContext();
    if (!context) {
      return REMIXAPI_ERROR_CODE_REMIX_DEVICE_WAS_NOT_REGISTERED;
    }
    std::lock_guard lock { s_mutex };
    dxvk::D3D11RemixApiAccess::EmitCs(context, [cHandle = handle](dxvk::DxvkContext* ctx) {
      ctx->getCommonObjects()->getSceneManager().destroyExternalMesh(dxvk::Rc<dxvk::DxvkContext>(ctx), cHandle);
    });
    return REMIXAPI_ERROR_CODE_SUCCESS;
  }

  remixapi_ErrorCode REMIXAPI_CALL d3d11_CreateMaterial(
    const remixapi_MaterialInfo* info, remixapi_MaterialHandle* outHandle) {
    auto* context = tryGetContext();
    if (!context) {
      return REMIXAPI_ERROR_CODE_REMIX_DEVICE_WAS_NOT_REGISTERED;
    }
    if (!outHandle || !info || info->sType != REMIXAPI_STRUCT_TYPE_MATERIAL_INFO) {
      return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
    }
    static_assert(sizeof(remixapi_MaterialHandle) == sizeof(info->hash));
    auto handle = reinterpret_cast<remixapi_MaterialHandle>(info->hash);
    if (!handle) {
      return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
    }
    std::lock_guard lock { s_mutex };
    dxvk::D3D11RemixApiAccess::EmitCs(context,
      [cHandle = handle,
       cMaterialData = convert::toRtMaterialWithoutTexturePreload(*info),
       cPreloadSrc = convert::makePreloadSource(*info)](dxvk::DxvkContext* ctx) {
        auto& assets = ctx->getCommonObjects()->getSceneManager().getAssetReplacer();
        assets->makeMaterialWithTexturePreload(*ctx, cHandle,
          convert::toRtMaterialFinalized(*ctx, cMaterialData, cPreloadSrc));
      });
    *outHandle = handle;
    return REMIXAPI_ERROR_CODE_SUCCESS;
  }

  remixapi_ErrorCode REMIXAPI_CALL d3d11_DestroyMaterial(remixapi_MaterialHandle handle) {
    auto* context = tryGetContext();
    if (!context) {
      return REMIXAPI_ERROR_CODE_REMIX_DEVICE_WAS_NOT_REGISTERED;
    }
    std::lock_guard lock { s_mutex };
    dxvk::D3D11RemixApiAccess::EmitCs(context, [cHandle = handle](dxvk::DxvkContext* ctx) {
      ctx->getCommonObjects()->getSceneManager().getAssetReplacer()->destroyExternalMaterial(cHandle);
    });
    return REMIXAPI_ERROR_CODE_SUCCESS;
  }

  remixapi_ErrorCode REMIXAPI_CALL d3d11_SetupCamera(const remixapi_CameraInfo* info) {
    auto* context = tryGetContext();
    if (!context) {
      return REMIXAPI_ERROR_CODE_REMIX_DEVICE_WAS_NOT_REGISTERED;
    }
    if (!info || info->sType != REMIXAPI_STRUCT_TYPE_CAMERA_INFO) {
      return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
    }
    std::lock_guard lock { s_mutex };
    // The host's projection must reach the renderer unaltered, so the depth it
    // reads back reprojects to the world positions it expects.
    if (dxvk::RtxOptions::enableNearPlaneOverride()) {
      const_cast<bool&>(dxvk::RtxOptions::enableNearPlaneOverride()) = false;
    }
    dxvk::D3D11RemixApiAccess::EmitCs(context,
      [cRtCamera = convert::toRtCamera(*info)](dxvk::DxvkContext* ctx) {
        ctx->getCommonObjects()->getSceneManager().getCameraManager()
          .processExternalCamera(cRtCamera.type, cRtCamera.worldToView, cRtCamera.viewToProjection);
      });
    return REMIXAPI_ERROR_CODE_SUCCESS;
  }

  remixapi_ErrorCode REMIXAPI_CALL d3d11_DrawInstance(const remixapi_InstanceInfo* info) {
    auto* context = tryGetContext();
    if (!context) {
      return REMIXAPI_ERROR_CODE_REMIX_DEVICE_WAS_NOT_REGISTERED;
    }
    if (!info || info->sType != REMIXAPI_STRUCT_TYPE_INSTANCE_INFO) {
      return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
    }
    auto drawState = dxvk::RemixAPIPrivateAccessor::toRtDrawState(*info);
    std::lock_guard lock { s_mutex };
    dxvk::D3D11RemixApiAccess::EmitCs(context, [cState = std::move(drawState)](dxvk::DxvkContext* ctx) mutable {
      static_cast<dxvk::RtxContext*>(ctx)->commitExternalGeometryToRT(std::move(cState));
    });
    return REMIXAPI_ERROR_CODE_SUCCESS;
  }

  remixapi_ErrorCode REMIXAPI_CALL d3d11_CreateLight(
    const remixapi_LightInfo* info, remixapi_LightHandle* outHandle) {
    auto* context = tryGetContext();
    if (!context) {
      return REMIXAPI_ERROR_CODE_REMIX_DEVICE_WAS_NOT_REGISTERED;
    }
    if (!outHandle || !info || info->sType != REMIXAPI_STRUCT_TYPE_LIGHT_INFO) {
      return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
    }
    static_assert(sizeof(remixapi_LightHandle) == sizeof(info->hash));
    auto handle = reinterpret_cast<remixapi_LightHandle>(info->hash);
    if (!handle) {
      return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
    }
    std::lock_guard lock { s_mutex };
    if (auto src = pnext::find<remixapi_LightInfoDomeEXT>(info)) {
      dxvk::D3D11RemixApiAccess::EmitCs(context,
        [cHandle = handle, cRadiance = convert::tovec3(info->radiance),
         cTransform = convert::tomat4(src->transform),
         cTexturePath = convert::topath(src->colorTexture)](dxvk::DxvkContext* ctx) {
          dxvk::DomeLight domeLight;
          domeLight.radiance = cRadiance;
          domeLight.worldToLight = inverse(cTransform);
          if (!cTexturePath.empty()) {
            auto assetData = dxvk::AssetDataManager::get().findAsset(cTexturePath.string().c_str());
            if (assetData != nullptr) {
              domeLight.texture = dxvk::TextureRef {
                ctx->getCommonObjects()->getTextureManager()
                  .preloadTextureAsset(assetData, dxvk::ColorSpace::AUTO, true) };
            }
          }
          // Keeps the texture resident.
          uint32_t unused;
          ctx->getCommonObjects()->getSceneManager().trackTexture(domeLight.texture, unused, true, true);
          ctx->getCommonObjects()->getSceneManager().getLightManager().addExternalDomeLight(cHandle, domeLight);
        });
    } else {
      const auto rtLight = convert::toRtLight(*info);
      // An empty optional means the description named no light this API knows.
      if (!rtLight.has_value()) {
        return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
      }
      dxvk::D3D11RemixApiAccess::EmitCs(context,
        [cHandle = handle, cRtLight = *rtLight](dxvk::DxvkContext* ctx) {
          ctx->getCommonObjects()->getSceneManager().getLightManager().addExternalLight(cHandle, cRtLight);
        });
    }
    *outHandle = handle;
    return REMIXAPI_ERROR_CODE_SUCCESS;
  }

  remixapi_ErrorCode REMIXAPI_CALL d3d11_DestroyLight(remixapi_LightHandle handle) {
    auto* context = tryGetContext();
    if (!context) {
      return REMIXAPI_ERROR_CODE_REMIX_DEVICE_WAS_NOT_REGISTERED;
    }
    std::lock_guard lock { s_mutex };
    dxvk::D3D11RemixApiAccess::EmitCs(context, [cHandle = handle](dxvk::DxvkContext* ctx) {
      ctx->getCommonObjects()->getSceneManager().getLightManager().removeExternalLight(cHandle);
    });
    return REMIXAPI_ERROR_CODE_SUCCESS;
  }

  remixapi_ErrorCode REMIXAPI_CALL d3d11_DrawLightInstance(remixapi_LightHandle handle) {
    auto* context = tryGetContext();
    if (!context) {
      return REMIXAPI_ERROR_CODE_REMIX_DEVICE_WAS_NOT_REGISTERED;
    }
    if (!handle) {
      return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
    }
    std::lock_guard lock { s_mutex };
    dxvk::D3D11RemixApiAccess::EmitCs(context, [handle](dxvk::DxvkContext* ctx) {
      ctx->getCommonObjects()->getSceneManager().getLightManager().addExternalLightInstance(handle);
    });
    return REMIXAPI_ERROR_CODE_SUCCESS;
  }

  remixapi_ErrorCode REMIXAPI_CALL d3d11_SetConfigVariable(const char* key, const char* value) {
    if (!key || key[0] == 0 || !value) {
      return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
    }
    std::lock_guard lock { s_mutex };
    std::string strKey { key };
    dxvk::RtxOptionImpl* option = dxvk::RtxOptionImpl::getOptionByFullName(strKey);
    if (!option) {
      return REMIXAPI_ERROR_CODE_GENERAL_FAILURE;
    }
    dxvk::Config newSetting;
    newSetting.setOptionMove(std::move(strKey), std::string { value });
    option->readOption(newSetting, dxvk::RtxOptionLayer::getUserLayer());
    return REMIXAPI_ERROR_CODE_SUCCESS;
  }

  remixapi_ErrorCode REMIXAPI_CALL d3d11_Shutdown() {
    std::lock_guard lock { s_mutex };
    s_device = nullptr;
    s_context = nullptr;
    return REMIXAPI_ERROR_CODE_SUCCESS;
  }
}

extern "C" {

  REMIXAPI remixapi_ErrorCode REMIXAPI_CALL remixapi_dxvk_RegisterD3D11Device(ID3D11Device* d3d11Device);


  // Hands the host the interface it drives the scene through. This is the D3D11
  // DLL's own, distinct from the D3D9 host's function of the same name in the
  // shared library: a host resolves it from whichever DLL it loaded.
  REMIXAPI remixapi_ErrorCode REMIXAPI_CALL remixapi_InitializeLibrary(
    const remixapi_InitializeLibraryInfo* info, remixapi_Interface* outResult) {
    if (!info || info->sType != REMIXAPI_STRUCT_TYPE_INITIALIZE_LIBRARY_INFO) {
      return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
    }
    if (!outResult) {
      return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
    }
    if (info->version < REMIXAPI_VERSION_MAKE(REMIXAPI_VERSION_MAJOR, 0, 0)) {
      return REMIXAPI_ERROR_CODE_INCOMPATIBLE_VERSION;
    }
    dxvk::g_remixApiVersion = info->version;

    remixapi_Interface interf {};
    interf.Shutdown = d3d11_Shutdown;
    interf.CreateMaterial = d3d11_CreateMaterial;
    interf.DestroyMaterial = d3d11_DestroyMaterial;
    interf.CreateMesh = d3d11_CreateMesh;
    interf.DestroyMesh = d3d11_DestroyMesh;
    interf.SetupCamera = d3d11_SetupCamera;
    interf.DrawInstance = d3d11_DrawInstance;
    interf.CreateLight = d3d11_CreateLight;
    interf.DestroyLight = d3d11_DestroyLight;
    interf.DrawLightInstance = d3d11_DrawLightInstance;
    interf.SetConfigVariable = d3d11_SetConfigVariable;
    interf.dxvk_RegisterD3D11Device = remixapi_dxvk_RegisterD3D11Device;
    // The remaining members stay null: creating a D3D9 device, presenting
    // through Remix and the object-picking helpers all belong to the D3D9 host,
    // and this one owns its swap chain and composites with csRemixRender.

    *outResult = interf;
    return REMIXAPI_ERROR_CODE_SUCCESS;
  }

  // A host that drives D3D11 registers its device here instead of calling
  // remixapi_Startup, which creates and owns a D3D9 device.
  REMIXAPI remixapi_ErrorCode REMIXAPI_CALL remixapi_dxvk_RegisterD3D11Device(ID3D11Device* d3d11Device) {
    if (!d3d11Device) {
      return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
    }
    auto* device = dynamic_cast<dxvk::D3D11Device*>(d3d11Device);
    if (!device) {
      return REMIXAPI_ERROR_CODE_REGISTERING_NON_REMIX_D3D9_DEVICE;
    }
    ID3D11DeviceContext* context = nullptr;
    device->GetImmediateContext(&context);
    if (!context) {
      return REMIXAPI_ERROR_CODE_GENERAL_FAILURE;
    }
    // GetImmediateContext returns a reference which is deliberately not released:
    // the registration lasts for the life of the process and every entry point
    // below dereferences the context.
    std::lock_guard lock { s_mutex };
    // A host that drives D3D11 is built against these headers rather than
    // negotiating a version through InitializeLibrary, so the conversion may
    // read every field the current API defines.
    dxvk::g_remixApiVersion = REMIXAPI_VERSION_MAKE(
      REMIXAPI_VERSION_MAJOR, REMIXAPI_VERSION_MINOR, REMIXAPI_VERSION_PATCH);
    s_device = device;
    s_context = static_cast<dxvk::D3D11ImmediateContext*>(context);
    return REMIXAPI_ERROR_CODE_SUCCESS;
  }

  // Skyrim authors its normal maps in model space, so the host asks for a mesh
  // whose normal attribute is a basis rather than a normal.
  REMIXAPI remixapi_ErrorCode REMIXAPI_CALL csRemixCreateMeshMSN(
    const remixapi_MeshInfo* info, remixapi_MeshHandle* outHandle) {
    return createMeshInternal(info, outHandle, true);
  }

  // Bit 0 model-space basis, bit 1 keep the supplied shading direction on both
  // faces, bit 2 expanded grass, bit 3 five-layer terrain. Grass is expanded
  // from one unskinned prototype, and terrain carries its own blend weights, so
  // neither combines with the others.
  REMIXAPI remixapi_ErrorCode REMIXAPI_CALL csRemixCreateMeshNative(
    const remixapi_MeshInfo* info, uint32_t flags, remixapi_MeshHandle* outHandle) {
    const bool grass = (flags & 4u) != 0;
    const bool landscape = (flags & 8u) != 0;
    if ((flags & ~15u)
        || (landscape && (flags & 7u))
        || (grass && ((flags & 3u) != 2u || !info || !info->surfaces_values
                      || info->surfaces_count != 1 || info->surfaces_values[0].skinning_hasvalue))) {
      return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
    }
    return createMeshInternal(info, outHandle, (flags & 1u) != 0, (flags & 2u) != 0, grass, landscape);
  }

  // Registers a draw that persists until the host destroys it. A host
  // describing a large world would otherwise re-send every instance every
  // frame, taking the lock, converting a draw state and queueing a
  // command-stream lambda thousands of times on its own thread.
  REMIXAPI remixapi_ErrorCode REMIXAPI_CALL csRemixCreateRetainedInstance(
    const remixapi_InstanceInfo* info, uint64_t* outHandle) {
    if (!outHandle) {
      return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
    }
    const uint64_t handle = ++s_nextRetainedInstance;
    const auto result = setRetainedInstance(info, handle, 0);
    if (result == REMIXAPI_ERROR_CODE_SUCCESS) {
      *outHandle = handle;
    }
    return result;
  }

  // As above, but every placement in the named set is drawn from this one
  // registration.
  REMIXAPI remixapi_ErrorCode REMIXAPI_CALL csRemixCreateRetainedInstanceSet(
    const remixapi_InstanceInfo* info, uint64_t setHandle, uint64_t* outHandle) {
    if (!outHandle || !setHandle) {
      return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
    }
    const uint64_t handle = ++s_nextRetainedInstance;
    const auto result = setRetainedInstance(info, handle, setHandle);
    if (result == REMIXAPI_ERROR_CODE_SUCCESS) {
      *outHandle = handle;
    }
    return result;
  }

  // Replaces a registered draw's whole description, for a genuine change of
  // mesh, material or category. setHandle may be zero for a single placement.
  REMIXAPI remixapi_ErrorCode REMIXAPI_CALL csRemixUpdateRetainedInstance(
    const remixapi_InstanceInfo* info, uint64_t handle, uint64_t setHandle) {
    return setRetainedInstance(info, handle, setHandle);
  }

  // Builds a material from textures the host already holds on this device.
  // textureAddressMode bit 1 repeats U, bit 0 repeats V; anything else clamps.
  REMIXAPI remixapi_ErrorCode REMIXAPI_CALL csRemixCreateMaterialD3D11V3(
    const remixapi_MaterialInfo* info, ID3D11ShaderResourceView* albedo, ID3D11ShaderResourceView* normal,
    uint32_t textureAddressMode, remixapi_MaterialHandle* outHandle) {
    auto* context = tryGetContext();
    if (!context) {
      return REMIXAPI_ERROR_CODE_REMIX_DEVICE_WAS_NOT_REGISTERED;
    }
    if (!info || !outHandle || !albedo || !info->hash
        || info->sType != REMIXAPI_STRUCT_TYPE_MATERIAL_INFO
        || !pnext::find<remixapi_MaterialInfoOpaqueEXT>(info)) {
      return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
    }
    auto albedoView = importedView(albedo);
    if (albedoView == nullptr) {
      return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
    }

    std::lock_guard lock { s_mutex };
    auto material = convert::toRtMaterialWithoutTexturePreload(*info);
    material.getOpaqueMaterialData().setAlbedoOpacityTexture(dxvk::TextureRef { albedoView });
    if (normal) {
      auto normalView = importedView(normal);
      if (normalView == nullptr) {
        return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
      }
      material.getOpaqueMaterialData().setNormalTexture(dxvk::TextureRef { normalView });
    }

    dxvk::DxvkSamplerCreateInfo sampler {};
    sampler.magFilter = sampler.minFilter = VK_FILTER_LINEAR;
    sampler.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    sampler.mipmapLodMax = VK_LOD_CLAMP_NONE;
    sampler.useAnisotropy = VK_TRUE;
    sampler.maxAnisotropy = 8.0f;
    sampler.addressModeU = (textureAddressMode & 2) ? VK_SAMPLER_ADDRESS_MODE_REPEAT : VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sampler.addressModeV = (textureAddressMode & 1) ? VK_SAMPLER_ADDRESS_MODE_REPEAT : VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sampler.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    material.getOpaqueMaterialData().setSamplerOverride(s_device->GetDXVKDevice()->createSampler(sampler));

    auto handle = reinterpret_cast<remixapi_MaterialHandle>(info->hash);
    dxvk::D3D11RemixApiAccess::EmitCs(context,
      [handle, data = std::move(material)](dxvk::DxvkContext* ctx) mutable {
        ctx->getCommonObjects()->getSceneManager().getAssetReplacer()
          ->makeMaterialWithTexturePreload(*ctx, handle, std::move(data));
      });
    *outHandle = handle;
    return REMIXAPI_ERROR_CODE_SUCCESS;
  }

  REMIXAPI remixapi_ErrorCode REMIXAPI_CALL csRemixCreateInstanceSet(
    const remixapi_Transform* transforms, const float* brightness, uint32_t count, uint64_t* outHandle) {
    return createInstanceSet(transforms, brightness, nullptr, count, outHandle);
  }

  // Grass additionally carries a per-placement phase, which the host uses to
  // keep neighbouring blades from moving in lockstep.
  REMIXAPI remixapi_ErrorCode REMIXAPI_CALL csRemixCreateGrassInstanceSet(
    const remixapi_Transform* transforms, const float* brightness, const float* phases,
    uint32_t count, uint64_t* outHandle) {
    if (!phases || !brightness) {
      return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
    }
    return createInstanceSet(transforms, brightness, phases, count, outHandle);
  }

  REMIXAPI remixapi_ErrorCode REMIXAPI_CALL csRemixDestroyInstanceSet(uint64_t handle) {
    std::lock_guard lock { s_mutex };
    return s_instanceSets.erase(handle) ? REMIXAPI_ERROR_CODE_SUCCESS : REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
  }

  // Draws every placement in a set once, without retaining the draw.
  REMIXAPI remixapi_ErrorCode REMIXAPI_CALL csRemixDrawInstanceSet(
    const remixapi_InstanceInfo* info, uint64_t handle) {
    auto* context = tryGetContext();
    if (!context) {
      return REMIXAPI_ERROR_CODE_REMIX_DEVICE_WAS_NOT_REGISTERED;
    }
    // The set is the instancing description, so the caller may not also supply
    // the public one.
    if (!info || info->sType != REMIXAPI_STRUCT_TYPE_INSTANCE_INFO || !info->mesh
        || pnext::find<remixapi_InstanceInfoGpuInstancingEXT>(info)) {
      return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
    }
    auto state = dxvk::RemixAPIPrivateAccessor::toRtDrawState(*info);
    if (!state) {
      return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
    }
    std::lock_guard lock { s_mutex };
    const auto found = s_instanceSets.find(handle);
    if (found == s_instanceSets.end() || !attachInstanceSet(*state, found->second)) {
      return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
    }
    dxvk::D3D11RemixApiAccess::EmitCs(context, [cState = std::move(state)](dxvk::DxvkContext* ctx) mutable {
      static_cast<dxvk::RtxContext*>(ctx)->commitExternalGeometryToRT(std::move(cState));
    });
    return REMIXAPI_ERROR_CODE_SUCCESS;
  }

  // As above for a grass set. worldWindAndTimer is the wind direction and the
  // clock the host animates against.
  REMIXAPI remixapi_ErrorCode REMIXAPI_CALL csRemixDrawGrassInstanceSet(
    const remixapi_InstanceInfo* info, uint64_t handle, const float* worldWindAndTimer) {
    auto* context = tryGetContext();
    if (!context) {
      return REMIXAPI_ERROR_CODE_REMIX_DEVICE_WAS_NOT_REGISTERED;
    }
    if (!info || !worldWindAndTimer || info->sType != REMIXAPI_STRUCT_TYPE_INSTANCE_INFO || !info->mesh
        || pnext::find<remixapi_InstanceInfoGpuInstancingEXT>(info)) {
      return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
    }
    for (uint32_t i = 0; i < 4; ++i) {
      if (!std::isfinite(worldWindAndTimer[i])) {
        return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
      }
    }
    auto state = dxvk::RemixAPIPrivateAccessor::toRtDrawState(*info);
    if (!state) {
      return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
    }
    std::lock_guard lock { s_mutex };
    const auto found = s_instanceSets.find(handle);
    if (found == s_instanceSets.end() || !found->second->hasGrassPhases
        || !attachInstanceSet(*state, found->second)) {
      return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
    }
    dxvk::D3D11RemixApiAccess::EmitCs(context, [cState = std::move(state)](dxvk::DxvkContext* ctx) mutable {
      static_cast<dxvk::RtxContext*>(ctx)->commitExternalGeometryToRT(std::move(cState));
    });
    return REMIXAPI_ERROR_CODE_SUCCESS;
  }

  // The common case: only the placement moved, so the registered description
  // still stands and nothing has to be converted again.
  REMIXAPI remixapi_ErrorCode REMIXAPI_CALL csRemixUpdateRetainedInstanceTransform(
    uint64_t handle, const remixapi_Transform* transform) {
    auto* context = tryGetContext();
    if (!context) {
      return REMIXAPI_ERROR_CODE_REMIX_DEVICE_WAS_NOT_REGISTERED;
    }
    if (!handle || !transform || !isFinite(*transform)) {
      return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
    }
    std::lock_guard lock { s_mutex };
    dxvk::D3D11RemixApiAccess::EmitCs(context,
      [cHandle = handle, cTransform = convert::tomat4(*transform)](dxvk::DxvkContext* dxvkCtx) {
        static_cast<dxvk::RtxContext*>(dxvkCtx)->setRetainedExternalGeometryTransform(cHandle, cTransform);
      });
    return REMIXAPI_ERROR_CODE_SUCCESS;
  }

  REMIXAPI remixapi_ErrorCode REMIXAPI_CALL csRemixDestroyRetainedInstance(uint64_t handle) {
    auto* context = tryGetContext();
    if (!context) {
      return REMIXAPI_ERROR_CODE_REMIX_DEVICE_WAS_NOT_REGISTERED;
    }
    if (!handle) {
      return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
    }
    std::lock_guard lock { s_mutex };
    dxvk::D3D11RemixApiAccess::EmitCs(context, [cHandle = handle](dxvk::DxvkContext* dxvkCtx) {
      static_cast<dxvk::RtxContext*>(dxvkCtx)->removeRetainedExternalGeometry(cHandle);
    });
    return REMIXAPI_ERROR_CODE_SUCCESS;
  }

  // Replays the whole retained table: one call per frame in place of one per
  // instance. sceneOrigin is where the host has placed the origin of the
  // absolute coordinates it registers, so the scene can be kept near the origin
  // as the camera travels without the host rewriting every transform.
  REMIXAPI remixapi_ErrorCode REMIXAPI_CALL csRemixDrawRetainedInstances(const float* sceneOrigin) {
    auto* context = tryGetContext();
    if (!context) {
      return REMIXAPI_ERROR_CODE_REMIX_DEVICE_WAS_NOT_REGISTERED;
    }
    if (!sceneOrigin) {
      return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
    }
    for (uint32_t i = 0; i < 3; ++i) {
      if (!std::isfinite(sceneOrigin[i])) {
        return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
      }
    }
    std::lock_guard lock { s_mutex };
    dxvk::D3D11RemixApiAccess::EmitCs(context,
      [cOrigin = dxvk::Vector3(sceneOrigin[0], sceneOrigin[1], sceneOrigin[2])](dxvk::DxvkContext* dxvkCtx) {
        auto* rtx = static_cast<dxvk::RtxContext*>(dxvkCtx);
        rtx->setRetainedExternalSceneOrigin(cOrigin);
        rtx->commitRetainedExternalGeometry();
      });
    return REMIXAPI_ERROR_CODE_SUCCESS;
  }

  // The graphics preset drives bounce counts, denoiser separation, neural cache
  // quality and volumetrics, and is otherwise applied once during
  // initialization. Its onChange callback is suppressed inside this DLL, so
  // writing the option alone does nothing; this applies it.
  // 0 Ultra, 1 High, 2 Medium, 3 Low, 4 Custom.
  REMIXAPI remixapi_ErrorCode REMIXAPI_CALL csRemixApplyGraphicsPreset(uint32_t preset) {
    auto* context = tryGetContext();
    if (!context) {
      return REMIXAPI_ERROR_CODE_REMIX_DEVICE_WAS_NOT_REGISTERED;
    }
    if (preset > static_cast<uint32_t>(dxvk::GraphicsPreset::Custom)) {
      return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
    }
    std::lock_guard lock { s_mutex };
    dxvk::D3D11RemixApiAccess::EmitCs(context, [cPreset = preset](dxvk::DxvkContext* dxvkCtx) {
      dxvk::RtxOptions::graphicsPreset.setImmediately(static_cast<dxvk::GraphicsPreset>(cPreset));
      dxvk::RtxOptions::updateGraphicsPresets(dxvkCtx->getDevice().ptr());
    });
    return REMIXAPI_ERROR_CODE_SUCCESS;
  }

  // Composites the ray-traced image into a texture the host owns. The
  // destination stays on this device, and the work is ordered on the same
  // command stream as the scene description and the host's subsequent UI draws.
  REMIXAPI remixapi_ErrorCode REMIXAPI_CALL csRemixRender(ID3D11Texture2D* destination) {
    auto* context = tryGetContext();
    if (!context) {
      return REMIXAPI_ERROR_CODE_REMIX_DEVICE_WAS_NOT_REGISTERED;
    }
    if (!destination) {
      return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
    }
    auto* texture = dxvk::GetCommonTexture(destination);
    if (!texture) {
      return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
    }
    std::lock_guard lock { s_mutex };
    dxvk::D3D11RemixApiAccess::EmitCs(context, [image = texture->GetImage()](dxvk::DxvkContext* dxvkCtx) {
      static_cast<dxvk::RtxContext*>(dxvkCtx)->injectRTX(0, image);
    });
    // Ray tracing binds its own shaders, descriptors and state. Re-emit the
    // cached D3D11 state on the same command stream before the host's next draw.
    dxvk::D3D11RemixApiAccess::RestoreState(context);
    return REMIXAPI_ERROR_CODE_SUCCESS;
  }

}
