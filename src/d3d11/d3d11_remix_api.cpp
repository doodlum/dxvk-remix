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

#include <remix/remix_c.h>

#include "d3d11_device.h"
#include "d3d11_context_imm.h"
#include "d3d11_texture.h"

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

  std::atomic<uint64_t> s_nextRetainedInstance { 0 };

  remixapi_ErrorCode setRetainedInstance(const remixapi_InstanceInfo* info, uint64_t handle) {
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
    dxvk::D3D11RemixApiAccess::EmitCs(context,
      [cHandle = handle, cState = std::move(state)](dxvk::DxvkContext* dxvkCtx) mutable {
        static_cast<dxvk::RtxContext*>(dxvkCtx)->setRetainedExternalGeometry(cHandle, std::move(*cState));
      });
    return REMIXAPI_ERROR_CODE_SUCCESS;
  }
}

extern "C" {

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
    const auto result = setRetainedInstance(info, handle);
    if (result == REMIXAPI_ERROR_CODE_SUCCESS) {
      *outHandle = handle;
    }
    return result;
  }

  // Replaces a registered draw's whole description, for a genuine change of
  // mesh, material or category.
  REMIXAPI remixapi_ErrorCode REMIXAPI_CALL csRemixUpdateRetainedInstance(
    const remixapi_InstanceInfo* info, uint64_t handle) {
    return setRetainedInstance(info, handle);
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
