#include "d3d11_rtx.h"

#include "d3d11_context.h"

#include "../dxvk/rtx_render/rtx_context.h"

namespace dxvk {
  void D3D11Rtx::Initialize() {
    Logger::info("[Remix.D3D11] API-only host: scene submitted through the Remix API, not captured from draw calls");
  }

  void D3D11Rtx::EndFrame(const Rc<DxvkImage>& backbuffer) {
    // callInjectRtx is false: the host composites before drawing its own UI
    // into the same backbuffer, so injecting at present would overwrite it.
    // Emitted onto the command-stream thread to order against queued draws.
    m_context->EmitCs([backbuffer](DxvkContext* ctx) {
      static_cast<RtxContext*>(ctx)->endFrame(0, backbuffer, false);
    });

    m_drawCallId = 0;
  }

  void D3D11Rtx::OnPresent(const Rc<DxvkImage>& swapchainImage) {
    m_context->EmitCs([swapchainImage](DxvkContext* ctx) {
      static_cast<RtxContext*>(ctx)->onPresent(swapchainImage);
    });
  }
}
