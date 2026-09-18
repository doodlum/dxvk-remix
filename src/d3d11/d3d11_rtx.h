#pragma once

#include "d3d11_include.h"

#include "../dxvk/dxvk_image.h"

namespace dxvk {
  class D3D11DeviceContext;

  // Remix host for a D3D11 device whose scene arrives through the Remix API
  // rather than being reconstructed from draw calls. Carries the frame
  // boundaries and nothing else: the application states its camera, meshes,
  // materials, lights and instances explicitly.
  //
  // The draw entry points count and return false, leaving the draw on the
  // ordinary D3D11 path. The count is the host's evidence that a game's own
  // rendering has been suppressed.
  class D3D11Rtx {
  public:
    explicit D3D11Rtx(D3D11DeviceContext* pContext) : m_context(pContext) { }

    void Initialize();

    bool OnDraw(UINT, UINT) { ++m_drawCallId; return false; }
    bool OnDrawIndexed(UINT, UINT, INT) { ++m_drawCallId; return false; }
    bool OnDrawInstanced(UINT, UINT, UINT, UINT) { ++m_drawCallId; return false; }
    bool OnDrawIndexedInstanced(UINT, UINT, UINT, INT, UINT) { ++m_drawCallId; return false; }
    void OnUpdateSubresource(ID3D11Resource*, const void*, UINT, UINT = 0, UINT = 0) { }

    // Queues the ray-traced composite into the given backbuffer. Called before
    // the swap chain's flush so the composite is submitted with the frame.
    void EndFrame(const Rc<DxvkImage>& backbuffer);

    // Names the presentable image the composite was blitted into.
    void OnPresent(const Rc<DxvkImage>& swapchainImage);

    // No-op: the D3D9 host tracks the backbuffer to infer when the world has
    // finished drawing, which explicit submission makes unnecessary.
    void SetSwapchainBackbuffer(const Rc<DxvkImage>&) { }

    uint32_t getDrawCallId() const { return m_drawCallId; }
    void resetDrawCallId() { m_drawCallId = 0; }
    void addDrawCallId(uint32_t count) { m_drawCallId += count; }

    // Reflex frame ID for the frame currently being submitted. Incremented once
    // per present, after every consumer for that frame has read it, so Reflex
    // sees one consistent ID per frame.
    uint64_t GetReflexFrameId() const { return m_reflexFrameId; }
    void IncrementReflexFrameId() { ++m_reflexFrameId; }

  private:
    D3D11DeviceContext* m_context;
    uint32_t m_drawCallId = 0;
    uint64_t m_reflexFrameId = 0;
  };
}
