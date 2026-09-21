#include <atomic>
#include <cmath>
/*
* Copyright (c) 2023-2026, NVIDIA CORPORATION. All rights reserved.
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
#include "dxvk_device.h"
#include "rtx_dlfg.h"
#include "rtx_nsight_capture.h"

namespace {
  // 6x frame generation: 1 rendered frame + up to 5 interpolated frames
  constexpr uint32_t kDLFGMaxInterpolatedFrames = 6;
  constexpr uint64_t kPacerDoNotWait = uint64_t(-1);
  // Sentinel for VkSetPresentConfigNV::presentConfigFeedback: any driver that
  // applies the config overwrites it (0 = accepted).
  constexpr uint32_t kPresentMeteringUntouched = 0xFFFFFFFFu;

  // debugging flags
  // Measurement lever: with metering off, this is the only remaining pacing. Both
  // disabled together answers whether the ~19 ms each present blocks for is
  // deliberate spacing or real GPU work.
  const bool kSkipPacerSemaphoreWait = std::getenv("CS_REMIX_NO_PACING") != nullptr;
  // The backbuffer's last use is the blit into the swapchain image, and that blit
  // signals m_backbufferAcquireSemaphores[i] -- the very semaphore the renderer
  // waits on before writing that backbuffer again. Holding the CPU-side in-flight
  // flag until both presents have returned is therefore redundant with a GPU
  // ordering guarantee that already exists, and it is what gates the renderer:
  // the ring sits full, every acquire blocks, and the render rate is pinned to
  // the present thread's rate. Releasing the slot once the blit is submitted
  // keeps the ordering (the semaphore still enforces it) and lets the game
  // thread run ahead of presenting.
  const bool kEarlyBackbufferRelease = std::getenv("CS_REMIX_EARLY_BB_RELEASE") != nullptr;
  // Present-to-present spacing, bucketed by position in the job, so pacing can
  // be judged from the log: correct x2 pacing is both buckets equal with a low
  // spread. Logged under the same switch as the phase timers.
  const bool kLogPresentPacing = std::getenv("CS_REMIX_GPU_PHASES") != nullptr;
  uint32_t g_dlfgPresentInJob = 0;  // do not wait on pacer semaphore; this disables frame pacing, but still runs the pacer code
};

namespace dxvk {
  extern std::atomic<uint32_t> g_injectSkippedSameFrame, g_injectSkippedCameraInvalid,
                               g_injectSkippedAsyncShaders, g_injectSkippedRtDisabled,
                               g_injectReachedRender, g_dlfgDispatched;
  template<uint numBarriers>
  class DxvkDLFGImageBarrierSet {
  public:
    void addBarrier(VkImage image,
                    VkImageAspectFlags aspect,
                    VkAccessFlags srcAccess,
                    VkAccessFlags dstAccess,
                    VkImageLayout sourceLayout,
                    VkImageLayout targetLayout,
                    uint32_t srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    uint32_t dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED) {
      assert(m_barrierCount < m_barriers.size());

      auto& b = m_barriers[m_barrierCount];

      VkImageSubresourceRange range;
      range.aspectMask = aspect;
      range.baseMipLevel = 0;
      range.levelCount = VK_REMAINING_MIP_LEVELS;
      range.baseArrayLayer = 0;
      range.layerCount = VK_REMAINING_ARRAY_LAYERS;

      b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
      b.pNext = nullptr;
      b.srcAccessMask = srcAccess;
      b.dstAccessMask = dstAccess;
      b.oldLayout = sourceLayout;
      b.newLayout = targetLayout;
      b.srcQueueFamilyIndex = srcQueueFamilyIndex;
      b.dstQueueFamilyIndex = dstQueueFamilyIndex;
      b.image = image;
      b.subresourceRange = range;
      b.subresourceRange.aspectMask = aspect;

      m_barrierCount++;
    };

    void record(DxvkDevice* device, DxvkDLFGCommandList& cmdList, VkPipelineStageFlags srcStage, VkPipelineStageFlags dstStage) {
      VkCommandBuffer cmdBuf = cmdList.getCmdBuffer();
      device->vkd()->vkCmdPipelineBarrier(cmdBuf,
                                          srcStage,
                                          dstStage,
                                          0,
                                          0, nullptr,
                                          0, nullptr,
                                          m_barrierCount, m_barriers.data());
      m_barrierCount = 0;
    }

  private:
    std::array<VkImageMemoryBarrier, numBarriers> m_barriers;
    uint32_t m_barrierCount = 0;
  };

  static void labelSemaphore(Rc<DxvkDevice>& m_device, VkSemaphore semaphore, const char* name) {
    if (m_device->vkd()->vkSetDebugUtilsObjectNameEXT) {
      VkDebugUtilsObjectNameInfoEXT nameInfo;
      nameInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT;
      nameInfo.pNext = nullptr;
      nameInfo.objectType = VK_OBJECT_TYPE_SEMAPHORE;
      nameInfo.objectHandle = (uint64_t) semaphore;
      nameInfo.pObjectName = name;
      m_device->vkd()->vkSetDebugUtilsObjectNameEXT(m_device->handle(), &nameInfo);
    }
  }

  void DxvkDLFGCommandList::submit() {
    ScopedCpuProfileZone();

    VkTimelineSemaphoreSubmitInfo timelineInfo;
    timelineInfo.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
    timelineInfo.pNext = nullptr;
    timelineInfo.waitSemaphoreValueCount = m_numWaitSemaphores;
    timelineInfo.pWaitSemaphoreValues = m_waitSemaphoreValues.data();
    timelineInfo.signalSemaphoreValueCount = m_numSignalSemaphores;
    timelineInfo.pSignalSemaphoreValues = m_signalSemaphoreValues.data();

    VkPipelineStageFlags waitMask[kMaxSemaphores];
    for (uint32_t i = 0; i < kMaxSemaphores; i++) {
      waitMask[i] = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    }

    VkSubmitInfo submitInfo;
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.pNext = &timelineInfo;
    submitInfo.waitSemaphoreCount = m_numWaitSemaphores;
    submitInfo.pWaitSemaphores = m_waitSemaphores.data();
    submitInfo.pWaitDstStageMask = waitMask;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &m_cmdBuf;
    submitInfo.signalSemaphoreCount = m_numSignalSemaphores;
    submitInfo.pSignalSemaphores = m_signalSemaphores.data();

    m_device->vkd()->vkQueueSubmit(m_device->queues().__DLFG_QUEUE.queueHandle, 1, &submitInfo, m_signalFence);
    // assert(m_device->vkd()->vkQueueWaitIdle(m_device->queues().__DLFG_QUEUE.queueHandle) == VK_SUCCESS);

    m_numWaitSemaphores = 0;
    m_numSignalSemaphores = 0;
    m_signalFence = nullptr;
  }

  inline void DxvkDLFGCommandList::reset() {
    m_resources.reset();
    m_numWaitSemaphores = 0;
    m_numSignalSemaphores = 0;
    m_signalFence = nullptr;
    m_device->vkd()->vkResetCommandBuffer(m_cmdBuf, 0);
  }

  DxvkDLFGPresenter::DxvkDLFGPresenter(Rc<DxvkDevice> device,
                                       Rc<DxvkContext> ctx,
                                       HWND window,
                                       const Rc<vk::InstanceFn>& vki,
                                       const Rc<vk::DeviceFn>& vkd,
                                       vk::PresenterDevice presenterDevice,
                                       const vk::PresenterDesc& desc)
    : vk::Presenter(window, vki, vkd, presenterDevice, desc)
    , m_device(device.ptr())
    , m_ctx(ctx)
    , m_backbufferIndex(0)
    , m_dlfgCommandLists(device.ptr(), 1 + kDLFGMaxInterpolatedFrames)
    , m_blitCommandLists(device.ptr(), 1 + kDLFGMaxInterpolatedFrames)
    , m_presentPacingCommandLists(device.ptr(), kDLFGMaxInterpolatedFrames)
    , m_dlfgFrameEndSemaphore(m_device->getCommon()->metaDLFG().getFrameEndSemaphore())
    , m_dlfgPacerSemaphore(RtxSemaphore::createTimeline(m_device, "DLFG pacer CPU semaphore"))
    , m_dlfgPacerToPresentSemaphore(RtxSemaphore::createBinary(m_device, "DLFG pacer present semaphore")) {
    
    // vk::Presenter ctor calls into the base class implementation of recreateSwapchain and not our override,
    // so we need to create the backbuffers explicitly
    std::unique_lock<dxvk::mutex> lock(m_presentThread.mutex);
    assert(m_presentQueue.empty());
    createBackbuffers();

    m_presentThread.threadHandle = dxvk::thread([this]() { runPresentThread(); });
    m_pacerThread.threadHandle = dxvk::thread([this]() { runPacerThread(); });

  }

  DxvkDLFGPresenter::~DxvkDLFGPresenter() {
    if (m_presentThread.threadHandle.joinable()) {
      {
        std::unique_lock<dxvk::mutex> lock(m_presentThread.mutex);
        m_presentThread.stopped.store(true);
        m_presentThread.condWorkAvailable.notify_all();
      }

      m_presentThread.threadHandle.join();
    }

    if (m_pacerThread.threadHandle.joinable()) {
      {
        std::unique_lock<dxvk::mutex> lock(m_pacerThread.mutex);
        m_pacerThread.stopped.store(true);
        m_pacerThread.condWorkAvailable.notify_all();
      }

      m_pacerThread.threadHandle.join();
    }
  }

  vk::PresenterImage DxvkDLFGPresenter::getImage(uint32_t index) const {
    vk::PresenterImage ret;
    ret.image = m_backbufferImages[index]->handle();
    ret.view = m_backbufferViews[index]->handle();

    return ret;
  }

  VkResult DxvkDLFGPresenter::acquireNextImage(vk::PresenterSync& sync, uint32_t& index, bool) {
    ScopedCpuProfileZone();

    VkResult lastStatus = m_lastPresentStatus;
    if (lastStatus != VK_SUCCESS) {
      return lastStatus;
    }

    m_backbufferIndex = (m_backbufferIndex + 1) % m_appRequestedImageCount;

    // stall until the image is available
    {
      // Splits this wait between taking the mutex and waiting on the flag. The
      // ring-occupancy probe below runs after the lock is held, so it cannot see
      // which of the two the game thread actually lost its frame to.
      const auto lockStart = std::chrono::steady_clock::now();
      std::unique_lock<dxvk::mutex> lock(m_presentThread.mutex);
      const auto lockEnd = std::chrono::steady_clock::now();
      // How full the ring is when this blocks says whether the game thread is
      // genuinely running ahead (ring full, present thread is the limiter) or is
      // stuck on one slot while others sit free (rotation is the limiter).
      {
        static double blockedTotal = 0.0, inFlightTotal = 0.0, queueTotal = 0.0;
        static uint32_t samples = 0, blockedSamples = 0;
        uint32_t inFlight = 0;
        for (bool f : m_backbufferInFlight) {
          inFlight += f ? 1u : 0u;
        }
        const bool willBlock = m_backbufferInFlight[m_backbufferIndex];
        inFlightTotal += inFlight;
        queueTotal += double(m_presentQueue.size());
        blockedSamples += willBlock ? 1u : 0u;
        if (++samples % 240 == 0) {
          Logger::info(str::format("[RTX.acquire] ring ", m_backbufferInFlight.size(),
            ", in flight ", inFlightTotal / 240.0, ", queued jobs ", queueTotal / 240.0,
            ", blocked ", 100.0 * blockedSamples / 240.0, "% of acquires (mean of 240)"));
          inFlightTotal = queueTotal = 0.0; samples = 0; blockedSamples = 0;
        }
      }
      m_presentThread.condWorkConsumed.wait(lock, [this] { return m_backbufferInFlight[m_backbufferIndex] == false; });
      {
        static double lockTotal = 0.0, waitTotal = 0.0;
        static uint32_t samples = 0;
        lockTotal += std::chrono::duration<double, std::milli>(lockEnd - lockStart).count();
        waitTotal += std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - lockEnd).count();
        if (++samples % 240 == 0) {
          Logger::info(str::format("[RTX.acquiresplit] take mutex ", lockTotal / 240.0,
            " ms | wait on flag ", waitTotal / 240.0, " ms (mean of 240)"));
          lockTotal = waitTotal = 0.0;
        }
      }
    }

    index = m_backbufferIndex;
    sync.acquire = m_backbufferAcquireSemaphores[m_backbufferIndex]->handle();
    sync.present = m_backbufferPresentSemaphores[m_backbufferIndex]->handle();

    return VK_SUCCESS;
  }

  VkResult DxvkDLFGPresenter::presentImage(std::atomic<VkResult>* status,
                                           const DxvkPresentInfo& presentInfo,
                                           const DxvkFrameInterpolationInfo& frameInterpolationInfo,
                                           std::uint32_t acquiredImageIndex,
                                           bool isDlfgPresenting,
                                           VkSetPresentConfigNV* presentMetering)
{
    // isDlfgPresenting flag must be false here: this method can only be called from the CS thread, which does not know about DLFG
    assert(isDlfgPresenting == false);

    VkResult lastStatus = m_lastPresentStatus;
    if (lastStatus != VK_SUCCESS) {
      *status = lastStatus;
      return lastStatus;
    }

    *status = VK_EVENT_SET;

    {
      std::unique_lock<dxvk::mutex> lock(m_presentThread.mutex);

      assert(m_backbufferInFlight[acquiredImageIndex] == false);
      m_backbufferInFlight[acquiredImageIndex] = true;

      m_presentQueue.push({ status, acquiredImageIndex, presentInfo, frameInterpolationInfo });

      m_presentThread.condWorkAvailable.notify_all();
    }

    // NV-DXVK start: rendered-frame rate counter
    // This entry point is reached exactly once per rendered frame, so it is the
    // only place that can separate render cost from generated-frame cadence.
    // The presenter-side counter sees every swapchain present and cannot.
    {
      static std::mutex renderRateMutex;
      static std::chrono::steady_clock::time_point windowStart = std::chrono::steady_clock::now();
      static uint32_t rendered = 0;
      std::lock_guard lock(renderRateMutex);
      ++rendered;
      const auto now = std::chrono::steady_clock::now();
      const double elapsed = std::chrono::duration<double>(now - windowStart).count();
      if (elapsed >= 2.0) {
        Logger::info(str::format("[RTX.render] ", rendered / elapsed, " rendered/s (",
                                 1000.0 * elapsed / rendered, " ms each, x",
                                 frameInterpolationInfo.interpolatedFrameCount + 1, " presented)"));
        windowStart = now;
        rendered = 0;
      }
    }
    // NV-DXVK end

    // stash the number of frames we will present, so the HUD can calculate FPS
    m_lastPresentFrameCount = frameInterpolationInfo.interpolatedFrameCount + 1;

    return VK_EVENT_SET;
  }

  VkResult DxvkDLFGPresenter::recreateSwapChain(const vk::PresenterDesc& desc) {
    std::unique_lock<dxvk::mutex> lock(m_presentThread.mutex);
    synchronize(lock);

    m_appRequestedImageCount = desc.imageCount;

    vk::PresenterDesc adjustedDesc = desc;
    // Allocate only as many swapchain images as the configured multiplier needs (interpolated + 1 rendered),
    // rather than always allocating the hardware maximum. Swapchain will be recreated if the user increases
    // the multiplier at runtime (see acquireNextImage).
    // One image per frame in a batch is the minimum that can be presented, not
    // the number that lets the present thread run ahead of the display. With the
    // minimum, every present waits for the presentation engine to hand an image
    // back, and the stall lands on whoever acquires next -- on D3D11 that is the
    // game thread, which sat in acquireNextImage for 16.8 ms of a 25 ms frame.
    // One spare image per batch lets the next batch be prepared while the
    // current one is on screen.
    // This is the real VkSwapchain, not the backbuffer ring: each job presents
    // interpolatedFrameCount + 1 images, so at this size vkQueuePresentKHR has
    // to wait for the presentation engine to hand one back, and that wait lands
    // on the present thread and then on whoever acquires next.
    {
      const uint32_t minimum = m_ctx->dlfgInterpolatedFrameCount() + 2;
      const char* v = std::getenv("CS_REMIX_DLFG_SWAPCHAIN_IMAGES");
      const uint32_t override = v ? uint32_t(std::max(0, atoi(v))) : 0u;
      adjustedDesc.imageCount = override ? std::max(override, minimum) : minimum;
      Logger::info(str::format("[RTX.dlfg] vulkan swapchain image count ", adjustedDesc.imageCount));
    }
    
    VkResult res = vk::Presenter::recreateSwapChain(adjustedDesc);
    if (res != VK_SUCCESS) {
      return res;
    }
    
    // Reset present status since we recreated the swapchain. This ensures we try to acquire
    // during the next present instead of returning a stale error value.
    m_lastPresentStatus = VK_SUCCESS;

    createBackbuffers();
    return res;
  }

  vk::PresenterInfo DxvkDLFGPresenter::info() const {
    vk::PresenterInfo ret = vk::Presenter::info();
    ret.imageCount = m_appRequestedImageCount;
    return ret;
  }

  void DxvkDLFGPresenter::createBackbuffers() {
    // note: assumes queue is idle and m_presentThread.mutex is locked
    assert(m_presentQueue.empty());

    DxvkImageCreateInfo info;
    info.type             = VK_IMAGE_TYPE_2D;
    info.format           = m_info.format.format;
    info.flags            = 0;
    info.sampleCount      = VK_SAMPLE_COUNT_1_BIT;
    info.extent           = { m_info.imageExtent.width, m_info.imageExtent.height, 1 };
    info.numLayers        = 1;
    info.mipLevels        = 1;
    info.usage            = VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                            VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                            VK_IMAGE_USAGE_SAMPLED_BIT |
                            VK_IMAGE_USAGE_STORAGE_BIT |
                            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    info.stages           = 0;
    info.access           = 0;
    info.tiling           = VK_IMAGE_TILING_OPTIMAL;
    info.layout           = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    info.shared           = VK_FALSE;

    DxvkImageViewCreateInfo viewInfo;
    viewInfo.type         = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format       = m_info.format.format;
    viewInfo.usage        = VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                            VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                            VK_IMAGE_USAGE_SAMPLED_BIT |
                            VK_IMAGE_USAGE_STORAGE_BIT |
                            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    viewInfo.aspect       = VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.minLevel     = 0;
    viewInfo.numLevels    = 1;
    viewInfo.minLayer     = 0;
    viewInfo.numLayers    = 1;

    m_backbufferImages.resize(m_appRequestedImageCount);
    m_backbufferViews.resize(m_appRequestedImageCount);
    m_backbufferAcquireSemaphores.resize(m_appRequestedImageCount);
    m_backbufferPresentSemaphores.resize(m_appRequestedImageCount);
    m_backbufferInFlight.resize(m_appRequestedImageCount);
    
    Rc<DxvkDLFGCommandList> dummyCmdList = new DxvkDLFGCommandList(m_device);
    dummyCmdList->beginRecording();
    
    for (std::uint32_t i = 0; i < m_appRequestedImageCount; i++) {
      m_backbufferImages[i] = m_device->createImage(info, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, DxvkMemoryStats::RTXRenderTarget, "DLFG backbuffer");
      m_backbufferViews[i] = m_device->createImageView(m_backbufferImages[i], viewInfo);

      char buf[32];
      snprintf(buf, sizeof(buf), "backbuffer acquire %u", i);
      m_backbufferAcquireSemaphores[i] = RtxSemaphore::createBinary(m_device, buf);

      snprintf(buf, sizeof(buf), "backbuffer present %u", i);
      m_backbufferPresentSemaphores[i] = RtxSemaphore::createBinary(m_device, buf);

      m_backbufferInFlight[i] = false;

      // we just created the images, so acquire semaphores need to be signaled
      dummyCmdList->addSignalSemaphore(m_backbufferAcquireSemaphores[i]->handle());
    }

    dummyCmdList->endRecording();

#if !__DLFG_USE_GRAPHICS_QUEUE
    dummyCmdList->submit();
    m_device->vkd()->vkQueueWaitIdle(m_device->queues().__DLFG_QUEUE.queueHandle);
#else
    // submitting to the graphics queue here means we're racing with the submit thread;
    // we'll wait on unsignaled semaphores for the first imageCount frames instead, which works fine in practice on Windows
#endif
    
    m_backbufferIndex = 0;

    // if we're here, the swapchain was just (re)created
    // create image/view wrappers and mark all swapchain images as undefined so we transition them properly
    m_swapchainImages.resize(m_info.imageCount);
    m_swapchainImageViews.resize(m_info.imageCount);
    m_swapchainImageLayouts.resize(m_info.imageCount);

    // these need to match the swapchain usage bits
    info.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
    viewInfo.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
    
    for (uint32_t i = 0; i < m_info.imageCount; i++) {
      vk::PresenterImage swapImage = vk::Presenter::getImage(i);

      m_swapchainImages[i] = new DxvkImage(m_device, info, swapImage.image);
      m_swapchainImageViews[i] = new DxvkImageView(m_device->vkd(), m_swapchainImages[i], viewInfo);
      m_swapchainImageLayouts[i] = VK_IMAGE_LAYOUT_UNDEFINED;
    }
  }

  void DxvkDLFGPresenter::synchronize() {
    std::unique_lock<dxvk::mutex> lock(m_presentThread.mutex);
    synchronize(lock);
  }

  void DxvkDLFGPresenter::synchronize(std::unique_lock<dxvk::mutex>& lock) {
    m_presentThread.condWorkConsumed.wait(lock, [this] { return m_presentQueue.empty(); });
    m_pacerThread.condWorkConsumed.wait(lock, [this] { return m_pacerQueue.empty(); });
  }

  bool DxvkDLFGPresenter::swapchainAcquire(SwapchainImage& swapchainImage) {
    // The present-thread job takes as long as a rendered frame; this says how
    // much of that is the presentation engine handing an image back.
    const auto acquireStart = std::chrono::steady_clock::now();
    m_lastPresentStatus = vk::Presenter::acquireNextImage(swapchainImage.sync, swapchainImage.index, true);
    {
      static double total = 0.0;
      static uint32_t samples = 0;
      total += std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - acquireStart).count();
      if (++samples % 240 == 0) {
        Logger::info(str::format("[RTX.dlfgacq] swapchain acquire inside present job ", total / 240.0, " ms (mean of 240)"));
        total = 0.0;
      }
    }
    if (m_lastPresentStatus != VK_SUCCESS) {
      // got an error, bail until it's handled
      // xxxnsubtil: may need to signal the frame end semaphore here
      return false;
    }

    assert(swapchainImage.index < m_blitCommandLists.size());
    swapchainImage.image = vk::Presenter::getImage(swapchainImage.index);
    return true;
  }

  bool DxvkDLFGPresenter::interpolateFrame(DxvkDLFGCommandList* commandList,
                                           SwapchainImage& swapchainImage,
                                           const PresentJob& present,
                                           uint32_t interpolatedFrameIndex) {
    DxvkDLFG& dlfg = m_device->getCommon()->metaDLFG();
    DxvkDLFGImageBarrierSet<4> barriers;

    {
      ScopedGpuProfileZone_Present(m_device, commandList->getCmdBuffer(), "DLFG pre-eval barriers");

      barriers.addBarrier(m_swapchainImages[swapchainImage.index]->handle(),
                          VK_IMAGE_ASPECT_COLOR_BIT,
                          VK_ACCESS_NONE,
                          VK_ACCESS_SHADER_WRITE_BIT,
                          m_swapchainImageLayouts[swapchainImage.index],
                          VK_IMAGE_LAYOUT_GENERAL);

      barriers.record(m_device, *commandList, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    }

    commandList->addWaitSemaphore(swapchainImage.sync.acquire);

    // run DLFG to populate the swapchain image
    dlfg.dispatch(m_ctx,
                  commandList,
                  present.frameInterpolation.camera,
                  m_swapchainImageViews[swapchainImage.index],
                  m_backbufferViews[present.acquiredImageIndex],
                  present.frameInterpolation.motionVectors,
                  present.frameInterpolation.depth,
                  interpolatedFrameIndex,
                  present.frameInterpolation.interpolatedFrameCount,
                  false);

    {
      ScopedGpuProfileZone_Present(m_device, commandList->getCmdBuffer(), "DLFG post-eval barriers");

      barriers.addBarrier(m_swapchainImages[swapchainImage.index]->handle(),
                          VK_IMAGE_ASPECT_COLOR_BIT,
                          VK_ACCESS_SHADER_WRITE_BIT,
                          VK_ACCESS_MEMORY_READ_BIT,
                          VK_IMAGE_LAYOUT_GENERAL,
                          VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);

      barriers.record(m_device, *commandList, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    }

    m_swapchainImageLayouts[swapchainImage.index] = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    return true;
  }

  void DxvkDLFGPresenter::blitRenderedFrame(DxvkDLFGCommandList* commandList,
                                            SwapchainImage& renderedSwapchainImage,
                                            const PresentJob& present, bool frameInterpolated) {
  
    ScopedGpuProfileZone_Present(m_device, commandList->getCmdBuffer(), "DLFG real frame blit");
    DxvkDLFGImageBarrierSet<4> barriers;

    barriers.addBarrier(m_backbufferImages[present.acquiredImageIndex]->handle(),
              VK_IMAGE_ASPECT_COLOR_BIT,
              VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
              VK_ACCESS_TRANSFER_READ_BIT,
              // Note: If a frame was interpolated the backbuffer will be in the shader read only optimal layout rather than the
              // present source optimal layout.
              frameInterpolated ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL : VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
              VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);

    barriers.addBarrier(renderedSwapchainImage.image.image,
                        VK_IMAGE_ASPECT_COLOR_BIT,
                        VK_ACCESS_MEMORY_READ_BIT,
                        VK_ACCESS_TRANSFER_WRITE_BIT,
                        m_swapchainImageLayouts[present.acquiredImageIndex],
                        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

    barriers.record(m_device, *commandList, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);

    VkImageCopy copy = {
      { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
      { 0, 0, 0 },
      { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
      { 0, 0, 0 },
      { m_info.imageExtent.width, m_info.imageExtent.height, 1 }
    };

    m_device->vkd()->vkCmdCopyImage(commandList->getCmdBuffer(),
                                    m_backbufferImages[present.acquiredImageIndex]->handle(),
                                    VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                    renderedSwapchainImage.image.image,
                                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                    1, &copy);

    barriers.addBarrier(m_backbufferImages[present.acquiredImageIndex]->handle(),
                        VK_IMAGE_ASPECT_COLOR_BIT,
                        VK_ACCESS_TRANSFER_READ_BIT,
                        VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                        VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);

    barriers.addBarrier(renderedSwapchainImage.image.image,
                        VK_IMAGE_ASPECT_COLOR_BIT,
                        VK_ACCESS_TRANSFER_WRITE_BIT,
                        VK_ACCESS_MEMORY_READ_BIT,
                        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                        VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);

    m_swapchainImageLayouts[present.acquiredImageIndex] = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    barriers.record(m_device, *commandList, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);

    commandList->addWaitSemaphore(renderedSwapchainImage.sync.acquire);
  }

  bool DxvkDLFGPresenter::submitPresent(SwapchainImage& image, const PresentJob& present, uint64_t pacerSemaphoreWaitValue, VkSetPresentConfigNV* presentMetering) {
    const auto& reflex = m_ctx->getCommonObjects()->metaReflex();

    if (!kSkipPacerSemaphoreWait && pacerSemaphoreWaitValue != kPacerDoNotWait) {
      assert(presentMetering == nullptr);

      // inject a command list that waits on the pacer semaphore and signals the present semaphore
      // this will cause the present below to wait on this timeline semaphore, which the pacer thread will signal from the CPU
      DxvkDLFGCommandList* commandList = m_presentPacingCommandLists.nextCmdList();
      commandList->endRecording();
      commandList->addWaitSemaphore(m_dlfgPacerSemaphore->handle(), pacerSemaphoreWaitValue);
      commandList->addSignalSemaphore(image.sync.present);
      commandList->submit();
    }

    reflex.beginOutOfBandPresent(present.present.cachedReflexFrameId);
    m_lastPresentStatus = vk::Presenter::presentImage(present.status, present.present, present.frameInterpolation, image.index, true, presentMetering);
    reflex.endOutOfBandPresent(present.present.cachedReflexFrameId);

    if (kLogPresentPacing) {
      using clock = std::chrono::steady_clock;
      static clock::time_point s_prevReturn;
      static bool s_havePrev = false;
      static double s_sum[2] = {}, s_sumSq[2] = {};
      static uint32_t s_n[2] = {};
      const auto now = clock::now();
      if (s_havePrev) {
        const uint32_t bucket = std::min<uint32_t>(g_dlfgPresentInJob, 1u);
        const double ms = std::chrono::duration<double, std::milli>(now - s_prevReturn).count();
        s_sum[bucket] += ms; s_sumSq[bucket] += ms * ms; ++s_n[bucket];
        if (s_n[0] + s_n[1] >= 240) {
          auto stat = [&](uint32_t b) {
            const double m = s_n[b] ? s_sum[b] / s_n[b] : 0.0;
            const double v = s_n[b] ? std::max(0.0, s_sumSq[b] / s_n[b] - m * m) : 0.0;
            return str::format(m, " ms sd ", m > 0.0 ? 100.0 * std::sqrt(v) / m : 0.0, "% (n=", s_n[b], ")");
          };
          Logger::info(str::format("[RTX.pacing] rendered->interp ", stat(0), " | interp->rendered ", stat(1)));
          s_sum[0] = s_sum[1] = s_sumSq[0] = s_sumSq[1] = 0.0; s_n[0] = s_n[1] = 0;
        }
      }
      s_prevReturn = now; s_havePrev = true;
      ++g_dlfgPresentInJob;
    }

    if (m_lastPresentStatus == VK_SUCCESS) {
      NsightGraphicsCapture::signalFrameBoundary(m_device->queues().__DLFG_QUEUE.queueHandle);
      NsightGraphicsCapture::processPendingCaptureRequest();
    }

    return m_lastPresentStatus == VK_SUCCESS;
  }

  void DxvkDLFGPresenter::runPresentThread() {
    ScopedCpuProfileZone();
    env::setThreadName("dxvk-dlfg-present");

    std::unique_lock<dxvk::mutex> lock(m_presentThread.mutex);

    DxvkDLFG& dlfg = m_device->getCommon()->metaDLFG();
    Rc<DxvkDLFGTimestampQueryPool> queryPoolDLFG = dlfg.getDLFGQueryPool();
    
    while (!m_presentThread.stopped.load()) {
      {
        ScopedCpuProfileZoneN("DLFG queue: wait");
        m_presentThread.condWorkAvailable.wait(lock, [this] { return m_presentThread.stopped.load() || !m_presentQueue.empty(); });
      }

      if (m_presentThread.stopped.load()) {
        // idle the queue here to ensure we can destroy objects if needed
        m_device->vkd()->vkQueueWaitIdle(m_device->queues().__DLFG_QUEUE.queueHandle);
        return;
      }

      PresentJob present = std::move(m_presentQueue.front());

      // Interpolating and presenting takes as long as a rendered frame, and none
      // of it touches state this mutex guards. Holding it across that work blocks
      // the game thread in acquireNextImage, which only needs the mutex to read a
      // flag -- measured at 29.6 ms of a 37.9 ms frame, with the backbuffer ring
      // empty the whole time. The job stays on the queue until the guard below
      // pops it, so synchronize() still waits for it.
      lock.unlock();

      // The game thread's acquire stall is as long as this job takes, so the job
      // needs its own breakdown rather than being inferred from the stall.
      const auto jobStart = std::chrono::steady_clock::now();
      struct JobTimer {
        std::chrono::steady_clock::time_point start;
        ~JobTimer() {
          static double total = 0.0;
          static uint32_t samples = 0;
          total += std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count();
          if (++samples % 120 == 0) {
            Logger::info(str::format("[RTX.dlfgjob] present-thread job ", total / 120.0, " ms (mean of 120)"));
            total = 0.0;
          }
        }
      } jobTimer { jobStart };

      // Phase split for the job timer above. nextCmdList blocks on a command
      // list's fence, so it is where GPU work shows up on this thread rather
      // than as GPU busy; whatever remains is interpolation and presenting.
      static double tCmdList = 0.0, tInterp = 0.0, tRecord = 0.0, tPresent1 = 0.0, tPresentRest = 0.0;
      static uint32_t jobPhaseSamples = 0;
      std::chrono::steady_clock::time_point interpStart = jobStart;
      std::chrono::steady_clock::time_point recordEnd = jobStart;
      std::chrono::steady_clock::time_point present1End = jobStart;
      bool backbufferReleased = false;

      // Frees the backbuffer ring slot as soon as the blit that consumes it has
      // been submitted, rather than after the presents. Safe because the blit
      // signals the semaphore the next writer of this backbuffer waits on.
      auto releaseBackbuffer = [&]() {
        if (backbufferReleased) {
          return;
        }
        backbufferReleased = true;
        std::unique_lock<dxvk::mutex> releaseLock(m_presentThread.mutex);
        m_backbufferInFlight[present.acquiredImageIndex] = false;
        m_presentThread.condWorkConsumed.notify_all();
      };

      DxvkDLFGScopeGuard signalWorkConsumed([&]() {
        // m_device->vkd()->vkQueueWaitIdle(m_device->queues().__DLFG_QUEUE.queueHandle);
        const auto jobEnd = std::chrono::steady_clock::now();
        tInterp += std::chrono::duration<double, std::milli>(jobEnd - interpStart).count();
        tRecord += std::chrono::duration<double, std::milli>(recordEnd - interpStart).count();
        tPresent1 += std::chrono::duration<double, std::milli>(present1End - recordEnd).count();
        tPresentRest += std::chrono::duration<double, std::milli>(jobEnd - present1End).count();
        if (++jobPhaseSamples % 120 == 0) {
          Logger::info(str::format("[RTX.dlfgphase] nextCmdList ", tCmdList / 120.0,
            " ms | record+submit interp ", tRecord / 120.0,
            " ms | present#1 ", tPresent1 / 120.0,
            " ms | remaining presents ", tPresentRest / 120.0,
            " ms | job ", tInterp / 120.0, " ms (mean of 120)"));
          tCmdList = tInterp = tRecord = tPresent1 = tPresentRest = 0.0;
        }
        present.status->store(m_lastPresentStatus);

        // Retake the mutex for the queue and flag updates the waiters watch.
        lock.lock();

        if (!backbufferReleased) {
          m_backbufferInFlight[present.acquiredImageIndex] = false;
        }

        m_presentQueue.pop();
        m_presentThread.condWorkConsumed.notify_all();
      });

      // if we have an error condition that hasn't been cleared yet, drop frames until recreateSwapchain is called
      if (m_lastPresentStatus != VK_SUCCESS) {
        continue;
      }

      SwapchainImage renderedSwapchainImage;

      const auto cmdListStart = std::chrono::steady_clock::now();
      DxvkDLFGCommandList* commandList = m_dlfgCommandLists.nextCmdList();
      const auto cmdListEnd = std::chrono::steady_clock::now();
      tCmdList += std::chrono::duration<double, std::milli>(cmdListEnd - cmdListStart).count();
      DxvkDLFGImageBarrierSet<4> barriers;

      VkSemaphore backbufferWaitSemaphore = m_backbufferPresentSemaphores[present.acquiredImageIndex]->handle();
      VkSemaphore backbufferSignalSemaphore = m_backbufferAcquireSemaphores[present.acquiredImageIndex]->handle();

      commandList->addWaitSemaphore(backbufferWaitSemaphore);

      interpStart = std::chrono::steady_clock::now();
      g_dlfgPresentInJob = 0;
      if (kLogPresentPacing) {
        static uint32_t s_jobs = 0, s_valid = 0, s_noMv = 0, s_noDepth = 0, s_count0 = 0;
        const auto& fi = present.frameInterpolation;
        ++s_jobs;
        if (fi.valid()) { ++s_valid; }
        else {
          if (!fi.motionVectors.ptr()) { ++s_noMv; }
          if (!fi.depth.ptr()) { ++s_noDepth; }
          if (fi.interpolatedFrameCount == 0) { ++s_count0; }
        }
        if (s_jobs == 120) {
          Logger::info(str::format("[RTX.fgjobs] jobs 120 | interpolated ", s_valid,
            " | no-FI: noMV ", s_noMv, " noDepth ", s_noDepth, " count0 ", s_count0,
            " | injectRTX: sameFrame ", g_injectSkippedSameFrame.exchange(0),
            " cameraInvalid ", g_injectSkippedCameraInvalid.exchange(0),
            " asyncShaders ", g_injectSkippedAsyncShaders.exchange(0),
            " rtDisabled ", g_injectSkippedRtDisabled.exchange(0),
            " reachedRender ", g_injectReachedRender.exchange(0),
            " dlfgDispatched ", g_dlfgDispatched.exchange(0)));
          s_jobs = s_valid = s_noMv = s_noDepth = s_count0 = 0;
        }
      }
      if (present.frameInterpolation.valid()) {
        ScopedCpuProfileZoneN("DLFG queue: interpolate");

        // If the frame generation multiplier changed, the swapchain image count no longer matches.
        // Return VK_ERROR_OUT_OF_DATE_KHR to force recreation. Checking here on the present thread
        // (rather than on the CS thread) ensures no command list waits/signals or presents get
        // submitted against the old swapchain semaphores.
        const uint32_t neededSwapchainImages = present.frameInterpolation.interpolatedFrameCount + 1;
        if (neededSwapchainImages > m_info.imageCount) {
          m_lastPresentStatus = VK_ERROR_OUT_OF_DATE_KHR;
          commandList->reset();
          continue;
        }

        PacerJob pacer;

        SwapchainImage interpolatedSwapchainImages[kDLFGMaxInterpolatedFrames];
        
        const auto& reflex = m_ctx->getCommonObjects()->metaReflex();

        reflex.beginOutOfBandRendering(present.present.cachedReflexFrameId);

        // pre-DLFG barriers
        // xxxnsubtil: missing queue transfers here
        //
        // the VK spec requires a queue ownership transfer barrier when switching an image created with
        // VK_SHARING_MODE_EXCLUSIVE (which is all of them in dxvk) between queues; not doing so is not
        // a spec violation, but it allows the driver to leave the image contents undefined on the target
        // queue
        //
        // we colud do the present side queue transfer here, but it needs a corresponding release from graphics,
        // which we can not do on this thread; doing it in dxvk-cs requires teaching dxvk about queues, which
        // is a wider change, and doing only one half of the queue transfer results in VL errors
        //
        // we can also get around this by using VK_SHARING_MODE_CONCURRENT, but that requires queues be set up
        // in DxvkDevice before any of the implicit singleton objects that hold images are constructed (as
        // we need to specify all the queue families up front when creating the image), which is also a wider change
        //
        // for now, SHARING_MODE_EXCLUSIVE + no queue transfer barriers works fine

        {
          ScopedGpuProfileZone_Present(m_device, commandList->getCmdBuffer(), "DLFG pre-eval barriers");

          barriers.addBarrier(m_backbufferImages[present.acquiredImageIndex]->handle(),
                              VK_IMAGE_ASPECT_COLOR_BIT,
                              VK_ACCESS_NONE,
                              VK_ACCESS_SHADER_READ_BIT,
                              VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                              VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

          barriers.addBarrier(present.frameInterpolation.motionVectors->image()->handle(),
                              VK_IMAGE_ASPECT_COLOR_BIT,
                              VK_ACCESS_NONE,
                              VK_ACCESS_SHADER_READ_BIT,
                              present.frameInterpolation.motionVectorsLayout,
                              VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

          barriers.addBarrier(present.frameInterpolation.depth->image()->handle(),
                              VK_IMAGE_ASPECT_COLOR_BIT,
                              VK_ACCESS_NONE,
                              VK_ACCESS_SHADER_READ_BIT,
                              present.frameInterpolation.depthLayout,
                              VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

          barriers.record(m_device, *commandList, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
        }

        for (uint32_t fgInterpolateIndex = 0; fgInterpolateIndex < present.frameInterpolation.interpolatedFrameCount; fgInterpolateIndex++) {
          SwapchainImage& swapchainImage = interpolatedSwapchainImages[fgInterpolateIndex];
          if (!swapchainAcquire(swapchainImage)) {
            // got an error, bail until it's handled
            commandList->reset();
            continue;
          }

          interpolateFrame(commandList, swapchainImage, present, fgInterpolateIndex);
          if (fgInterpolateIndex == 0) {
            // emit the timestamp query that the pacer will read
            pacer.dlfgQueryIndex = queryPoolDLFG->writeTimestamp(commandList->getCmdBuffer(), VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);

            // first interpolated frame presents immediately, so we signal the present semaphore here
            commandList->addSignalSemaphore(swapchainImage.sync.present);
          }
        }

        // Skip post-interpolate barriers and commandlist submission if swapchain acquire failed above,
        // since the command list was reset and is no longer in a recording state. Bail until it's handled.
        if (m_lastPresentStatus == VK_SUCCESS) {
          {
            // Note: the profile zone must end before endRecording()/submit() below, since
            // its destructor records a vkCmdWriteTimestamp into the command buffer when
            // Tracy is connected.
            ScopedGpuProfileZone_Present(m_device, commandList->getCmdBuffer(), "DLFG post-interpolate barriers");

            barriers.addBarrier(present.frameInterpolation.motionVectors->image()->handle(),
                      VK_IMAGE_ASPECT_COLOR_BIT,
                      VK_ACCESS_SHADER_READ_BIT,
                      VK_ACCESS_SHADER_WRITE_BIT,
                      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                      present.frameInterpolation.motionVectorsLayout);

            barriers.addBarrier(present.frameInterpolation.depth->image()->handle(),
                                VK_IMAGE_ASPECT_COLOR_BIT,
                                VK_ACCESS_SHADER_READ_BIT,
                                VK_ACCESS_SHADER_WRITE_BIT,
                                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                present.frameInterpolation.depthLayout);

            barriers.record(m_device, *commandList, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
          }

          // queue interpolated frame presents before doing the rendered frame, to avoid stalling on swapchain acquire at the bottom

          // pacer thread will do a CPU wait on this command list before signaling the semaphores below
          pacer.lastCmdListFence = commandList->getSignalFence();
          commandList->endRecording();
          commandList->submit();
        }
        // Diagnostic: waiting on the interpolation command list here splits the
        // ~19 ms each present costs into GPU work and everything after it. If the
        // fence wait takes the time and the presents then return quickly, the
        // presents were waiting on the GPU; if the fence is instant and the
        // presents still cost 19 ms, they are blocked on the presentation engine.
        static const bool kFenceSplit = std::getenv("CS_REMIX_PRESENT_FENCE_SPLIT") != nullptr;
        if (kFenceSplit && pacer.lastCmdListFence != VK_NULL_HANDLE) {
          const auto fenceStart = std::chrono::steady_clock::now();
          m_device->vkd()->vkWaitForFences(m_device->vkd()->device(), 1,
            &pacer.lastCmdListFence, VK_TRUE, 1000ull * 1000ull * 1000ull);
          static double fenceTotal = 0.0;
          static uint32_t fenceSamples = 0;
          fenceTotal += std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - fenceStart).count();
          if (++fenceSamples % 120 == 0) {
            Logger::info(str::format("[RTX.fencesplit] interp GPU wait ", fenceTotal / 120.0,
              " ms (mean of 120)"));
            fenceTotal = 0.0;
          }
        }
        commandList = nullptr;
        recordEnd = std::chrono::steady_clock::now();

        reflex.endOutOfBandRendering(present.present.cachedReflexFrameId);

        // try to use present metering if enabled, fall back to CPU metering if it fails
        bool usePresentMetering = DxvkDLFG::enablePresentMetering();
        uint64_t pacerSemaphoreValue = kPacerDoNotWait;
        VkSetPresentConfigNV presentMetering;

        if (usePresentMetering) {
          presentMetering.sType = VK_STRUCTURE_TYPE_SET_PRESENT_CONFIG_NV;
          presentMetering.pNext = nullptr;
          presentMetering.numFramesPerBatch = 1 + present.frameInterpolation.interpolatedFrameCount;
          // The driver reports acceptance by writing 0 here. Left uninitialised,
          // a driver that ignores the config leaves stack contents behind, and a
          // stray 0 means "metering accepted": Remix then skips its own pacer and
          // both presents leave back to back with nothing metering their display.
          presentMetering.presentConfigFeedback = kPresentMeteringUntouched;
          pacerSemaphoreValue = kPacerDoNotWait;
        }

        // present the first interpolated frame
        // if we're using CPU pacing, this frame is presented immediately;
        // if we're using hardware pacing, this present sends down the pacing info
        // Diagnostic: skipping the interpolated present leaves interpolation
        // running but halves the number of presents per job. If the job time
        // halves with it, the two presents are additive serial waits; if it does
        // not, the job is bounded by something one present already waits on.
        // Output is wrong while set -- the interpolated frame is never shown.
        static const bool kSkipInterpolatedPresent = std::getenv("CS_REMIX_SKIP_INTERP_PRESENT") != nullptr;
        if (!kSkipInterpolatedPresent &&
            !submitPresent(interpolatedSwapchainImages[0], present, kPacerDoNotWait, usePresentMetering ? &presentMetering : nullptr)) {
          // got an error, bail until it's handled
          continue;
        }
        present1End = std::chrono::steady_clock::now();

        if (usePresentMetering) {
          // if we tried present metering and it failed, fall back to CPU pacing
          if (presentMetering.presentConfigFeedback != 0) {
            usePresentMetering = false;
          }
          {
            static uint32_t s_lastFeedback = ~0u - 1;
            static uint32_t s_reports = 0;
            if (presentMetering.presentConfigFeedback != s_lastFeedback && s_reports < 8) {
              ++s_reports;
              s_lastFeedback = presentMetering.presentConfigFeedback;
              Logger::info(str::format("[RTX.metering] presentConfigFeedback=",
                presentMetering.presentConfigFeedback == kPresentMeteringUntouched ? std::string("untouched-by-driver") : std::to_string(presentMetering.presentConfigFeedback),
                " -> ", usePresentMetering ? "hardware metering" : "CPU pacer"));
            }
          }
        }

        if (!usePresentMetering) {
          // if we are using CPU pacing, kick off the pacer job for this frame
          // we have to do this before present, since VK overlays may assume
          // it's safe to idle the queue during present, which would otherwise cause
          // the GPU to get stuck waiting on the pacer job
          pacerSemaphoreValue = m_dlfgPacerSemaphoreValue;

          assert(pacer.lastCmdListFence != nullptr);
          pacer.semaphoreSignalValue = pacerSemaphoreValue;
          pacer.interpolatedFrameCount = present.frameInterpolation.interpolatedFrameCount;
          {
            std::unique_lock<dxvk::mutex> lock(m_pacerThread.mutex);
            m_pacerQueue.push(pacer);
            m_pacerThread.condWorkAvailable.notify_all();
          }
        }

        // subsequent interpolated frames are paced
        // note that if we're using present metering, only the first frame gets the metering token
        // (and in that case, pacerSemaphoreValue is the do-not-wait token for the CPU pacer)
        for (uint32_t fgInterpolateIndex = 1; fgInterpolateIndex < present.frameInterpolation.interpolatedFrameCount; fgInterpolateIndex++) {
          if (!submitPresent(interpolatedSwapchainImages[fgInterpolateIndex], present, pacerSemaphoreValue, nullptr)) {
            // got an error, bail until it's handled
            continue;
          }

          if (!usePresentMetering) {
            pacerSemaphoreValue++;
          }
        }

        // do the rendered frame blit into the swapchain
        if (!swapchainAcquire(renderedSwapchainImage)) {
          // got an error, bail until it's handled
          continue;
        }

        commandList = m_dlfgCommandLists.nextCmdList();
        blitRenderedFrame(commandList, renderedSwapchainImage, present, true);

        commandList->addWaitSemaphore(renderedSwapchainImage.sync.acquire);
        commandList->addSignalSemaphore(backbufferSignalSemaphore);
        commandList->endRecording();
        commandList->submit();
        commandList = nullptr;

        if (kEarlyBackbufferRelease) {
          releaseBackbuffer();
        }

        // rendered frame present
        if (!submitPresent(renderedSwapchainImage, present, pacerSemaphoreValue, nullptr)) {
          // got an error, bail until it's handled
          continue;
        }

        pacerSemaphoreValue++;

        if (!usePresentMetering) {
          m_dlfgPacerSemaphoreValue += present.frameInterpolation.interpolatedFrameCount;
          assert(pacerSemaphoreValue == m_dlfgPacerSemaphoreValue);
        }
      } else {
        // FG was enabled but no interpolation info, present the backbuffer without FG

        if (!swapchainAcquire(renderedSwapchainImage)) {
          // got an error, bail until it's handled
          commandList->reset();
          continue;
        }

        blitRenderedFrame(commandList, renderedSwapchainImage, present, false);

        commandList->addWaitSemaphore(renderedSwapchainImage.sync.acquire);
        commandList->addSignalSemaphore(backbufferSignalSemaphore);
        commandList->endRecording();
        commandList->submit();
        commandList = nullptr;

        if (kEarlyBackbufferRelease) {
          releaseBackbuffer();
        }

        // rendered frame present
        if (!submitPresent(renderedSwapchainImage, present, kPacerDoNotWait, nullptr)) {
          // got an error, bail until it's handled
          continue;
        }
      }
    }
  }

  void DxvkDLFGPresenter::runPacerThread() {
    ScopedCpuProfileZone();
    env::setThreadName("dxvk-dlfg-pacer");

    std::unique_lock<dxvk::mutex> lock(m_pacerThread.mutex);

    Rc<DxvkDLFGTimestampQueryPool> queryPoolDLFG = m_device->getCommon()->metaDLFG().getDLFGQueryPool();

    VkPhysicalDeviceLimits limits = m_device->adapter()->deviceProperties().limits;
    const double nsPerGpuIncrement = limits.timestampPeriod;

    const int64_t qpcIncrementsPerSecond = high_resolution_clock::getFrequency();
    const double qpcIncrementsPerNs = double(qpcIncrementsPerSecond) / 1000000000.0;
    const double nsPerQpcIncrement = 1000000000.0 / double(qpcIncrementsPerSecond);

    uint64_t referenceTimestampGpu = 0;
    uint64_t referenceTimestampQpc = 0;
    uint64_t referenceMaxDeviation = 0;

    auto signalPresentSemaphore = [&](uint64_t value) {
      VkSemaphoreSignalInfo info;
      info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO;
      info.pNext = nullptr;
      info.semaphore = m_dlfgPacerSemaphore->handle();
      info.value = value;
      VkResult res = m_device->vkd()->vkSignalSemaphore(m_device->handle(), &info);
      if (res != VK_SUCCESS) {
        Logger::err("DxvkDLFGPresenter::runPacerThread: vkSignalSemaphore failed");
      }
    };

    auto calibrateTimestamps = [&]() {
      VkCalibratedTimestampInfoEXT info[2] = {
        {
          VK_STRUCTURE_TYPE_CALIBRATED_TIMESTAMP_INFO_EXT,
          nullptr,
          VK_TIME_DOMAIN_DEVICE_EXT,
        }, {
          VK_STRUCTURE_TYPE_CALIBRATED_TIMESTAMP_INFO_EXT,
          nullptr,
          VK_TIME_DOMAIN_QUERY_PERFORMANCE_COUNTER_EXT,
        }
      };

      uint64_t timestamps[2];
      VkResult res;
      res = m_device->vkd()->vkGetCalibratedTimestampsEXT(m_device->handle(),
                                                          2,
                                                          info,
                                                          timestamps, &referenceMaxDeviation);
      if (res != VK_SUCCESS) {
        throw DxvkError("DxvkDLFGPresenter::runPacerThread(): vkGetCalibratedTimestampsEXT failed");
      }

      referenceTimestampGpu = timestamps[0];
      referenceTimestampQpc = timestamps[1];
    };

    auto qpcTicksToNs = [&](int64_t ticks) -> double {
      return ticks * nsPerQpcIncrement;
    };

    auto nsToQpcTicks = [&](double ns) -> uint64_t {
      return uint64_t(ns * qpcIncrementsPerNs);
    };

    auto gpuTicksToNs = [&](uint64_t ticks) -> double {
      return ticks * nsPerGpuIncrement;
    };

    auto nsToMs = [](double ns) -> double {
      return ns / 1000000.0;
    };

    auto msToNs = [](double ms) -> double {
      return ms * 1e6;
    };

    auto gpuTicksToQpc = [&](uint64_t gpuTicks) -> int64_t {
      double deltaToReferenceNs;
      int64_t deltaToReferenceQpcTicks;
      
      if (gpuTicks > referenceTimestampGpu) {
        deltaToReferenceNs = gpuTicksToNs(gpuTicks - referenceTimestampGpu);
        deltaToReferenceQpcTicks = nsToQpcTicks(deltaToReferenceNs);
        return referenceTimestampQpc + deltaToReferenceQpcTicks;
      } else {
        deltaToReferenceNs = gpuTicksToNs(referenceTimestampGpu - gpuTicks);
        deltaToReferenceQpcTicks = nsToQpcTicks(deltaToReferenceNs);
        return referenceTimestampQpc - deltaToReferenceQpcTicks;
      }
    };

    uint64_t lastFrameDlfgEndGpuTicks = 0;

    while (!m_pacerThread.stopped.load()) {
      {
        ScopedCpuProfileZoneN("DLFG pacer: wait");
        m_pacerThread.condWorkAvailable.wait(lock, [this] { return m_pacerThread.stopped.load() || !m_pacerQueue.empty(); });
      }

      if (m_pacerThread.stopped.load()) {
        break;
      }

      PacerJob pacer = std::move(m_pacerQueue.front());
      lock.unlock();

      uint64_t dlfgTimestamp = 0;
      bool pacerActive = true;

      // wait on the GPU and read back timestamp query
      {
        ScopedCpuProfileZoneN("DLFG pacer: query readback");

        // instead of using the WAIT bit for GetQueryPoolResults, wait on a fence that resolves after the query results
        // this ensures we can timeout if the queries don't resolve, instead of hanging forever
        VkResult res = m_device->vkd()->vkWaitForFences(m_device->handle(), 1, &pacer.lastCmdListFence, VK_TRUE, 1000000000);

        if (res != VK_SUCCESS) {
          Logger::warn("DLFG pacer: fence timed out");
          lastFrameDlfgEndGpuTicks = 0;
          pacerActive = false;
        } else {
          pacerActive = queryPoolDLFG->readTimestamp(&dlfgTimestamp, pacer.dlfgQueryIndex);

          if (lastFrameDlfgEndGpuTicks == 0) {
            lastFrameDlfgEndGpuTicks = dlfgTimestamp;
            // no data for previous frame available, do not pace this frame
            pacerActive = false;
          }
        }
      }

      if (pacerActive) {
        {
          ScopedCpuProfileZoneN("DLFG pacer: timestamp calibration");
          calibrateTimestamps();
        }

        const double frameToFrame = nsToMs(gpuTicksToNs(dlfgTimestamp - lastFrameDlfgEndGpuTicks));
        ProfilerPlotValue("DLFG pacer: frame-to-frame time (ms)", frameToFrame);

        // this determines the maximum amount of time we're willing to sleep, as a backstop in case something goes wrong
        // this is based on the minimum input frame rate required for FG; our max sleep time is 2x that value to provide
        // enough margin for variance in frame times
        constexpr double kMinOutputFPS = 20;
        constexpr double kMaxFrameTimeMs = 2000.0 / kMinOutputFPS;

        // skip the pacer logic if the timestamps don't make sense
        if (frameToFrame > 0.0 && frameToFrame < kMaxFrameTimeMs) {
          // time the present to land at the halfway point between the two DLFG interpolated frames
          const uint64_t frameTimeGpuTicks = dlfgTimestamp - lastFrameDlfgEndGpuTicks;
          uint64_t deltaGpuPresentTicks = frameTimeGpuTicks / (1 + pacer.interpolatedFrameCount);

          const double deltaPresentNs = gpuTicksToNs(deltaGpuPresentTicks);
          const double deltaPresentQpcNs = qpcTicksToNs(gpuTicksToQpc(dlfgTimestamp + deltaGpuPresentTicks * pacer.interpolatedFrameCount) - high_resolution_clock::getCounter());
          ProfilerPlotValue("DLFG pacer: measured GPU sleep time (ms)", nsToMs(deltaPresentNs));
          ProfilerPlotValue("DLFG pacer: remaining CPU sleep time (ms)", nsToMs(deltaPresentQpcNs));

          for (uint32_t frameIndex = 0; frameIndex < pacer.interpolatedFrameCount; frameIndex++) {

            // ignore sleeps longer than kMaxFrameTimeMs in case something goes wrong with the math above
            if (deltaPresentQpcNs < msToNs(kMaxFrameTimeMs)) {
              // convert the GPU timestamp to a CPU timestamp
              const int64_t targetQpcPresentTicks = gpuTicksToQpc(dlfgTimestamp + deltaGpuPresentTicks * (frameIndex+1));

              ScopedCpuProfileZoneN("DLFG pacer: sleep");
              while (high_resolution_clock::getCounter() < targetQpcPresentTicks) {
                _mm_pause();
              }

              // signal the present semaphore
              {
                ScopedCpuProfileZoneN("DLFG pacer: signal semaphore");
                signalPresentSemaphore(pacer.semaphoreSignalValue + frameIndex);
              }
            } else {
              pacerActive = false;
            }
          }

          lastFrameDlfgEndGpuTicks = dlfgTimestamp;
        } else {
          // timings don't make sense, reset history
          lastFrameDlfgEndGpuTicks = 0;
          pacerActive = false;
        }
      }

      ProfilerPlotValueI64("DLFG pacer: active", pacerActive ? 1 : 0);

      // if the pacer was inactive, signal all semaphores here to ensure forward progress
      if (!pacerActive) {
        for (uint32_t frameIndex = 0; frameIndex < pacer.interpolatedFrameCount; frameIndex++) {
          ScopedCpuProfileZoneN("DLFG pacer (inactive): signal semaphore");
          signalPresentSemaphore(pacer.semaphoreSignalValue + frameIndex);
        }
      }

      lock.lock();
      m_pacerQueue.pop();
      m_pacerThread.condWorkConsumed.notify_all();
    }

    // release all pending frames in the queue before leaving
    assert(lock.owns_lock());

    while (m_pacerQueue.size()) {
      PacerJob pacer = std::move(m_pacerQueue.front());

      for (uint32_t frameIndex = 0; frameIndex < pacer.interpolatedFrameCount; frameIndex++) {
        signalPresentSemaphore(pacer.semaphoreSignalValue + frameIndex);
      }

      m_pacerQueue.pop();
      m_pacerThread.condWorkConsumed.notify_all();
    }
  }

  DxvkDLFGCommandList::DxvkDLFGCommandList(DxvkDevice* device)
    : m_device(device) {
    VkCommandPoolCreateInfo poolInfo;
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.pNext = nullptr;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolInfo.queueFamilyIndex = m_device->queues().__DLFG_QUEUE.queueFamily;

    if (m_device->vkd()->vkCreateCommandPool(m_device->handle(), &poolInfo, nullptr, &m_cmdPool) != VK_SUCCESS) {
      throw DxvkError("DxvkDLFGCommandList: failed to create command pool");
    }

    VkCommandBufferAllocateInfo cmdInfo;
    cmdInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmdInfo.pNext = nullptr;
    cmdInfo.commandPool = m_cmdPool;
    cmdInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmdInfo.commandBufferCount = 1;
    
    if (m_device->vkd()->vkAllocateCommandBuffers(m_device->handle(), &cmdInfo, &m_cmdBuf) != VK_SUCCESS) {
      throw DxvkError("DxvkDLFGCommandList: failed to create command list");
    }
  }

  DxvkDLFGCommandList::~DxvkDLFGCommandList() {
    m_resources.reset();
    if (m_cmdPool) {
      m_device->vkd()->vkDestroyCommandPool(m_device->handle(), m_cmdPool, nullptr);
      m_cmdPool = nullptr;
      m_cmdBuf = nullptr;
    }
  }

  void DxvkDLFGCommandList::beginRecording() {
    assert(m_cmdPool);
    assert(m_cmdBuf);
    
    VkCommandBufferBeginInfo info;
    info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    info.pNext = nullptr;
    info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    info.pInheritanceInfo = nullptr;

    if (m_device->vkd()->vkBeginCommandBuffer(m_cmdBuf, &info) != VK_SUCCESS) {
      Logger::err("DxvkDLFGCommandList::beginRecording: vkBeginCommandBuffer failed");
    }
  }

  void DxvkDLFGCommandList::endRecording() {
    TracyVkCollect(m_device->queues().__DLFG_QUEUE.tracyCtx, m_cmdBuf);

    if (m_device->vkd()->vkEndCommandBuffer(m_cmdBuf) != VK_SUCCESS) {
      Logger::err("DxvkDLFGCommandList::endRecording: vkEndCommandBuffer failed");
    }
  }

  void DxvkDLFGCommandList::addWaitSemaphore(VkSemaphore sem, uint64_t value) {
    assert(m_numWaitSemaphores < kMaxSemaphores);

    if (sem) {
      m_waitSemaphores[m_numWaitSemaphores] = sem;
      m_waitSemaphoreValues[m_numWaitSemaphores] = value;
      ++m_numWaitSemaphores;
    }
  }

  void DxvkDLFGCommandList::addSignalSemaphore(VkSemaphore sem, uint64_t value) {
    assert(m_numSignalSemaphores < kMaxSemaphores);

    if (sem) {
      m_signalSemaphores[m_numSignalSemaphores] = sem;
      m_signalSemaphoreValues[m_numSignalSemaphores] = value;
      ++m_numSignalSemaphores;
    }
  }

  void DxvkDLFGCommandList::setSignalFence(VkFence fence) {
    assert(!m_signalFence);
    assert(fence);
    m_signalFence = fence;
  }

  DxvkDLFGCommandListArray::DxvkDLFGCommandListArray(DxvkDevice* device, uint32_t numCmdLists)
    : m_device(device) {
    m_commandLists.resize(dxvk::kMaxFramesInFlight * numCmdLists);
    m_fences.resize(dxvk::kMaxFramesInFlight * numCmdLists);
  }

  DxvkDLFGCommandList *DxvkDLFGCommandListArray::nextCmdList() {
    ScopedCpuProfileZone();

    // note: we can't create this in the ctor as our parent object is constructed before the VK device is created
    if (m_commandLists[m_currentCommandListIndex] == nullptr) {
      m_commandLists[m_currentCommandListIndex] = new DxvkDLFGCommandList(m_device);
      m_fences[m_currentCommandListIndex] = new RtxFence(m_device);
    }

    DxvkDLFGCommandList *ret = m_commandLists[m_currentCommandListIndex].ptr();
    VkFence fence = m_fences[m_currentCommandListIndex]->handle();
    assert(fence);
    VkResult res = m_device->vkd()->vkWaitForFences(m_device->handle(), 1, &fence, VK_TRUE, 1'000'000'000ull);
    if (res != VK_SUCCESS) {
      ONCE(Logger::err("DxvkDLFGCommandListArray::nextCmdList: vkWaitForFences failed"));
    }

    res = m_device->vkd()->vkResetFences(m_device->handle(), 1, &fence);
    assert(res == VK_SUCCESS);
    
    ret->reset();
    ret->setSignalFence(m_fences[m_currentCommandListIndex]->handle());
    ret->beginRecording();
    
    m_currentCommandListIndex = (m_currentCommandListIndex + 1) % m_commandLists.size();

    return ret;
  }

  DxvkDLFGTimestampQueryPool::DxvkDLFGTimestampQueryPool(DxvkDevice* device, const uint32_t numQueries)
    : m_device(device)
    , m_queryPoolSize(numQueries) {

    VkQueryPoolCreateInfo info;
    info.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    info.pNext = nullptr;
    info.flags = 0;
    info.queryType = VK_QUERY_TYPE_TIMESTAMP;
    info.queryCount = numQueries;
    info.pipelineStatistics = 0;

    VkResult res;
    res = m_device->vkd()->vkCreateQueryPool(m_device->handle(), &info, nullptr, &m_queryPool);
    if (res != VK_SUCCESS) {
      throw DxvkError("DxvkDLFGTimestampQueryPool: vkCreateQueryPool failed");
    }

    m_device->vkd()->vkResetQueryPool(m_device->handle(), m_queryPool, 0, numQueries);
  }

  DxvkDLFGTimestampQueryPool::~DxvkDLFGTimestampQueryPool() {
    if (m_queryPool) {
      m_device->vkd()->vkDestroyQueryPool(m_device->handle(), m_queryPool, nullptr);
      m_queryPool = nullptr;
    }
  }

  uint32_t DxvkDLFGTimestampQueryPool::writeTimestamp(VkCommandBuffer cmdList, VkPipelineStageFlagBits stage) {
    assert(m_queryPool);

    uint32_t idx = m_nextQueryIndex;
    m_device->vkd()->vkCmdResetQueryPool(cmdList, m_queryPool, idx, 1);
    m_device->vkd()->vkCmdWriteTimestamp(cmdList, stage, m_queryPool, idx);

    m_nextQueryIndex = (m_nextQueryIndex + 1) % m_queryPoolSize;
    return idx;
  }

  bool DxvkDLFGTimestampQueryPool::readTimestamp(uint64_t* queryResult, uint32_t queryIndex) {
    VkResult res;

    res = m_device->vkd()->vkGetQueryPoolResults(m_device->handle(),
                                                 m_queryPool,
                                                 queryIndex,
                                                 1,
                                                 sizeof(uint64_t),
                                                 queryResult,
                                                 sizeof(uint64_t),
                                                 VK_QUERY_RESULT_64_BIT);

    if (res != VK_SUCCESS) {
      return false;
    }

    return true;
  }

  DxvkDLFG::DxvkDLFG(DxvkDevice* device)
    : CommonDeviceObject(device)
    // xxxnsubtil: use swapchain frame count here
    , m_dlfgEvalCommandLists(device, 1)
    , m_dlfgFrameEndSemaphore(RtxSemaphore::createTimeline(device, "DLFG frame end"))
    , m_currentDisplaySize{0, 0} {

    m_queryPoolDLFG = new DxvkDLFGTimestampQueryPool(m_device, kMaxFramesInFlight);

    if (!supportsPresentMetering()) {
      Logger::warn("NV_present_metering extension not supported");
      enablePresentMetering.setDeferred(false);
    }
  }

  void DxvkDLFG::onDestroy() {
    if (m_dlfgContext) {
      m_dlfgContext->releaseNGXFeature();
    }
    m_dlfgContext = nullptr;
    m_queryPoolDLFG = nullptr;
  }

  bool DxvkDLFG::supportsDLFG() {
    return m_device->getCommon()->metaNGXContext().supportsDLFG();
  }

  const std::string& DxvkDLFG::getDLFGNotSupportedReason() {
    return m_device->getCommon()->metaNGXContext().getDLFGNotSupportedReason();
  }

  void DxvkDLFG::setDisplaySize(uint2 displaySize) {
    if (m_currentDisplaySize[0] != displaySize.x ||
        m_currentDisplaySize[1] != displaySize.y) {
      m_currentDisplaySize[0] = displaySize.x;
      m_currentDisplaySize[1] = displaySize.y;
      m_contextDirty = true;
    }
  }

  // note: we expect that the input semaphore is already waited on by commandList
  void DxvkDLFG::dispatch(Rc<DxvkContext> ctx,
                          DxvkDLFGCommandList* commandList,
                          const RtCamera& camera,
                          Rc<DxvkImageView> outputImage,                       // VK_IMAGE_LAYOUT_GENERAL
                          Rc<DxvkImageView> colorBuffer,                       // VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
                          Rc<DxvkImageView> primaryScreenSpaceMotionVector,    // VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
                          Rc<DxvkImageView> primaryDepth,                      // VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
                          uint32_t interpolatedFrameIndex,                     // starts at 0
                          uint32_t interpolatedFrameCount,                     // total number of frames we will interpolate before the next rendered frame
                          bool resetHistory) {
    ScopedCpuProfileZone();

    if (!m_dlfgContext) {
      m_dlfgContext = m_device->getCommon()->metaNGXContext().createDLFGContext();
      m_contextDirty = true;
    }

    // check if the output extents have changed
    VkExtent3D outputExtent = outputImage->imageInfo().extent;
    if (outputExtent.width != m_currentDisplaySize[0] ||
        outputExtent.height != m_currentDisplaySize[1]) {
      // note: this is the size of the window client area, which isn't necessarily the same as the D3D9 swapchain size
      setDisplaySize(uint2(outputExtent.width, outputExtent.height));
      m_contextDirty = true;
    }

    if (m_contextDirty) {
      assert(m_dlfgContext != nullptr);

      if (m_device->vkd()->vkQueueWaitIdle(m_device->queues().__DLFG_QUEUE.queueHandle) != VK_SUCCESS) {
        Logger::err("DxvkDLFG::dispatch: vkQueueWaitIdle failed");
      }

      m_dlfgContext->releaseNGXFeature();
      m_dlfgContext->initialize(ctx,
                                commandList->getCmdBuffer(),
                                m_currentDisplaySize,
                                outputImage->info().format);
      m_contextDirty = false;
    }

    NGXDLFGContext::EvaluateResult res;

    commandList->trackResource<DxvkAccess::Write>(outputImage);
    commandList->trackResource<DxvkAccess::Read>(colorBuffer);
    commandList->trackResource<DxvkAccess::Read>(primaryScreenSpaceMotionVector);
    commandList->trackResource<DxvkAccess::Read>(primaryDepth);

    {
      ScopedGpuProfileZone_Present(m_device, commandList->getCmdBuffer(), "DLFG evaluate");

      assert(m_dlfgContext != nullptr);

      res = m_dlfgContext->evaluate(Rc<DxvkContext>(ctx.ptr()),
                                                    commandList->getCmdBuffer(),
                                                    outputImage,
                                                    colorBuffer,
                                                    primaryScreenSpaceMotionVector,
                                                    primaryDepth,
                                                    camera,
                                                    Vector2(1.0f, 1.0f),
                                                    interpolatedFrameIndex,
                                                    interpolatedFrameCount,
                                                    resetHistory);

      switch (res) {
      case NGXDLFGContext::EvaluateResult::Failure:
        Logger::err("NGX DLFG evaluate failed");
        m_hasDLFGFailed = true;
        break;

      case NGXDLFGContext::EvaluateResult::Success:
        break;
      }
    }
  }

  bool DxvkDLFG::supportsPresentMetering() const {
    return m_device->extensions().nvPresentMetering;
  }

  uint32_t DxvkDLFG::getMaxSupportedInterpolatedFrameCount() {
    return std::min(maxInterpolatedFrames(), m_device->getCommon()->metaNGXContext().dlfgMaxInterpolatedFrames());
  }

  uint32_t DxvkDLFG::getInterpolatedFrameCount() {
    return std::min(maxInterpolatedFrames(), getMaxSupportedInterpolatedFrameCount());
  }
} // namespace dxvk
