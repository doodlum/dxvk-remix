#include <algorithm>
#include <array>

#include "../../test_utils.h"
#include "rtx_render/rtx_instance_manager.h"
#include "rtx_render/rtx_draw_call_tracker.h"

namespace dxvk {
  Logger Logger::s_instance("test_retained_retirement.log");

  void require(bool condition, const char* message) {
    if (!condition) {
      throw DxvkError(message);
    }
  }

  void testHostRetirement() {
    RtInstance first(1, 0);
    RtInstance second(2, 1);
    ReplacementInstance node(ReplacementInstance::LookupKey{}, 1, 100);
    node.hostOwned = true;
    node.hostRetainedHandle = 47;
    node.frameLastSeen = 100;
    node.prims = { PrimInstance(&first), PrimInstance(), PrimInstance(&second) };
    node.root = PrimInstance(&first);
    node.prims[0].setReplacementInstance(&node, 0);
    node.prims[2].setReplacementInstance(&node, 2);
    require(!first.isMarkedForGC() && !second.isMarkedForGC(), "Live instances began retired");

    // Removal in the same frame must not wait for frame age or a camera cut.
    node.releaseHost();
    require(!node.hostOwned && !node.hostRetainedHandle, "Host ownership survived removal");
    require(first.isMarkedForGC() && second.isMarkedForGC(), "Removed prims survived for another frame");
    require(node.prims.empty() && !node.root.getInstance(), "Retired prim slots survived removal");
    require(!node.geometryBoundingBox.isValid() && !node.lightBoundingBox.isValid(), "Removed bounds survived for anti-culling");
    require(node.frameLastSeen == 100, "Retirement falsified the frame history");
    node.releaseHost();
    require(node.prims.empty(), "Repeated retirement was not idempotent");

    // A tracker node can subsequently be reused for a new host registration.
    RtInstance replacement(3, 2);
    node.setup(PrimInstance(&replacement), 1, nullptr);
    node.prims[0] = PrimInstance(&replacement);
    node.prims[0].setReplacementInstance(&node, 0);
    node.hostOwned = true;
    node.hostRetainedHandle = 48;
    require(!replacement.isMarkedForGC(), "Reused node prematurely retired its new prim");
    node.releaseHost();
    require(replacement.isMarkedForGC(), "Reused node failed to retire its new prim");
  }

  void testRetainedTracking() {
    RtInstance first(1, 0);
    RtInstance second(2, 1);
    DrawCallTracker tracker(nullptr);
    ReplacementInstance::LookupKey key{};
    key.identityHash = 123;
    key.spatialMapHash = 456;
    key.materialHash = 789;
    auto* pFirst = tracker.trackRetainedDraw(key, 1, 100, nullptr);
    auto* pSecond = tracker.trackRetainedDraw(key, 2, 100, nullptr);
    require(pFirst != pSecond, "Identical retained draws shared ownership");
    require(!tracker.findReplacementInstanceByIdentity(key.identityHash), "Retained draw leaked into heuristic lookup");
    pFirst->setup(PrimInstance(&first), 1, nullptr);
    pFirst->prims[0] = PrimInstance(&first);
    pFirst->prims[0].setReplacementInstance(pFirst, 0);
    pSecond->setup(PrimInstance(&second), 1, nullptr);
    pSecond->prims[0] = PrimInstance(&second);
    pSecond->prims[0].setReplacementInstance(pSecond, 0);
    pFirst->frameLastSeen = pSecond->frameLastSeen = 100;
    pFirst->objectToWorld = key.transform;
    require(tracker.trackRetainedDraw(key, 1, 101, pFirst) == pFirst, "Stable registration changed node");
    require((pFirst->dirtyFlags & ReplacementInstance::kLookupDriftMask).isClear(), "Stable draw remained dirty");
    key.identityHash++;
    key.transform[3].x = 10.f;
    require(tracker.trackRetainedDraw(key, 1, 102, pFirst) == pFirst, "Moving registration changed node");
    require(pFirst->dirtyFlags.test(ReplacementInstance::DirtyFlag::Transform), "Transform update lost dirty flag");
    require(!first.isMarkedForGC() && !second.isMarkedForGC(), "Motion retired live geometry");

    unsigned notifications = 0;
    tracker.setHostNodeDestroyedCallback([&](uint64_t handle) {
      ++notifications;
      if (handle == 1) { pFirst = nullptr; }
      if (handle == 2) { pSecond = nullptr; }
    });
    tracker.clear();
    require(!pFirst && !pSecond && notifications == 2, "Bulk clear left dangling host pointers");
    require(first.isMarkedForGC() && second.isMarkedForGC(), "Bulk clear left live prims");
    tracker.clear();
    require(notifications == 2, "Empty clear repeated destruction callbacks");

    pFirst = tracker.trackRetainedDraw(key, 1, 103, pFirst);
    pSecond = tracker.trackRetainedDraw(key, 2, 103, pSecond);
    key.spatialMapHash++;
    require(tracker.trackRetainedDraw(key, 1, 104, pFirst) == pFirst, "Mesh change changed retained identity");
    tracker.removeReplacementInstancesWithSpatialMapHash(key.spatialMapHash);
    require(!pFirst && pSecond && notifications == 3, "Mesh removal invalidated the wrong registration");
    pSecond->releaseHost();
    pSecond = nullptr;
    tracker.clear();
    require(notifications == 3, "Released registration retained ownership");
  }

  class InstanceRetirementTest {
  public:
    static void run() {
      std::array<uint32_t, 5> order { 0, 1, 2, 3, 4 };
      unsigned permutations = 0;
      do {
        unsigned destroyedCallbacks = 0;
        InstanceManager manager(nullptr, nullptr);
        struct Cleanup {
          InstanceManager& manager;
          ~Cleanup() { manager.clear(); }
        } cleanup { manager };
        std::array<RtInstance*, 5> instances {};
        for (const auto id : order) {
          auto* pInstance = new RtInstance(id + 1, uint32_t(manager.m_instances.size()));
          pInstance->m_isCreatedByRenderer = id > 0 && id < 4;
          manager.m_instances.push_back(pInstance);
          instances[id] = pInstance;
        }
        manager.m_persistentViewModelInstances[instances[0]] = instances[1];
        manager.m_persistentVirtualViewModelInstances[instances[1]] = instances[2];
        manager.m_persistentPlayerModelClones[instances[0]] = instances[3];
        InstanceEventHandler events(&manager);
        events.onInstanceDestroyedCallback = [&](RtInstance&) { ++destroyedCallbacks; };
        manager.addEventHandler(events);

        instances[0]->markForGarbageCollection();
        manager.garbageCollection();
        if (manager.getActiveCount() != 1) {
          std::cerr << "Retirement order:";
          for (const auto id : order) { std::cerr << ' ' << id; }
          std::cerr << "; survivors=" << manager.getActiveCount() << std::endl;
          throw DxvkError("Dependent clones survived same-frame collection");
        }
        require(manager.getInstanceTable()[0] == instances[4], "Collection removed an unrelated instance");
        require(instances[4]->getVectorIdx() == 0, "Swap removal left an incorrect vector index");
        require(!instances[4]->isMarkedForGC(), "Unrelated instance was marked");
        require(manager.m_persistentViewModelInstances.empty() &&
            manager.m_persistentVirtualViewModelInstances.empty() &&
            manager.m_persistentPlayerModelClones.empty(), "Persistent map retained deleted instances");
        require(destroyedCallbacks == 1, "Renderer copies emitted unbalanced destruction callbacks");
        require(manager.getSceneGeneration() == 4, "Collection did not invalidate scene generation for each deletion");
        manager.garbageCollection();
        require(manager.getActiveCount() == 1 && manager.getSceneGeneration() == 4, "Repeated GC changed the surviving scene");
        ++permutations;
      } while (std::next_permutation(order.begin(), order.end()));
      require(permutations == 120, "Missing retirement order coverage");
      std::cout << "All 120 source/view-model/virtual/player/survivor orderings passed." << std::endl;

      InstanceManager manager(nullptr, nullptr);
      auto* pSource = new RtInstance(1, 0);
      auto* pClone = new RtInstance(2, 1);
      pClone->m_isCreatedByRenderer = true;
      manager.m_instances = { pSource, pClone };
      manager.m_persistentViewModelInstances[pSource] = pClone;
      pClone->markForGarbageCollection();
      manager.garbageCollection();
      const bool sourceSurvived = manager.getActiveCount() == 1 &&
          manager.getInstanceTable()[0] == pSource &&
          manager.m_persistentViewModelInstances.empty();
      manager.clear();
      manager.garbageCollection();
      require(sourceSurvived, "Direct clone removal retired its source or retained a map entry");
      require(!manager.getActiveCount(), "Empty GC created an instance");
    }
  };
}

int main() {
  try {
    dxvk::testHostRetirement();
    dxvk::testRetainedTracking();
    dxvk::InstanceRetirementTest::run();
  } catch (const dxvk::DxvkError& error) {
    std::cerr << "TEST FAILED: " << error.message() << std::endl;
    return -1;
  } catch (const std::exception& error) {
    std::cerr << "TEST FAILED: " << error.what() << std::endl;
    return -1;
  }
  std::cout << "Retained retirement, unique tracking, motion, mesh removal and bulk invalidation passed." << std::endl;
  return 0;
}
