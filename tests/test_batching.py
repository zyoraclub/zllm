"""
Test continuous batching functionality.
"""

import time
import threading
from concurrent.futures import ThreadPoolExecutor


def test_imports():
    """Test that batching modules can be imported."""
    print("Testing imports...")
    
    from zllm.core.batching import (
        BatchingEngine,
        ContinuousBatchScheduler,
        GenerationRequest,
        RequestStatus,
        StopReason,
        KVCachePool,
        BatchState,
    )
    print("  ✓ All batching modules imported")
    return True


def test_request_creation():
    """Test creating generation requests."""
    print("\nTesting request creation...")
    
    from zllm.core.batching import GenerationRequest, RequestStatus
    
    request = GenerationRequest(
        request_id="test-001",
        prompt="Hello, world!",
        input_ids=[1, 2, 3, 4, 5],
        max_new_tokens=100,
        temperature=0.7,
    )
    
    assert request.status == RequestStatus.PENDING
    assert request.num_prompt_tokens == 5
    assert request.num_generated_tokens == 0
    assert not request.is_finished
    
    print(f"  ✓ Request created: {request.request_id}")
    print(f"  ✓ Prompt tokens: {request.num_prompt_tokens}")
    print(f"  ✓ Status: {request.status.value}")
    
    return True


def test_kv_cache_pool():
    """Test KV cache pool allocation."""
    print("\nTesting KV cache pool...")
    
    import torch
    from zllm.core.batching import KVCachePool
    
    pool = KVCachePool(
        num_slots=4,
        num_layers=2,
        num_heads=8,
        head_dim=64,
        max_seq_length=128,
        dtype=torch.float32,
        device="cpu",
    )
    
    # Test allocation
    assert pool.num_free_slots == 4
    
    slot1 = pool.allocate_slot()
    assert slot1 is not None
    assert pool.num_free_slots == 3
    assert pool.num_used_slots == 1
    print(f"  ✓ Allocated slot: {slot1}")
    
    slot2 = pool.allocate_slot()
    slot3 = pool.allocate_slot()
    slot4 = pool.allocate_slot()
    assert pool.num_free_slots == 0
    print(f"  ✓ Allocated all 4 slots")
    
    # Can't allocate more
    slot5 = pool.allocate_slot()
    assert slot5 is None
    print(f"  ✓ Correctly returns None when full")
    
    # Free a slot
    pool.free_slot(slot1)
    assert pool.num_free_slots == 1
    print(f"  ✓ Freed slot {slot1}")
    
    # Can allocate again
    slot_new = pool.allocate_slot()
    assert slot_new is not None
    print(f"  ✓ Re-allocated slot: {slot_new}")
    
    return True


def test_scheduler():
    """Test the continuous batch scheduler."""
    print("\nTesting scheduler...")
    
    import torch
    from zllm.core.batching import (
        ContinuousBatchScheduler,
        GenerationRequest,
        RequestStatus,
    )
    
    scheduler = ContinuousBatchScheduler(
        max_batch_size=4,
        max_waiting_requests=10,
    )
    
    # Initialize KV pool
    scheduler.initialize_kv_pool(
        num_layers=2,
        num_heads=8,
        head_dim=64,
        dtype=torch.float32,
        device="cpu",
    )
    
    # Submit requests
    req1 = GenerationRequest(
        request_id="req-1",
        prompt="Hello",
        input_ids=[1, 2, 3],
    )
    req2 = GenerationRequest(
        request_id="req-2",
        prompt="World",
        input_ids=[4, 5, 6],
    )
    
    scheduler.submit(req1)
    scheduler.submit(req2)
    
    print(f"  ✓ Submitted 2 requests")
    
    # Schedule step should move requests to batch
    active = scheduler.schedule_step()
    assert len(active) == 2
    assert req1.status == RequestStatus.RUNNING
    assert req2.status == RequestStatus.RUNNING
    print(f"  ✓ Scheduled {len(active)} requests")
    
    # Verify slots assigned
    assert req1.kv_slot is not None
    assert req2.kv_slot is not None
    assert req1.kv_slot != req2.kv_slot
    print(f"  ✓ KV slots assigned: {req1.kv_slot}, {req2.kv_slot}")
    
    # Complete one request
    from zllm.core.batching import StopReason
    scheduler.complete_request(req1, StopReason.MAX_TOKENS)
    
    assert req1.status == RequestStatus.COMPLETED
    assert req1.is_finished
    print(f"  ✓ Completed request {req1.request_id}")
    
    # Check stats
    stats = scheduler.get_stats()
    assert stats["completed_requests"] == 1
    assert stats["active_requests"] == 1
    print(f"  ✓ Stats: {stats['completed_requests']} completed, {stats['active_requests']} active")
    
    return True


def test_batch_state():
    """Test batch state management."""
    print("\nTesting batch state...")
    
    from zllm.core.batching import BatchState, GenerationRequest
    
    batch = BatchState()
    
    assert batch.batch_size == 0
    
    req = GenerationRequest(
        request_id="test",
        prompt="Test",
        input_ids=[1, 2, 3],
    )
    
    batch.add_request(slot=0, request=req)
    assert batch.batch_size == 1
    assert 0 in batch.active_slots
    print(f"  ✓ Added request to batch, size: {batch.batch_size}")
    
    removed = batch.remove_request(slot=0)
    assert removed == req
    assert batch.batch_size == 0
    print(f"  ✓ Removed request from batch")
    
    return True


def test_concurrent_requests():
    """Test handling concurrent requests."""
    print("\nTesting concurrent request handling...")
    
    import torch
    from zllm.core.batching import (
        ContinuousBatchScheduler,
        GenerationRequest,
        StopReason,
    )
    
    scheduler = ContinuousBatchScheduler(max_batch_size=4)
    scheduler.initialize_kv_pool(
        num_layers=2,
        num_heads=8,
        head_dim=64,
        dtype=torch.float32,
        device="cpu",
    )
    
    # Submit many requests concurrently
    requests = []
    
    def submit_request(i):
        req = GenerationRequest(
            request_id=f"concurrent-{i}",
            prompt=f"Request {i}",
            input_ids=[i, i+1, i+2],
        )
        scheduler.submit(req)
        requests.append(req)
    
    # Use thread pool to submit concurrently
    with ThreadPoolExecutor(max_workers=10) as executor:
        list(executor.map(submit_request, range(10)))
    
    print(f"  ✓ Submitted {len(requests)} concurrent requests")
    
    # Schedule should only activate max_batch_size
    active = scheduler.schedule_step()
    assert len(active) == 4  # max_batch_size
    print(f"  ✓ First batch: {len(active)} active")
    
    # Complete some
    for req in active[:2]:
        scheduler.complete_request(req, StopReason.EOS_TOKEN)
    
    # Next schedule should fill slots
    active = scheduler.schedule_step()
    assert len(active) == 4  # Should still be 4 (2 old + 2 new)
    print(f"  ✓ After completing 2: still {len(active)} active (slots refilled)")
    
    stats = scheduler.get_stats()
    print(f"  ✓ Total submitted: {stats['total_requests']}")
    print(f"  ✓ Completed: {stats['completed_requests']}")
    print(f"  ✓ Pending: {stats['pending_requests']}")
    
    return True


def test_request_cancellation():
    """Test cancelling requests."""
    print("\nTesting request cancellation...")
    
    import torch
    from zllm.core.batching import (
        ContinuousBatchScheduler,
        GenerationRequest,
        RequestStatus,
    )
    
    scheduler = ContinuousBatchScheduler(max_batch_size=4)
    scheduler.initialize_kv_pool(
        num_layers=2,
        num_heads=8,
        head_dim=64,
        dtype=torch.float32,
        device="cpu",
    )
    
    req = GenerationRequest(
        request_id="to-cancel",
        prompt="Cancel me",
        input_ids=[1, 2, 3],
    )
    
    scheduler.submit(req)
    scheduler.schedule_step()  # Activate it
    
    assert req.status == RequestStatus.RUNNING
    
    # Cancel
    cancelled = scheduler.cancel("to-cancel")
    assert cancelled
    assert req.status == RequestStatus.CANCELLED
    print(f"  ✓ Cancelled running request")
    
    # Cancel non-existent
    cancelled = scheduler.cancel("does-not-exist")
    assert not cancelled
    print(f"  ✓ Correctly handles non-existent request")
    
    return True


def main():
    """Run all tests."""
    print("=" * 50)
    print("Continuous Batching Tests")
    print("=" * 50)
    
    tests = [
        ("Imports", test_imports),
        ("Request Creation", test_request_creation),
        ("KV Cache Pool", test_kv_cache_pool),
        ("Scheduler", test_scheduler),
        ("Batch State", test_batch_state),
        ("Concurrent Requests", test_concurrent_requests),
        ("Request Cancellation", test_request_cancellation),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_fn in tests:
        try:
            if test_fn():
                passed += 1
        except Exception as e:
            print(f"\n❌ FAILED: {name}")
            print(f"   Error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 50)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 50)
    
    if failed == 0:
        print("\n✅ All continuous batching tests passed!")
        return 0
    else:
        print(f"\n❌ {failed} test(s) failed")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
