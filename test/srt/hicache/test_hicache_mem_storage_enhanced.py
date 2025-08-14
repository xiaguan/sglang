import threading
import time
import psutil
import unittest
from itertools import product
from typing import List

import torch

from sglang.srt.mem_cache.hicache_storage import get_hash_str
from sglang.srt.distributed import (
    get_world_group,
    init_distributed_environment,
    initialize_model_parallel,
)
from sglang.srt.managers.cache_controller import (
    HiCacheController,
    StorageOperation,
)
from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, MLATokenToKVPool
from sglang.srt.mem_cache.memory_pool_host import MHATokenToKVPoolHost, MLATokenToKVPoolHost


# =============================================================================
# Configuration Constants
# =============================================================================

# Test configuration matrix
TEST_CONFIGURATIONS = list(product(
    ["layer_first", "page_first"],  # layouts
    [1, 64],                        # page_sizes  
    ["kernel", "direct"],           # io_backends
    ["MHA", "MLA"]                  # attention_types
))

# # Simple matric
# TEST_CONFIGURATIONS = list(product(
#     ["page_first"],  # layouts
#     [64],              # page_sizes  
#     ["kernel"],       # io_backends
#     ["MHA"]           # attention_types
# ))

# Test constants
MAX_TOTAL_NUM_TOKENS = 12 * 1024
KV_CACHE_DTYPE = torch.bfloat16
LAYER_NUM = 64
HEAD_NUM, HEAD_DIM = 8, 128
KV_LORA_RANK = 512  # MLA specific
QK_ROPE_HEAD_DIM = 64  # MLA specific
DEVICE = "cpu"
HICACHE_RATIO = 2
HICACHE_SIZE = 0
HICACHE_WRITE_POLICY = "write_through"
HICACHE_STORAGE_BACKEND = "test_storage"
PREFETCH_THRESHOLD = 256

# Test parameters
ENHANCED_OP_SIZE = 1024
ENHANCED_OP_NUM = 10

assert ENHANCED_OP_SIZE * ENHANCED_OP_NUM <= MAX_TOTAL_NUM_TOKENS, "Too many tokens for test"


# =============================================================================
# Helper Classes and Functions  
# =============================================================================

class TestConfig:
    """Test configuration container"""
    def __init__(self, layout: str, page_size: int, io_backend: str, attention_type: str):
        self.layout = layout
        self.page_size = page_size
        self.io_backend = io_backend
        self.attention_type = attention_type
        self.max_total_num_tokens = MAX_TOTAL_NUM_TOKENS
        self.kv_cache_dtype = KV_CACHE_DTYPE
        self.layer_num = LAYER_NUM
        self.device = DEVICE
        self.hicache_ratio = HICACHE_RATIO
        self.hicache_size = HICACHE_SIZE
        self.hicache_write_policy = HICACHE_WRITE_POLICY
        self.hicache_storage_backend = HICACHE_STORAGE_BACKEND
        self.prefetch_threshold = max(PREFETCH_THRESHOLD, page_size)
        
        # Attention type specific parameters
        if attention_type == "MHA":
            self.head_num = HEAD_NUM
            self.head_dim = HEAD_DIM
        elif attention_type == "MLA":
            self.kv_lora_rank = KV_LORA_RANK
            self.qk_rope_head_dim = QK_ROPE_HEAD_DIM
        else:
            raise ValueError(f"Unsupported attention type: {attention_type}")

    def __str__(self):
        return f"layout={self.layout}_page={self.page_size}_io={self.io_backend}_attn={self.attention_type}"


def setup_distributed():
    """Setup distributed environment for testing"""
    import socket
    
    # Find an available port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        port = s.getsockname()[1]
    
    init_distributed_environment(
        world_size=1,
        rank=0,
        distributed_init_method=f"tcp://127.0.0.1:{port}",
        local_rank=0,
        backend="gloo",
    )
    
    initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
    )
    
    return get_world_group().cpu_group


def create_cache_components(config: TestConfig):
    """Create cache components for testing"""
    if config.attention_type == "MHA":
        token_to_kv_pool = MHATokenToKVPool(
            config.max_total_num_tokens,
            page_size=config.page_size,
            dtype=config.kv_cache_dtype,
            head_num=config.head_num,
            head_dim=config.head_dim,
            layer_num=config.layer_num,
            device=config.device,
            enable_memory_saver=True,
        )
        
        token_to_kv_pool_allocator = TokenToKVPoolAllocator(
            config.max_total_num_tokens,
            dtype=config.kv_cache_dtype,
            device=config.device,
            kvcache=token_to_kv_pool,
            need_sort=False,
        )

        kv_cache = token_to_kv_pool_allocator.get_kvcache()
        token_to_kv_pool_host = MHATokenToKVPoolHost(
            kv_cache,
            config.hicache_ratio,
            config.hicache_size,
            config.page_size,
            config.layout,
        )
        
    elif config.attention_type == "MLA":
        token_to_kv_pool = MLATokenToKVPool(
            config.max_total_num_tokens,
            page_size=config.page_size,
            dtype=config.kv_cache_dtype,
            kv_lora_rank=config.kv_lora_rank,
            qk_rope_head_dim=config.qk_rope_head_dim,
            layer_num=config.layer_num,
            device=config.device,
            enable_memory_saver=True,
        )
        
        token_to_kv_pool_allocator = TokenToKVPoolAllocator(
            config.max_total_num_tokens,
            dtype=config.kv_cache_dtype,
            device=config.device,
            kvcache=token_to_kv_pool,
            need_sort=False,
        )

        kv_cache = token_to_kv_pool_allocator.get_kvcache()
        token_to_kv_pool_host = MLATokenToKVPoolHost(
            kv_cache,
            config.hicache_ratio,
            config.hicache_size,
            config.page_size,
            config.layout,
        )
    else:
        raise ValueError(f"Unsupported attention type: {config.attention_type}")

    load_cache_event = threading.Event()
    cache_controller = HiCacheController(
        token_to_kv_pool_allocator,
        token_to_kv_pool_host,
        config.page_size,
        get_world_group().cpu_group,
        load_cache_event=load_cache_event,
        write_policy=config.hicache_write_policy,
        io_backend=config.io_backend,
        storage_backend=config.hicache_storage_backend,
        prefetch_threshold=config.prefetch_threshold,
    )

    return cache_controller, load_cache_event


def create_storage_operations(op_num: int, op_size: int, page_size: int) -> List[StorageOperation]:
    """Create storage operations for testing"""
    operations = []
    for i in range(0, op_num * op_size, op_size):
        # Generate proper hash values for each page
        hash_values = []
        last_hash = None
        for j in range(i, i + op_size, page_size):
            page_tokens = list(range(j, j + page_size))
            hash_value = get_hash_str(page_tokens, last_hash)
            hash_values.append(hash_value)
            last_hash = hash_value

        operations.append(StorageOperation(
            torch.tensor(list(range(i, i + op_size))),
            list(range(i, i + op_size)),
            hash_value=hash_values,
        ))
    return operations

# Removed unused create_prefetch_operations function



# =============================================================================
# Test Class
# =============================================================================

class TestHiCacheMemStorageEnhanced(unittest.TestCase):
    """Enhanced HiCache Memory Storage Tests"""

    @classmethod
    def setUpClass(cls):
        """Setup class-level resources"""
        cls.group = setup_distributed()

    def _run_configuration_test(self, layout: str, page_size: int, io_backend: str, attention_type: str):
        """Test a specific configuration combination"""
        print(f"\n=== Testing: layout={layout}, page_size={page_size}, io_backend={io_backend}, attention={attention_type} ===")
        
        config = TestConfig(layout, page_size, io_backend, attention_type)
        cache_controller, _ = create_cache_components(config)
        
        # Test storage operations
        storage_ops = create_storage_operations(ENHANCED_OP_NUM, ENHANCED_OP_SIZE, config.page_size)
        
        start_time = time.monotonic()
        backup_ids = []
        for operation in storage_ops:
            backup_id = cache_controller.write_storage(operation.host_indices, operation.token_ids, operation.hash_value)
            backup_ids.append(backup_id)

        # should receive all acks
        # get cachecontroller's ack_backup_queue
        ack_backup_queue = cache_controller.ack_backup_queue
        for backup_id in backup_ids:
            ack_id, completed_tokens = ack_backup_queue.get()
            self.assertEqual(ack_id, backup_id)
            self.assertEqual(completed_tokens, len(operation.hash_value) * config.page_size)
        
        storage_time = time.monotonic() - start_time
        
        # Test prefetch operations
        # Create prefetch operations through the cache controller
        start_time = time.monotonic()
        prefetch_ops = []
        for i in range(0, ENHANCED_OP_NUM * ENHANCED_OP_SIZE, ENHANCED_OP_SIZE):
            # Allocate host indices for prefetch
            host_indices = torch.tensor(list(range(i, i + ENHANCED_OP_SIZE)))
            token_ids = list(range(i, i + ENHANCED_OP_SIZE))
            request_id = f"{i}"

            # Use the cache controller's prefetch method which returns the operation
            operation = cache_controller.prefetch(request_id, host_indices, token_ids, None)
            prefetch_ops.append(operation)

        # Wait for prefetch operations to complete
        for operation in prefetch_ops:
            timeout_secs = 10
            operation_start_time = time.monotonic()
            while operation.completed_tokens < len(operation.hash_value) * config.page_size:
                time.sleep(0.01)
                if time.monotonic() - operation_start_time > timeout_secs:
                    raise TimeoutError(f"Prefetch operation {operation.request_id} timed out")
            # Note: completed_tokens may be 0 if no data was found in storage, which is expected for test

        prefetch_time = time.monotonic() - start_time
        print(f"Storage time: {storage_time:.6f}s")
        print(f"Prefetch time: {prefetch_time:.6f}s")
        print(f"Configuration {config} - PASSED")
        
        # Assertions
        self.assertTrue(storage_time > 0, "Storage operations should take some time")
        self.assertTrue(prefetch_time > 0, "Prefetch operations should take some time")

    def test_all_configurations(self):
        """Test all configuration combinations using parameterized approach"""
        for layout, page_size, io_backend, attention_type in TEST_CONFIGURATIONS:
            with self.subTest(layout=layout, page_size=page_size, io_backend=io_backend, attention_type=attention_type):
                self._run_configuration_test(layout, page_size, io_backend, attention_type)


# =============================================================================
# Main Program
# =============================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)
