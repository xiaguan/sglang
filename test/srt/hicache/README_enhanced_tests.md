# HiCache Enhanced Test Suite

这是SGLang HiCache功能的增强测试套件，提供了全面的参数组合测试、长时间运行测试和代码覆盖率分析。

## 功能特性

### 1. 参数组合测试覆盖
- **内存布局**: `layer_first`, `page_first`
- **页面大小**: `1`, `64`
- **IO后端**: `kernel`, `direct`
- **存储后端**: `test_storage` (内存模拟)
- **写入策略**: `write_through` (固定)

总共 **8种配置组合** 的全面测试。

### 2. 长时间运行测试
- 默认运行5分钟的持续操作测试
- 内存使用监控和泄漏检测
- 大量操作执行 (100个操作组 × 1024个token)

### 3. 内存压力测试
- 针对不同页面大小的内存压力场景
- 峰值内存使用监控
- 大批量操作处理测试

### 4. 混合操作模式测试
- 存储和预取操作的交错执行
- 不同配置下的稳定性验证

## 文件结构

```
sglang/test/srt/hicache/
├── test_hicache_mem_storage.py              # 原始测试文件
├── test_hicache_mem_storage_enhanced.py     # 增强测试文件 (unittest)
├── run_with_coverage.py                     # 覆盖率运行脚本
└── README_enhanced_tests.md                 # 本文档
```

## 使用方法

### 1. 基础运行 (无覆盖率)

```bash
# 运行所有增强测试
cd sglang/test/srt/hicache/
python test_hicache_mem_storage_enhanced.py

# 运行特定测试方法
python -m unittest TestHiCacheMemStorageEnhanced.test_config_page_first_page64_kernel -v
```

### 2. 带覆盖率分析的运行

```bash
# 完整运行 (包括长时间测试，约5分钟)
python run_with_coverage.py

# 快速测试 (跳过长时间测试)
python run_with_coverage.py --quick

# 无覆盖率快速测试
python run_with_coverage.py --no-coverage

# 自定义输出目录
python run_with_coverage.py --output-dir my_coverage_reports
```

### 3. 原始测试运行 (对比)

```bash
# 运行原始测试进行对比
python run_with_coverage.py --test-file test_hicache_mem_storage.py
```

## 测试配置

### 可调参数
在 `test_hicache_mem_storage_enhanced.py` 中可以调整以下参数：

```python
# 测试规模参数
ENHANCED_OP_SIZE = 1024          # 每个操作的token数量
ENHANCED_OP_NUM = 100            # 操作组数量
LONG_DURATION_MINUTES = 5        # 长时间测试的分钟数

# 系统参数
MAX_TOTAL_NUM_TOKENS = 12 * 1024 # 最大token数
LAYER_NUM = 64                   # 层数
HEAD_NUM, HEAD_DIM = 8, 128      # 注意力头配置
```

### 环境要求

确保安装了必要的依赖：
```bash
pip install coverage psutil tqdm torch
```

## 测试报告

### 1. 控制台输出
运行时会显示：
- 每个配置的测试进度和性能指标
- 内存使用情况和增长监控
- 测试通过/失败状态

### 2. 覆盖率报告
运行 `run_with_coverage.py` 会生成：
- **HTML报告**: `coverage_reports/html/index.html` (可视化报告)
- **XML报告**: `coverage_reports/coverage.xml` (CI/CD集成)
- **控制台报告**: 直接显示覆盖率百分比

示例输出：
```
COVERAGE REPORT
============================================================
Name                                           Stmts   Miss  Cover   Missing
----------------------------------------------------------------------------
sglang/srt/mem_cache/hicache_storage.py        120      8    93%     45-47, 89
sglang/srt/managers/cache_controller.py        450     45    90%     123-125, ...
sglang/srt/mem_cache/memory_pool_host.py       280     20    93%     67-69, ...
----------------------------------------------------------------------------
TOTAL                                           850     73    91%
```

## 测试方法说明

### 配置矩阵测试
- `test_config_*`: 测试8种不同的配置组合
- 验证每种配置的基本功能和性能

### 长时间运行测试
- `test_long_duration_operations`: 连续5分钟的操作测试
- 监控内存泄漏和系统稳定性

### 内存压力测试
- `test_memory_pressure_page_size_*`: 大批量操作的内存压力测试
- 验证内存管理的正确性

### 混合操作测试
- `test_mixed_operation_patterns`: 存储和预取操作的混合执行
- 测试并发场景下的稳定性

## 性能基准

### 典型性能指标
在标准硬件配置下 (CUDA GPU):
- 存储操作: ~0.1-1.0秒 (16操作组 × 1024token)
- 预取操作: ~0.1-1.0秒 (16操作组 × 1024token)
- 内存增长: <100MB (5分钟长时间测试)

### 配置差异
- `page_first` vs `layer_first`: 性能差异通常<20%
- `kernel` vs `direct`: kernel后端通常性能更好
- `page_size=1` vs `page_size=64`: 大页面通常更高效

## 故障排除

### 常见问题

1. **CUDA内存不足**
   - 减少 `MAX_TOTAL_NUM_TOKENS` 或 `LAYER_NUM`
   - 确保GPU有足够的可用内存

2. **分布式初始化失败**
   - 检查端口23456是否被占用
   - 修改 `distributed_init_method` 中的端口

3. **长时间测试超时**
   - 减少 `LONG_DURATION_MINUTES`
   - 使用 `--quick` 选项跳过长时间测试

4. **覆盖率报告生成失败**
   - 确保安装了 `coverage` 包
   - 检查写入权限

### 调试模式

启用详细日志：
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 贡献指南

### 添加新测试
1. 在 `TestHiCacheMemStorageEnhanced` 类中添加新的测试方法
2. 遵循命名规范: `test_<功能描述>`
3. 使用助手方法减少代码重复
4. 添加适当的断言验证

### 扩展配置矩阵
1. 修改全局变量 `LAYOUTS`, `PAGE_SIZES`, `IO_BACKENDS`
2. 相应地添加新的测试方法
3. 更新文档说明

## 版本历史

- **v1.0**: 初始版本，支持8种配置组合测试
- **v1.1**: 添加长时间运行测试和内存监控
- **v1.2**: 增加覆盖率报告和混合操作测试

## 相关文档

- [SGLang HiCache文档](../../../docs/hicache.md)
- [原始测试文件](test_hicache_mem_storage.py)
- [覆盖率工具文档](https://coverage.readthedocs.io/)
