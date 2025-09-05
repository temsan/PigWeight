#!/usr/bin/env python3
import torch

print("=== GPU Memory Information ===")
print(f"GPU: {torch.cuda.get_device_name(0)}")

# Получаем свойства устройства
props = torch.cuda.get_device_properties(0)
total_memory_gb = props.total_memory // 1024 // 1024 // 1024
total_memory_mb = props.total_memory // 1024 // 1024

print(f"Total memory: {total_memory_gb} GB ({total_memory_mb} MB)")

# Текущая память
allocated_mb = torch.cuda.memory_allocated(0) // 1024 // 1024
reserved_mb = torch.cuda.memory_reserved(0) // 1024 // 1024
free_mb = (props.total_memory - torch.cuda.memory_reserved(0)) // 1024 // 1024

print(f"Allocated: {allocated_mb} MB")
print(f"Reserved: {reserved_mb} MB")
print(f"Free: {free_mb} MB")

# Процент использования
usage_percent = (reserved_mb / total_memory_mb) * 100
print(".1f")

# Максимальный размер блока памяти
max_memory_mb = torch.cuda.get_device_properties(0).max_memory_allocated // 1024 // 1024
print(f"Max memory ever allocated: {max_memory_mb} MB")

print("\n=== Memory Summary ===")
print(f"GPU Memory Usage: {usage_percent:.1f}%")
if usage_percent < 50:
    print("✅ Memory usage is low - good for processing")
elif usage_percent < 80:
    print("⚠️  Memory usage is moderate")
else:
    print("🔴 Memory usage is high - consider optimization")
