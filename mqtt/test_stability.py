#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试超声波稳定性检测功能
"""

# 超声波稳定性检测
class UltrasonicStabilityDetector:
    def __init__(self, window_size=8, threshold=3):  # 减小窗口大小，减小阈值
        self.window_size = window_size
        self.threshold = threshold
        self.front_readings = []
        self.left_readings = []
        self.right_readings = []
    
    def add_reading(self, front, left, right):
        """添加新的距离读数"""
        self.front_readings.append(front)
        self.left_readings.append(left)
        self.right_readings.append(right)
        
        # 保持窗口大小
        if len(self.front_readings) > self.window_size:
            self.front_readings.pop(0)
        if len(self.left_readings) > self.window_size:
            self.left_readings.pop(0)
        if len(self.right_readings) > self.window_size:
            self.right_readings.pop(0)
    
    def is_stable(self, readings):
        """检查读数是否稳定"""
        if len(readings) < 5:  # 至少需要5个读数
            return False
        
        # 计算最近读数的平均值和标准差
        recent_readings = readings[-5:]  # 取最近5个读数
        avg = sum(recent_readings) / len(recent_readings)
        
        # 计算标准差
        variance = sum((x - avg) ** 2 for x in recent_readings) / len(recent_readings)
        std_dev = variance ** 0.5
        
        # 检查标准差是否在阈值内（更严格的稳定性判断）
        if std_dev > self.threshold:
            return False
        
        # 额外检查：最新的3个读数都必须在平均值±阈值范围内
        for reading in recent_readings[-3:]:
            if abs(reading - avg) > self.threshold:
                return False
        
        return True
    
    def get_stable_distances(self):
        """获取稳定的距离值"""
        front_stable = self.is_stable(self.front_readings)
        left_stable = self.is_stable(self.left_readings)
        right_stable = self.is_stable(self.right_readings)
        
        return {
            'front': self.front_readings[-1] if front_stable and self.front_readings else None,
            'left': self.left_readings[-1] if left_stable and self.left_readings else None,
            'right': self.right_readings[-1] if right_stable and self.right_readings else None,
            'front_stable': front_stable,
            'left_stable': left_stable,
            'right_stable': right_stable
        }

def test_stability_detector():
    """测试稳定性检测器"""
    print("Testing Enhanced Ultrasonic Stability Detector")
    print("=" * 50)
    
    detector = UltrasonicStabilityDetector(window_size=8, threshold=2)
    
    # 测试1: 稳定的读数
    print("\nTest 1: Stable readings")
    stable_readings = [25, 25, 26, 25, 25, 26, 25, 24]
    
    for i, (f, l, r) in enumerate([(x, x+10, x+5) for x in stable_readings]):
        detector.add_reading(f, l, r)
        result = detector.get_stable_distances()
        print(f"Reading {i+1}: F={f}, L={l}, R={r}")
        print(f"  Stable: F={result['front_stable']}, L={result['left_stable']}, R={result['right_stable']}")
        print(f"  Values: F={result['front']}, L={result['left']}, R={result['right']}")
    
    # 测试2: 不稳定的读数
    print("\nTest 2: Unstable readings")
    detector = UltrasonicStabilityDetector(window_size=8, threshold=2)
    
    unstable_readings = [25, 35, 20, 40, 15, 30, 45, 10]
    
    for i, (f, l, r) in enumerate([(x, x+10, x+5) for x in unstable_readings]):
        detector.add_reading(f, l, r)
        result = detector.get_stable_distances()
        print(f"Reading {i+1}: F={f}, L={l}, R={r}")
        print(f"  Stable: F={result['front_stable']}, L={result['left_stable']}, R={result['right_stable']}")
        print(f"  Values: F={result['front']}, L={result['left']}, R={result['right']}")
    
    # 测试3: 轻微波动的读数（应该被认为是稳定的）
    print("\nTest 3: Slightly fluctuating readings (should be stable)")
    detector = UltrasonicStabilityDetector(window_size=8, threshold=2)
    
    slight_fluctuation = [25, 26, 24, 25, 27, 25, 24, 26]
    
    for i, (f, l, r) in enumerate([(x, x+10, x+5) for x in slight_fluctuation]):
        detector.add_reading(f, l, r)
        result = detector.get_stable_distances()
        print(f"Reading {i+1}: F={f}, L={l}, R={r}")
        print(f"  Stable: F={result['front_stable']}, L={result['left_stable']}, R={result['right_stable']}")
        print(f"  Values: F={result['front']}, L={result['left']}, R={result['right']}")
    
    # 测试4: 中等波动的读数（应该被认为是不稳定的）
    print("\nTest 4: Medium fluctuating readings (should be unstable)")
    detector = UltrasonicStabilityDetector(window_size=8, threshold=2)
    
    medium_fluctuation = [25, 28, 22, 27, 23, 29, 21, 26]
    
    for i, (f, l, r) in enumerate([(x, x+10, x+5) for x in medium_fluctuation]):
        detector.add_reading(f, l, r)
        result = detector.get_stable_distances()
        print(f"Reading {i+1}: F={f}, L={l}, R={r}")
        print(f"  Stable: F={result['front_stable']}, L={result['left_stable']}, R={result['right_stable']}")
        print(f"  Values: F={result['front']}, L={result['left']}, R={result['right']}")

if __name__ == "__main__":
    test_stability_detector()
