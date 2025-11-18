# -*- coding: utf-8 -*-
"""
专家经验采集脚本
用于生成和保存专家轨迹数据，供RL训练使用
"""

import random
import time
import pickle
import os
from typing import List, Optional, Tuple
from datetime import datetime

# 导入必要的类
from pytorch_solver_demo import (
    Status, Card, BaselineSolver, EnhancedFeatureExtractor
)


class ExpertDataCollector:
    """专家数据采集器"""
    
    def __init__(self, save_dir: str = "expert_data"):
        self.save_dir = save_dir
        self.feature_extractor = EnhancedFeatureExtractor()
        os.makedirs(save_dir, exist_ok=True)
        
    def create_random_layout(self) -> Status:
        """创建随机盘面"""
        s = Status()
        cards = [Card(i) for i in range(40)]
        random.shuffle(cards)
        for i in range(8):
            for j in range(5):
                s.stacks[i].push(cards[i*5 + j])
        s.auto_remove()
        return s
    
    def collect_trajectories(self, 
                           num_games: int = 100,
                           timeout_per_game: float = 30,
                           verbose: bool = True) -> Tuple[List, dict]:
        """
        收集专家轨迹
        
        Returns:
            experiences: 经验列表，每个经验是 (state_features, target_value, priority)
            metadata: 元数据字典
        """
        if verbose:
            print("="*70)
            print("专家经验采集器")
            print("="*70)
            print(f"目标盘面数: {num_games}")
            print(f"每个盘面超时: {timeout_per_game}秒")
            print("="*70)
        
        experiences = []
        successful = 0
        failed = 0
        total_states = 0
        total_time = 0
        
        start_time = time.time()
        
        for game_num in range(1, num_games + 1):
            if verbose:
                print(f"\n[{game_num}/{num_games}] 生成盘面...", end=' ')
            
            # 创建随机盘面
            status = self.create_random_layout()
            
            # 使用A*求解器生成专家轨迹
            solver = BaselineSolver(status)
            game_start = time.time()
            solution = solver.solve(timeout=timeout_per_game)
            game_time = time.time() - game_start
            total_time += game_time
            
            if solution:
                successful += 1
                num_states = len(solution)
                total_states += num_states
                
                if verbose:
                    print(f"✓ 成功 (步数:{num_states}, 时间:{game_time:.1f}s, 探索:{solver.nodes_explored}节点)")
                
                # 提取特征并添加到经验池
                for state, steps_to_goal in solution:
                    features = self.feature_extractor.extract(state)
                    target_value = -steps_to_goal / 50.0  # 归一化价值
                    priority = abs(target_value) + 0.1  # 初始优先级
                    
                    experiences.append({
                        'features': features,
                        'value': target_value,
                        'priority': priority,
                        'steps_to_goal': steps_to_goal
                    })
            else:
                failed += 1
                if verbose:
                    print(f"✗ 失败 (超时)")
            
            # 显示进度统计
            if verbose and game_num % 10 == 0:
                success_rate = 100 * successful / game_num
                avg_time = total_time / game_num
                print(f"    进度: 成功率={success_rate:.1f}%, 平均用时={avg_time:.1f}s, 总样本={total_states}")
        
        elapsed = time.time() - start_time
        
        # 元数据
        metadata = {
            'num_games': num_games,
            'successful': successful,
            'failed': failed,
            'total_states': total_states,
            'total_time': elapsed,
            'timeout_per_game': timeout_per_game,
            'collection_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'feature_dim': 128
        }
        
        if verbose:
            print("\n" + "="*70)
            print("采集完成！")
            print("="*70)
            print(f"成功盘面: {successful}/{num_games} ({100*successful/num_games:.1f}%)")
            print(f"失败盘面: {failed}/{num_games}")
            print(f"总经验数: {total_states}")
            print(f"总用时: {elapsed:.1f}秒")
            print(f"平均每盘: {elapsed/num_games:.1f}秒")
            if successful > 0:
                print(f"平均步数: {total_states/successful:.1f}")
            print("="*70)
        
        return experiences, metadata
    
    def save_experiences(self, experiences: List, metadata: dict, 
                        filename: Optional[str] = None) -> str:
        """保存经验到文件"""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"expert_data_{timestamp}.pkl"
        
        filepath = os.path.join(self.save_dir, filename)
        
        data = {
            'experiences': experiences,
            'metadata': metadata
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        file_size = os.path.getsize(filepath) / (1024 * 1024)  # MB
        print(f"\n✓ 经验已保存: {filepath}")
        print(f"  文件大小: {file_size:.2f} MB")
        print(f"  经验数量: {len(experiences)}")
        
        return filepath
    
    @staticmethod
    def load_experiences(filepath: str) -> Tuple[List, dict]:
        """加载经验文件"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"经验文件不存在: {filepath}")
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        experiences = data['experiences']
        metadata = data['metadata']
        
        print(f"✓ 成功加载经验: {filepath}")
        print(f"  采集日期: {metadata.get('collection_date', '未知')}")
        print(f"  经验数量: {len(experiences)}")
        print(f"  成功盘面: {metadata.get('successful', '?')}/{metadata.get('num_games', '?')}")
        
        return experiences, metadata
    
    def list_available_data(self) -> List[str]:
        """列出可用的经验数据文件"""
        if not os.path.exists(self.save_dir):
            return []
        
        files = [f for f in os.listdir(self.save_dir) if f.endswith('.pkl')]
        files.sort(reverse=True)  # 最新的在前
        return files


def main():
    """主程序"""
    print("="*70)
    print(" " * 20 + "专家经验采集脚本")
    print("="*70)
    print("\n选择操作:")
    print("  1. 收集新的专家经验")
    print("  2. 查看已有的经验文件")
    print("  3. 加载并查看经验文件详情")
    print("="*70)
    
    choice = input("\n请选择 (1-3): ").strip()
    
    collector = ExpertDataCollector()
    
    if choice == "1":
        # 收集新经验
        print("\n" + "="*70)
        print("收集新的专家经验")
        print("="*70)
        
        num_games = int(input("要采集多少个盘面？(推荐100-500): ") or "100")
        timeout = float(input("每个盘面求解超时/秒 (推荐30-60): ") or "30")
        
        # 收集轨迹
        experiences, metadata = collector.collect_trajectories(
            num_games=num_games,
            timeout_per_game=timeout,
            verbose=True
        )
        
        if len(experiences) > 0:
            # 保存
            save_choice = input("\n是否保存这批经验？(y/n): ").strip().lower()
            if save_choice == 'y':
                filename = input("文件名（留空自动生成）: ").strip()
                if not filename:
                    filename = None
                elif not filename.endswith('.pkl'):
                    filename += '.pkl'
                
                collector.save_experiences(experiences, metadata, filename)
        else:
            print("\n✗ 没有收集到任何经验")
    
    elif choice == "2":
        # 查看已有文件
        print("\n" + "="*70)
        print("已有的经验文件")
        print("="*70)
        
        files = collector.list_available_data()
        
        if not files:
            print("未找到任何经验文件")
        else:
            print(f"\n找到 {len(files)} 个文件:\n")
            for i, f in enumerate(files, 1):
                filepath = os.path.join(collector.save_dir, f)
                size = os.path.getsize(filepath) / (1024 * 1024)
                print(f"  {i}. {f} ({size:.2f} MB)")
    
    elif choice == "3":
        # 加载并查看详情
        print("\n" + "="*70)
        print("加载经验文件")
        print("="*70)
        
        files = collector.list_available_data()
        
        if not files:
            print("未找到任何经验文件")
        else:
            print("\n可用文件:\n")
            for i, f in enumerate(files, 1):
                print(f"  {i}. {f}")
            
            file_num = int(input("\n选择文件编号: ").strip())
            
            if 1 <= file_num <= len(files):
                filepath = os.path.join(collector.save_dir, files[file_num - 1])
                
                try:
                    experiences, metadata = collector.load_experiences(filepath)
                    
                    print("\n详细信息:")
                    print("-"*70)
                    for key, value in metadata.items():
                        print(f"  {key}: {value}")
                    
                    # 统计分析
                    if experiences:
                        values = [e['value'] for e in experiences]
                        steps = [e['steps_to_goal'] for e in experiences]
                        
                        print("\n经验统计:")
                        print(f"  价值范围: [{min(values):.4f}, {max(values):.4f}]")
                        print(f"  步数范围: [{min(steps)}, {max(steps)}]")
                        print(f"  平均步数: {sum(steps)/len(steps):.1f}")
                except Exception as e:
                    print(f"\n✗ 加载失败: {e}")
            else:
                print("无效的文件编号")
    
    else:
        print("\n✗ 无效的选择")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n程序已中断")
    except Exception as e:
        print(f"\n\n✗ 发生错误: {e}")
        import traceback
        traceback.print_exc()
