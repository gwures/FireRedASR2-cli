#!/usr/bin/env python3
"""
独立测试模块 - 直接集成测试，不依赖子进程
完全独立，可随时删除此文件而不影响项目
"""
import sys
import os
import time
import random
import statistics
import json
import gc
from pathlib import Path


def setup_env():
    """设置环境，确保能正确导入项目模块"""
    script_dir = Path(__file__).parent
    fire_red_asr_path = script_dir / "FireRedASR2S"
    
    if fire_red_asr_path.exists():
        sys.path.insert(0, str(fire_red_asr_path))
    
    sys.path.insert(0, str(script_dir))
    
    if 'PYTHONPATH' in os.environ:
        os.environ['PYTHONPATH'] = str(fire_red_asr_path) + ";" + os.environ['PYTHONPATH']
    else:
        os.environ['PYTHONPATH'] = str(fire_red_asr_path)


setup_env()

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s"
)
logger = logging.getLogger("test_direct")


def cleanup_resources():
    """清理资源，释放显存和内存"""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        logger.debug("PyTorch cache cleared")
    except ImportError:
        pass
    
    gc.collect()
    logger.debug("Garbage collection done")


class DirectTester:
    """直接测试器，集成核心逻辑"""
    
    def __init__(self, nfp=False, pu=False, ts=False, auto_batch=False, auto_batch_initial=None, auto_batch_low=None, auto_batch_high=None):
        self.nfp = nfp
        self.pu = pu
        self.ts = ts
        self.auto_batch = auto_batch
        self.auto_batch_initial = auto_batch_initial
        self.auto_batch_low = auto_batch_low
        self.auto_batch_high = auto_batch_high
        self.asr_system = None
        self._initialized = False
    
    def _initialize(self):
        """懒加载初始化"""
        if self._initialized:
            return
        
        logger.info("Initializing ASR system...")
        
        from core.worker import MixedASRSystem
        
        import config
        
        worker_config = {
            "model_paths": config.MODEL_PATHS,
            "asr": config.ASR_CONFIG.copy(),
            "vad": config.VAD_CONFIG,
            "punc": config.PUNC_CONFIG.copy(),
            "results_dir": str(config.RESULTS_DIR),
            "asr_batch_size": self.auto_batch_initial if self.auto_batch_initial is not None else 1,
            "auto_batch": self.auto_batch,
            "auto_batch_low": self.auto_batch_low,
            "auto_batch_high": self.auto_batch_high
        }
        
        worker_config["asr"]["use_half"] = not self.nfp
        worker_config["asr"]["return_timestamp"] = self.ts
        worker_config["punc"]["use_gpu"] = True
        
        self.asr_system = MixedASRSystem(worker_config)
        self._initialized = True
        logger.info("ASR system initialized!")
    
    def run_single_test(self, file_path):
        """运行单次测试并返回RTF和使用的批次大小"""
        self._initialize()
        
        try:
            import soundfile as sf
            
            start_time = time.time()
            
            result = self.asr_system.process(
                file_path, 
                uttid=f"test_{int(time.time())}",
                enable_punc=self.pu
            )
            
            process_duration = time.time() - start_time
            audio_duration = result.get("dur_s", 0)
            
            if audio_duration > 0:
                rtf = process_duration / audio_duration
                used_bs = self.asr_system.current_asr_batch
                logger.info(f"  \u2713 Test completed: Process={process_duration:.2f}s, Audio={audio_duration:.2f}s, RTF={rtf:.4f}, used_bs={used_bs}")
                return rtf, used_bs
            else:
                logger.warning(f"  \u26a0 Audio duration is 0")
                return None, None
                
        except Exception as e:
            logger.error(f"  \u2717 Error: {e}", exc_info=True)
            return None, None
    
    def cleanup(self):
        """清理当前测试器的资源"""
        if self.asr_system is not None:
            logger.debug("Cleaning up ASR system resources...")
            try:
                if hasattr(self.asr_system, 'vad'):
                    self.asr_system.vad = None
                if hasattr(self.asr_system, 'asr'):
                    self.asr_system.asr = None
                if hasattr(self.asr_system, 'punc'):
                    self.asr_system.punc = None
                if hasattr(self.asr_system, '_batch_manager'):
                    self.asr_system._batch_manager = None
            except:
                pass
            self.asr_system = None
        self._initialized = False


def save_checkpoint(checkpoint_path, all_results, total_tests, completed_tests):
    """保存中间结果到检查点"""
    try:
        checkpoint = {
            "all_results": all_results,
            "total_tests": total_tests,
            "completed_tests": completed_tests,
            "timestamp": time.time()
        }
        with open(checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump(checkpoint, f)
        logger.debug(f"Checkpoint saved to {checkpoint_path}")
    except Exception as e:
        logger.warning(f"Failed to save checkpoint: {e}")


def load_checkpoint(checkpoint_path):
    """加载检查点"""
    if not Path(checkpoint_path).exists():
        return None
    try:
        with open(checkpoint_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load checkpoint: {e}")
        return None


def auto_batch_range_type(s):
    """解析 --abr 参数，格式为 'low-high'"""
    try:
        parts = s.split('-')
        if len(parts) != 2:
            raise argparse.ArgumentTypeError("auto-batch-range must be in format 'low-high'")
        low = float(parts[0])
        high = float(parts[1])
        if not (0 < low < high < 1):
            raise argparse.ArgumentTypeError("auto-batch-range must be 0 < low < high < 1")
        return low, high
    except Exception as e:
        raise argparse.ArgumentTypeError(f"Invalid auto-batch-range: {e}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Direct Batch Size Performance Test')
    parser.add_argument('-f', '--file', type=str, required=True, help='Input video/audio file path')
    
    mode_group = parser.add_mutually_exclusive_group(required=False)
    mode_group.add_argument('--min-bs', type=int, help='Fixed batch mode: minimum batch size')
    mode_group.add_argument('--ab', '--auto-batch', type=int, help='Auto batch mode: initial batch size')
    mode_group.add_argument('--auto-bs', type=str, help='Auto batch mode with multiple initials, comma-separated (e.g., 8,16,32)')
    
    parser.add_argument('--max-bs', type=int, default=10, help='Fixed batch mode: maximum batch size (default: 10)')
    parser.add_argument('--repeat', type=int, default=10, help='Number of repeats per test case (default: 10)')
    parser.add_argument('--abr', '--auto-batch-range', type=auto_batch_range_type, help='Auto batch mode: target memory utilization range (e.g., 0.8-0.9)')
    parser.add_argument('--min-success', type=int, default=1, help='Minimum successful tests per case to include in report (default: 1)')
    parser.add_argument('--order', type=str, choices=['random', 'sequential'], default='random', help='Test order (random or sequential, default: random)')
    parser.add_argument('--no-checkpoint', action='store_true', help='Disable checkpoint saving')
    parser.add_argument('--resume', action='store_true', help='Resume from last checkpoint if available')
    parser.add_argument('--no-cleanup', action='store_true', help='Disable full cleanup between tests (faster but may have interference)')
    parser.add_argument('--cooldown', type=float, default=2.0, help='Cooldown time between tests in seconds (default: 2.0)')
    parser.add_argument('--nfp', action='store_true', default=False, help='Disable FP16 (default: False, FP16 enabled)')
    parser.add_argument('--fp', action='store_false', dest='nfp', help='Enable FP16 (default)')
    parser.add_argument('--pu', action='store_true', default=False, help='Enable punctuation prediction (default: False)')
    parser.add_argument('--npu', action='store_false', dest='pu', help='Disable punctuation prediction (default)')
    parser.add_argument('--ts', action='store_true', default=False, help='Enable timestamp output (default: False)')
    parser.add_argument('--nts', action='store_false', dest='ts', help='Disable timestamp output (default)')
    parser.add_argument('-o', '--output', type=str, default='test_results_direct.md', help='Output markdown file (default: test_results_direct.md)')
    parser.add_argument('--checkpoint-file', type=str, default='test_checkpoint.json', help='Checkpoint file path (default: test_checkpoint.json)')
    
    args = parser.parse_args()
    
    file_path = Path(args.file)
    if not file_path.exists():
        print(f"Error: File not found: {file_path}")
        return
    
    checkpoint_path = Path(args.checkpoint_file)
    
    all_results = {}
    completed_tests = 0
    test_cases = []
    test_mode = "fixed"
    
    if args.ab is not None:
        test_mode = "auto"
        test_cases = [("auto", args.ab)]
    elif args.auto_bs is not None:
        test_mode = "auto"
        auto_bs_list = [int(x.strip()) for x in args.auto_bs.split(',')]
        test_cases = [("auto", bs) for bs in auto_bs_list]
    else:
        test_mode = "fixed"
        min_bs = args.min_bs if args.min_bs is not None else 1
        max_bs = args.max_bs
        test_cases = [("fixed", bs) for bs in range(min_bs, max_bs + 1)]
    
    total_tests = args.repeat * len(test_cases)
    
    if args.resume and not args.no_checkpoint:
        checkpoint = load_checkpoint(checkpoint_path)
        if checkpoint:
            print(f"\u21bb Resuming from checkpoint...")
            all_results = checkpoint.get("all_results", {})
            completed_tests = checkpoint.get("completed_tests", 0)
            print(f"  Loaded {completed_tests}/{checkpoint.get('total_tests', total_tests)} completed tests")
    
    print("=" * 80)
    print("FireRedASR Direct Batch Size Performance Test")
    print("=" * 80)
    print(f"File: {file_path}")
    print(f"Test mode: {test_mode}")
    if test_mode == "fixed":
        min_bs = args.min_bs if args.min_bs is not None else 1
        print(f"Fixed batch range: {min_bs} to {args.max_bs}")
    else:
        if args.ab is not None:
            print(f"Auto batch initial: {args.ab}")
        else:
            print(f"Auto batch initials: {args.auto_bs}")
        if args.abr:
            print(f"Auto batch range: {args.abr[0]}-{args.abr[1]}")
    print(f"Repeats per test case: {args.repeat}")
    print(f"Minimum successful tests per case: {args.min_success}")
    print(f"Test order: {args.order}")
    print(f"Checkpoint: {'disabled' if args.no_checkpoint else f'enabled ({checkpoint_path})'}")
    print(f"Full cleanup between tests: {'disabled' if args.no_cleanup else 'enabled'}")
    print(f"Cooldown between tests: {args.cooldown}s")
    print(f"Total tests: {total_tests}")
    if completed_tests > 0:
        print(f"Already completed: {completed_tests}")
        print(f"Remaining: {total_tests - completed_tests}")
    fp_status = "FP16 disabled" if args.nfp else "FP16 enabled (default)"
    pu_status = "punctuation enabled" if args.pu else "punctuation disabled (default)"
    ts_status = "timestamp enabled" if args.ts else "timestamp disabled (default)"
    print(f"Flags: {fp_status}, {pu_status}, {ts_status}")
    print("=" * 80)
    print()
    
    test_order = []
    for case in test_cases:
        for i in range(args.repeat):
            case_key = f"{case[0]}_{case[1]}"
            if case_key in all_results:
                if len(all_results[case_key]) > i:
                    continue
            test_order.append(case)
    
    if args.order == 'random':
        random.shuffle(test_order)
        print("Testing in random order...")
    else:
        print("Testing in sequential order...")
    print()
    
    start_time = time.time()
    tester = None
    
    try:
        for idx, case in enumerate(test_order):
            print(f"[{completed_tests + 1}/{total_tests}] ", end="")
            
            mode, bs_val = case
            
            case_label = f"bs={bs_val}"
            if mode == "auto":
                case_label = f"auto_bs={bs_val}"
            print(f"{case_label} - ", end="")
            
            ab_low = None
            ab_high = None
            if args.abr:
                ab_low, ab_high = args.abr
            
            tester = DirectTester(
                nfp=args.nfp,
                pu=args.pu,
                ts=args.ts,
                auto_batch=(mode == "auto"),
                auto_batch_initial=bs_val if mode == "auto" else bs_val,
                auto_batch_low=ab_low,
                auto_batch_high=ab_high
            )
            
            rtf, used_bs = tester.run_single_test(str(file_path))
            
            if rtf is not None:
                case_key = f"{mode}_{bs_val}"
                if case_key not in all_results:
                    all_results[case_key] = []
                result_entry = {
                    "rtf": rtf, 
                    "used_bs": used_bs,
                    "expected_mode": mode,
                    "expected_bs": bs_val,
                    "test_idx": idx,
                    "timestamp": time.time()
                }
                all_results[case_key].append(result_entry)
            
            completed_tests += 1
            
            if not args.no_checkpoint:
                save_checkpoint(checkpoint_path, all_results, total_tests, completed_tests)
            
            tester.cleanup()
            del tester
            tester = None
            
            if not args.no_cleanup:
                cleanup_resources()
            
            if args.cooldown > 0:
                time.sleep(args.cooldown)
            
            print()
    except KeyboardInterrupt:
        print("\n\n\u26a0 Test interrupted by user")
        print("Saving current results...")
        if tester:
            tester.cleanup()
            del tester
        cleanup_resources()
    except Exception as e:
        print(f"\n\n\u274c Error during testing: {e}")
        print("Saving current results...")
        if tester:
            tester.cleanup()
            del tester
        cleanup_resources()
    
    total_duration = time.time() - start_time
    
    print("\n" + "=" * 80)
    print("Generating report...")
    print("=" * 80)
    
    md_content = []
    md_content.append("# FireRedASR Direct Batch Size Performance Test Results\n")
    md_content.append("- **Test File**: " + file_path.name)
    md_content.append("- **Test Mode**: " + test_mode)
    md_content.append("- **Test Duration**: " + str(round(total_duration, 1)) + "s")
    md_content.append("- **Repeats per Case**: " + str(args.repeat))
    md_content.append("- **Minimum Success per Case**: " + str(args.min_success))
    md_content.append("- **Total Tests Planned**: " + str(total_tests))
    md_content.append("- **Total Tests Completed**: " + str(completed_tests))
    md_content.append("- **Full Cleanup Between Tests**: " + ("No" if args.no_cleanup else "Yes"))
    md_content.append("- **Cooldown Between Tests**: " + str(args.cooldown) + "s")
    fp_flag = "--nfp" if args.nfp else "--fp (default)"
    pu_flag = "--pu" if args.pu else "--npu (default)"
    ts_flag = "--ts" if args.ts else "--nts (default)"
    md_content.append("- **Flags**: " + fp_flag + ", " + pu_flag + ", " + ts_flag + "\n")
    
    valid_cases = []
    
    for case_key, results in all_results.items():
        if len(results) >= args.min_success:
            valid_cases.append((case_key, results))
    
    md_content.append("## Test Status Summary\n")
    if not valid_cases:
        md_content.append("\u26a0 **No valid test results available!**\n")
    else:
        md_content.append("| Test Case | Mode | Planned | Successful | Success Rate | Status |")
        md_content.append("|-------------|------|---------|------------|--------------|--------|")
        
        for case in test_cases:
            mode, bs_val = case
            case_key = f"{mode}_{bs_val}"
            planned = args.repeat
            successful = len(all_results.get(case_key, []))
            success_rate = (successful / planned * 100) if planned > 0 else 0
            
            status = "\u2713 Included"
            if successful < args.min_success:
                status = "\u26a0 Excluded (insufficient data)"
            elif successful < planned:
                status = "\u26a0 Partial data"
            
            case_label = f"bs={bs_val}"
            if mode == "auto":
                case_label = f"auto_bs={bs_val}"
            
            row = "| " + case_label + " | " + mode + " | " + str(planned) + " | " + str(successful) + " | " + "{:.1f}".format(success_rate) + "% | " + status + " |"
            md_content.append(row)
        
        md_content.append("")
    
    if valid_cases:
        md_content.append("## Detailed Results (Valid Data Only)\n")
        md_content.append("| Test Case | Mode | Repeats | Mean RTF | Std Dev | Min RTF | Max RTF | Mean Used BS | RTF vs First |")
        md_content.append("|-------------|------|---------|----------|---------|---------|---------|-------------|--------------|")
        
        def get_sort_key(case):
            case_key, _ = case
            mode, bs_str = case_key.split("_", 1)
            try:
                bs_val = int(bs_str)
            except:
                bs_val = 0
            return (mode, bs_val)
        
        sorted_valid_cases = sorted(valid_cases, key=get_sort_key)
        
        first_case_mean = None
        if sorted_valid_cases:
            first_case_key, first_case_results = sorted_valid_cases[0]
            first_rtfs = [r["rtf"] for r in first_case_results]
            first_case_mean = statistics.mean(first_rtfs)
        
        for case_key, results in sorted_valid_cases:
            rtfs = [r["rtf"] for r in results]
            used_bs_list = [r["used_bs"] for r in results if r["used_bs"] is not None]
            
            mean_rtf = statistics.mean(rtfs)
            std_dev = 0
            if len(rtfs) > 1:
                std_dev = statistics.stdev(rtfs)
            min_rtf = min(rtfs)
            max_rtf = max(rtfs)
            mean_used_bs = 0
            if used_bs_list:
                mean_used_bs = statistics.mean(used_bs_list)
            
            vs_first = ""
            if first_case_mean is not None:
                ratio = mean_rtf / first_case_mean
                vs_first = str(round(ratio, 2)) + "x"
            
            mode, bs_val = case_key.split("_", 1)
            case_label = f"bs={bs_val}"
            if mode == "auto":
                case_label = f"auto_bs={bs_val}"
            
            row = "| " + case_label + " | " + mode + " | " + str(len(rtfs)) + " | " + "{:.4f}".format(mean_rtf) + " | " + "{:.4f}".format(std_dev) + " | " + "{:.4f}".format(min_rtf) + " | " + "{:.4f}".format(max_rtf) + " | " + "{:.1f}".format(mean_used_bs) + " | " + vs_first + " |"
            md_content.append(row)
        
        md_content.append("\n## Raw Data\n")
        
        def get_sort_key(case):
            case_key, _ = case
            mode, bs_str = case_key.split("_", 1)
            try:
                bs_val = int(bs_str)
            except:
                bs_val = 0
            return (mode, bs_val)
        
        sorted_valid_cases = sorted(valid_cases, key=get_sort_key)
        
        for case_key, results in sorted_valid_cases:
            mode, bs_val = case_key.split("_", 1)
            case_label = f"bs={bs_val}"
            if mode == "auto":
                case_label = f"auto_bs={bs_val}"
            
            md_content.append("### " + case_label + "\n")
            md_content.append("- **Mode**: " + mode)
            md_content.append("- **Valid tests**: " + str(len(results)) + "/" + str(args.repeat))
            md_content.append("- **Used batch sizes**: " + ", ".join([str(r["used_bs"]) for r in results]))
            md_content.append("\n| Test # | RTF | Used BS | Expected Mode | Expected BS | Test Index | Timestamp |")
            md_content.append("|--------|-----|---------|---------------|-------------|------------|-----------|")
            
            for i, result in enumerate(results):
                test_num = i + 1
                rtf = result["rtf"]
                used_bs = result["used_bs"]
                exp_mode = result.get("expected_mode", "N/A")
                exp_bs = result.get("expected_bs", "N/A")
                test_idx = result.get("test_idx", "N/A")
                ts = result.get("timestamp", "N/A")
                if ts != "N/A":
                    ts = time.strftime("%H:%M:%S", time.localtime(ts))
                
                row = "| " + str(test_num) + " | " + "{:.4f}".format(rtf) + " | " + str(used_bs) + " | " + str(exp_mode) + " | " + str(exp_bs) + " | " + str(test_idx) + " | " + str(ts) + " |"
                md_content.append(row)
            
            md_content.append("\n")
        
        md_content.append("\n## Result Integrity Check\n")
        integrity_errors = []
        total_results = 0
        validated_results = 0
        
        for case_key, results in all_results.items():
            mode, bs_val = case_key.split("_", 1)
            total_results += len(results)
            
            for i, result in enumerate(results):
                validated_results += 1
                expected_mode = result.get("expected_mode")
                expected_bs = result.get("expected_bs")
                
                if expected_mode != mode:
                    error_msg = f"Case {case_key} result {i}: expected_mode={expected_mode}, but case mode={mode}"
                    integrity_errors.append(error_msg)
                
                if expected_bs is not None and str(expected_bs) != str(bs_val):
                    error_msg = f"Case {case_key} result {i}: expected_bs={expected_bs}, but case bs={bs_val}"
                    integrity_errors.append(error_msg)
        
        md_content.append("- **Total results collected**: " + str(total_results))
        md_content.append("- **Results validated**: " + str(validated_results))
        
        if not integrity_errors:
            md_content.append("- **Integrity Status**: \u2713 All results are correctly assigned!")
        else:
            md_content.append("- **Integrity Status**: \u274c " + str(len(integrity_errors)) + " error(s) found!")
            md_content.append("\n### Integrity Errors\n")
            md_content.append("```")
            for error in integrity_errors:
                md_content.append(error)
            md_content.append("```")
        
        md_content.append("\n## Conclusion\n")
        if first_case_mean is not None:
            best_case_key = min(sorted_valid_cases, key=lambda x: statistics.mean([r["rtf"] for r in x[1]]))
            best_case_results = best_case_key[1]
            best_mean = statistics.mean([r["rtf"] for r in best_case_results])
            
            best_mode, best_bs = best_case_key[0].split("_", 1)
            best_case_label = f"bs={best_bs}"
            if best_mode == "auto":
                best_case_label = f"auto_bs={best_bs}"
            
            md_content.append("- **Best test case**: " + best_case_label + " (mode=" + best_mode + ", mean RTF=" + "{:.4f}".format(best_mean) + ")")
            md_content.append("- **First case mean RTF**: " + "{:.4f}".format(first_case_mean))
            if best_case_key != sorted_valid_cases[0]:
                ratio = best_mean / first_case_mean
                if ratio > 1:
                    md_content.append("- **Improvement vs first case**: " + "{:.2f}".format(ratio) + "x (worse)")
                else:
                    md_content.append("- **Improvement vs first case**: " + "{:.2f}".format(ratio) + "x (better)")
            
            if integrity_errors:
                md_content.append("\n\u26a0 **Note**: There are integrity errors in the results. Please verify the data.")
    else:
        md_content.append("\n## Conclusion\n")
        md_content.append("\u26a0 **No valid data to draw conclusions from.**\n")
        md_content.append("- Try increasing `--min-success` or checking for errors in test execution.")
    
    output_path = Path(args.output)
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(md_content))
        print("\n\u2713 Report written to: " + str(output_path))
    except Exception as e:
        print(f"\n\u274c Failed to write report: {e}")
        fallback_path = output_path.with_name(output_path.stem + "_fallback.md")
        try:
            with open(fallback_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(md_content))
            print(f"\n\u2713 Report written to fallback path: {fallback_path}")
        except Exception as e2:
            print(f"\n\u274c Fallback also failed: {e2}")
            print("\nReport content:\n")
            print('\n'.join(md_content))
    
    if not args.no_checkpoint and completed_tests < total_tests:
        print("\n\u2139 To resume this test, run with --resume")
    
    cleanup_resources()
    
    print("\n" + "=" * 80)
    print("Test process completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
