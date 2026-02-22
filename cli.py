#!/usr/bin/env python3

import sys
import os
import logging
import asyncio
import platform
import argparse
from pathlib import Path
from typing import List, Set

if platform.system() == 'Windows':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

try:
    import pkg_resources
except ImportError:
    try:
        from setuptools.extern import pkg_resources
        sys.modules['pkg_resources'] = pkg_resources
    except Exception:
        try:
            import pkg_resources
        except Exception:
            pass

fire_red_asr_path = Path(__file__).parent / "FireRedASR2S"
if fire_red_asr_path.exists():
    sys.path.insert(0, str(fire_red_asr_path))

sys.path.insert(0, str(Path(__file__).parent))

os.environ['PYTHONPATH'] = f"{str(fire_red_asr_path)};{os.environ.get('PYTHONPATH', '')}"

from core import TaskQueue, WorkerManager
from core.auto_batch import AutoBatchManager
from services import FileService
import config

root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

for handler in root_logger.handlers[:]:
    root_logger.removeHandler(handler)

formatter = logging.Formatter("%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
root_logger.addHandler(stream_handler)

logger = logging.getLogger("fireredasr2s.cli")

QUIET_MODE = False


def print_progress(msg: str):
    """打印进度信息，即使在 --quiet 模式下也显示"""
    print(msg, flush=True)


def print_error(msg: str):
    """打印错误信息"""
    print(f"ERROR: {msg}", file=sys.stderr, flush=True)

SUPPORTED_AUDIO_EXTENSIONS = {
    '.wav', '.mp3', '.flac', '.aac', '.ogg', '.m4a', '.wma',
    '.mp4', '.avi', '.mkv', '.mov', '.wmv', '.webm'
}

def auto_batch_range_type(value):
    try:
        low, high = value.split('-')
        low = float(low)
        high = float(high)
        if not (0 < low < high < 1):
            raise ValueError
        return low, high
    except:
        raise argparse.ArgumentTypeError(
            "auto-batch-range must be in format 'low-high', e.g., 0.75-0.95, where 0 < low < high < 1"
        )

def parse_args():
    parser = argparse.ArgumentParser(
        description='FireRedASR2S 命令行工具 - 语音转文字',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  处理单个文件(固定批次):
    python cli.py -f audio.mp3 --bs 8
  
  处理单个文件(自动批次，从 16 开始):
    python cli.py -f audio.mp3 --ab 16
  
  处理多个文件(自动批次，从 32 开始，自定义显存目标):
    python cli.py -f file1.mp3 file2.wav --ab 32 --abr 0.75-0.95
  
  处理整个目录(递归):
    python cli.py -d /path/to/audio -r --ab 16
  
  启用标点预测:
    python cli.py -f audio.mp3 --ab 16 --pu
  
  带时间戳输出:
    python cli.py -f audio.mp3 --ab 16 --ts
  
  禁用 FP16:
    python cli.py -f audio.mp3 --ab 16 --nfp
  
  自定义输出目录:
    python cli.py -f audio.mp3 --ab 16 -o my_output
        """
    )

    input_group = parser.add_argument_group('输入选项')
    input_group.add_argument(
        '-f', '--files',
        nargs='+',
        help='一个或多个音频/视频文件路径'
    )
    input_group.add_argument(
        '-d', '--directory',
        help='目录路径'
    )
    input_group.add_argument(
        '-r', '--recursive',
        action='store_true',
        help='递归搜索子目录中的文件'
    )

    output_group = parser.add_argument_group('输出选项')
    output_group.add_argument(
        '-o', '--output-dir',
        help='输出目录(默认: results)'
    )

    config_group = parser.add_argument_group('配置选项')
    config_group.add_argument(
        '--fp',
        '--fp16',
        action='store_true',
        help='启用 FP16 精度加速(默认)'
    )
    config_group.add_argument(
        '--nfp',
        '--no-fp16',
        action='store_true',
        help='禁用 FP16'
    )
    config_group.add_argument(
        '--pu',
        '--punc',
        action='store_true',
        help='启用标点预测'
    )
    config_group.add_argument(
        '--npu',
        '--no-punc',
        action='store_true',
        help='禁用标点预测(默认)'
    )
    config_group.add_argument(
        '--nts',
        '--no-timestamp',
        action='store_true',
        help='无时间戳(默认)'
    )
    config_group.add_argument(
        '--ts',
        '--timestamp',
        action='store_true',
        help='返回时间戳'
    )
    
    batch_group = config_group.add_mutually_exclusive_group(required=True)
    batch_group.add_argument(
        '--ab',
        '--auto-batch',
        type=int,
        metavar='N',
        help='自动优化批次大小，从 N 开始'
    )
    batch_group.add_argument(
        '--bs',
        '--batch-size',
        type=int,
        metavar='N',
        help='固定 ASR 批次大小，不自动调整'
    )
    
    config_group.add_argument(
        '--abr',
        '--auto-batch-range',
        type=auto_batch_range_type,
        metavar='low-high',
        help='目标显存利用率区间，格式为 low-high，如 --abr 0.75-0.95 (默认: 0.8-0.9)'
    )

    other_group = parser.add_argument_group('其他选项')
    other_group.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='显示详细日志（包括调试信息）'
    )
    other_group.add_argument(
        '-q', '--quiet',
        action='store_true',
        help='静默模式：仅显示进度、统计和错误，隐藏详细日志'
    )

    return parser.parse_args()


def collect_files(args) -> List[Path]:
    files: List[Path] = []
    seen: Set[Path] = set()
    
    extensions = SUPPORTED_AUDIO_EXTENSIONS
    
    if args.files:
        for file_path in args.files:
            path = Path(file_path)
            if path.exists() and path.is_file():
                if path.suffix.lower() in extensions:
                    abs_path = path.resolve()
                    if abs_path not in seen:
                        files.append(abs_path)
                        seen.add(abs_path)
                else:
                    logger.warning(f"跳过不支持的文件格式: {file_path}")
            elif not path.exists():
                logger.error(f"文件不存在: {file_path}")
            else:
                logger.warning(f"跳过目录(使用 -d 参数处理目录): {file_path}")
    
    if args.directory:
        dir_path = Path(args.directory)
        if dir_path.exists() and dir_path.is_dir():
            pattern = '**/*' if args.recursive else '*'
            for path in dir_path.glob(pattern):
                if path.is_file() and path.suffix.lower() in extensions:
                    abs_path = path.resolve()
                    if abs_path not in seen:
                        files.append(abs_path)
                        seen.add(abs_path)
        else:
            logger.error(f"目录不存在或不是目录: {args.directory}")
    
    return sorted(files)


def validate_file(file_path: Path) -> tuple[bool, str]:
    if not file_path.exists():
        return False, "文件不存在"
    
    if not file_path.is_file():
        return False, "不是文件"
    
    if file_path.suffix.lower() not in SUPPORTED_AUDIO_EXTENSIONS:
        return False, f"不支持的文件格式: {file_path.suffix}"
    
    return True, "有效"


def init_system(use_fp16=True, punc_use_gpu=True, 
                asr_batch_size=4, return_timestamp=True,
                auto_batch=False, auto_batch_initial=None,
                auto_batch_low=None, auto_batch_high=None,
                results_dir=None):
    if results_dir is None:
        results_dir = str(config.RESULTS_DIR)
    
    logger.info(f"初始化系统... (FP16={use_fp16}, PuncGPU={punc_use_gpu}, AutoBatch={auto_batch})")
    
    try:
        task_queue = TaskQueue()
        
        if auto_batch:
            final_asr_batch = auto_batch_initial
            logger.info(f"Auto batch enabled: ASR={final_asr_batch} (FP16={use_fp16})")
        else:
            final_asr_batch = asr_batch_size

        asr_config = config.ASR_CONFIG.copy()
        asr_config["use_half"] = use_fp16
        asr_config["return_timestamp"] = return_timestamp
        
        punc_config = config.PUNC_CONFIG.copy()
        punc_config["use_gpu"] = punc_use_gpu

        worker_config = {
            "model_paths": config.MODEL_PATHS,
            "asr": asr_config,
            "vad": config.VAD_CONFIG,
            "punc": punc_config,
            "results_dir": str(results_dir),
            "asr_batch_size": final_asr_batch,
            "auto_batch": auto_batch,
            "auto_batch_low": auto_batch_low,
            "auto_batch_high": auto_batch_high
        }

        worker_manager = WorkerManager(
            task_queue.get_queue(),
            task_queue.get_status_dict(),
            worker_config
        )
        worker_manager.start()

        file_service = FileService(str(config.UPLOAD_DIR), str(results_dir))
        
        logger.info(f"系统初始化完成！")
        return task_queue, worker_manager, file_service
        
    except Exception as e:
        logger.error(f"系统初始化失败: {e}", exc_info=True)
        return None, None, None


async def process_files_async(files: List[Path], args):
    punc_use_gpu = True
    
    if args.nfp:
        use_fp16 = False
    else:
        use_fp16 = True
    
    if args.pu:
        enable_punc = True
    else:
        enable_punc = False
    
    if args.ts:
        plain_text_mode = False
    else:
        plain_text_mode = True
    
    if args.ab is not None:
        auto_batch = True
        auto_batch_initial = args.ab
        asr_batch_size = 4
    else:
        auto_batch = False
        auto_batch_initial = None
        asr_batch_size = args.bs
    
    auto_batch_low = None
    auto_batch_high = None
    if args.abr:
        auto_batch_low, auto_batch_high = args.abr
    
    return_timestamp = not plain_text_mode
    
    output_dir = Path(args.output_dir) if args.output_dir else config.RESULTS_DIR
    output_dir.mkdir(exist_ok=True)
    
    valid_files = []
    for file_path in files:
        is_valid, reason = validate_file(file_path)
        if is_valid:
            valid_files.append(file_path)
        else:
            if not QUIET_MODE:
                logger.warning(f"跳过文件 {file_path.name}: {reason}")
    
    if not valid_files:
        print_error("没有有效的文件需要处理")
        return
    
    print_progress(f"准备处理 {len(valid_files)} 个文件")
    
    task_queue, worker_manager, file_service = init_system(
        use_fp16=use_fp16,
        punc_use_gpu=punc_use_gpu,
        asr_batch_size=asr_batch_size,
        return_timestamp=return_timestamp,
        auto_batch=auto_batch,
        auto_batch_initial=auto_batch_initial,
        auto_batch_low=auto_batch_low,
        auto_batch_high=auto_batch_high,
        results_dir=str(output_dir)
    )
    
    if not task_queue or not worker_manager or not file_service:
        print_error("系统初始化失败")
        return
    
    wav_files_to_cleanup = []
    
    try:
        print_progress("正在转换音频格式...")
        wav_paths = file_service.convert_multiple_to_wav([str(f) for f in valid_files])
        
        for wav_path in wav_paths:
            if wav_path:
                wav_files_to_cleanup.append(Path(wav_path))
        
        task_ids = []
        task_stats = {}
        
        for i, (original_file, wav_path) in enumerate(zip(valid_files, wav_paths)):
            if not wav_path:
                print_progress(f"警告: 音频转换失败: {original_file.name}")
                continue
            
            task = task_queue.create_task(wav_path, enable_punc=enable_punc)
            task_ids.append(task["id"])
            task_stats[task["id"]] = {
                "original_file": original_file,
                "status": "queued"
            }
        
        if not task_ids:
            print_error("没有任务被创建")
            return
        
        print_progress(f"已添加 {len(task_ids)} 个任务到队列，开始处理...")
        
        completed_count = 0
        failed_count = 0
        
        while True:
            all_done = True
            for task_id in task_ids:
                task = task_queue.get_task(task_id)
                if task:
                    status = task.get("status")
                    if status == "queued" or status == "processing":
                        all_done = False
                    elif status == "completed" and task_stats[task_id]["status"] != "completed":
                        task_stats[task_id]["status"] = "completed"
                        completed_count += 1
                        audio_dur = task.get("audio_duration", 0)
                        process_dur = task.get("process_duration", 0)
                        rtf = task.get("rtf", 0)
                        print_progress(f"[{completed_count}/{len(task_ids)}] 完成: {task_stats[task_id]['original_file'].name} "
                                      f"(音频: {audio_dur:.2f}s, 处理: {process_dur:.2f}s, RTF: {rtf:.3f})")
                    elif status == "failed" and task_stats[task_id]["status"] != "failed":
                        task_stats[task_id]["status"] = "failed"
                        failed_count += 1
                        error_msg = task.get("error_msg", "未知错误")
                        print_progress(f"[!] 失败: {task_stats[task_id]['original_file'].name} - {error_msg}")
            
            if all_done:
                break
            
            await asyncio.sleep(0.5)
        
        print_progress("=" * 60)
        print_progress(f"处理完成! 成功: {completed_count}, 失败: {failed_count}")
        print_progress(f"输出目录: {output_dir.resolve()}")
        
    finally:
        cleanup_errors = []
        try:
            worker_manager.stop()
        except Exception as e:
            cleanup_errors.append(f"worker_manager.stop: {e}")
        try:
            task_queue.shutdown()
        except Exception as e:
            cleanup_errors.append(f"task_queue.shutdown: {e}")
        try:
            file_service.shutdown()
        except Exception as e:
            cleanup_errors.append(f"file_service.shutdown: {e}")
        try:
            if wav_files_to_cleanup:
                if not QUIET_MODE:
                    print_progress("清理临时 WAV 文件...")
                for wav_file in wav_files_to_cleanup:
                    try:
                        if wav_file.exists():
                            wav_file.unlink()
                            if not QUIET_MODE:
                                logger.debug(f"已删除: {wav_file}")
                    except Exception as e:
                        logger.warning(f"删除临时文件失败 {wav_file}: {e}")
        except Exception as e:
            cleanup_errors.append(f"temp_files_cleanup: {e}")
        if cleanup_errors and not QUIET_MODE:
            logger.warning(f"清理过程中出现错误: {'; '.join(cleanup_errors)}")


def main():
    global QUIET_MODE
    args = parse_args()
    
    QUIET_MODE = args.quiet
    
    if args.verbose:
        root_logger.setLevel(logging.DEBUG)
    elif args.quiet:
        root_logger.setLevel(logging.ERROR)
    
    if not args.files and not args.directory:
        print_error("请指定输入文件 (-f) 或目录 (-d)")
        sys.exit(1)
    
    files = collect_files(args)
    
    if not files:
        print_error("没有找到符合条件的文件")
        sys.exit(1)
    
    print_progress(f"找到 {len(files)} 个文件")
    
    try:
        asyncio.run(process_files_async(files, args))
    except KeyboardInterrupt:
        print_progress("\n用户中断")
        sys.exit(130)
    except Exception as e:
        print_error(f"处理出错: {e}")
        if not QUIET_MODE:
            logger.error(f"处理出错: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
