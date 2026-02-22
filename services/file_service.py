import os
import logging
from pathlib import Path
from typing import Optional, List
from concurrent.futures import ThreadPoolExecutor
import subprocess

logger = logging.getLogger("fireredasr2s.file_service")


class FileService:
    def __init__(self, upload_dir: str, results_dir: str):
        self.upload_dir = Path(upload_dir)
        self.results_dir = Path(results_dir)
        self.upload_dir.mkdir(exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)
        cpu_count = os.cpu_count() or 4
        max_workers = min(cpu_count * 2, 16)
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="ffmpeg")

    def _get_safe_filename(self, directory: Path, filename: str) -> str:
        dangerous_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|', '\0']
        safe_filename = ''.join(c if c not in dangerous_chars else '_' for c in filename)
        
        if not safe_filename or safe_filename.strip() == '':
            import time
            safe_filename = f"file_{int(time.time())}"
        
        base_path = directory / safe_filename
        if not base_path.exists():
            return safe_filename
        
        stem = Path(safe_filename).stem
        ext = Path(safe_filename).suffix
        counter = 1
        while (directory / f"{stem}_{counter}{ext}").exists():
            counter += 1
        
        return f"{stem}_{counter}{ext}"

    def convert_to_wav(self, input_path: str) -> Optional[str]:
        input_path = Path(input_path)
        output_path = self.upload_dir / f"{input_path.stem}.wav"

        counter = 1
        while output_path.exists() and output_path != input_path:
            output_path = self.upload_dir / f"{input_path.stem}_{counter}.wav"
            counter += 1

        try:
            result = subprocess.run([
                'ffmpeg',
                '-y',
                '-i', str(input_path),
                '-threads', str(os.cpu_count() or 4),
                '-ar', '16000',
                '-ac', '1',
                '-acodec', 'pcm_s16le',
                '-f', 'wav',
                '-hide_banner',
                '-loglevel', 'error',
                str(output_path)
            ], check=True, capture_output=True, encoding='utf-8', errors='ignore')
            logger.info(f"Converted {input_path} to {output_path}")
            return str(output_path)
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg conversion failed: {e.stderr}")
            return None
        except Exception as e:
            logger.error(f"Conversion error: {e}")
            return None

    def convert_multiple_to_wav(self, input_paths: List[str]) -> List[Optional[str]]:
        futures = [self._executor.submit(self.convert_to_wav, path) for path in input_paths]
        results = [future.result() for future in futures]
        return results

    def shutdown(self):
        self._executor.shutdown(wait=False)
        logger.info("FileService executor shutdown")
