import sys
import os
import time
import traceback
import logging
import re
from pathlib import Path
from multiprocessing import Process, Queue
from typing import Dict, Any, Optional

import soundfile as sf


def _fix_pkg_resources():
    try:
        import pkg_resources
        return True
    except ImportError:
        pass
    
    try:
        from setuptools._distutils import errors
    except ImportError:
        pass
    
    try:
        import importlib.metadata
        sys.modules['pkg_resources'] = type(sys)('pkg_resources')
        
        class Distribution:
            def __init__(self, name, version):
                self.project_name = name
                self.version = version
        
        def get_distribution(name):
            try:
                metadata = importlib.metadata.metadata(name)
                return Distribution(name, metadata.get('Version', '0.0.0'))
            except Exception as e:
                print(f"Warning: Could not get distribution for {name}: {e}", file=sys.stderr)
                return Distribution(name, '0.0.0')
        
        sys.modules['pkg_resources'].get_distribution = get_distribution
        sys.modules['pkg_resources'].Distribution = Distribution
        return True
    except Exception as e:
        print(f"Warning: Could not fix pkg_resources: {e}", file=sys.stderr)
        return False

_fix_pkg_resources()

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")
logger = logging.getLogger("fireredasr2s.worker")


class MixedASRSystem:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_paths = config["model_paths"]
        
        self.vad = None
        self.asr = None
        self.punc = None
        
        self.auto_batch = config.get("auto_batch", False)
        self.current_asr_batch = config.get("asr_batch_size", 8)
        self._batch_manager = None
        
        if self.auto_batch:
            from core.auto_batch import AutoBatchManager, BatchConfig
            self._batch_manager = AutoBatchManager(
                target_low=config.get("auto_batch_low"),
                target_high=config.get("auto_batch_high")
            )
            self._batch_manager.config = BatchConfig(asr_batch_size=self.current_asr_batch)
            logger.info(f"Auto-batch starting from: {self.current_asr_batch}, target range: [{self._batch_manager._target_low:.2f}, {self._batch_manager._target_high:.2f}]")
        
        logger.info(f"MixedASRSystem created (ASR batch={self.current_asr_batch}, auto={self.auto_batch})")
        
        self._load_vad()
        self._load_asr()
        
        logger.info("VAD and ASR preloaded! Punc will be loaded on demand.")
    
    def _load_vad(self):
        if self.vad is not None:
            return
        logger.info("Loading VAD model...")
        from fireredasr2s.fireredvad import FireRedVad, FireRedVadConfig
        vad_config = FireRedVadConfig(**self.config["vad"])
        self.vad = FireRedVad.from_pretrained(self.model_paths["vad"], vad_config)
        logger.info("VAD model loaded!")
    
    def _load_asr(self):
        if self.asr is not None:
            return
        logger.info("Loading ASR model...")
        from fireredasr2s.fireredasr2 import FireRedAsr2, FireRedAsr2Config
        asr_config = FireRedAsr2Config(**self.config["asr"])
        self.asr = FireRedAsr2.from_pretrained("aed", self.model_paths["asr"], asr_config)
        logger.info("ASR model loaded!")
    
    def _load_punc(self):
        if self.punc is not None:
            return
        logger.info("Loading Punc model...")
        from fireredasr2s.fireredpunc import FireRedPunc, FireRedPuncConfig
        punc_config = FireRedPuncConfig(**self.config["punc"])
        self.punc = FireRedPunc.from_pretrained(self.model_paths["punc"], punc_config)
        logger.info("Punc model loaded!")
    
    def process(self, wav_path: str, uttid: str = "tmpid", enable_punc: bool = True):
        if enable_punc:
            self._load_punc()
        
        wav_np, sample_rate = sf.read(wav_path, dtype="int16")
        dur = wav_np.shape[0] / sample_rate

        vad_start = time.time()
        vad_result, prob = self.vad.detect((sample_rate, wav_np))
        vad_segments = vad_result["timestamps"]
        vad_duration = time.time() - vad_start
        logger.info(f"VAD detected {len(vad_segments)} segments in {vad_duration:.2f}s")

        asr_results = []
        assert sample_rate == 16000
        
        batch_sizes_used = []
        
        if self.auto_batch and self._batch_manager:
            self.current_asr_batch, reason = self._batch_manager.adjust_batch_sizes(
                self.current_asr_batch
            )
            if reason:
                logger.info(f"Auto batch adjust: {reason}")
        
        BATCH_SIZE = self.current_asr_batch
        logger.info(f"Processing with ASR batch={BATCH_SIZE}")
        batch_sizes_used.append(BATCH_SIZE)
        
        asr_start = time.time()
        i = 0
        total_segments = len(vad_segments)
        logger.info(f"Starting ASR transcription for {total_segments} segments...")
        
        while i < len(vad_segments):
            batch_segments = vad_segments[i:i + BATCH_SIZE]
            batch_asr_uttid = []
            batch_asr_wav = []
            
            for j, (start_s, end_s) in enumerate(batch_segments):
                wav_segment = wav_np[int(start_s * sample_rate):int(end_s * sample_rate)]
                vad_uttid = f"{uttid}_s{int(start_s * 1000)}_e{int(end_s * 1000)}"
                batch_asr_uttid.append(vad_uttid)
                batch_asr_wav.append((sample_rate, wav_segment))
            
            if batch_asr_wav:
                try:
                    batch_num = i // BATCH_SIZE + 1
                    logger.info(f"ASR batch {batch_num}: processing {len(batch_asr_wav)} segments ({i + 1}-{min(i + BATCH_SIZE, total_segments)}/{total_segments})")
                    
                    batch_asr_results = self.asr.transcribe(batch_asr_uttid, batch_asr_wav)
                    logger.debug(f"ASR batch {batch_num} result: {len(batch_asr_results)} items")
                    batch_asr_results = [a for a in batch_asr_results if not re.search(r"(<blank>)|(<sil>)", a["text"])]
                    asr_results.extend(batch_asr_results)
                    i += BATCH_SIZE
                    
                    if self.auto_batch and self._batch_manager and i < len(vad_segments):
                        new_batch, reason = self._batch_manager.adjust_batch_sizes(BATCH_SIZE)
                        if new_batch != BATCH_SIZE:
                            BATCH_SIZE = new_batch
                            batch_sizes_used.append(BATCH_SIZE)
                            if reason:
                                logger.info(f"Auto batch adjust mid-processing: {reason}")
                    
                except RuntimeError as e:
                    if "out of memory" in str(e).lower() and self.auto_batch and self._batch_manager:
                        logger.warning(f"OOM during ASR batch {i//BATCH_SIZE + 1}, reducing batch size...")
                        self.current_asr_batch, should_abort = self._batch_manager.handle_oom(
                            self.current_asr_batch
                        )
                        if should_abort:
                            raise
                        BATCH_SIZE = self.current_asr_batch
                        batch_sizes_used.append(BATCH_SIZE)
                        logger.info(f"Retrying batch at index {i} with size {BATCH_SIZE}")
                        continue
                    raise
            else:
                i += BATCH_SIZE
        
        asr_duration = time.time() - asr_start
        

        avg_batch_size = sum(batch_sizes_used) / len(batch_sizes_used) if batch_sizes_used else self.current_asr_batch
        logger.info(f"Task {uttid} completed - ASR took {asr_duration:.2f}s, batch sizes used: {batch_sizes_used}, average: {avg_batch_size:.1f}")

        batch_info = {
            "batch_sizes_used": batch_sizes_used,
            "avg_batch_size": avg_batch_size
        }

        punc_duration = 0.0
        if enable_punc and self.punc is not None:
            punc_start = time.time()
            punc_results = []
            
            PUNC_BATCH_SIZE = 32
            
            for i in range(0, len(asr_results), PUNC_BATCH_SIZE):
                batch_results = asr_results[i:i + PUNC_BATCH_SIZE]
                batch_asr_text = []
                batch_asr_uttid = []
                batch_asr_timestamp = []
                
                for j, asr_result in enumerate(batch_results):
                    batch_asr_text.append(asr_result["text"])
                    batch_asr_uttid.append(asr_result["uttid"])
                    if "timestamp" in asr_result:
                        batch_asr_timestamp.append(asr_result["timestamp"])

                if batch_asr_text:
                    if self.config["asr"].get("return_timestamp", True):
                        batch_punc_results = self.punc.process_with_timestamp(batch_asr_timestamp, batch_asr_uttid)
                    else:
                        batch_punc_results = self.punc.process(batch_asr_text, batch_asr_uttid)
                    logger.debug(f"Punc batch {i//PUNC_BATCH_SIZE + 1} result: {len(batch_punc_results)} items")
                    punc_results.extend(batch_punc_results)
            
            punc_duration = time.time() - punc_start
            logger.info(f"Task {uttid} - Punc took {punc_duration:.2f}s")
        else:
            punc_results = asr_results

        sentences = []
        words = []
        for asr_result, punc_result in zip(asr_results, punc_results):
            if enable_punc and self.punc is not None:
                assert asr_result["uttid"] == punc_result["uttid"], f"fix code: {asr_result} | {punc_result}"
            
            start_ms, end_ms = asr_result["uttid"].split("_")[-2:]
            assert start_ms.startswith("s") and end_ms.startswith("e")
            start_ms, end_ms = int(start_ms[1:]), int(end_ms[1:])
            
            if self.config["asr"].get("return_timestamp", True):
                sub_sentences = []
                if enable_punc and self.punc is not None:
                    for i, punc_sent in enumerate(punc_result["punc_sentences"]):
                        start = start_ms + int(punc_sent["start_s"] * 1000)
                        end = start_ms + int(punc_sent["end_s"] * 1000)
                        if i == 0:
                            start = start_ms
                        if i == len(punc_result["punc_sentences"]) - 1:
                            end = end_ms
                        sub_sentence = {
                            "start_ms": start,
                            "end_ms": end,
                            "text": punc_sent["punc_text"],
                            "asr_confidence": asr_result["confidence"],
                        }
                        sub_sentences.append(sub_sentence)
                else:
                    sub_sentences = [{
                        "start_ms": start_ms,
                        "end_ms": end_ms,
                        "text": asr_result["text"],
                        "asr_confidence": asr_result["confidence"],
                    }]
                sentences.extend(sub_sentences)
            else:
                text = punc_result["punc_text"] if enable_punc and self.punc is not None else asr_result["text"]
                sentence = {
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                    "text": text,
                    "asr_confidence": asr_result["confidence"],
                }
                sentences.append(sentence)
            
            if "timestamp" in asr_result:
                for w, s, e in asr_result["timestamp"]:
                    word = {"start_ms": int(s * 1000 + start_ms), "end_ms": int(e * 1000 + start_ms), "text": w}
                    words.append(word)

        vad_segments_ms = [(int(s * 1000), int(e * 1000)) for s, e in vad_result["timestamps"]]
        text = "".join(s["text"] for s in sentences)

        result = {
            "uttid": uttid,
            "text": text,
            "sentences": sentences,
            "vad_segments_ms": vad_segments_ms,
            "dur_s": dur,
            "words": words,
            "wav_path": wav_path,
            "batch_info": batch_info,
            "timing": {
                "vad_s": vad_duration,
                "asr_s": asr_duration,
                "punc_s": punc_duration
            }
        }
        
        return result


def worker_entry(input_queue: Queue, status_dict: Dict[str, Any], config: Dict[str, Any], worker_id: int = 0):
    fire_red_asr_path = Path(__file__).parent.parent / "FireRedASR2S"
    if fire_red_asr_path.exists():
        sys.path.insert(0, str(fire_red_asr_path))
    
    asr_system = MixedASRSystem(config)
    logger.info(f"[Worker-{worker_id}] Ready (VAD+ASR preloaded, Punc on demand)")

    while True:
        try:
            task_id = input_queue.get()
            if task_id is None:
                logger.info(f"[Worker-{worker_id}] Received shutdown signal")
                break

            task_data = status_dict.get(task_id)
            if not task_data:
                logger.warning(f"[Worker-{worker_id}] Task {task_id} not found in status_dict")
                continue

            file_path = task_data.get("file_path")
            enable_punc = task_data.get("enable_punc", True)
            logger.info(f"[Worker-{worker_id}] Processing task {task_id}: {file_path} (punc={enable_punc})")

            status_dict[task_id] = {
                "id": task_id,
                "file_path": file_path,
                "status": "processing",
                "enable_punc": enable_punc,
                "worker_id": worker_id
            }

            try:
                start_time = time.time()
                result = asr_system.process(file_path, task_id, enable_punc=enable_punc)
                end_time = time.time()
                
                process_duration = end_time - start_time
                audio_duration = result.get("dur_s", 0)
                rtf = process_duration / audio_duration if audio_duration > 0 else 0

                output_dir = Path(config["results_dir"])
                output_srt = output_dir / f"{task_id}.srt"
                output_txt = output_dir / f"{task_id}.txt"

                with open(output_srt, 'w', encoding='utf-8') as f:
                    for i, sentence in enumerate(result.get('sentences', []), 1):
                        start_ms = sentence.get('start_ms', 0)
                        end_ms = sentence.get('end_ms', 0)
                        text = sentence.get('text', '')

                        start_time_srt = format_srt_time(start_ms)
                        end_time_srt = format_srt_time(end_ms)

                        f.write(f"{i}\n")
                        f.write(f"{start_time_srt} --> {end_time_srt}\n")
                        f.write(f"{text}\n\n")

                with open(output_txt, 'w', encoding='utf-8') as f:
                    f.write(result.get('text', ''))

                batch_info = result.get("batch_info", {})
                timing_info = result.get("timing", {})
                status_dict[task_id] = {
                    "id": task_id,
                    "file_path": file_path,
                    "status": "completed",
                    "result_path": str(output_srt),
                    "enable_punc": enable_punc,
                    "audio_duration": audio_duration,
                    "process_duration": process_duration,
                    "rtf": rtf,
                    "worker_id": worker_id,
                    "batch_sizes_used": batch_info.get("batch_sizes_used", []),
                    "avg_batch_size": batch_info.get("avg_batch_size", 0),
                    "timing_vad": timing_info.get("vad_s", 0),
                    "timing_asr": timing_info.get("asr_s", 0),
                    "timing_punc": timing_info.get("punc_s", 0)
                }
                logger.info(f"[Worker-{worker_id}] Task {task_id} completed! Audio: {audio_duration:.2f}s, Process: {process_duration:.2f}s, RTF: {rtf:.3f}")

            except Exception as e:
                logger.error(f"[Worker-{worker_id}] Error processing task {task_id}: {e}", exc_info=True)
                status_dict[task_id] = {
                    "id": task_id,
                    "file_path": file_path,
                    "status": "failed",
                    "error_msg": str(e),
                    "enable_punc": enable_punc,
                    "worker_id": worker_id
                }

        except Exception as e:
            logger.error(f"[Worker-{worker_id}] Worker error: {e}", exc_info=True)


def format_srt_time(ms: int) -> str:
    hours = ms // 3600000
    ms %= 3600000
    minutes = ms // 60000
    ms %= 60000
    seconds = ms // 1000
    milliseconds = ms % 1000
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"


class WorkerProcess:
    def __init__(self, input_queue: Queue, status_dict: Dict[str, Any], config: Dict[str, Any], worker_id: int = 0):
        self.input_queue = input_queue
        self.status_dict = status_dict
        self.config = config
        self.worker_id = worker_id
        self.process: Process = None

    def start(self):
        self.process = Process(
            target=worker_entry,
            args=(self.input_queue, self.status_dict, self.config, self.worker_id)
        )
        self.process.daemon = True
        self.process.start()
        logger.info(f"Worker-{self.worker_id} process started (PID: {self.process.pid})")

    def stop(self):
        if self.process and self.process.is_alive():
            self.input_queue.put(None)
            self.process.join(timeout=10)
            if self.process.is_alive():
                logger.warning(f"Worker-{self.worker_id} did not stop gracefully, terminating...")
                self.process.terminate()
            else:
                logger.info(f"Worker-{self.worker_id} stopped gracefully")


class WorkerManager:
    def __init__(self, input_queue: Queue, status_dict: Dict[str, Any], config: Dict[str, Any]):
        self.input_queue = input_queue
        self.status_dict = status_dict
        self.config = config
        self.worker: WorkerProcess = None

    def start(self):
        self.worker = WorkerProcess(self.input_queue, self.status_dict, self.config, worker_id=0)
        self.worker.start()
        logger.info("WorkerManager started")

    def stop(self):
        if self.worker:
            self.worker.stop()
            self.worker = None
        logger.info("WorkerManager stopped")

    def is_running(self) -> bool:
        return self.worker is not None and self.worker.process is not None and self.worker.process.is_alive()
