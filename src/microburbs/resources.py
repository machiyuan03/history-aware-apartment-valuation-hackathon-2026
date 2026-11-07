import contextlib
import json
import os
import subprocess
import tempfile
import threading
import time
import psutil
import sys

class Monitor:
    """Process RSS plus device-wide GPU occupancy (not per-process VRAM)."""
    def __init__(self,gpu_every=10):
        self.gpu_every=gpu_every
    def __enter__(self):
        self.start=time.perf_counter(); self.peak_rss=psutil.Process().memory_info().rss; self.gpu=[]; self.stop=threading.Event()
        def sample():
            proc=psutil.Process(); tick=0
            while not self.stop.is_set():
                self.peak_rss=max(self.peak_rss,proc.memory_info().rss)
                if tick%self.gpu_every==0:
                    try:
                        r=subprocess.run(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=3)
                        if r.returncode==0: self.gpu.append(int(r.stdout.splitlines()[0]))
                    except (OSError,ValueError,subprocess.TimeoutExpired): pass
                tick+=1; self.stop.wait(.2)
        torch=sys.modules.get('torch')
        if torch and torch.cuda.is_initialized(): torch.cuda.reset_peak_memory_stats()
        self.thread=threading.Thread(target=sample,daemon=True); self.thread.start(); return self
    def __exit__(self,*args):
        self.stop.set(); self.thread.join(timeout=4)
        self.result={'seconds':time.perf_counter()-self.start,'peak_process_rss_mib':self.peak_rss/2**20,
                     'sampled_peak_device_vram_mib':max(self.gpu) if self.gpu else None,
                     'gpu_note':'Device-wide occupancy includes unrelated processes; model device recorded separately.'}
        torch=sys.modules.get('torch')
        if torch and torch.cuda.is_initialized():
            torch.cuda.synchronize()
            self.result['torch_peak_allocated_mib']=torch.cuda.max_memory_allocated()/2**20
            self.result['torch_peak_reserved_mib']=torch.cuda.max_memory_reserved()/2**20

@contextlib.contextmanager
def gpu_lock(wait=False):
    # Use a local host lock shared by GPU workers.
    import fcntl
    path=os.environ.get('MICROBURBS_GPU_LOCK',os.path.join(tempfile.gettempdir(),'microburbs-gpu-0.lock'))
    with open(path,'a') as f:
        try: fcntl.flock(f,fcntl.LOCK_EX | (0 if wait else fcntl.LOCK_NB))
        except BlockingIOError: raise RuntimeError('Another Microburbs GPU experiment is running')
        try: yield
        finally: fcntl.flock(f,fcntl.LOCK_UN)
