"""Resource checks and cleanup scoped to children created by this run only."""
import os,signal,subprocess,time
import psutil

def external_test_pids():
    result=[]
    for p in psutil.process_iter(['pid','cmdline']):
        try:
            args=p.info['cmdline'] or []
            if p.pid!=os.getpid() and args and 'python' in args[0].lower() and ('FingerEyeLab' in ' '.join(args) or 'FingerEyeLab' in p.cwd()):result.append(p.pid)
        except (psutil.NoSuchProcess,psutil.AccessDenied):pass
    return result

def wait_external(deadline,stop,note):
    idle=0
    while idle<2:
        if time.time()>=deadline or stop.exists():raise TimeoutError('Finite deadline or own stop while awaiting external test')
        pids=external_test_pids();note(pids)
        idle=idle+1 if not pids else 0
        if idle<2:time.sleep(30)

def run_owned(command,stream,deadline,stop):
    child=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    try:
        while child.poll() is None:
            if time.time()>=deadline or stop.exists():raise TimeoutError('Finite deadline or own stop during child')
            time.sleep(5)
    finally:
        if child.poll() is None:
            # Capture process identities before graceful shutdown. No process-name matches grant ownership.
            try:descendants=psutil.Process(child.pid).children(recursive=True)
            except psutil.NoSuchProcess:descendants=[]
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=100)
            except subprocess.TimeoutExpired:
                for process in reversed(descendants):
                    try:process.terminate()
                    except psutil.NoSuchProcess:pass
                _,remaining=psutil.wait_procs(descendants,timeout=30)
                for process in remaining:
                    try:process.kill()
                    except psutil.NoSuchProcess:pass
                psutil.wait_procs(remaining,timeout=15)
                if child.poll() is None:os.killpg(child.pid,signal.SIGKILL)
                child.wait(timeout=15)
    return child.returncode
