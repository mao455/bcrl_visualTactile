"""CPU-only publication receipts after the registered evaluation owner exits."""
import subprocess,sys,time,traceback,os
from scripts.tight_recovery_20261118.run import ROOT,read,write

def main():
    assert not (ROOT/'publication_review_started.json').exists()
    write(ROOT/'publication_review_started.json',dict(utc=time.time(),pid=os.getpid()))
    try:
        while not (ROOT/'eval_recovery_exit.json').exists():
            if time.time()>read(ROOT/'session.json')['deadline']:raise TimeoutError('Original deadline')
            time.sleep(5)
        assert read(ROOT/'eval_recovery_exit.json')['children_reaped']
        assert read(ROOT/'eval_recovery_complete.json')['complete']
        for name in ['finalize','transfer','close']:
            began=time.time()
            with (ROOT/('publication_'+name+'.log')).open('x') as handle:
                subprocess.run([sys.executable,'-m','scripts.tight_recovery_20261118.'+name],stdout=handle,stderr=subprocess.STDOUT,check=True)
            write(ROOT/('publication_'+name+'_complete.json'),dict(utc=time.time(),seconds=time.time()-began))
        write(ROOT/'publication_review_complete.json',dict(utc=time.time(),complete=True,scientific_review_and_push_still_required=True))
    except BaseException:
        write(ROOT/'publication_review_failure.json',dict(utc=time.time(),error=traceback.format_exc()));raise
    finally:write(ROOT/'publication_review_exit.json',dict(utc=time.time(),children_reaped=True))
if __name__=='__main__':main()
