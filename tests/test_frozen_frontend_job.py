from pathlib import Path

from tools.tight_frozen_frontend import training_job
from tools.tight_project import commands
from visuotactile.anchor_continue import argument_parser


def test_rear_jobs_satisfy_actual_worker_parser():
    job = training_job(Path('/tmp/registered experiment'))
    queue = commands(job, 'train')
    assert len(queue) == 3
    assert queue[0][3] == 'residual_cache'
    parser = argument_parser()
    for argv,mode in zip(queue[1:], ['residual-train','residual-offline']):
        args = parser.parse_args(argv[3:])
        assert args.mode == mode
        assert args.job == 'image'
        assert args.root == Path('/tmp/registered experiment/B/backend/B')
        assert args.output.parent == args.root
    assert all('tight_fresh_train' not in part for argv in queue for part in argv)
