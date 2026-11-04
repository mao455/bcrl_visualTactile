"""Final checkpoint evaluation: isolated Kit paths and read-only controller audit."""
import argparse
from pathlib import Path
import time

from tools.state_image_queue import read, write, sha


def run(root, output, job_path):
    import torch
    from isaaclab import app as launcher
    from visuotactile import state_benchmark as base, native_controller
    from tools.state_image_kit_diagnostic import run as evaluate
    job = read(job_path)
    assert job['mode'] == 'dp' and len(job['specs']) == 1
    assert not any(k in job for k in ('takeover_steps', 'replay'))
    session = read(root/'session.json')
    original_launcher = launcher.AppLauncher

    def isolated_launcher(*args, **kwargs):
        # Preserve the exact prior persistent values, but never write shared paths.
        # Fresh simulator processes share only this run's seeded shader cache.
        # No physical/environment/controller state is reused between workers.
        private = root/'private_runtime'
        private.mkdir(parents=True, exist_ok=True)
        kwargs['kit_args'] = kwargs.get('kit_args', '') + (
            f' --portable-root {private}'
            f' --/app/userConfigPath={root / "freeze/user.config.json"}'
            ' --/app/settings/persistent=false')
        app = original_launcher(*args, **kwargs)
        import carb.tokens
        tokens = carb.tokens.get_tokens_interface()
        write(output/'isolated_paths.json', dict(
            tokens={k:tokens.resolve('${'+k+'}') for k in ('data', 'cache', 'logs')},
            copied_user_config_sha256=sha(root/'freeze/user.config.json')))
        return app
    launcher.AppLauncher = isolated_launcher
    audit = dict(model=job['model'], updates=10000, checkpoint_sha256=job['checkpoint_sha256'],
        act_calls=0, execution_acks=0, episode_ids=[], ticks=[], model_output_only=True,
        teacher_forcing=False, expert_takeover=False, created_utc=time.time())

    def instrument(cls, state):
        original_act, original_ack = cls.act, cls.observe_execution
        def act(self, *args, **kwargs):
            packet = args[1] if state else args[0]
            tick, episode = int(packet['step_id'][0]), int(packet['episode_id'][0])
            if not audit['act_calls']:
                assert tick == 0 and self.episode is None
                assert self.plan is None
                audit['history_initially_empty'] = True
                audit['model_class'] = type(self.model).__name__
                audit['model_contract'] = self.model.contract
            assert tick == audit['act_calls'], 'Skipped/duplicated native controller tick'
            if episode not in audit['episode_ids']: audit['episode_ids'].append(episode)
            assert len(audit['episode_ids']) == 1, 'Unexpected episode reset'
            result = original_act(self, *args, **kwargs)
            assert torch.isfinite(result).all()
            audit['act_calls'] += 1
            if tick % 100 == 0: write(output/'controller_audit.json', audit)
            return result
        def ack(self, *args, **kwargs):
            result = original_ack(self, *args, **kwargs)
            audit['execution_acks'] += 1
            return result
        cls.act, cls.observe_execution = act, ack
    instrument(base.HistoryController, True)
    instrument(native_controller.NativeController, False)
    original_trial = base.trial
    def trial(*args, **kwargs):
        result = original_trial(*args, **kwargs)
        audit['completed'] = bool(result.get('legal') and audit['act_calls'] == result['steps']
            and audit['execution_acks'] == result['steps'])
        write(output/'controller_audit.json', audit)
        assert audit['completed'], 'Controller execution/history audit incomplete'
        return result
    base.trial = trial
    original_video = base.RolloutVideo
    class Video(original_video):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.title = f"{job['model']} | updates 10000 | {job['specs'][0]['identity']} | autonomous"
    base.RolloutVideo = Video
    evaluate(root, output, job_path)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--job', type=Path, required=True)
    a = p.parse_args()
    import torch
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    run(a.root, a.output, a.job)
