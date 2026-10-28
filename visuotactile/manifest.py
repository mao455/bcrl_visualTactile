"""Explicit multi-episode index over legacy or compact lossless HDF5 records."""
import json
from pathlib import Path
from .compact import sha256
from .dataset import EpisodeSequenceDataset
from .recording import EpisodeReader


def _calibration_identity(metadata):
    """Stable identity for all sensor geometry used by a manifest split."""
    cameras = []
    for camera in metadata.get('cameras', []):
        cameras.append({k: camera.get(k) for k in
                        ('id', 'resolution', 'intrinsics',
                         'clipping_range_m', 'extrinsic_role')})
    # A wrist camera's world pose legitimately changes with the initial robot
    # state; it is not a calibration change.  The rig contract carries its
    # fixed local mount and fixed-view eyes/targets, so use that identity while
    # excluding per-episode dynamic `pos_env`/quaternion values.
    rig = metadata.get('camera_rig', {})
    rig_identity = {k: rig.get(k) for k in ('version', 'name', 'fixed_views',
                                            'fixed_focal_length', 'wrist')}
    return json.dumps({'rig': rig_identity, 'cameras': cameras},
                      sort_keys=True, separators=(',', ':'))


class ManifestReader:
    def __init__(self, path, verify=False):
        self.path = Path(path).resolve()
        self.manifest = json.loads(self.path.read_text())
        if self.manifest.get("manifest_version") != 1:
            raise ValueError("Unknown manifest version")
        self.entries = {r["id"]: r for r in self.manifest["episodes"]}
        if len(self.entries) != len(self.manifest["episodes"]):
            raise ValueError("Duplicate episode identity")
        for row in self.entries.values():
            if not row.get("valid"):
                raise ValueError("Unvalidated row in committed manifest")
            views=len(row.get('camera_ids',['global_oblique','local_opposite']))
            expected_video_names={f'camera_{v}.mp4' for v in range(views)}
            declared=set(row.get('video_files', expected_video_names))
            if declared - expected_video_names:
                raise ValueError("Manifest declares an unknown camera video")
            if set(row["files"]) != {'data.h5'} | declared:
                raise ValueError("A committed episode must contain data.h5 and only declared optional camera videos; legacy rows require one video per declared camera")
            for filename, record in row["files"].items():
                file = (self.path.parent/row["directory"]/filename).resolve()
                if not file.is_relative_to(self.path.parent):
                    raise ValueError("Manifest path traversal")
                if file.stat().st_size != record["bytes"] or (verify and sha256(file) != record["sha256"]):
                    raise ValueError("Manifest file checksum/size mismatch: "+str(file))

    def reader(self, episode_id):
        return EpisodeReader(self.path.parent/self.entries[episode_id]["directory"]/"data.h5")

    def steps(self, episode_id):
        yield from self.reader(episode_id).steps(self.entries[episode_id]["episode"])


class ManifestSequenceDataset:
    """Successful scan episodes by default; acceptance trials stay separate.

    Split whole initial-state/pair groups, not rows. Explicit stage=None allows
    other stages but still rejects mixed wrench compatibility configurations.

    Pure expert captures have no own-plan distribution. This wrapper does not
    resolve that research gap; a future trainer must explicitly choose a
    no-own-plan baseline or provide identified causal model replay predictions.
    """
    def __init__(self, manifest, *, allow_unsuccessful=False, allow_incomplete=False,
                 ids=None, horizon=12, stage="scan"):
        self.reader = ManifestReader(manifest)
        if ids is not None and set(ids)-self.reader.entries.keys():
            raise ValueError("Unknown requested episode identity")
        self.episodes = []
        self.datasets = {}
        for key, row in self.reader.entries.items():
            if stage is not None and row.get("stage","scan") != stage:
                continue
            if ids is not None and key not in ids:
                continue
            if not allow_unsuccessful and row["outcome"] != "success":
                continue
            if not allow_incomplete and row["outcome"] == "censored":
                continue
            self.datasets[key] = EpisodeSequenceDataset(
                self.reader.path.parent/row["directory"]/"data.h5", horizon=horizon,
                allow_unsuccessful=allow_unsuccessful, allow_incomplete=allow_incomplete)
            self.episodes.append(key)
        if not self.episodes:
            raise ValueError("No eligible episodes")
        modes={self.reader.entries[k].get("wrench_frame_mode","legacy") for k in self.episodes}
        if len(modes)>1:
            raise ValueError("Do not silently mix legacy/world_hold collection configurations")
        layouts={(d.metadata.get('camera_rig',{}).get('version',2),
                  tuple(d.metadata.get('camera_ids',[])),
                  tuple(tuple(c['resolution']) for c in d.metadata.get('cameras',[])),
                  _calibration_identity(d.metadata))
                 for d in self.datasets.values()}
        if len(layouts)>1:
            raise ValueError('Do not mix camera layouts/resolutions in one Dataset')

    def chunks(self, episode, chunk_size=8):
        row = self.reader.entries[episode]
        for chunk in self.datasets[episode].chunks(row["episode"],chunk_size):
            yield {**chunk,"episode":episode,"initial_state_group":row["pair_id"]}

    def require_self_plan_training(self):
        for key in self.episodes:
            meta = self.datasets[key].metadata
            if meta.get("nominal_plan_source") != "trained_checkpoint" or not meta.get("nominal_model_sha256"):
                raise ValueError("Self-plan-conditioned training needs genuine causal model replay; expert labels are not plans")


def assert_group_disjoint(reader, train_ids, validation_ids):
    groups = lambda ids: {reader.entries[k]["pair_id"] for k in ids}
    if groups(train_ids) & groups(validation_ids):
        raise ValueError("Initial-state variants leak across train/validation")
