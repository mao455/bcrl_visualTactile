"""Episode-based HDF5 recorder; model inputs and privileged labels are separate."""
import json
import hashlib
import os
import pickle
from pathlib import Path
import h5py
from .compact import EpisodeFile, VirtualArray, IMAGE_FIELDS, rgb_u8
import numpy as np


def model_fingerprint(model):
    """Parameter/buffer identity, independent of checkpoint filename."""
    digest = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        digest.update(key.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()

def numpy_tree(value):
    if isinstance(value, dict):
        return {k: numpy_tree(v) for k, v in value.items()}
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy().copy()
    return np.asarray(value)


def flatten(value, prefix=""):
    result = {}
    for key, child in value.items():
        path = f"{prefix}/{key}" if prefix else key
        if isinstance(child, dict):
            result.update(flatten(child, path))
        else:
            result[path] = child
    return result


class EpisodeWriter:
    def __init__(self, path, metadata):
        self.file = h5py.File(path, "x")
        self.file.attrs["schema_version"] = metadata.get("perception_version", 1)
        self.file.attrs["trace_version"] = metadata.get("trace_version",1)
        self.file.attrs["metadata_json"] = json.dumps(metadata, default=str)
        self.pending = {}
        self.count = 0

    def append(self, observation, transition, terminated, truncated, action_source):
        obs, tr = numpy_tree(observation), numpy_tree(transition)
        terminated, truncated = numpy_tree(terminated), numpy_tree(truncated)
        for i in range(len(obs["episode_id"])):
            key = (i, int(obs["episode_id"][i]))
            if int(tr["episode_id"][i]) != key[1] or tr["step_id"][i] != obs["step_id"][i]:
                raise ValueError("Observation/action episode or step misalignment")
            record = flatten({"obs": obs, "transition": tr})
            record = {k: v[i] for k, v in record.items()}
            record["terminated"] = terminated[i]
            record["truncated"] = truncated[i]
            group_name = f"episodes/env{key[0]:03d}_ep{key[1]:06d}"
            group = self.file.require_group(group_name)
            length = int(group.attrs.get("length",0))
            if int(obs["step_id"][i]) != length:
                raise ValueError("Episode must start at zero and record every control step once")
            if length == 0:
                group.attrs.update(complete=False,action_source=action_source,length=0,
                                   terminal_image_available=False)
            for name, value in record.items():
                arr = np.asarray(value)
                if length == 0:
                    ds = group.create_dataset(name,shape=(0,*arr.shape),maxshape=(None,*arr.shape),
                                              chunks=(1,*arr.shape),dtype=arr.dtype,
                                              compression="gzip",compression_opts=1)
                else:
                    ds = group[name]
                ds.resize(length+1,axis=0)
                ds[length] = arr
            # Publish length last: readers use only committed rows after a partial write.
            group.attrs["length"] = length+1
            self.pending[key] = group_name
            if bool(terminated[i] or truncated[i]):
                group.attrs["complete"] = True
                self.pending.pop(key)
                self.count += 1
            self.file.flush()

    def close(self, action_source):
        self.file.close()


class CompactEpisodeStream:
    """One independent episode, lossless storage2, bounded eight-row writes.

    Called by one writer thread only. All original fields have incremental
    source hashes; the finalizer reconstructs and verifies them before the
    manifest publishes the episode. No duplicate raw HDF5/conversion pass.
    """
    def __init__(self, directory, metadata, episode, block_size=8):
        self.directory=Path(directory);self.directory.mkdir(parents=True,exist_ok=False)
        self.path=self.directory/'data.h5.partial'
        self.file=h5py.File(self.path,'x')
        self.file.attrs.update(schema_version=2,trace_version=2,storage_version=2,
            metadata_json=json.dumps(metadata,default=str),stream_writer_version=1,
            video_role='optional_viewing_derivative_not_training_RGB')
        self.group=self.file.create_group('episodes/'+episode)
        self.group.attrs.update(length=0,complete=False,action_source='released_rl_checkpoint',
                                terminal_image_available=False)
        self.group.require_group('frames').attrs.update(extra_final_post=False,missing_frame_index=-1)
        self.rows=[];self.block_size=block_size;self.length=0;self.previous=None
        self.hashes={};self.specs={}

    def append(self, row):
        step=int(row['obs/step_id'])
        if step!=self.length or step!=int(row['transition/step_id']):
            raise ValueError('Independent stream step mismatch')
        if row['obs/episode_id']!=row['transition/episode_id']:
            raise ValueError('Independent stream episode mismatch')
        if self.previous is not None:
            if not self.previous['transition/observation_after/available']:
                raise ValueError('Append after terminal row')
            for key in ('image','depth_valid','camera_time_s','camera_frame','episode_id','step_id',
                        'camera_pose_w','camera_parent_pose_w'):
                if 'obs/'+key in row and not np.array_equal(
                        self.previous['transition/observation_after/'+key],row['obs/'+key]):
                    raise ValueError('Post/pre dedup misalignment: '+key)
        if not np.array_equal(row['obs/image'][:,3:],row['obs/depth_m']):
            raise ValueError('Duplicate depth differs')
        if not row['transition/observation_after/available']:
            for key in ('image','depth_valid'):
                if np.any(row['transition/observation_after/'+key]):
                    raise ValueError('Terminal image contains reset data')
        for key,value in row.items():
            a=np.asarray(value)
            if key not in self.hashes:
                self.hashes[key]=hashlib.sha256()
                self.specs[key]=dict(dtype=a.dtype.str,row_shape=list(a.shape))
            if self.specs[key]!=dict(dtype=a.dtype.str,row_shape=list(a.shape)):
                raise ValueError('Field shape/dtype changed: '+key)
            self.hashes[key].update(np.ascontiguousarray(a).tobytes())
        if set(row)!=set(self.hashes):raise ValueError('Stream field set changed')
        self.rows.append(row);self.previous=row;self.length+=1
        if len(self.rows)>=self.block_size:self.flush()

    def _extend(self,key,values):
        values=np.asarray(values)
        if key not in self.group:
            # LZF is lossless; less CPU work than the former raw gzip plus
            # second gzip conversion. Exact metric depth is retained.
            chunks=(1,*values.shape[1:]) if key.startswith('frames/') and values.ndim>2 else (self.block_size,*values.shape[1:])
            self.group.create_dataset(key,shape=(0,*values.shape[1:]),maxshape=(None,*values.shape[1:]),
                dtype=values.dtype,chunks=chunks,compression='lzf',shuffle=True)
        ds=self.group[key];start=len(ds);ds.resize(start+len(values),axis=0);ds[start:]=values

    def flush(self):
        if not self.rows:return
        for key in self.rows[0]:
            if key not in IMAGE_FIELDS:self._extend(key,np.stack([r[key] for r in self.rows]))
        images=np.stack([r['obs/image'] for r in self.rows])
        self._extend('frames/rgb_u8',rgb_u8(images[:,:,:3]))
        self._extend('frames/depth_m',images[:,:,3:])
        self._extend('frames/depth_valid',np.stack([r['obs/depth_valid'] for r in self.rows]))
        self._extend('frames/post_index',np.array([
            int(r['obs/step_id'])+1 if r['transition/observation_after/available'] else -1
            for r in self.rows],dtype='i8'))
        self.group.attrs['length']=self.length
        self.file.flush();self.rows.clear()

    def finish(self, result):
        self.flush()
        last=self.previous
        if last is None:raise ValueError('Cannot publish empty episode')
        extra=bool(last['transition/observation_after/available'])
        if extra:
            im=last['transition/observation_after/image'][None]
            self._extend('frames/rgb_u8',rgb_u8(im[:,:,:3]))
            self._extend('frames/depth_m',im[:,:,3:])
            self._extend('frames/depth_valid',last['transition/observation_after/depth_valid'][None])
        self.group.attrs['complete']=bool(last['terminated'] or last['truncated'])
        self.group['frames'].attrs['extra_final_post']=extra
        self.file.attrs['result_json']=json.dumps(result)
        self.file.attrs['source_field_digests_json']=json.dumps({k:dict(self.specs[k],sha256=v.hexdigest()) for k,v in self.hashes.items()})
        self.file.flush();self.file.close()
        with self.path.open('rb') as stream:os.fsync(stream.fileno())
        self.path.rename(self.directory/'data.h5')
        self.previous=None
        return self.directory


def verify_stream_fields(path):
    """Compare every reconstructed field with its exact online source digest."""
    with EpisodeFile(path) as f:
        specs=json.loads(f.attrs['source_field_digests_json'])
        g=f['episodes'][next(iter(f['episodes']))];n=int(g.attrs['length'])
        actual=[]
        def verify(key,obj):
            if not isinstance(obj,(h5py.Dataset,VirtualArray)):return
            actual.append(key);spec=specs[key]
            if obj.dtype.str!=spec['dtype'] or obj.shape!=(n,*spec['row_shape']):
                raise ValueError('Source/reconstruction shape or dtype differs: '+key)
            digest=hashlib.sha256()
            for start in range(0,n,8):digest.update(np.ascontiguousarray(obj[start:start+8]).tobytes())
            if digest.hexdigest()!=spec['sha256']:raise ValueError('Source/reconstruction content differs: '+key)
        g.visititems(verify)
        if set(actual)!=set(specs):raise ValueError('Source/reconstruction fields differ')
        return len(actual)


def post_observation(packet,done):
    """Never attach automatic-reset imagery/encoders to the preceding episode."""
    import torch
    keys = ("image","depth_valid","camera_valid","camera_time_s","camera_frame",
            "proprio","time_s","episode_id","step_id","camera_pose_w","camera_parent_pose_w")
    result = {"available":~done}
    for key in keys:
        if key in packet:
            value = packet[key]
            mask = done.reshape(-1,*([1]*(value.ndim-1)))
            result[key] = torch.where(mask,torch.zeros_like(value),value)
    return result


class EpisodeReader:
    """Inspection API: raw arrays, provenance and validity, not network features.

    Reads only committed rows. HDF5 flush is not a power-loss transaction;
    abrupt filesystem/process failure can still require HDF5 recovery.
    """
    def __init__(self,path):
        self.path = str(path)
        with EpisodeFile(self.path) as f:
            self.metadata = json.loads(f.attrs["metadata_json"])
            self.schema_version = int(f.attrs["schema_version"])
            self.trace_version = int(f.attrs.get("trace_version",1))
            self.episodes = list(f["episodes"])

    def attributes(self,episode):
        with EpisodeFile(self.path) as f:
            return dict(f["episodes"][episode].attrs)

    def read(self,episode,index=slice(None)):
        with EpisodeFile(self.path) as f:
            group = f["episodes"][episode]
            length = int(group.attrs["length"])
            if isinstance(index,int) and not 0 <= index < length:
                raise IndexError(index)
            selection = np.arange(length)[index]
            result = {}
            def visit(name,obj):
                if isinstance(obj,(h5py.Dataset,VirtualArray)):
                    cursor = result
                    parts = name.split("/")
                    for part in parts[:-1]:
                        cursor = cursor.setdefault(part,{})
                    cursor[parts[-1]] = obj[selection]
            group.visititems(visit)
            return result

    def steps(self,episode):
        for i in range(int(self.attributes(episode)["length"])):
            yield self.read(episode,i)

    def export_pickle(self,episode,path):
        """Trusted local inspection only: never load arbitrary/untrusted pickle."""
        path = Path(path)
        if path.exists():
            raise FileExistsError(path)
        payload = dict(export_version=1,schema_version=self.schema_version,trace_version=self.trace_version,
                       metadata=self.metadata,episode=episode,attributes=self.attributes(episode),
                       arrays=self.read(episode),layout="nested arrays; first axis is control step")
        temporary = path.with_suffix(path.suffix+".partial")
        created = False
        try:
            with temporary.open("xb") as stream:
                created = True
                pickle.dump(payload,stream,protocol=5)
                stream.flush()
                os.fsync(stream.fileno())
            os.link(temporary,path)  # publish atomically; never overwrite an existing export
        finally:
            if created:
                temporary.unlink(missing_ok=True)
        return path


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Export one complete or incomplete episode for trusted local inspection")
    parser.add_argument("--data",required=True)
    parser.add_argument("--episode",required=True)
    parser.add_argument("--pkl",required=True)
    args = parser.parse_args()
    EpisodeReader(args.data).export_pickle(args.episode,args.pkl)


if __name__ == "__main__":
    main()
