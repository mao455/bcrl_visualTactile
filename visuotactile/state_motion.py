"""Target continuity and recorded-contact diagnostics, independent of training."""
import json
import h5py
import numpy as np

def target_metrics(targets):
    x = np.asarray(targets)
    result = {}
    for name, sl in [('arm', slice(0, 7)), ('hand', slice(7, 29))]:
        delta = np.diff(x[:, sl], axis=0)
        boundary = np.arange(1, len(x)) % 12 == 0
        for label, mask in [('boundary', boundary), ('internal', ~boundary)]:
            result[f'{name}_{label}_jump_rms_rad'] = float(np.sqrt(np.mean(delta[mask]**2))) if mask.any() else None
        if len(delta)>1:
            acceleration = np.diff(delta, axis=0)*60**2
            result[f'{name}_acceleration_rms_rad_s2'] = float(np.sqrt(np.mean(acceleration**2)))
            for label, mask in [('boundary', (np.arange(2,len(x))%12<=1)),
                                ('internal', (np.arange(2,len(x))%12>1))]:
                result[f'{name}_{label}_acceleration_rms_rad_s2'] = float(np.sqrt(np.mean(acceleration[mask]**2))) if mask.any() else None
        result[f'{name}_speed_rms_rad_s'] = float(np.sqrt(np.mean(delta**2))*60) if len(delta) else None
    return result


def plan_metrics(plans, valid=None):
    plans = np.asarray(plans)
    if len(plans)<2:
        return {}
    overlap = plans[:-1, 12:] - plans[1:, :-12]
    if valid is None:valid=np.ones(plans.shape[:2],dtype=bool)
    aligned_valid=valid[:-1,12:] & valid[1:,:-12]
    result = target_metrics(plans[:, :12][valid[:,:12]])
    for name, sl in [('arm', slice(0, 7)), ('hand', slice(7, 29))]:
        result[name+'_overlap_rms_rad'] = float(np.sqrt(np.mean(overlap[..., sl][aligned_valid]**2))) if aligned_valid.any() else None
        result[name+'_aligned_first_rms_rad'] = float(np.sqrt(np.mean(overlap[:, 0, sl][aligned_valid[:,0]]**2))) if aligned_valid[:,0].any() else None
    return result


def metadata_and_contact(entry):
    with h5py.File(entry['path'], 'r') as stream:
        g=stream['episodes'][entry['episode']]
        meta=json.loads(stream.attrs['metadata_json'])
        force=g['obs/contact_substeps_w_N'][:]
        contact=np.linalg.norm(force,axis=-1).reshape(len(force),-1).max(-1)>.1
        # Source400 lacks object-filtered contact. Never call this object contact.
        return meta, np.maximum.accumulate(contact)
