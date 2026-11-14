"""Normalize actual optimizer receipts to visits without inventing optimizer steps."""
BATCH_ROW_VERSION='full_trajectory_batch_learning_v1'


def _count(value,name,*,positive=False):
    if type(value) is not int or value < (1 if positive else 0):
        raise ValueError('Invalid learning counter: '+name)
    return value


def iter_visits(rows,*,limit=None):
    if limit is None:limit=len(rows)
    if type(limit) is not int or not 0<=limit<=len(rows):raise ValueError('Invalid optimizer receipt limit')
    previous=dict(anchors_seen=0,target_ticks=0);seen=0
    for update,row in enumerate(rows[:limit],1):
        if type(row.get('updates')) is not int or row['updates']!=update:
            raise ValueError('Noncontiguous optimizer receipt')
        da=_count(row['anchors_seen'],'anchors_seen')-previous['anchors_seen']
        dt=_count(row['target_ticks'],'target_ticks')-previous['target_ticks']
        if da<=0 or dt<=0:raise ValueError('Nonpositive learning exposure increment')
        if row.get('version')==BATCH_ROW_VERSION:
            from .trajectory_training import validate_profile
            if validate_profile(row['full_trajectory_batch']) is None:raise ValueError('Missing B2 profile')
            visits=row['visits']
            _count(row['visual_frames'],'visual_frames',positive=True)
            if not isinstance(visits,list) or len(visits) not in (1,2):raise ValueError('Invalid B2 visit list')
            for visit in visits:
                for key in ('anchors','target_ticks','visual_frames'):_count(visit[key],key,positive=True)
                if visit.get('loss_weight')!=1/len(visits):raise ValueError('B2 visit loss weight differs')
            if (sum(v['anchors'] for v in visits)!=da or sum(v['target_ticks'] for v in visits)!=dt or
                    sum(v['visual_frames'] for v in visits)!=row['visual_frames'] or
                    _count(row['visits_seen'],'visits_seen')!=seen+len(visits)):
                raise ValueError('B2 local visits differ from cumulative optimizer counters')
        elif 'visits' in row or 'full_trajectory_batch' in row or row.get('version') is not None:
            raise ValueError('Unknown learning receipt version')
        else:
            visits=[dict(episode=row['episode'],anchors=da,target_ticks=dt,visual_frames=row['visual_frames'])]
            _count(row['visual_frames'],'visual_frames',positive=True)
        for visit in visits:
            if not isinstance(visit.get('episode'),str) or not visit['episode']:raise ValueError('Missing visit episode')
            seen+=1
            yield dict(visit,updates=update,visit_index=seen)
        previous=row


def learning_counters(rows,*,limit=None):
    if limit is None:limit=len(rows)
    visits=list(iter_visits(rows,limit=limit))
    return dict(updates=limit,anchors_seen=sum(v['anchors'] for v in visits),
        target_ticks=sum(v['target_ticks'] for v in visits),visits_seen=len(visits))
