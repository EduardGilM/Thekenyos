"""Acceptance gates and bounded practice from physically reached graph states."""
from .reward_graph import STAGE_NAMES


def graph_rank(result):
    return tuple(result.get(key, 0.) for key in (
        'success','completed/deposit','completed/carry','completed/extract',
        'completed/grip','completed/position')) + (
        -result.get('physical_failure',0.),-result.get('ground_drop',0.),
        -result.get('invalid_extraction',0.),-result.get('closest_distance_m',1.))


def acceptance(candidate, accepted, tolerance=.05, floor=None):
    """Absolute-rate guard, an engineering tolerance rather than a significance test."""
    reasons=[]
    protected=accepted if floor is None else floor
    for key in ('success',*[f'completed/{name}' for name in STAGE_NAMES[:-1]]):
        if candidate.get(key,0.) < protected.get(key,0.)-tolerance:
            reasons.append(key)
    for key in ('physical_failure','task_failure'):
        if candidate.get(key,0.) > protected.get(key,0.)+tolerance:
            reasons.append(key)
    # Preserve the continuous approach skill before the first enclosure exists.
    if candidate.get('closest_distance_m',1.) > protected.get('closest_distance_m',1.)+.02:
        reasons.append('closest_distance_m')
    if 'training_scene/episodes' in candidate:
        scene_candidate={k.removeprefix('training_scene/'):v for k,v in candidate.items() if k.startswith('training_scene/')}
        scene_accepted={k.removeprefix('training_scene/'):v for k,v in accepted.items() if k.startswith('training_scene/')}
        scene_floor={k.removeprefix('training_scene/'):v for k,v in protected.items() if k.startswith('training_scene/')}
        _,scene_reasons=acceptance(scene_candidate,scene_accepted,tolerance,floor=scene_floor)
        reasons.extend('training_scene/'+key for key in scene_reasons)
    gains=[candidate.get(key,0.)-accepted.get(key,0.) for key in
        ('success',*[f'completed/{name}' for name in STAGE_NAMES[:-1]])]
    gains += [accepted.get(key,0.)-candidate.get(key,0.) for key in ('physical_failure','task_failure')]
    meaningful=max(gains)>=tolerance or accepted.get('closest_distance_m',1.)-candidate.get('closest_distance_m',1.)>=.01
    return not reasons and meaningful and graph_rank(candidate)>graph_rank(accepted), reasons


def protect_skills(floor, result):
    """High-water marks prevent a sequence of small accepted regressions."""
    protected=dict(floor)
    for key,value in result.items():
        metric=key.removeprefix('training_scene/')
        if metric=='success' or metric.startswith('completed/'):
            protected[key]=max(protected.get(key,0.),value)
        elif metric in ('physical_failure','task_failure','closest_distance_m'):
            protected[key]=min(protected.get(key,float('inf')),value)
        else: protected[key]=value
    return protected


class StagePractice:
    """Small in-memory bank; never used for full-task evaluation or success rates."""
    def __init__(self, fraction=.25):
        self.fraction=fraction
        self.bank={}
        self.cursor=0
        self.starts=0

    def capture(self, collector):
        import torch
        from .counterfactual import capture_worlds
        p=collector.progress
        for stage in (0,1,2,3,4):
            valid=(p.graph_stage==stage) & ~p.graph_invalid_extract & ~p.graph_ground_drop
            if stage==0: valid &= (p.graph_score>=1.6) & ~p.graph_detached
            eligible=valid.nonzero().flatten()
            if not len(eligible): continue
            ids=eligible[:8]
            history={name:getattr(p,name)[ids].clone() for name in (
                'graph_dwell','graph_position_dwell','graph_grip','graph_detached','graph_valid_extract','graph_valid_release')}
            self.bank[stage]=capture_worlds(collector.runtime,ids,application_state=history)

    def restore(self, collector, ending):
        import torch
        from .counterfactual import restore_worlds
        from .harvest_training import signals
        if not self.bank: return
        ids=ending.nonzero().flatten()
        # Rotate slots so that practice remains a bounded fraction, even when
        # episodes finish individually; all other starts are ordinary resets.
        selected=ids[(ids+collector.episode_ids[ids])%round(1/self.fraction)==0]
        if not len(selected): return
        stages=sorted(self.bank)
        snapshot=self.bank[stages[self.cursor%len(stages)]];self.cursor+=1
        mapping=torch.arange(len(selected),device=selected.device)%len(snapshot.source_world_ids)
        history=restore_worlds(collector.runtime,snapshot,selected,source_indices=mapping)
        mask=torch.zeros_like(ending);mask[selected]=True
        p=collector.progress
        p.reset(mask,signals(collector.runtime))
        for name,value in history.items(): getattr(p,name)[selected]=value[mapping]
        # Score is recomputed from restored prerequisites before the first action.
        from .reward_graph import graph_step
        # Compute on a separate copy: never advance unrelated worlds' dwell.
        from copy import deepcopy
        restored=deepcopy(p)
        graph_step(restored,signals(collector.runtime))
        for name in ('graph_score','graph_stage'):
            getattr(p,name)[selected]=getattr(restored,name)[selected]
        for name in STAGE_NAMES: p.graph_components[name][selected]=restored.graph_components[name][selected]
        p.graph_best[selected]=p.graph_score[selected]
        p.graph_reference[selected]=p.graph_score[selected]
        collector.practice_mask[selected]=True
        self.starts+=len(selected)


def save_acceptance(path, checkpoint, result, floor):
    import json
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(dict(checkpoint=checkpoint,evaluation=result,skill_floor=floor),indent=2)+'\n')
    temporary.replace(path)
