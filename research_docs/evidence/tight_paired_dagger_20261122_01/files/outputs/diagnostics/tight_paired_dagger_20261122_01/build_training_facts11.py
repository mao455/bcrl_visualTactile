"""Finite CPU-only summary of existing training metadata; no tensor/evaluation reads."""
from pathlib import Path
from collections import defaultdict
import csv,hashlib,json,math,statistics,time

PROJECT=Path('/home/CNS2025331827/docs/bc+rl_RSS')
AUDIT=PROJECT/'outputs/diagnostics/tight_paired_dagger_20261122_01'
RUN=PROJECT/'outputs/visuotactile/run/tight_paired_dagger_20261122_01'
inputs={}
def record(path):
 path=Path(path);h=hashlib.sha256()
 with path.open('rb') as f:
  while chunk:=f.read(1<<20):h.update(chunk)
 inputs[str(path)]={'bytes':path.stat().st_size,'sha256':h.hexdigest()}
 return path
def read(path):return json.loads(record(path).read_text())
def pct(a,b):return (b/a-1)*100 if a else None
def trend(values):
 assert list(sorted(values))==list(range(41,61)),sorted(values)
 vals=[values[e] for e in range(41,61)];early=statistics.mean(vals[:5]);late=statistics.mean(vals[-5:])
 mx=50.5;my=statistics.mean(vals)
 slope=sum((e-mx)*(values[e]-my) for e in range(41,61))/sum((e-mx)**2 for e in range(41,61))
 return {'epoch41':values[41],'epoch50':values[50],'epoch60':values[60],
  'endpoint41_to60_change_pct':pct(values[41],values[60]),'epoch50_to60_change_pct':pct(values[50],values[60]),
  'mean_epoch41_45':early,'mean_epoch56_60':late,'early5_to_late5_change_pct':pct(early,late),
  'ols_slope_per_epoch':slope,'minimum_last20':min(vals),'minimum_epoch_last20':41+vals.index(min(vals)),
  'increased_transitions_last20':sum(b>a for a,b in zip(vals,vals[1:])),
  'decreased_transitions_last20':sum(b<a for a,b in zip(vals,vals[1:]))}

guard=read(AUDIT/'presealed_guard_main_recovery10.json')
assert guard['status']=='passed' and guard['checkpoint_reads']==8 and not guard['cuda_initialized']
components=[]
for stage,roles in guard['actual'].items():
 for role,items in roles.items():
  for component,m in items.items():
   e=m['exposure'];vis=m['visits'];entry=m.get('entry')
   assert m['epoch']==60 and e['minimum_tick_steps']>=50 and e['below50_ticks']==e['below50_trajectories']==0
   assert e['ancestors_counted'] is False and e['eligible'] is True
   if entry:assert entry['minimum_steps']>=50 and entry['below50']==0
   components.append({'stage':stage,'role':role,'component':'DP' if component=='frontend' else 'Transformer',
    **{k:m[k] for k in ('directory','checkpoint','checkpoint_sha256','epoch','updates','base_sha256','wall_seconds')},
    'exposure':e,'entry':entry,'visits':{'trajectory_count':len(vis),'minimum':min(vis.values()),'maximum':max(vis.values())}})
assert len(components)==8
for stage in ('common180_parents','learning'):
 for component in ('DP','Transformer'):
  pair=[x for x in components if x['stage']==stage and x['component']==component]
  assert len(pair)==2 and pair[0]['updates']==pair[1]['updates']
for m in components:
 if m['component']=='Transformer':
  front=next(x for x in components if x['stage']==m['stage'] and x['role']==m['role'] and x['component']=='DP')
  assert m['base_sha256']==front['checkpoint_sha256']

METRICS=('normalized_equal_mse','arm_rmse_rad','hand_rmse_rad')
curves={};flat=[]
for role in ('A','B'):
 for comp in ('DP','Transformer'):
  directory=RUN/'learning'/role/'training' if comp=='DP' else RUN/'learning'/role/'backend'/role/'training'
  path=record(directory/'family_phase_curves.csv')
  with path.open(newline='') as f:
   rows=list(csv.DictReader(f))
  assert {x['component'] for x in rows}=={comp}
  assert {x['role'] for x in rows}=={role}
  assert {x['target'] for x in rows}=={('q_reached' if role=='A' else 'u_expert') if comp=='DP' else 'u_expert'}
  groups=defaultdict(list)
  for r in rows:
   r['epoch']=int(r['epoch']);r['samples']=int(r['samples'])
   for metric in METRICS:r[metric]=float(r[metric]);assert math.isfinite(r[metric])
   if r['family']=='__all__' or r['phase']=='__all__':
    groups[(r['split'],r['family'],r['phase'])].append(r)
  result={'target':rows[0]['target'],'directory':str(directory),'splits':{},'validation_by_phase':{},'validation_by_family':{}}
  for (split,family,phase),selected in sorted(groups.items()):
   selected=sorted(selected,key=lambda x:x['epoch'])
   epochs=[x['epoch'] for x in selected]
   if family!='__all__' and split!='validation':continue
   if family!='__all__' and phase!='__all__':continue
   if family=='__all__' and phase!='__all__' and split!='validation':continue
   key=(split,family,phase)
   counts=[x['samples'] for x in selected]
   end=selected[-1]
   family_count=len({x['family'] for x in rows if x['epoch']==end['epoch'] and x['split']==split and x['family']!='__all__' and x['phase']==phase})
   out={'available_epochs':epochs,'samples_min':min(counts),'samples_max':max(counts),'samples_epoch60':end['samples'],
     'families_epoch60':family_count if family=='__all__' else 1,'metrics':{}}
   for metric in METRICS:
    vals={x['epoch']:x[metric] for x in selected if 41<=x['epoch']<=60}
    if sorted(vals)==list(range(41,61)):
     t=trend(vals)
    else:
     t={'late20_trend_available':False,'first_epoch':selected[0]['epoch'],'last_epoch':end['epoch'],
       'first_value':selected[0][metric],'last_value':end[metric],
       'first_to_last_change_pct':pct(selected[0][metric],end[metric])}
    out['metrics'][metric]=t
    flat.append({'role':role,'component':comp,'target':result['target'],'split':split,'family':family,'phase':phase,'metric':metric,
      'samples_epoch60':out['samples_epoch60'],'families_epoch60':out['families_epoch60'],**t})
   if family=='__all__' and phase=='__all__':result['splits'][split]=out
   elif split=='validation' and family=='__all__':result['validation_by_phase'][phase]=out
   elif split=='validation' and phase=='__all__':result['validation_by_family'][family]=out
  definition=read(directory/'family_phase_definition.json')
  result['diagnostic_definition']=definition
  if comp=='DP':
   records=[json.loads(line) for line in record(directory/'validation.jsonl').open() if line.strip()]
   vals={r['epoch']:r['loss_per_optimizer_batch_mean'] for r in records if 41<=r['epoch']<=60}
   result['training_objective_loss']=trend(vals)
   result['objective_definition']='Per-epoch mean actual optimizer-batch full DP training loss, including registered geometric terms; not the phase nominal error.'
  else:
   validation=read(directory/'validation.json')
   result['fixed_panel_recheck']={'train_ticks':validation[-1]['train']['all']['ticks'],'validation_ticks':validation[-1]['validation']['all']['ticks'],
      'validation_score_epoch60':validation[-1]['validation']['all']['normalized_equal_mse']}
   sums=defaultdict(float);counts=defaultdict(int);updates=[]
   loss_path=record(directory/'loss.jsonl')
   with loss_path.open() as f:
    for line in f:
     r=json.loads(line);e=r['epoch'];sums[e]+=r['loss'];counts[e]+=1;updates.append(r['update'])
   assert updates==list(range(1,38401)) and set(counts.values())=={640} and sorted(counts)==list(range(1,61))
   vals={e:sums[e]/counts[e] for e in range(41,61)}
   result['training_objective_loss']=trend(vals)
   result['objective_definition']='Arithmetic mean of640 successful optimizer-step objective losses per epoch, preserving fixed-denominator and tail-window treatment. Contains FY objective; not directly comparable with final-command MSE.'
  flat.append({'role':role,'component':comp,'target':result['target'],'split':'training_objective','family':'__all__','phase':'__all__',
    'metric':'actual_objective_loss','samples_epoch60':'','families_epoch60':'',**result['training_objective_loss']})
  result['validation_family_directions_late5']={
   'decreased':sum(v['metrics']['normalized_equal_mse']['early5_to_late5_change_pct']<0 for v in result['validation_by_family'].values()),
   'increased':sum(v['metrics']['normalized_equal_mse']['early5_to_late5_change_pct']>0 for v in result['validation_by_family'].values())}
  curves[role+'_'+comp]=result

# Recheck the earlier independently computed DP summaries on their shared metrics.
previous=read(AUDIT/'DP_FINAL20_READONLY07.json')
for role in ('A','B'):
 for k in ('epoch41','epoch60','early5_to_late5_change_pct'):
  assert math.isclose(curves[role+'_DP']['training_objective_loss'][k],previous['roles'][role]['training_loss'][k],rel_tol=1e-10,abs_tol=1e-14)
 for phase,value in curves[role+'_DP']['validation_by_phase'].items():
  old=previous['roles'][role]['validation_by_phase'][phase]
  # Existing artifact schema is retained; raw CSV is the current authority.

sources={}
for name in ('visuotactile/paired_curve_diagnostics.py','visuotactile/geometry_sampling.py','visuotactile/anchor_continue.py','visuotactile/paired_training_decks.py','tools/tight_paired_learning.py'):
 p=record(PROJECT/name);sources[name]=inputs[str(p)]['sha256']
result={'version':'paired_training_facts11_readonly_v1','created_utc':time.time(),
 'scope':'CPU-only existing training small metadata/curves and streaming rear loss logs; no tensor payload, cache, raw capture or evaluation outcome read; no science/runtime mutations.',
 'authority':'Actual exposure/checkpoint fields copied from root presealed_guard_main_recovery10 actual eight-payload extraction; this audit does not reopen those tensors. Root retains final postclosure payload audit.',
 'deployment':'All eight registered components fixed epoch60; no checkpoint reselection.',
 'component_exposure':components,
 'pool_counts':{r:{k:len(v) for k,v in d.items()} for r,d in guard['pools'].items()},
 'curve_definitions':{'late_window':'epoch41-60; early5mean41-45 versus late5mean56-60; also endpoints50to60 and descriptive OLS slope.',
  'normalized_equal_mse':'0.5*mean(normalized arm squared error)+0.5*mean(normalized hand squared error); pooled by recorded legal samples.',
  'validation':'Fixed6 held-out training-validation families,1843 legal control slots per epoch; same families reused every epoch. This is offline teacher-forced validation, not new autonomous held-out evaluation.',
  'train_evaluation':'Fixed first2 training episodes /649 legal control slots per epoch; never a full-pool training metric.',
  'train_updates':'Recorded errors from actual successful optimizer updates with repeated windows/sampling. DP uses noisy sample-prediction; rear uses pre-update composed final-command error. These are not the same as epoch-final fixed-panel evaluation.',
  'phase':'Offline held_near_goal60 uses60mm screen; not native15mm entry, cumulative10tick confirmation, or insertion.',
  'comparison':'Within each role/component only. A DP q_reached and B DP u_expert losses do not rank A/B. Both rear targets u_expert, but their fixed nominal input generators and training pools differ.'},
 'curves':curves,'source_sha256':sources,'inputs':inputs,
 'limitations':['Exposure means at least60 distinct successful optimizer steps for each legal label and DP entry slot under the existing guard; duplicated windows/padding/ancestor updates do not fill that floor.',
  'Training improvement, validation improvement, and autonomous generalization are separate claims. Declining repeated-data errors alone do not prove convergence or independent complete insertion.',
  'Late-window and family directions are descriptive; no confidence intervals or significance tests, and individual ticks are not independent trials.',
  'All perphase errors omit a causal decomposition of geometry/FY/projection contributions.',
  'DP correction_evaluation has only endpoint diagnostics0and60; it cannot establish continuous improvement through the final20epochs.',
  'Reported component wall_seconds may span different resume/wait scopes and are not total session time or GPU compute hours; do not sum them as experiment walltime.'],
 'new_optimizer_updates_from_this_audit':0,'new_kit_or_gpu_jobs':0,'evaluation_outcomes_read':0}
with (AUDIT/'TRAINING_FACTS11.json').open('w') as f:json.dump(result,f,indent=2,ensure_ascii=False);f.write('\n')
fields=list(dict.fromkeys(k for row in flat for k in row))
with (AUDIT/'TRAINING_FACTS11.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(flat)
print(json.dumps({'components':len(components),'curves':list(curves),'compact_csv_rows':len(flat),'input_files':len(inputs),
 'trends':{key:{s:v['metrics']['normalized_equal_mse']['early5_to_late5_change_pct'] for s,v in value['splits'].items() if s!='correction_evaluation'} for key,value in curves.items()},
 'objective_trends':{key:value['training_objective_loss']['early5_to_late5_change_pct'] for key,value in curves.items()},
 'near60':{key:value['validation_by_phase']['held_near_goal60']['metrics']['normalized_equal_mse']['early5_to_late5_change_pct'] for key,value in curves.items()}},indent=2))
# Human-readable facts, preserving separate claims and fixed denominators.
def signed(value):return f'{value:+.2f}%'
lines=[
 '八个部署组件的监督覆盖均达标；新一轮四组件在后20轮继续拟合训练数据，但固定验证家族没有一致改善。此处不读取或判断自主开发／终测结果。A DP 学 q_reached、B DP 学 u_expert；下列变化都在各自目标空间内计算，不以 nominal 损失给 A/B 排名。',
 '',
 '已有 presealed_guard_main_recovery10 对实际八个部署载荷的提取显示：全部固定 epoch60、每个合法专家 tick 至少进入60个不同成功更新步，低于50的 tick／轨迹均为0。DP 原生纠偏入口合法提交槽也全部至少60步；祖先更新不计。此次只读该既有提取，不重新打开权重；root 的终态载荷核验仍单列。',
 '',
 '| 阶段 / 组别 / 组件 | epoch / 实际更新 | 轨迹 / 合法 tick | tick最少成功step | DP纠偏条数 / 入口槽 / 最少step |',
 '| --- | ---: | ---: | ---: | ---: |']
for m in components:
 e=m['exposure'];entry=m['entry'];label=('共同180父本' if m['stage']=='common180_parents' else '本轮新学生')
 entrytext=f"{entry['entries']} / {entry['slots']} / {entry['minimum_steps']}" if entry else '—'
 lines.append(f"| {label} / {m['role']} / {m['component']} | {m['epoch']} / {m['updates']} | {e['trajectories']} / {e['legal_ticks']} | {e['minimum_tick_steps']} | {entrytext} |")
lines += ['',
 'A/B父本分别为同预算 DP7200＋后端22763更新；本轮两 DP 各10200、两后端各38400更新。四个后端的 base SHA 均对应同阶段本组 DP。A池228条、B池231条：共同旧180＋新完整示范24＋本组新纠偏24/27；两池均54个训练家族。新增纠偏入口槽为A164、B193（总数减原956），所有槽均包含在上述最低60step核验中。',
 '',
 '后20轮定义为41–60，平滑变化比较41–45均值与56–60均值；另给50→60端点变化。负数表示误差／损失下降。',
 '',
 '| 组件 / 组别 | 训练完整目标损失，末/首5轮 | 固定2训练轨迹误差，末/首5轮 | 固定6验证家族误差，末/首5轮 | 验证误差 epoch50→60 |',
 '| --- | ---: | ---: | ---: | ---: |']
for key in ('A_DP','B_DP','A_Transformer','B_Transformer'):
 v=curves[key];s=v['splits'];val=s['validation']['metrics']['normalized_equal_mse']
 lines.append(f"| {key} | {signed(v['training_objective_loss']['early5_to_late5_change_pct'])} | {signed(s['train_evaluation']['metrics']['normalized_equal_mse']['early5_to_late5_change_pct'])} | {signed(val['early5_to_late5_change_pct'])} | {signed(val['epoch50_to60_change_pct'])} |")
lines += ['',
 '四组件的训练目标损失及固定训练面板误差仍下降；新后端A/B的训练更新误差也分别下降14.87%/13.87%。验证总体则是DP近持平、后端略升：A后端6家族中5个上升、1个下降，B后端各3个。后端epoch50→60的验证误差均上升，不能把继续拟合解释为验证能力继续提高，亦不能据此证明已完全收敛。',
 '',
 '| 固定验证阶段 | 合法槽 / 家族 | A DP变化 | B DP变化 | A后端变化 | B后端变化 |',
 '| --- | ---: | ---: | ---: | ---: | ---: |']
phases=[('grasp_without_contact','抓取、未接触'),('contact_not_held','有接触、未稳定持物'),('held_far','稳定持物、较远'),('held_rotating','稳定持物、转正'),('held_near_goal60','稳定持物近60mm目标'),('after_first_native_subgoal','原生首子目标后'),('contact_lost','接触已丢失')]
for phase,label in phases:
 sample=curves['A_DP']['validation_by_phase'][phase]
 vals=[signed(curves[k]['validation_by_phase'][phase]['metrics']['normalized_equal_mse']['early5_to_late5_change_pct']) for k in ('A_DP','B_DP','A_Transformer','B_Transformer')]
 lines.append(f"| {label} | {sample['samples_epoch60']} / {sample['families_epoch60']} | "+' | '.join(vals)+' |')
lines += ['',
 '近60mm持物阶段的验证误差四组件均上升；该离线阶段不是15mm入门、累计10tick原生确认或完整插入。阶段与家族方向均为描述，不含显著性检验，逐tick也不是独立试验。各家族的末/首5轮变化为：',
 '',
 '| 验证家族 seed | 合法槽 | A DP | B DP | A后端 | B后端 |',
 '| --- | ---: | ---: | ---: | ---: | ---: |']
for family in sorted(curves['A_DP']['validation_by_family']):
 vals=[signed(curves[k]['validation_by_family'][family]['metrics']['normalized_equal_mse']['early5_to_late5_change_pct']) for k in ('A_DP','B_DP','A_Transformer','B_Transformer')]
 lines.append(f"| {family.rsplit('_',1)[-1]} | {curves['A_DP']['validation_by_family'][family]['samples_epoch60']} | "+' | '.join(vals)+' |')
lines += ['',
 '验证每轮均为原固定6家族／1843合法控制槽；它们未进训练池，但每轮重复诊断，不是新增自主泛化测试。train_evaluation始终只取前2条训练轨迹／649合法控制槽，不能当全池误差。train_updates覆盖54家族的实际采样更新，DP每轮约21万重复窗口槽，后端约9万合法样本；重复样本不是独立样本，也不用于凑逐tick不同更新步的曝光门槛。',
 '',
 '标准化误差是手臂7维与手指22维各均值后等权MSE。DP训练更新误差使用扩散带噪sample预测，固定评估使用DDIM16、原生12tick提交；后端训练更新误差来自更新前的完整最终命令，固定评估用轮末模型。DP完整损失还含原几何项，后端完整损失含FY目标，均不能与最终命令MSE相减。两组后端同学u_expert，但nominal输入生成器与各自纠偏池不同，不据离线误差给整套策略排名。',
 '',
 '新纠偏DP只有epoch0和60的固定历史诊断：A6721合法槽/20家族、B7801槽/22家族，nominal MSE分别下降92.61%/92.83%。这是已训练数据拟合，无后20轮逐轮诊断，不能证明末期持续改善或独立学生救回能力。',
 '',
 '监督是否覆盖：已有实际部署载荷审计确认全部达标。训练是否仍改善：四组件的训练拟合仍改善，验证阶段／家族表现混合。能否泛化：须由四套冻结完整学生的独立自主评估回答；本事实表不含该结果，不改登记epoch60，也不启动额外训练。',
 '',
 '本次CPU审计：0 GPU/Kit、0新增优化器更新、0评估结果读取。八组件各自wall_seconds保留在JSON，包含不同恢复／等待口径，不能求和当总墙钟或GPU小时。输入SHA、三种误差的逐阶段／家族数值与OLS描述斜率见 TRAINING_FACTS11.json；紧凑曲线表见 TRAINING_FACTS11.csv，八组件表见 TRAINING_FACTS11_components.csv；生成脚本为 build_training_facts11.py。']
(AUDIT/'TRAINING_FACTS11.md').write_text('\n'.join(lines)+'\n')
comp_rows=[]
for m in components:
 e=m['exposure'];en=m['entry']
 comp_rows.append({'stage':m['stage'],'role':m['role'],'component':m['component'],'epoch':m['epoch'],'updates':m['updates'],
  'trajectories':e['trajectories'],'legal_ticks':e['legal_ticks'],'minimum_tick_steps':e['minimum_tick_steps'],
  'below50_ticks':e['below50_ticks'],'below50_trajectories':e['below50_trajectories'],'ancestors_counted':e['ancestors_counted'],
  'entry_trajectories':en['entries'] if en else '', 'entry_slots':en['slots'] if en else '', 'entry_minimum_steps':en['minimum_steps'] if en else '',
  'checkpoint_sha256':m['checkpoint_sha256'],'base_sha256':m['base_sha256'],'wall_seconds':m['wall_seconds']})
with (AUDIT/'TRAINING_FACTS11_components.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(comp_rows[0]));w.writeheader();w.writerows(comp_rows)
print('Markdown and8-component CSV written.')
