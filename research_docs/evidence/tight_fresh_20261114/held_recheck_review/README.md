# Delayed held-state takeover review

This is a retrospective review of **student prefixes**, followed by an implemented
but not yet live-validated collection candidate. It adds no recovered trajectory
or autonomous success. [summary.json](summary.json) binds the private audit and
its program by SHA256, and retains the original plan/results-index hashes.
Large numeric payload hashes were not recomputed; the private audit recorded
their paths, sizes and modification times. All source captures were closed.

“Held” here means the latest 12 completed control transitions each retained at
least two finger contacts above 0.1 N in both physics samples, with object height
at least 2 cm above the median of completed transitions 25–48. Palm-relative
rotation is allowed. This collection condition is separate from native lift,
native dropped/fall termination and final task success. Completed transition
count N can authorize only the next zero-based action N.

## What the existing data shows

| Cohort | Allocated / legal | Student prefixes ever held | Actual expert takeovers | Held at takeover |
| --- | ---: | ---: | ---: | ---: |
| Collection 06, all schedules | 16 / 16 | 9 | 9 | 6 |
| Collection 07, all schedules | 16 / 16 | 7 | 7 | 7 |
| Collection 08, all schedules | 32 / 29 | 7 | 19 | 5 |

Collection 08 includes 14 successful early fixed-tick takeovers. Six fixed40
student prefixes end before the height-reference window is available; these
are **unknown**, not evidence that the student failed the held condition.
There are four native-lift cases among the 29 legal starts. The table does not
equate lift with held or compare rates causally across different cohorts.

The old delay12 rule was allocated six opportunities in each of collections 06
and 07: **12 opportunities, six actual takeovers, three still held at the
takeover boundary**. The rule locked first-held + 12 and never rechecked. All
three cases without the held window at takeover were nevertheless successfully
rescued by the expert. Of the three held takeovers, two succeeded and one timed
out. This supports rechecking when deliberately targeting *currently held*
states; it does not show that rejecting other recoverable states improves
overall collection yield.

Collection 08's near60 schedule had **eight allocated, eight legal opportunities**:
three student prefixes ever held, one actually triggered and was rescued, and
five never met the held window. Two of the three held prefixes would qualify
under a minimum60-tick/current-held rule. Only **one** is additional to the
already executed near60 rescue:

| Student prefix | First held | At least 60 ticks later | Evidence and limit |
| --- | ---: | --- | --- |
| A, seed28400003 | 160 | Held at 220; native max-keypoint distance 68.37 mm, orientation error 0.533 rad | Existing near60 takeover at 222 succeeded under expert control. Three intervening ticks missed the instantaneous condition, so uninterrupted holding would exclude this useful interval. Student +120 is censored by the expert. |
| A, seed28400007 | 167 | Not held at 227; held again at 239, 141.64 mm, 1.303 rad | An additional **unexecuted** recovery candidate. It also held at +120 (287), 174.56 mm, 2.786 rad. Its original student run timed out. |
| B, seed28400003 | 132 | Not held at 192; no later held window | Also no held window after +120; native timeout, not a recorded dropped termination. |

Thus the +60 retrospective has three observable due points and two later-held
candidates among all eight near60 opportunities. The +120 retrospective has
two observable due points, one candidate, and one student future censored by
the actual expert takeover. **Neither retrospective executed a new rescue.**
B seed28400015 met native lift but never held, showing why these criteria must
remain separate.

Contact0 was allocated eight opportunities, seven legal, with four actual
held takeovers. At those boundaries the native maximum keypoint distances were
177–403 mm. This metric includes orientation; it is not literal object-centre
distance to the hole. Immediate expert control makes their later *student*
behavior unavailable.

## Entry schedules are not action-stage labels

The 14 accepted fixed40/fixed80 trajectories contain **4,244 / 5,795 = 73.24%**
of collection 08's legal expert-action supervision. This counts their complete
expert suffixes, including 939 held-window expert steps. It does **not** mean
73.24% of actions occur before lifting or turning. No per-step “grasp / turn /
align / insert” classifier was inferred from an early takeover label.

## Explicit candidate and remaining validation

`contact_height_recheck_after_min_delay_v1` exposes `held60` and `held120`.
After the first completed held window it waits at least 60 or 120 control ticks,
then requires the latest 12 completed **student** transitions to be held before
authorizing the next action. Temporary condition loss does not restart the
first timer. There is no fixed fallback and no requirement that holding remain
uninterrupted. Native goal changes may occur during the wait and are recorded;
the candidate does not use goal distance. Existing contact0, contact12 and
near60 rules remain unchanged.

No first held window, waiting for the delay, delay expired without current held,
natural termination, actual expert recovery and genuine student final success
remain separate outcomes. A missing held window is never relabeled as native
drop, and a first native subgoal is never relabeled as final success.

The candidate has CPU boundary/replay/serial-batch bridge tests; **no new live
collection or autonomous evaluation has accepted it yet**. Initial families,
policy sources and execution arrangements differ across the reviewed cohorts;
A/B opportunities on one family are not two independent initial families.
The evidence motivates a limited new collection schedule, not a success-rate
prediction or proof of a better policy. All 60 Hz observations, original
120/60 Hz physics/control and native success criteria remain unchanged.
