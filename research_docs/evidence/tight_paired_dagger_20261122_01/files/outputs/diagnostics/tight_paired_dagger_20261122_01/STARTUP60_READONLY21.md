# Wave60 readonly initialization diagnosis

Exact own worker1251659, rawbirth1795655525.47, remains in environment_constructor_started. Automatic120s Python dumps repeatedly stop at SimulationContext.play:917 `self._app.update()`, called by reset inside DirectRLEnv constructor. Student initialization/progress files are absent; this is before student control, while native physical initialization may already run.

A 2.007s sample measured 902.8% process CPU: eight carb.tasking threads plus main thread near one core each, all active wchan0, and every process I/O counter unchanged. RAM/GPU headroom and pressure samples do not indicate current resource exhaustion.

Current OV, DerivedData, shader, cooked-mesh and Optix cache locks are held through this own worker's open descriptors. No blocked filesystem-lock row for their inodes, old58/59 process, or old-PID lock row was observed. Kit's last writes concern initial OmniPBR material/render updates, without an Error-tagged entry. These facts do not establish cache corruption or a removable stale lock.

The same play/reset stack was observed in historicalwave41 before cleanup58. The defensible finding remains a native app.update/tasking initialization stall of unknown exact C++ cause. No concrete cache deletion or other repair is supported. Preserve evidence and retain the registered finite startup timeout. No signal, affinity/source/cache/sysctl change, helper activation or scientific-result read occurred. Adjacent JSON contains identities, CPU/I/O samples, own descriptor/lock evidence and source/log SHA.
