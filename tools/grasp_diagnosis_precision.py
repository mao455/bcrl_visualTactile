"""Opt-in current-encoder reconstruction avoiding normalized-proprio round trip.

Both fields are existing current robot measurements. No object/tactile state,
expert data, future target, joint-state writes, or new sensor information is used.
Installed only in the explicitly versioned diagnostic worker process.
"""
from contextvars import ContextVar

import torch


def current_position(packet):
    target=packet['actuator_target_rad'][0].cpu()
    tracking=packet['tracking_error_rad'][0].cpu()
    if target.shape!=(29,) or tracking.shape!=(29,) or not torch.isfinite(target).all() or not torch.isfinite(tracking).all():
        raise ValueError('Canonical current target and encoder tracking error required')
    return target-tracking


def install():
    from visuotactile import state_history,native_controller
    context=ContextVar('diagnostic_current_encoder',default=None)
    for module,cls,state in ((state_history,state_history.HistoryController,True),
                             (native_controller,native_controller.NativeController,False)):
        original=module.proprio_radians
        def decode(proprio,limits,_original=original):
            result=_original(proprio,limits)
            position=context.get()
            if position is not None:result[...,:29]=position.to(result)
            return result
        module.proprio_radians=decode
        previous=cls.act
        def act(self,*a,_previous=previous,_state=state,**kw):
            packet=a[1] if _state else a[0]
            token=context.set(current_position(packet))
            try:return _previous(self,*a,**kw)
            finally:context.reset(token)
        cls.act=act
