"""Unintegrated optimization probe: no current experiment reads this module."""
import torch
from torch.nn.utils.rnn import pack_sequence,PackedSequence,pad_packed_sequence

def sequences(cell,values):
    packed=pack_sequence(values,enforce_sorted=False)
    h=values[0].new_zeros((1,len(values),cell.hidden_size))
    out,_=torch._VF.gru(packed.data,packed.batch_sizes,h,
        [cell.weight_ih,cell.weight_hh,cell.bias_ih,cell.bias_hh],True,1,0.,cell.training,False)
    padded,lengths=pad_packed_sequence(PackedSequence(out,packed.batch_sizes,packed.sorted_indices,packed.unsorted_indices))
    return [padded[:int(n),i] for i,n in enumerate(lengths)]

def candidate():
    from visuotactile.trajectory_batch import encode_trajectories
    from visuotactile.recurrent_execution import profile
    def encode(model,*args,**kwargs):
        old=getattr(model,'_offline_recurrent_execution',None)
        model._offline_recurrent_execution=profile()
        try:return encode_trajectories(model,*args,**kwargs)
        finally:
            if old is None:del model._offline_recurrent_execution
            else:model._offline_recurrent_execution=old
    return encode
