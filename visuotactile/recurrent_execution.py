"""Explicit offline GRU execution optimization with unchanged cell parameters.

Packing consumes each real frame once. It never runs padded recurrent steps,
mixes histories, caches learned states, or changes the deployed online cells.
CuDNN and cell kernels have normal floating point rounding differences.
"""
import json
from pathlib import Path


def profile():
    return dict(version='full_prefix_packed_gru_v1', parameters='existing_GRUCell_weights',
        history='all_real_ticks_once_per_independent_trajectory', padding='not_executed',
        deployment='original_online_GRUCell', precision='existing_fp32_policy')


def validate(value):
    if value is not None and value != profile():
        raise ValueError('Exact explicit packed GRU execution profile required')
    return dict(value) if value is not None else None


def load(root):
    path=Path(root)/'recurrent_execution.json'
    return validate(json.loads(path.read_text())) if path.exists() else None


def sequences(cell, values):
    import torch
    from torch.nn.utils.rnn import pack_sequence, PackedSequence, pad_packed_sequence
    if (not isinstance(cell,torch.nn.GRUCell) or not values
            or any(x.ndim != 2 or not len(x) or x.shape[1] != cell.input_size for x in values)):
        raise ValueError('Aligned nonempty real sequences and the existing GRUCell required')
    packed=pack_sequence(values,enforce_sorted=False)
    initial=values[0].new_zeros((1,len(values),cell.hidden_size))
    output,_=torch._VF.gru(packed.data,packed.batch_sizes,initial,
        [cell.weight_ih,cell.weight_hh,cell.bias_ih,cell.bias_hh],True,1,0.,cell.training,False)
    output,lengths=pad_packed_sequence(PackedSequence(output,packed.batch_sizes,
        packed.sorted_indices,packed.unsorted_indices))
    return [output[:int(length),lane] for lane,length in enumerate(lengths)]
