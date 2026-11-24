"""CPU-only architecture and original pruning source regression tests."""
import inspect
import torch
from src import utils
from src.data_io import read_config
from src.model_factory import instantiate,model_counts
from src.dpn import DPN,FlexibleDPN


def test_original_outer_pruning_preserved():
    body=inspect.getsource(utils.apply_lth_pruning)
    assert 'apply_lth_pruning_param(' in body
    assert 'while hidden_dims != hidden_dims_pruned' in body


def test_geometry_four_datasets():
    for ds in ('covtype','jannis','helena','higgs_small'):
        cfg=read_config('.',ds);ex=cfg['expected']
        d=ex['input_features'];out=ex['output_logits'];ref=cfg['pruning']['reference_target_diagnostic_only']
        mlp=utils.match_mlp_to_ticket(d,out,cfg['reference_mlp']['hidden_widths'],ref)
        assert len(mlp['hidden_widths'])==len(cfg['reference_mlp']['hidden_widths'])
        blk=utils.match_shallow_block_dpn(d,out,mlp['hidden_widths'],mlp['actual_allocated'])
        assert blk['hidden_blocks']==len(mlp['hidden_widths'])
        assert blk['relative_error']<=.005
        maximum=utils.match_maximal_singleton_dpn(d,out,ref)
        assert maximum['relative_error']<=.01
        staged=utils.staged_dpn_candidates(d,out,ref)
        assert all(v['status']=='candidate' for v in staged.values())


def test_masked_flexible_dpn_forward_backward():
    m=DPN(5,11,2,minimal=False,hidden_dims=[3,3,3,2])
    with torch.no_grad():m.mask_0[:,:]=1;m.mask_1[:,0]=0
    f=FlexibleDPN.from_canonical(m,normalization='pre_linear_ln',dropout=.1)
    x=torch.randn(6,5)
    z=f(x)
    assert z.shape==(6,2)
    z.square().mean().backward()
    assert f.weights[0].grad is not None
    assert f.accounting()['parameters_active']<f.accounting()['parameters_allocated']
