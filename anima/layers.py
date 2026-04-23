import torch
from ..utils import nag

def nag_self_attn_forward(self_attn_module, *args, **kwargs):
    """
    Intercepts the output of Anima's self-attention to apply Negative Attention Guidance (NAG).
    """
    # 1. Run standard self-attention on the combined batch
    out = self_attn_module.original_forward(*args, **kwargs)
    
    # ComfyUI attention outputs can sometimes be a tuple depending on optimization used
    is_tuple = isinstance(out, tuple)
    attn_out = out[0] if is_tuple else out

    # 2. Check if NAG should be applied for this step
    if hasattr(self_attn_module, 'nag_scale') and getattr(self_attn_module, 'origin_bsz', 0) > 0:
        origin_bsz = self_attn_module.origin_bsz
        pos_bsz = attn_out.shape[0] - origin_bsz
        
        if pos_bsz > 0:
            # 3. Split positive and negative paths
            out_pos = attn_out[:pos_bsz]
            out_neg = attn_out[-origin_bsz:]
            
            # Make sure negative path matches positive batch size
            if origin_bsz < pos_bsz:
                repeat_times = (pos_bsz + origin_bsz - 1) // origin_bsz
                out_neg_expanded = out_neg.repeat(repeat_times, 1, 1)[:pos_bsz]
            elif origin_bsz > pos_bsz:
                out_neg_expanded = out_neg[:pos_bsz]
            else:
                out_neg_expanded = out_neg
                
            # 4. Apply the NAG formula
            out_guided = nag(
                out_pos, 
                out_neg_expanded, 
                self_attn_module.nag_scale, 
                self_attn_module.nag_tau, 
                self_attn_module.nag_alpha
            )
            
            # 5. Re-concatenate the guided positive and unmodified negative paths
            attn_out = torch.cat([out_guided, out_neg], dim=0)
            
            if attn_out.dtype == torch.float16:
                attn_out = torch.nan_to_num(attn_out, nan=0.0, posinf=65504, neginf=-65504)
            
            if is_tuple:
                out = (attn_out,) + out[1:]
            else:
                out = attn_out
                
    return out