import torch
import torch.nn.functional as F

dev = "cuda"
D = 1536


def two_dropouts(rows_first, rows_pad):
    """Dropout on a tensor of rows_pad rows (first rows_first real), then a fixed 512-row dropout.
    Returns (real rows of the first mask, the second mask)."""
    torch.cuda.manual_seed(7)
    a = torch.ones(rows_pad, D, device=dev, dtype=torch.float16)
    da = F.dropout(a, 0.05, True)
    b = torch.ones(512, D, device=dev, dtype=torch.float16)
    db = F.dropout(b, 0.05, True)
    return da[:rows_first].clone(), db.clone()


base_a, base_b = two_dropouts(300, 300)
for pad in [300, 320, 384, 512]:
    a, b = two_dropouts(300, pad)
    print(f"first dropout padded 300->{pad}: real rows equal base: {torch.equal(a, base_a)}; "
          f"next dropout equal base: {torch.equal(b, base_b)}", flush=True)
