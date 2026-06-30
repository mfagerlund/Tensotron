import torch

def g(t):
    return None if t.grad is None else t.grad.tolist()

print("=== POW zero-base: base grad and exponent grad ===")
for base, exp in [(0.0,2.0),(0.0,1.0),(0.0,3.0),(0.0,0.5),(0.0,0.0)]:
    a = torch.tensor([base], requires_grad=True)
    b = torch.tensor([exp], requires_grad=True)
    y = torch.pow(a, b)
    y.backward(torch.ones_like(y))
    print(f"pow({base},{exp}): y={y.item():.4g}  da={g(a)}  db={g(b)}")

print("\n=== MAX/MIN tie gradient distribution ===")
x = torch.tensor([2.0,2.0,1.0], requires_grad=True)
y = torch.max(x); y.backward()
print(f"torch.max([2,2,1]) full-reduction grad = {g(x)}")

x = torch.tensor([2.0,2.0,1.0], requires_grad=True)
y = torch.amax(x, dim=0); y.backward()
print(f"torch.amax([2,2,1], dim=0) grad = {g(x)}")

x = torch.tensor([[2.0,2.0],[1.0,3.0]], requires_grad=True)
y = torch.amax(x, dim=(0,1)); y.backward()
print(f"torch.amax 2d dim=(0,1) ties grad = {g(x)}")

x = torch.tensor([2.0,2.0,1.0], requires_grad=True)
vals, idx = torch.max(x, dim=0); vals.backward()
print(f"torch.max([2,2,1], dim=0).values grad = {g(x)}  (idx={idx.item()})")

x = torch.tensor([1.0,1.0,3.0], requires_grad=True)
y = torch.min(x); y.backward()
print(f"torch.min([1,1,3]) full-reduction grad = {g(x)}")

print("\n=== KLDiv target=0 gradient (log_target=False, target requires grad) ===")
inp = torch.tensor([-1.0,-2.0], requires_grad=True)          # log-probabilities
tgt = torch.tensor([0.0,0.5], requires_grad=True)            # probabilities, one exactly 0
out = torch.nn.functional.kl_div(inp, tgt, reduction='sum', log_target=False)
out.backward()
print(f"kl_div sum, target=[0,0.5]: out={out.item():.4g}  d_input={g(inp)}  d_target={g(tgt)}")
