"""
Tensotron parity-fixture generator.

This is the ONLY place torch is needed. Run it to (re)generate the committed JSON
fixtures under tests/Tensotron.Tests/Fixtures/. The C# tests read those JSON files
and never import torch — so the build and test run have no torch dependency.

Each fixture embeds:
  - torch_version       : provenance
  - source              : the exact Python that produced the cases (via inspect)
  - cases[]             : recorded inputs, grad_output, expected output, expected grads

So the "exact code required to regenerate" lives inside the test hierarchy; you only
run this when you want to add or change a fixture.

Usage:  python tools/fixtures/gen.py
"""

import torch, json, inspect, os

torch.manual_seed(0)

OUT = os.path.join(os.path.dirname(__file__), "..", "..",
                   "tests", "Tensotron.Tests", "Fixtures")


def _t(x):
    return {"shape": list(x.shape), "data": x.detach().flatten().tolist()}


def run_case(name, input_shapes, fn, meta=None):
    inputs = [torch.randn(*s, requires_grad=True) if s else torch.randn((), requires_grad=True)
              for s in input_shapes]
    out = fn(*inputs)
    grad_output = torch.randn_like(out)
    out.backward(grad_output)
    return {
        "name": name,
        "meta": meta,
        "inputs": [{"shape": list(i.shape), "data": i.detach().flatten().tolist()} for i in inputs],
        "grad_output": _t(grad_output),
        "output": _t(out),
        "grads": [_t(i.grad) for i in inputs],
    }


def emit(op, gen_fn):
    data = {
        "op": op,
        "torch_version": torch.__version__,
        "source": inspect.getsource(gen_fn),
        "cases": gen_fn(),
    }
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, op + ".json"), "w") as f:
        json.dump(data, f, indent=2)
    print(f"wrote {op}.json ({len(data['cases'])} cases)")


# ---- one generator per op; its source is embedded in the fixture ----

def gen_add():
    f = lambda a, b: a + b
    return [
        run_case("2x3 + 2x3", [[2, 3], [2, 3]], f),
        run_case("2x3 + 3 (broadcast row)", [[2, 3], [3]], f),
        run_case("2x1 + 1x3 (broadcast both)", [[2, 1], [1, 3]], f),
        run_case("4 + 4", [[4], [4]], f),
    ]


def gen_sum():
    return [
        run_case("sum all 2x3", [[2, 3]], lambda x: x.sum(),
                 meta={"dims": None, "keepdim": False}),
        run_case("sum dim0 keepdim 2x3", [[2, 3]], lambda x: x.sum(dim=0, keepdim=True),
                 meta={"dims": [0], "keepdim": True}),
        run_case("sum dim1 2x3", [[2, 3]], lambda x: x.sum(dim=1),
                 meta={"dims": [1], "keepdim": False}),
        run_case("sum dim1 keepdim 2x3x4", [[2, 3, 4]], lambda x: x.sum(dim=1, keepdim=True),
                 meta={"dims": [1], "keepdim": True}),
    ]


def run_unary(name, fn, domain):
    base = torch.randn(2, 3)
    if domain == "pos":
        x = base.abs() + 0.2
    elif domain == "nz":  # away from non-differentiable kink at 0
        x = base.sign() * (base.abs() + 0.3)
    else:
        x = base
    x = x.clone().detach().requires_grad_(True)
    out = fn(x)
    grad_output = torch.randn_like(out)
    out.backward(grad_output)
    return {
        "name": name,
        "meta": {"op": name},
        "inputs": [{"shape": list(x.shape), "data": x.detach().flatten().tolist()}],
        "grad_output": _t(grad_output),
        "output": _t(out),
        "grads": [_t(x.grad)],
    }


def run_unary_explicit(op, name, data, shape, fn, grad=None):
    """Unary case with hand-crafted input (so boundary values like exact 0 are tested).
    grad_output defaults to ones for an unambiguous, deterministic recorded gradient."""
    x = torch.tensor(data, dtype=torch.float32).reshape(shape).clone().detach().requires_grad_(True)
    out = fn(x)
    go = torch.ones_like(out) if grad is None else torch.tensor(grad, dtype=torch.float32).reshape(out.shape)
    out.backward(go)
    return {
        "name": name,
        "meta": {"op": op},
        "inputs": [{"shape": list(x.shape), "data": x.detach().flatten().tolist()}],
        "grad_output": _t(go),
        "output": _t(out),
        "grads": [_t(x.grad)],
    }


def gen_unary():
    import torch.nn.functional as TF
    ops = [
        ("neg", lambda x: -x, "real"),
        ("abs", lambda x: x.abs(), "nz"),
        ("sign", lambda x: x.sign(), "nz"),
        ("reciprocal", lambda x: x.reciprocal(), "pos"),
        ("square", lambda x: x * x, "real"),
        ("sqrt", lambda x: x.sqrt(), "pos"),
        ("rsqrt", lambda x: x.rsqrt(), "pos"),
        ("exp", lambda x: x.exp(), "real"),
        ("log", lambda x: x.log(), "pos"),
        ("log1p", lambda x: x.log1p(), "pos"),
        ("sin", lambda x: x.sin(), "real"),
        ("cos", lambda x: x.cos(), "real"),
        ("tanh", lambda x: x.tanh(), "real"),
        ("sigmoid", lambda x: x.sigmoid(), "real"),
        ("relu", lambda x: TF.relu(x), "nz"),
        ("gelu", lambda x: TF.gelu(x, approximate="tanh"), "real"),
        ("softplus", lambda x: TF.softplus(x), "real"),
    ]
    cases = [run_unary(n, f, d) for n, f, d in ops]
    # Boundary cases: deliberately hit the non-differentiable kink at exactly 0 that the
    # randn-based "nz" domain steers away from. torch defines the subgradient there; these
    # would have caught the relu/abs/sign grad-at-0 behavior the random fixtures could not.
    kink = [-2.0, -1.0, 0.0, 1.0, 2.0]
    cases += [
        run_unary_explicit("relu", "relu @ kink (incl 0)", kink, [1, 5], TF.relu),
        run_unary_explicit("abs", "abs @ kink (incl 0)", kink, [1, 5], torch.abs),
        run_unary_explicit("sign", "sign @ kink (incl 0)", kink, [1, 5], torch.sign),
    ]
    return cases


def gen_gelu():
    """Exact (erf) GELU — torch's DEFAULT F.gelu(approximate='none') / nn.GELU(), which is what
    Tensotron's Gelu() uses by default. The tanh approximation is covered by the 'gelu' case in
    unary.json (Tensotron's Gelu(approximateTanh: true)). Self-seeded so it regenerates identically
    whether run alone or from the full pipeline."""
    import torch.nn.functional as TF
    torch.manual_seed(0)
    cases = [run_unary("gelu", lambda x: TF.gelu(x), "real")]
    # Boundary: include exactly 0 (gelu(0)=0, grad(0)=0.5) and a spread around it.
    edge = [-3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 3.0]
    cases.append(run_unary_explicit("gelu", "gelu(erf) @ incl 0", edge, [1, 7], lambda x: TF.gelu(x)))
    return cases


def _dom(shape, dom):
    base = torch.randn(*shape)
    if dom == "pos":
        return base.abs() + 0.3
    if dom == "small":
        return base * 0.5
    return base


def run_binary_case(op, name, ashape, bshape, fn, adom, bdom):
    a = _dom(ashape, adom).clone().detach().requires_grad_(True)
    b = _dom(bshape, bdom).clone().detach().requires_grad_(True)
    out = fn(a, b)
    grad_output = torch.randn_like(out)
    out.backward(grad_output)
    return {
        "name": name,
        "meta": {"op": op},
        "inputs": [{"shape": list(a.shape), "data": a.detach().flatten().tolist()},
                   {"shape": list(b.shape), "data": b.detach().flatten().tolist()}],
        "grad_output": _t(grad_output),
        "output": _t(out),
        "grads": [_t(a.grad), _t(b.grad)],
    }


def run_binary_explicit(op, name, adata, bdata, shape, fn, grad=None):
    """Binary case with hand-crafted inputs (so exact ties a==b are tested)."""
    a = torch.tensor(adata, dtype=torch.float32).reshape(shape).clone().detach().requires_grad_(True)
    b = torch.tensor(bdata, dtype=torch.float32).reshape(shape).clone().detach().requires_grad_(True)
    out = fn(a, b)
    go = torch.ones_like(out) if grad is None else torch.tensor(grad, dtype=torch.float32).reshape(out.shape)
    out.backward(go)
    return {
        "name": name,
        "meta": {"op": op},
        "inputs": [{"shape": list(a.shape), "data": a.detach().flatten().tolist()},
                   {"shape": list(b.shape), "data": b.detach().flatten().tolist()}],
        "grad_output": _t(go),
        "output": _t(out),
        "grads": [_t(a.grad), _t(b.grad)],
    }


def gen_binary():
    specs = [
        ("div", lambda a, b: a / b, "real", "pos"),
        ("pow", lambda a, b: a ** b, "pos", "small"),
        ("maximum", lambda a, b: torch.maximum(a, b), "real", "real"),
        ("minimum", lambda a, b: torch.minimum(a, b), "real", "real"),
    ]
    cases = []
    for op, fn, ad, bd in specs:
        cases.append(run_binary_case(op, f"{op} 2x3,2x3", [2, 3], [2, 3], fn, ad, bd))
        cases.append(run_binary_case(op, f"{op} 2x3,3 (bcast)", [2, 3], [3], fn, ad, bd))
    # Ties: torch splits the gradient 0.5/0.5 where a == b; independent randn never ties.
    # (positions 0 and 1 below are exact ties.)
    cases.append(run_binary_explicit("maximum", "maximum with ties",
                                     [1., 2., 3., 2.], [1., 2., 1., 5.], [2, 2], torch.maximum))
    cases.append(run_binary_explicit("minimum", "minimum with ties",
                                     [1., 2., 3., 2.], [1., 2., 1., 5.], [2, 2], torch.minimum))
    return cases


def gen_composite():
    import torch.nn.functional as TF
    # op (for C# dispatch) is kept separate from the display name so we can sweep parameters
    # and add boundary inputs without breaking the meta.op switch.
    def c(op, name, fn, params, data=None, grad=None):
        x = torch.randn(2, 3) if data is None else torch.tensor(data, dtype=torch.float32)
        x = x.clone().detach().requires_grad_(True)
        out = fn(x)
        go = torch.ones_like(out) if grad is None else torch.tensor(grad, dtype=torch.float32).reshape(out.shape)
        out.backward(go)
        return {
            "name": name,
            "meta": {"op": op, "params": params},
            "inputs": [{"shape": list(x.shape), "data": x.detach().flatten().tolist()}],
            "grad_output": _t(go),
            "output": _t(out),
            "grads": [_t(x.grad)],
        }
    # Spans negatives, exact 0 (the kink), and positives — including the bounds for clamp.
    kink = [-2.0, -0.5, 0.0, 0.5, 2.0]
    return [
        c("clamp", "clamp", lambda x: x.clamp(-0.5, 0.5), [-0.5, 0.5]),
        c("clamp", "clamp @ bounds", lambda x: x.clamp(-0.5, 0.5), [-0.5, 0.5], data=kink),
        # leaky_relu: sweep the slope across the regimes a naive max(x, slope*x) gets wrong —
        # <1 (where it accidentally agrees), >1, and negative — each hitting x==0.
        c("leaky_relu", "leaky_relu 0.1", lambda x: TF.leaky_relu(x, 0.1), [0.1]),
        c("leaky_relu", "leaky_relu 0.1 @ kink", lambda x: TF.leaky_relu(x, 0.1), [0.1], data=kink),
        c("leaky_relu", "leaky_relu 2.0 @ kink", lambda x: TF.leaky_relu(x, 2.0), [2.0], data=kink),
        c("leaky_relu", "leaky_relu -0.5 @ kink", lambda x: TF.leaky_relu(x, -0.5), [-0.5], data=kink),
        # elu: sweep alpha; the gradient at x==0 must equal alpha (not a tie-split).
        c("elu", "elu 1.0 @ kink", lambda x: TF.elu(x, 1.0), [1.0], data=kink),
        c("elu", "elu 1.3 @ kink", lambda x: TF.elu(x, 1.3), [1.3], data=kink),
        c("elu", "elu 0.5 @ kink", lambda x: TF.elu(x, 0.5), [0.5], data=kink),
    ]


def run_reduce(op, name, shape, fn, meta, no_grad=False):
    x = torch.randn(*shape)
    if no_grad:
        out = fn(x)
        return {
            "name": name, "meta": meta,
            "inputs": [{"shape": list(x.shape), "data": x.flatten().tolist()}],
            "grad_output": None,
            "output": {"shape": list(out.shape), "data": out.float().flatten().tolist()},
            "grads": [],
        }
    x = x.clone().detach().requires_grad_(True)
    out = fn(x)
    grad_output = torch.randn_like(out)
    out.backward(grad_output)
    return {
        "name": name, "meta": meta,
        "inputs": [{"shape": list(x.shape), "data": x.detach().flatten().tolist()}],
        "grad_output": _t(grad_output),
        "output": _t(out),
        "grads": [_t(x.grad)],
    }


def gen_reduce():
    import torch.nn.functional as TF
    return [
        run_reduce("mean", "mean all", [2, 3], lambda x: x.mean(), {"op": "mean", "dims": None, "keepdim": False}),
        run_reduce("mean", "mean dim1 keep", [2, 3], lambda x: x.mean(dim=1, keepdim=True), {"op": "mean", "dims": [1], "keepdim": True}),
        run_reduce("max", "max dim1", [2, 3], lambda x: x.max(dim=1).values, {"op": "max", "dims": [1], "keepdim": False}),
        run_reduce("max", "max dim0 keep", [2, 3], lambda x: x.max(dim=0, keepdim=True).values, {"op": "max", "dims": [0], "keepdim": True}),
        run_reduce("min", "min dim1", [2, 3], lambda x: x.min(dim=1).values, {"op": "min", "dims": [1], "keepdim": False}),
        run_reduce("prod", "prod dim1", [2, 3], lambda x: x.prod(dim=1), {"op": "prod", "dims": [1], "keepdim": False}),
        run_reduce("var", "var dim1", [2, 4], lambda x: x.var(dim=1), {"op": "var", "dims": [1], "keepdim": False}),
        run_reduce("std", "std dim1", [2, 4], lambda x: x.std(dim=1), {"op": "std", "dims": [1], "keepdim": False}),
        run_reduce("logsumexp", "lse dim1", [2, 3], lambda x: x.logsumexp(dim=1), {"op": "logsumexp", "dims": [1], "keepdim": False}),
        run_reduce("softmax", "softmax dim1", [2, 3], lambda x: TF.softmax(x, dim=1), {"op": "softmax", "dims": [1], "keepdim": False}),
        run_reduce("log_softmax", "log_softmax dim1", [2, 3], lambda x: TF.log_softmax(x, dim=1), {"op": "log_softmax", "dims": [1], "keepdim": False}),
        run_reduce("argmax", "argmax dim1", [2, 3], lambda x: x.argmax(dim=1), {"op": "argmax", "dims": [1], "keepdim": False}, no_grad=True),
        run_reduce("argmin", "argmin dim1", [2, 3], lambda x: x.argmin(dim=1), {"op": "argmin", "dims": [1], "keepdim": False}, no_grad=True),
    ]


def run_reduce_explicit(op, name, data, shape, fn, gout, meta):
    x = torch.tensor(data, dtype=torch.float32).reshape(shape).clone().detach().requires_grad_(True)
    out = fn(x)
    grad_output = torch.tensor(gout, dtype=torch.float32).reshape(out.shape)
    out.backward(grad_output)
    return {
        "name": name, "meta": meta,
        "inputs": [{"shape": list(x.shape), "data": x.detach().flatten().tolist()}],
        "grad_output": _t(grad_output),
        "output": _t(out),
        "grads": [_t(x.grad)],
    }


def run_reduce_explicit_nograd(op, name, data, shape, fn, meta):
    """No-grad reduce (e.g. argmax/argmin) with hand-crafted input to test tie routing."""
    x = torch.tensor(data, dtype=torch.float32).reshape(shape)
    out = fn(x)
    return {
        "name": name, "meta": meta,
        "inputs": [{"shape": list(x.shape), "data": x.flatten().tolist()}],
        "grad_output": None,
        "output": {"shape": list(out.shape), "data": out.float().flatten().tolist()},
        "grads": [],
    }


def gen_reduce_edge():
    # Deterministic edge cases the random fixtures can't hit: zeros under prod,
    # ties under max/min (first-winner routing), and argmax/argmin tie indices.
    return [
        run_reduce_explicit("prod", "prod one-zero row", [0., 2., 3.], [1, 3],
                            lambda x: x.prod(dim=1), [1.0],
                            {"op": "prod", "dims": [1], "keepdim": False}),
        run_reduce_explicit("prod", "prod zero+nonzero rows", [0., 2., 3., 1., 0., 0.], [2, 3],
                            lambda x: x.prod(dim=1), [1.0, 1.0],
                            {"op": "prod", "dims": [1], "keepdim": False}),
        run_reduce_explicit("max", "max tie first-wins", [2., 2., 1.], [1, 3],
                            lambda x: x.max(dim=1).values, [10.0],
                            {"op": "max", "dims": [1], "keepdim": False}),
        run_reduce_explicit("min", "min tie first-wins", [3., 1., 1.], [1, 3],
                            lambda x: x.min(dim=1).values, [10.0],
                            {"op": "min", "dims": [1], "keepdim": False}),
        # torch returns the FIRST tied index; the random argmax/argmin fixtures never tie.
        run_reduce_explicit_nograd("argmax", "argmax tie first-index", [1., 3., 3., 2., 5., 5., 1., 5.], [2, 4],
                                   lambda x: x.argmax(dim=1), {"op": "argmax", "dims": [1], "keepdim": False}),
        run_reduce_explicit_nograd("argmin", "argmin tie first-index", [2., 1., 1., 3., 4., 4., 4., 9.], [2, 4],
                                   lambda x: x.argmin(dim=1), {"op": "argmin", "dims": [1], "keepdim": False}),
    ]


def gen_structural():
    sq = lambda *s: torch.randn(*s).clone().detach().requires_grad_(True)

    def one(name, op, shape, fn, params):
        return run_movement(op, name, shape, fn, params)

    def two(name, op, ashape, bshape, fn, params):
        a = sq(*ashape); b = sq(*bshape)
        out = fn(a, b)
        grad_output = torch.randn_like(out)
        out.backward(grad_output)
        return {
            "name": name, "meta": {"op": op, "params": params},
            "inputs": [{"shape": list(a.shape), "data": a.detach().flatten().tolist()},
                       {"shape": list(b.shape), "data": b.detach().flatten().tolist()}],
            "grad_output": _t(grad_output),
            "output": _t(out),
            "grads": [_t(a.grad), _t(b.grad)],
        }

    cases = [
        one("squeeze all 1x3x1", "squeeze", [1, 3, 1], lambda x: x.squeeze(), None),
        one("squeeze dim0 1x3x1", "squeeze", [1, 3, 1], lambda x: x.squeeze(0), [0]),
        one("unsqueeze 2x3 @1", "unsqueeze", [2, 3], lambda x: x.unsqueeze(1), [1]),
        one("flatten all 2x3x4", "flatten", [2, 3, 4], lambda x: x.flatten(), [0, -1]),
        one("flatten 1..2 2x3x4", "flatten", [2, 3, 4], lambda x: x.flatten(1, 2), [1, 2]),
        one("expand 1x3 -> 4x3", "expand", [1, 3], lambda x: x.expand(4, 3), [4, 3]),
        one("expand 3x1 -> 3x5", "expand", [3, 1], lambda x: x.expand(3, 5), [3, 5]),
        one("narrow dim1 2x5 [1,4)", "narrow", [2, 5], lambda x: x.narrow(1, 1, 3), [1, 1, 3]),
        two("cat dim0 2x3,1x3", "cat", [2, 3], [1, 3], lambda a, b: torch.cat([a, b], 0), [0]),
        two("cat dim1 2x2,2x3", "cat", [2, 2], [2, 3], lambda a, b: torch.cat([a, b], 1), [1]),
        two("stack dim0 2x3,2x3", "stack", [2, 3], [2, 3], lambda a, b: torch.stack([a, b], 0), [0]),
        two("stack dim1 2x3,2x3", "stack", [2, 3], [2, 3], lambda a, b: torch.stack([a, b], 1), [1]),
    ]
    return cases


def gen_matmul():
    f = lambda a, b: a @ b
    return [
        run_case("2x3 @ 3x4", [[2, 3], [3, 4]], f),
        run_case("vec3 @ 3x4", [[3], [3, 4]], f),
        run_case("2x3 @ vec3", [[2, 3], [3]], f),
        run_case("vec3 @ vec3 (dot)", [[3], [3]], f),
    ]


def gen_matmul_nd():
    mm = lambda a, b: a @ b
    return [
        run_case("bmm 4x2x3 @ 4x3x5", [[4, 2, 3], [4, 3, 5]], mm,
                 meta={"op": "bmm"}),
        run_case("4x2x3 @ 3x5 (bcast b)", [[4, 2, 3], [3, 5]], mm,
                 meta={"op": "matmul"}),
        run_case("2x1x2x3 @ 2x3x5 (bcast batch)", [[2, 1, 2, 3], [2, 3, 5]], mm,
                 meta={"op": "matmul"}),
        run_case("4x2x3 @ vec3 (nd @ 1d)", [[4, 2, 3], [3]], mm,
                 meta={"op": "matmul"}),
        run_case("vec3 @ 4x3x5 (1d @ nd)", [[3], [4, 3, 5]], mm,
                 meta={"op": "matmul"}),
        run_case("outer 4,5", [[4], [5]], lambda a, b: torch.outer(a, b),
                 meta={"op": "outer"}),
    ]


def gen_select():
    def case(name, op, ashape, bshape, fn, params=None):
        a = torch.randn(*ashape).clone().detach().requires_grad_(True)
        b = torch.randn(*bshape).clone().detach().requires_grad_(True)
        out = fn(a, b)
        grad_output = torch.randn_like(out)
        out.backward(grad_output)
        return {
            "name": name,
            "meta": {"op": op, "params": params},
            "inputs": [{"shape": list(a.shape), "data": a.detach().flatten().tolist()},
                       {"shape": list(b.shape), "data": b.detach().flatten().tolist()}],
            "grad_output": _t(grad_output),
            "output": _t(out),
            "grads": [_t(a.grad), _t(b.grad)],
        }
    # where: cond derived from a>0; grads flow to a and b.
    where_fn = lambda a, b: torch.where(a > 0, a, b)
    # masked_fill: mask derived from b>0 (no grad to b); grad flows to a only.
    def masked_case(name, shape):
        a = torch.randn(*shape).clone().detach().requires_grad_(True)
        mask = torch.randn(*shape) > 0
        out = a.masked_fill(mask, 0.5)
        grad_output = torch.randn_like(out)
        out.backward(grad_output)
        return {
            "name": name,
            "meta": {"op": "masked_fill", "params": [0.5]},
            "inputs": [{"shape": list(a.shape), "data": a.detach().flatten().tolist()},
                       {"shape": list(mask.shape), "data": mask.float().flatten().tolist()}],
            "grad_output": _t(grad_output),
            "output": _t(out),
            "grads": [_t(a.grad)],
        }
    return [
        case("where 2x3,2x3", "where", [2, 3], [2, 3], where_fn),
        case("where 2x3,3 (bcast)", "where", [2, 3], [3], where_fn),
        masked_case("masked_fill 2x3", [2, 3]),
    ]


def run_movement(op, name, shape, fn, params):
    x = torch.randn(*shape).clone().detach().requires_grad_(True)
    out = fn(x)
    grad_output = torch.randn_like(out)
    out.backward(grad_output)
    return {
        "name": name,
        "meta": {"op": op, "params": params},
        "inputs": [{"shape": list(x.shape), "data": x.detach().flatten().tolist()}],
        "grad_output": _t(grad_output),
        "output": _t(out),
        "grads": [_t(x.grad)],
    }


def gen_movement():
    return [
        run_movement("t", "t 2x3", [2, 3], lambda x: x.t(), None),
        run_movement("transpose", "transpose 2x3x4 (0,2)", [2, 3, 4], lambda x: x.transpose(0, 2), [0, 2]),
        run_movement("transpose", "transpose 2x3x4 (1,2)", [2, 3, 4], lambda x: x.transpose(1, 2), [1, 2]),
        run_movement("permute", "permute 2x3x4 (2,0,1)", [2, 3, 4], lambda x: x.permute(2, 0, 1), [2, 0, 1]),
    ]


def gen_index():
    # Indices live in meta (int lists); the C# side uploads them as int buffers.
    # Cases deliberately include duplicate indices to exercise atomic scatter-add.
    cases = []

    def grad_case(name, meta, inputs, fn):
        out = fn(*inputs)
        grad_output = torch.randn_like(out)
        out.backward(grad_output)
        return {
            "name": name, "meta": meta,
            "inputs": [{"shape": list(i.shape), "data": i.detach().flatten().tolist()} for i in inputs],
            "grad_output": _t(grad_output),
            "output": _t(out),
            "grads": [_t(i.grad) for i in inputs],
        }

    def rg(*s):
        return torch.randn(*s).clone().detach().requires_grad_(True)

    # index_select (1D index addressed by the axis coordinate; duplicate 2 -> accumulate)
    x = rg(4, 3); idx = [0, 2, 2, 1]
    cases.append(grad_case("index_select dim0 4x3 idx[0,2,2,1]",
        {"op": "index_select", "dim": 0, "index": idx},
        [x], lambda x: x.index_select(0, torch.tensor(idx))))
    x = rg(2, 5); idx = [4, 0, 1]
    cases.append(grad_case("index_select dim1 2x5 idx[4,0,1]",
        {"op": "index_select", "dim": 1, "index": idx},
        [x], lambda x: x.index_select(1, torch.tensor(idx))))

    # gather (full-shaped index)
    x = rg(3, 4); index = torch.tensor([[0, 1, 2, 3], [3, 2, 1, 0], [0, 0, 0, 0]])
    cases.append(grad_case("gather dim1 3x4",
        {"op": "gather", "dim": 1, "index": index.flatten().tolist(), "index_shape": list(index.shape)},
        [x], lambda x: x.gather(1, index)))
    x = rg(3, 4); index = torch.tensor([[0, 1, 2, 0], [2, 2, 0, 1]])
    cases.append(grad_case("gather dim0 3x4 -> 2x4",
        {"op": "gather", "dim": 0, "index": index.flatten().tolist(), "index_shape": list(index.shape)},
        [x], lambda x: x.gather(0, index)))
    x = rg(2, 3); index = torch.tensor([[2], [0]])  # cross_entropy label pick
    cases.append(grad_case("gather dim1 2x3 -> 2x1 (label pick)",
        {"op": "gather", "dim": 1, "index": index.flatten().tolist(), "index_shape": list(index.shape)},
        [x], lambda x: x.gather(1, index)))

    # scatter_add (duplicate destinations -> accumulate)
    s = rg(3, 4); src = rg(2, 4); index = torch.tensor([[0, 1, 2, 0], [2, 2, 0, 1]])
    cases.append(grad_case("scatter_add dim0 self3x4 src2x4",
        {"op": "scatter_add", "dim": 0, "index": index.flatten().tolist(), "index_shape": list(index.shape)},
        [s, src], lambda s, sr: s.scatter_add(0, index, sr)))
    s = rg(2, 5); src = rg(2, 3); index = torch.tensor([[0, 2, 4], [1, 1, 3]])
    cases.append(grad_case("scatter_add dim1 self2x5 src2x3",
        {"op": "scatter_add", "dim": 1, "index": index.flatten().tolist(), "index_shape": list(index.shape)},
        [s, src], lambda s, sr: s.scatter_add(1, index, sr)))

    # repeat (tile; includes a new leading axis)
    x = rg(2, 3)
    cases.append(grad_case("repeat 2x3 by (2,2)",
        {"op": "repeat", "sizes": [2, 2]}, [x], lambda x: x.repeat(2, 2)))
    x = rg(3)
    cases.append(grad_case("repeat vec3 by (2,2) -> 2x6",
        {"op": "repeat", "sizes": [2, 2]}, [x], lambda x: x.repeat(2, 2)))
    x = rg(2, 3)
    cases.append(grad_case("repeat 2x3 by (1,3)",
        {"op": "repeat", "sizes": [1, 3]}, [x], lambda x: x.repeat(1, 3)))

    return cases


def gen_losses():
    import torch.nn.functional as TF
    cases = []

    def gout_for(out):
        return torch.randn(()) if out.dim() == 0 else torch.randn_like(out)

    # input requires grad; target is a plain (no-grad) tensor.
    def elt(name, op, shape, fn, reduction, params=None, tgt_fn=None):
        x = torch.randn(*shape).clone().detach().requires_grad_(True)
        tgt = (tgt_fn(shape) if tgt_fn else torch.randn(*shape)).clone().detach()
        out = fn(x, tgt)
        g = gout_for(out)
        out.backward(g)
        return {
            "name": name,
            "meta": {"op": op, "reduction": reduction, "params": params},
            "inputs": [{"shape": list(x.shape), "data": x.detach().flatten().tolist()},
                       {"shape": list(tgt.shape), "data": tgt.flatten().tolist()}],
            "grad_output": _t(g), "output": _t(out), "grads": [_t(x.grad)],
        }

    for red in ["mean", "sum", "none"]:
        cases.append(elt(f"mse {red}", "mse", [2, 3],
                         lambda a, b, r=red: TF.mse_loss(a, b, reduction=r), red))
    cases.append(elt("l1 mean", "l1", [2, 3], lambda a, b: TF.l1_loss(a, b), "mean"))
    cases.append(elt("l1 none", "l1", [2, 3], lambda a, b: TF.l1_loss(a, b, reduction="none"), "none"))
    cases.append(elt("huber mean d1", "huber", [2, 4],
                     lambda a, b: TF.huber_loss(a, b, delta=1.0), "mean", [1.0]))
    cases.append(elt("huber none d0.5", "huber", [2, 4],
                     lambda a, b: TF.huber_loss(a, b, delta=0.5, reduction="none"), "none", [0.5]))

    # bce_with_logits: target is soft labels in [0,1].
    soft = lambda shape: torch.rand(*shape)
    cases.append(elt("bce mean", "bce_with_logits", [2, 3],
                     lambda a, b: TF.binary_cross_entropy_with_logits(a, b), "mean", tgt_fn=soft))
    cases.append(elt("bce none", "bce_with_logits", [2, 3],
                     lambda a, b: TF.binary_cross_entropy_with_logits(a, b, reduction="none"), "none", tgt_fn=soft))

    # nll: input is a leaf of log-probabilities; target = class indices in meta.
    def nll(name, n, c, target, reduction):
        logp = TF.log_softmax(torch.randn(n, c), dim=1).clone().detach().requires_grad_(True)
        out = TF.nll_loss(logp, torch.tensor(target), reduction=reduction)
        g = gout_for(out); out.backward(g)
        return {"name": name, "meta": {"op": "nll", "reduction": reduction, "index": target},
                "inputs": [{"shape": list(logp.shape), "data": logp.detach().flatten().tolist()}],
                "grad_output": _t(g), "output": _t(out), "grads": [_t(logp.grad)]}

    # cross_entropy: input is raw logits leaf.
    def ce(name, n, c, target, reduction):
        x = torch.randn(n, c).clone().detach().requires_grad_(True)
        out = TF.cross_entropy(x, torch.tensor(target), reduction=reduction)
        g = gout_for(out); out.backward(g)
        return {"name": name, "meta": {"op": "cross_entropy", "reduction": reduction, "index": target},
                "inputs": [{"shape": list(x.shape), "data": x.detach().flatten().tolist()}],
                "grad_output": _t(g), "output": _t(out), "grads": [_t(x.grad)]}

    cases.append(nll("nll mean", 4, 3, [0, 2, 1, 2], "mean"))
    cases.append(nll("nll none", 4, 3, [0, 2, 1, 2], "none"))
    cases.append(ce("cross_entropy mean", 4, 3, [0, 2, 1, 2], "mean"))
    cases.append(ce("cross_entropy sum", 4, 3, [0, 2, 1, 2], "sum"))

    # kl_div: input = log-probs leaf, target = probs (no grad).
    def kl(name, shape, reduction):
        inp = TF.log_softmax(torch.randn(*shape), dim=-1).clone().detach().requires_grad_(True)
        tgt = TF.softmax(torch.randn(*shape), dim=-1)
        out = TF.kl_div(inp, tgt, reduction=reduction)
        g = gout_for(out); out.backward(g)
        return {"name": name, "meta": {"op": "kl_div", "reduction": reduction},
                "inputs": [{"shape": list(inp.shape), "data": inp.detach().flatten().tolist()},
                           {"shape": list(tgt.shape), "data": tgt.flatten().tolist()}],
                "grad_output": _t(g), "output": _t(out), "grads": [_t(inp.grad)]}

    cases.append(kl("kl_div mean", [2, 3], "mean"))
    cases.append(kl("kl_div sum", [2, 3], "sum"))

    # kl_div with EXACT zero-probability targets: torch treats target·log(target) as 0 there
    # (lim t→0 t·log t = 0). The naive target·(log target − input) yields 0·−∞ = NaN, so this
    # case directly guards the bug the softmax-only fixtures missed.
    def kl_zero(name, reduction):
        inp = TF.log_softmax(torch.tensor([[0.5, 0.3, 0.2], [0.1, 0.6, 0.3]]), dim=-1) \
            .clone().detach().requires_grad_(True)
        tgt = torch.tensor([[0.0, 0.4, 0.6], [1.0, 0.0, 0.0]])  # exact zeros (and a one-hot row)
        out = TF.kl_div(inp, tgt, reduction=reduction)
        g = gout_for(out); out.backward(g)
        return {"name": name, "meta": {"op": "kl_div", "reduction": reduction},
                "inputs": [{"shape": list(inp.shape), "data": inp.detach().flatten().tolist()},
                           {"shape": list(tgt.shape), "data": tgt.flatten().tolist()}],
                "grad_output": _t(g), "output": _t(out), "grads": [_t(inp.grad)]}

    cases.append(kl_zero("kl_div zero-target mean", "mean"))
    cases.append(kl_zero("kl_div zero-target sum", "sum"))
    return cases


def gen_optim():
    import torch.optim as O

    def make(kind, params, c):
        if kind == "sgd":
            return O.SGD(params, lr=c["lr"], momentum=c.get("momentum", 0.0),
                         weight_decay=c.get("weight_decay", 0.0),
                         dampening=c.get("dampening", 0.0), nesterov=bool(c.get("nesterov", 0)))
        if kind == "adam":
            return O.Adam(params, lr=c["lr"], betas=(c.get("beta1", 0.9), c.get("beta2", 0.999)),
                          eps=c.get("eps", 1e-8), weight_decay=c.get("weight_decay", 0.0))
        if kind == "adamw":
            return O.AdamW(params, lr=c["lr"], betas=(c.get("beta1", 0.9), c.get("beta2", 0.999)),
                           eps=c.get("eps", 1e-8), weight_decay=c.get("weight_decay", 0.01))
        if kind == "rmsprop":
            return O.RMSprop(params, lr=c["lr"], alpha=c.get("alpha", 0.99), eps=c.get("eps", 1e-8),
                             weight_decay=c.get("weight_decay", 0.0), momentum=c.get("momentum", 0.0),
                             centered=bool(c.get("centered", 0)))
        raise ValueError(kind)

    # Convex replay problem: loss = 0.5*||p - target||². grad = (p - target), evolving
    # as p moves — exercises momentum/moment adaptation. p0/target are recorded so the
    # C# replay is identical (no shared RNG).
    def run(name, kind, cfg, shape=[3, 4]):
        torch.manual_seed(123)
        p = torch.randn(*shape, requires_grad=True)
        p0 = p.detach().clone()
        target = torch.randn(*shape)
        steps = int(cfg["steps"])
        opt = make(kind, [p], cfg)
        for _ in range(steps):
            opt.zero_grad()
            loss = 0.5 * ((p - target) ** 2).sum()
            loss.backward()
            opt.step()
        return {
            "name": name, "meta": {"op": kind, "config": cfg},
            "inputs": [{"shape": list(p0.shape), "data": p0.flatten().tolist()},
                       {"shape": list(target.shape), "data": target.flatten().tolist()}],
            "grad_output": None, "output": _t(p.detach()), "grads": [],
        }

    return [
        run("sgd lr0.1", "sgd", {"lr": 0.1, "steps": 5}),
        run("sgd momentum0.9", "sgd", {"lr": 0.1, "momentum": 0.9, "steps": 5}),
        run("sgd nesterov+wd", "sgd", {"lr": 0.1, "momentum": 0.9, "weight_decay": 0.01, "nesterov": 1, "steps": 5}),
        run("adam lr0.05", "adam", {"lr": 0.05, "steps": 6}),
        run("adam wd", "adam", {"lr": 0.05, "weight_decay": 0.01, "steps": 6}),
        run("adamw", "adamw", {"lr": 0.05, "weight_decay": 0.05, "steps": 6}),
        run("rmsprop", "rmsprop", {"lr": 0.01, "steps": 6}),
        run("rmsprop momentum", "rmsprop", {"lr": 0.01, "momentum": 0.9, "steps": 6}),
        run("rmsprop centered", "rmsprop", {"lr": 0.01, "centered": 1, "steps": 6}),
    ]


def gen_init():
    # Scalar parity for the scale formulas (the random draws themselves can't match
    # torch's bit-stream, so we verify fan/gain/bound/std against torch's own values).
    import torch.nn.init as NI
    import math
    cases = []

    def fan(shape):
        w = torch.empty(*shape)
        fi, fo = NI._calculate_fan_in_and_fan_out(w)
        return {"name": f"fan {shape}", "meta": {"op": "fan", "dims": shape},
                "inputs": [], "grad_output": None,
                "output": {"shape": [2], "data": [float(fi), float(fo)]}, "grads": []}

    def gain(nl, a=0.0):
        g = NI.calculate_gain(nl, a) if nl == "leaky_relu" else NI.calculate_gain(nl)
        return {"name": f"gain {nl} a={a}", "meta": {"op": "gain", "reduction": nl, "params": [a]},
                "inputs": [], "grad_output": None,
                "output": {"shape": [1], "data": [float(g)]}, "grads": []}

    def ku_bound(shape, a, nl):
        w = torch.empty(*shape)
        fan_in, _ = NI._calculate_fan_in_and_fan_out(w)
        std = NI.calculate_gain(nl, a) / math.sqrt(fan_in)
        bound = math.sqrt(3.0) * std
        return {"name": f"kaiming_uniform bound {shape} a={a} {nl}",
                "meta": {"op": "kaiming_uniform_bound", "dims": shape, "reduction": nl, "params": [a]},
                "inputs": [], "grad_output": None,
                "output": {"shape": [1], "data": [float(bound)]}, "grads": []}

    def xu_bound(shape, gain_v):
        w = torch.empty(*shape)
        fan_in, fan_out = NI._calculate_fan_in_and_fan_out(w)
        std = gain_v * math.sqrt(2.0 / (fan_in + fan_out))
        bound = math.sqrt(3.0) * std
        return {"name": f"xavier_uniform bound {shape} gain={gain_v}",
                "meta": {"op": "xavier_uniform_bound", "dims": shape, "params": [gain_v]},
                "inputs": [], "grad_output": None,
                "output": {"shape": [1], "data": [float(bound)]}, "grads": []}

    cases += [fan([4, 6]), fan([2, 3, 4])]
    cases += [gain("relu"), gain("tanh"), gain("linear"), gain("selu"),
              gain("leaky_relu", 0.2), gain("leaky_relu", math.sqrt(5))]
    cases += [ku_bound([4, 6], 0.0, "relu"), ku_bound([8, 5], math.sqrt(5), "leaky_relu")]
    cases += [xu_bound([4, 6], 1.0), xu_bound([8, 5], 1.5)]
    return cases


def gen_norm():
    import torch.nn.functional as TF
    cases = []

    def layer_norm(name, shape, nshape):
        x = torch.randn(*shape).clone().detach().requires_grad_(True)
        w = torch.randn(*nshape).clone().detach().requires_grad_(True)
        b = torch.randn(*nshape).clone().detach().requires_grad_(True)
        out = TF.layer_norm(x, nshape, w, b, eps=1e-5)
        g = torch.randn_like(out)
        out.backward(g)
        return {"name": name, "meta": {"op": "layer_norm", "dims": nshape},
                "inputs": [{"shape": list(x.shape), "data": x.detach().flatten().tolist()},
                           {"shape": list(w.shape), "data": w.detach().flatten().tolist()},
                           {"shape": list(b.shape), "data": b.detach().flatten().tolist()}],
                "grad_output": _t(g), "output": _t(out),
                "grads": [_t(x.grad), _t(w.grad), _t(b.grad)]}

    def batch_norm(name, n, c):
        bn = torch.nn.BatchNorm1d(c, eps=1e-5, momentum=0.1)
        w = torch.randn(c); b = torch.randn(c)
        with torch.no_grad():
            bn.weight.copy_(w); bn.bias.copy_(b)
        x = torch.randn(n, c).clone().detach().requires_grad_(True)
        out = bn(x)
        g = torch.randn_like(out)
        out.backward(g)
        return {"name": name, "meta": {"op": "batch_norm", "dims": [c]},
                "inputs": [{"shape": list(x.shape), "data": x.detach().flatten().tolist()},
                           {"shape": list(w.shape), "data": w.flatten().tolist()},
                           {"shape": list(b.shape), "data": b.flatten().tolist()}],
                "grad_output": _t(g), "output": _t(out),
                "grads": [_t(x.grad), _t(bn.weight.grad), _t(bn.bias.grad)],
                "running_mean": _t(bn.running_mean.detach()),
                "running_var": _t(bn.running_var.detach())}

    def group_norm(name, shape, num_groups):
        x = torch.randn(*shape).clone().detach().requires_grad_(True)
        c = shape[1]
        w = torch.randn(c).clone().detach().requires_grad_(True)
        b = torch.randn(c).clone().detach().requires_grad_(True)
        out = TF.group_norm(x, num_groups, w, b, eps=1e-5)
        g = torch.randn_like(out)
        out.backward(g)
        return {"name": name, "meta": {"op": "group_norm", "dim": num_groups},
                "inputs": [{"shape": list(x.shape), "data": x.detach().flatten().tolist()},
                           {"shape": list(w.shape), "data": w.detach().flatten().tolist()},
                           {"shape": list(b.shape), "data": b.detach().flatten().tolist()}],
                "grad_output": _t(g), "output": _t(out),
                "grads": [_t(x.grad), _t(w.grad), _t(b.grad)]}

    def batch_norm2d(name, n, c, h, w):
        bn = torch.nn.BatchNorm2d(c, eps=1e-5, momentum=0.1)
        gw = torch.randn(c); gb = torch.randn(c)
        with torch.no_grad():
            bn.weight.copy_(gw); bn.bias.copy_(gb)
        x = torch.randn(n, c, h, w).clone().detach().requires_grad_(True)
        out = bn(x)
        g = torch.randn_like(out)
        out.backward(g)
        return {"name": name, "meta": {"op": "batch_norm_2d", "dims": [c]},
                "inputs": [{"shape": list(x.shape), "data": x.detach().flatten().tolist()},
                           {"shape": list(gw.shape), "data": gw.flatten().tolist()},
                           {"shape": list(gb.shape), "data": gb.flatten().tolist()}],
                "grad_output": _t(g), "output": _t(out),
                "grads": [_t(x.grad), _t(bn.weight.grad), _t(bn.bias.grad)],
                "running_mean": _t(bn.running_mean.detach()),
                "running_var": _t(bn.running_var.detach())}

    cases.append(layer_norm("layer_norm 4x6 over [6]", [4, 6], [6]))
    cases.append(layer_norm("layer_norm 2x3x4 over [3,4]", [2, 3, 4], [3, 4]))
    cases.append(batch_norm("batch_norm 8x4", 8, 4))
    cases.append(batch_norm("batch_norm 16x3", 16, 3))
    cases.append(group_norm("group_norm 2x6x4 g3", [2, 6, 4], 3))
    cases.append(group_norm("group_norm 2x8x5x5 g4", [2, 8, 5, 5], 4))
    cases.append(group_norm("group_norm 2x6x4 g1 (layernorm-like)", [2, 6, 4], 1))
    # num_groups == C: the InstanceNorm regime (one group per channel) — a distinct
    # group-indexing path the g3/g4/g1 cases don't exercise.
    cases.append(group_norm("group_norm 2x6x4 g6 (instancenorm-like)", [2, 6, 4], 6))
    cases.append(batch_norm2d("batch_norm2d 4x3x5x5", 4, 3, 5, 5))
    cases.append(batch_norm2d("batch_norm2d 8x4x3x3", 8, 4, 3, 3))
    return cases


def gen_conv():
    import torch.nn.functional as TF

    def conv(name, xshape, wshape, stride, pad, dil, bias=True):
        x = torch.randn(*xshape).clone().detach().requires_grad_(True)
        wt = torch.randn(*wshape).clone().detach().requires_grad_(True)
        b = torch.randn(wshape[0]).clone().detach().requires_grad_(True) if bias else None
        out = TF.conv2d(x, wt, b, stride=stride, padding=pad, dilation=dil)
        g = torch.randn_like(out)
        out.backward(g)
        inputs = [{"shape": list(x.shape), "data": x.detach().flatten().tolist()},
                  {"shape": list(wt.shape), "data": wt.detach().flatten().tolist()}]
        grads = [_t(x.grad), _t(wt.grad)]
        if bias:
            inputs.append({"shape": list(b.shape), "data": b.detach().flatten().tolist()})
            grads.append(_t(b.grad))
        return {"name": name,
                "meta": {"op": "conv2d", "config": {"stride": stride, "padding": pad,
                                                    "dilation": dil, "bias": 1 if bias else 0}},
                "inputs": inputs, "grad_output": _t(g), "output": _t(out), "grads": grads}

    return [
        conv("3x3 s1 p0", [2, 3, 8, 8], [4, 3, 3, 3], 1, 0, 1),
        conv("3x3 s2 p1", [2, 3, 9, 9], [5, 3, 3, 3], 2, 1, 1),
        conv("3x3 dil2 p2", [1, 2, 7, 7], [3, 2, 3, 3], 1, 2, 2),
        conv("2x2 s1 nobias", [2, 4, 6, 6], [6, 4, 2, 2], 1, 0, 1, bias=False),
        conv("1x1 s1", [2, 3, 5, 5], [8, 3, 1, 1], 1, 0, 1),
    ]


def gen_pool():
    import torch.nn.functional as TF

    def pool(name, mode, xshape, k, stride, pad, dil=1):
        x = torch.randn(*xshape).clone().detach().requires_grad_(True)
        if mode == "max":
            out = TF.max_pool2d(x, kernel_size=k, stride=stride, padding=pad, dilation=dil)
        else:
            out = TF.avg_pool2d(x, kernel_size=k, stride=stride, padding=pad)
        g = torch.randn_like(out)
        out.backward(g)
        return {"name": name,
                "meta": {"op": mode + "_pool2d",
                         "config": {"kernel": k, "stride": stride, "padding": pad, "dilation": dil}},
                "inputs": [{"shape": list(x.shape), "data": x.detach().flatten().tolist()}],
                "grad_output": _t(g), "output": _t(out), "grads": [_t(x.grad)]}

    def pool_explicit(name, mode, data, xshape, k, stride, pad, dil=1):
        # Hand-crafted input so windows contain TIED maxima — randn never ties, so the
        # gradient-routing convention (torch sends it to the first max, row-major) is
        # otherwise untested. Overlapping stride also exercises gradient accumulation.
        x = torch.tensor(data, dtype=torch.float32).reshape(xshape).clone().detach().requires_grad_(True)
        out = TF.max_pool2d(x, kernel_size=k, stride=stride, padding=pad, dilation=dil)
        g = torch.ones_like(out)
        out.backward(g)
        return {"name": name,
                "meta": {"op": "max_pool2d",
                         "config": {"kernel": k, "stride": stride, "padding": pad, "dilation": dil}},
                "inputs": [{"shape": list(x.shape), "data": x.detach().flatten().tolist()}],
                "grad_output": _t(g), "output": _t(out), "grads": [_t(x.grad)]}

    return [
        pool("max 2x2 s2", "max", [2, 3, 8, 8], 2, 2, 0),
        pool("max 3x3 s2 p1", "max", [2, 3, 9, 9], 3, 2, 1),
        pool("max 3x3 s1 dil2", "max", [1, 2, 9, 9], 3, 1, 0, dil=2),
        pool("max 2x2 s1 overlap", "max", [2, 4, 6, 6], 2, 1, 0),
        pool("avg 2x2 s2", "avg", [2, 3, 8, 8], 2, 2, 0),
        pool("avg 3x3 s2 p1", "avg", [2, 3, 9, 9], 3, 2, 1),
        pool("avg 2x2 s1 overlap", "avg", [2, 4, 6, 6], 2, 1, 0),
        # tie routing + overlap accumulation (1x1x2x3, k2 s1 -> two overlapping windows).
        pool_explicit("max ties + overlap", "max", [3., 3., 1., 2., 3., 3.], [1, 1, 2, 3], 2, 1, 0),
    ]


def gen_sched():
    L = torch.optim.lr_scheduler

    def make(kind, opt, c):
        if kind == "step": return L.StepLR(opt, step_size=int(c["step_size"]), gamma=c["gamma"])
        if kind == "exp": return L.ExponentialLR(opt, gamma=c["gamma"])
        if kind == "cosine": return L.CosineAnnealingLR(opt, T_max=int(c["t_max"]), eta_min=c.get("eta_min", 0.0))
        if kind == "linear": return L.LinearLR(opt, start_factor=c["start"], end_factor=c["end"], total_iters=int(c["total"]))
        raise ValueError(kind)

    def seq(name, kind, cfg, n=8):
        p = torch.zeros(1, requires_grad=True)
        p.grad = torch.zeros_like(p)
        opt = torch.optim.SGD([p], lr=cfg["lr"])
        sch = make(kind, opt, cfg)
        lrs = [opt.param_groups[0]["lr"]]
        for _ in range(n):
            opt.step()
            sch.step()
            lrs.append(opt.param_groups[0]["lr"])
        return {"name": name, "meta": {"op": kind, "config": cfg},
                "inputs": [], "grad_output": None,
                "output": {"shape": [len(lrs)], "data": [float(v) for v in lrs]}, "grads": []}

    return [
        seq("step size2 gamma0.5", "step", {"lr": 0.1, "step_size": 2, "gamma": 0.5}),
        seq("exp gamma0.9", "exp", {"lr": 0.1, "gamma": 0.9}),
        seq("cosine tmax5 etamin0.01", "cosine", {"lr": 0.1, "t_max": 5, "eta_min": 0.01}),
        seq("linear warmup", "linear", {"lr": 0.1, "start": 0.1, "end": 1.0, "total": 4}),
    ]


if __name__ == "__main__":
    emit("add", gen_add)
    emit("sum", gen_sum)
    emit("matmul", gen_matmul)
    emit("unary", gen_unary)
    emit("gelu", gen_gelu)
    emit("binary", gen_binary)
    emit("composite", gen_composite)
    emit("reduce", gen_reduce)
    emit("reduce_edge", gen_reduce_edge)
    emit("matmul_nd", gen_matmul_nd)
    emit("select", gen_select)
    emit("movement", gen_movement)
    emit("structural", gen_structural)
    emit("index", gen_index)
    emit("losses", gen_losses)
    emit("optim", gen_optim)
    emit("init", gen_init)
    emit("norm", gen_norm)
    emit("sched", gen_sched)
    emit("conv", gen_conv)
    emit("pool", gen_pool)
