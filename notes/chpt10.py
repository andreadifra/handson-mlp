import marimo

__generated_with = "0.23.0"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    ## Import common libraries
    import torch
    import torch.nn as nn
    import torch.optim as optim

    return mo, nn, optim, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Book Questions
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2. Creating tensors
    """)
    return


@app.cell
def _(torch):
    X = torch.tensor([[1.0, 5.6, 2.3], [2.0, 3.4, 4.5], [3.0, 1.2, 6.7]], device="cuda")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 5. Runtime Error Check
    """)
    return


@app.cell
def _(torch):
    t = torch.tensor(2.0, requires_grad=True)
    z = t.cos().exp_()
    z.backward()
    return t, z


@app.cell
def _(t, torch):
    t2 = torch.tensor(2.0, requires_grad=True)
    z2 = t.cos_().exp()
    z2.backward()
    return


@app.cell
def _(z):
    z.grad
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 6. Linear module
    """)
    return


@app.cell
def _(nn):
    test_mod = nn.Linear(100, 200)
    return (test_mod,)


@app.cell
def _(test_mod):
    print(test_mod.bias.shape)
    print(test_mod.weight.shape)
    return


@app.cell
def _(test_mod):
    test_mod
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 8. Testing moving optimizer to GPU after creating it
    """)
    return


@app.cell
def _(nn, optim):
    # 1. Define a simple model
    model = nn.Linear(10, 1)

    # 2. WRONG WAY: Initialize optimizer while model is on CPU
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    # 3. Move model to GPU
    model.to("cuda:0")
    return model, optimizer


@app.cell
def _(model, optimizer):
    # 4. Check the mismatch
    model_p = next(model.parameters())
    opt_p = optimizer.param_groups[0]['params'][0]

    print(f"Model parameter device: {model_p.device}")
    print(f"Optimizer tracked parameter device: {opt_p.device}")

    # This check tells us if they are the same actual object in memory
    print(f"Are they the same object? {id(model_p) == id(opt_p)}")
    return


@app.cell
def _(model, optimizer, torch):
    # 5. Attempt a dummy training step
    try:
        input_data = torch.randn(5, 10).to("cuda:0")
        output = model(input_data)
        loss = output.sum()
        loss.backward()
        optimizer.step()
        print("\nStep successful (but might be updating the wrong memory!)")
    except Exception as e:
        print(f"\nCaught expected error: {e}")
    return


@app.cell
def _(optimizer):
    optimizer.load_state_dict
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 13. Use autograd to find gradient of $f(x,y) = sin(x^2y)$ at the point $(x,y) = (1.2,3.4)$.
    """)
    return


@app.cell
def _():
    from torch.autograd import grad

    return (grad,)


@app.cell
def _(torch):
    ## Define the function
    def f(x,y):
        return torch.sin(x**2 * y)

    x = torch.tensor(1.2, requires_grad=True)
    y = torch.tensor(3.4, requires_grad=True)

    z = f(x,y)
    return x, y, z


@app.cell
def _(grad, x, y, z):
    grad(z, [x,y])
    return


@app.cell
def _(torch, x, y):
    ## Alternatively 

    out = torch.sin(x**2 * y)
    out.backward()
    print(x.grad, y.grad)
    return


if __name__ == "__main__":
    app.run()
