import modal

image = modal.Image.debian_slim().pip_install("torch")
app = modal.App("cuda-test", image=image)


@app.function(gpu="T4", image=image)
def run_torch():
    import torch
    has_cuda = torch.cuda.is_available()
    print(f"It is {has_cuda} {torch.cuda.current_device()} that torch can access CUDA")
    return has_cuda