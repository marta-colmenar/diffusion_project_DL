import torch


def c_funcs(sigma: torch.Tensor, sigma_data: float):
    # sigma: (B,)
    sd = sigma_data
    denom = torch.sqrt(sd**2 + sigma**2)
    c_in = 1.0 / denom
    c_out = sigma * sd / denom
    c_skip = (sd**2) / (denom**2)
    c_noise = torch.log(sigma) / 4.0
    return c_in, c_out, c_skip, c_noise


def euler_sample(
    model,
    sigmas,
    n,
    channels,
    H,
    sigma_data,
    device,
    class_label=None,
    guidance_scale=3.0,
):
    assert not model.training

    # sigmas: 1D tensor [sigma_0, sigma_1, ..., sigma_T] (assumed decreasing)
    x = torch.randn(n, channels, H, H, device=device) * sigmas[0].to(device)

    if class_label is not None:
        labels_cond = torch.full((n,), class_label, dtype=torch.long, device=device)
    else:
        labels_cond = torch.full((n,), -1, dtype=torch.long, device=device)

    # Labels for unconditional path
    labels_uncond = torch.full((n,), -1, dtype=torch.long, device=device)

    for i, sigma in enumerate(sigmas):
        sigma = sigma.to(device)
        sigma_next = (
            sigmas[i + 1].to(device)
            if i + 1 < len(sigmas)
            else torch.tensor(0.0, device=device)
        )

        sigma_b = sigma.repeat(n)
        c_in, c_out, c_skip, c_noise = c_funcs(sigma_b, sigma_data)
        cin_x = c_in.view(-1, 1, 1, 1) * x

        with torch.no_grad():
            # TODO: here we run the model twice not to complicate the model code with CFG logic. We might want to optimize this later.
            pred_uncond = model(cin_x, c_noise.to(device), labels_uncond)
            pred_cond = model(cin_x, c_noise.to(device), labels_cond)

            # if labels_uncond == labels_cond we practically have unconditioned predictions -> OKAY
            pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)

        x_denoised = c_skip.view(-1, 1, 1, 1) * x + c_out.view(-1, 1, 1, 1) * pred
        d = (x - x_denoised) / sigma.view(1, 1, 1, 1)
        x = x + d * (sigma_next - sigma).view(1, 1, 1, 1)

    return x
