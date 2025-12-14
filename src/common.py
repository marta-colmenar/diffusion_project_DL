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

def heun_sample(
    model,
    sigmas,
    n,
    channels,
    H,
    sigma_data,
    device,
    labels=None,
    guidance_scale=1.0,
    s_churn=0.0,     # 0.1 enables stochasticity
    s_tmin=0.0,
    s_tmax=50.0,
    s_noise=1.0,
):
    """
    Stochastic Heun sampler (EDM / Karras et al. 2022, Alg. 2)
    with classifier-free guidance.

    Args:
        model: denoiser, called as model(x, c_noise, labels)
        sigmas: 1D tensor of noise levels (descending), shape [N]
        n: batch size
        channels: image channels
        H: image height/width
        sigma_data: dataset std
        device: torch.device
        labels: class labels tensor or None (unconditional)
        guidance_scale: CFG scale (1.0 = no guidance)
        s_churn, s_tmin, s_tmax, s_noise: stochasticity parameters

    Returns:
        x: sampled images, shape (n, channels, H, H)
    """

    assert not model.training

    sigmas = sigmas.to(device).flatten()
    N = sigmas.numel()

    # Initial sample
    x = torch.randn(n, channels, H, H, device=device) * sigmas[0]

    # CFG labels
    if labels is not None:
        labels_cond = labels.to(device)
    else:
        labels_cond = torch.full((n,), -1, dtype=torch.long, device=device)

    labels_uncond = torch.full((n,), -1, dtype=torch.long, device=device)

    def edm_coeffs(sigma_b):
        denom = torch.sqrt(sigma_data**2 + sigma_b**2)
        c_in = 1.0 / denom
        c_out = sigma_b * sigma_data / denom
        c_skip = (sigma_data**2) / (sigma_data**2 + sigma_b**2)
        c_noise = torch.log(sigma_b) / 4.0
        return c_in, c_out, c_skip, c_noise

    for i in range(N):
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1] if i + 1 < N else torch.tensor(0.0, device=device)

        # --- Stochasticity (EDM gamma) ---
        gamma = 0.0
        if s_tmin <= sigma <= s_tmax:
            gamma = min(s_churn / max(N - 1, 1), 2**0.5 - 1.0)

        sigma_hat = sigma * (1.0 + gamma)

        if gamma > 0:
            eps = torch.randn_like(x) * s_noise
            x_hat = x + eps * torch.sqrt(sigma_hat**2 - sigma**2)
        else:
            x_hat = x

        # --- Denoiser at sigma_hat ---
        sigma_b = sigma_hat.expand(n)
        c_in, c_out, c_skip, c_noise = edm_coeffs(sigma_b)

        cin_x = c_in.view(-1, 1, 1, 1) * x_hat

        with torch.no_grad():
            denoised_uncond = model(cin_x, c_noise, labels_uncond)
            denoised_cond = model(cin_x, c_noise, labels_cond)
            denoised = denoised_uncond + guidance_scale * (
                denoised_cond - denoised_uncond
            )

        x_denoised = (
            c_skip.view(-1, 1, 1, 1) * x_hat
            + c_out.view(-1, 1, 1, 1) * denoised
        )

        d = (x_hat - x_denoised) / sigma_hat

        # --- Euler step ---
        dt = sigma_next - sigma_hat
        x_euler = x_hat + d * dt

        # --- Heun correction ---
        if sigma_next > 0:
            sigma_b_next = sigma_next.expand(n)
            c_in_n, c_out_n, c_skip_n, c_noise_n = edm_coeffs(sigma_b_next)

            cin_x_next = c_in_n.view(-1, 1, 1, 1) * x_euler

            with torch.no_grad():
                denoised_uncond_next = model(cin_x_next, c_noise_n, labels_uncond)
                denoised_cond_next = model(cin_x_next, c_noise_n, labels_cond)
                denoised_next = denoised_uncond_next + guidance_scale * (
                    denoised_cond_next - denoised_uncond_next
                )

            x_denoised_next = (
                c_skip_n.view(-1, 1, 1, 1) * x_euler
                + c_out_n.view(-1, 1, 1, 1) * denoised_next
            )

            d_next = (x_euler - x_denoised_next) / sigma_next
            x = x_hat + 0.5 * (d + d_next) * dt
        else:
            x = x_euler

    return x
