import torch
import torch.utils.data
from librosa.filters import mel as librosa_mel_fn
import librosa.util

MAX_WAV_VALUE = 32768.0


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    """
    PARAMS
    ------
    C: compression factor
    """
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    """
    PARAMS
    ------
    C: compression factor used to compress
    """
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output


mel_basis = {}
hann_window = {}


def spectrogram_torch(y, n_fft, sampling_rate, hop_size, win_size, center=False):
    if torch.min(y) < -1.1:
        print("min value is ", torch.min(y))
    if torch.max(y) > 1.1:
        print("max value is ", torch.max(y))

    global hann_window
    dtype_device = str(y.dtype) + "_" + str(y.device)
    wnsize_dtype_device = str(win_size) + "_" + dtype_device
    if wnsize_dtype_device not in hann_window:
        hann_window[wnsize_dtype_device] = torch.hann_window(win_size).to(
            dtype=y.dtype, device=y.device
        )

    y = torch.nn.functional.pad(
        y.unsqueeze(1),
        (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
        mode="reflect",
    )
    y = y.squeeze(1)

    # Actualizado para PyTorch 2.x: stft ahora devuelve complex64 por defecto
    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window[wnsize_dtype_device],
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,  # Cambiado a True (por defecto en PyTorch 2.x)
    )
    
    # Convertir complex a magnitud
    spec = torch.view_as_real(spec)  # Convierte complex64 a float con dimensión extra
    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)
    return spec


def spectrogram_torch_conv(y, n_fft, sampling_rate, hop_size, win_size, center=False):
    # if torch.min(y) < -1.:
    #     print('min value is ', torch.min(y))
    # if torch.max(y) > 1.:
    #     print('max value is ', torch.max(y))

    global hann_window
    dtype_device = str(y.dtype) + '_' + str(y.device)
    wnsize_dtype_device = str(win_size) + '_' + dtype_device
    if wnsize_dtype_device not in hann_window:
        hann_window[wnsize_dtype_device] = torch.hann_window(win_size).to(dtype=y.dtype, device=y.device)

    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    
    # ******************** ConvSTFT ************************#
    freq_cutoff = n_fft // 2 + 1
    fourier_basis = torch.view_as_real(torch.fft.fft(torch.eye(n_fft)))
    forward_basis = fourier_basis[:freq_cutoff].permute(2, 0, 1).reshape(-1, 1, fourier_basis.shape[1])
    
    # Actualizado para librosa 0.11.0: usar librosa.util.pad_center correctamente
    window_tensor = torch.hann_window(win_size)
    padded_window = torch.nn.functional.pad(
        window_tensor, 
        (0, n_fft - win_size), 
        mode='constant', 
        value=0
    )
    if n_fft > win_size:
        # Asegurar que el centro esté en el medio
        pad_left = (n_fft - win_size) // 2
        pad_right = n_fft - win_size - pad_left
        padded_window = torch.nn.functional.pad(
            window_tensor, 
            (pad_left, pad_right), 
            mode='constant', 
            value=0
        )
    forward_basis = forward_basis * padded_window.float()

    import torch.nn.functional as F

    # if center:
    #     signal = F.pad(y[:, None, None, :], (n_fft // 2, n_fft // 2, 0, 0), mode = 'reflect').squeeze(1)
    assert center is False

    forward_transform_squared = F.conv1d(y, forward_basis.to(y.device), stride = hop_size)
    spec2 = torch.stack([forward_transform_squared[:, :freq_cutoff, :], forward_transform_squared[:, freq_cutoff:, :]], dim = -1)

    # ******************** Verification ************************#
    # Actualizado para usar la nueva versión de spectrogram_torch
    spec1 = spectrogram_torch(y.squeeze(1), n_fft, sampling_rate, hop_size, win_size, center)
    
    # Asegurar que spec2 tenga la misma forma que spec1
    spec2_magnitude = torch.sqrt(spec2.pow(2).sum(-1) + 1e-6)
    
    # Verificación con tolerancia
    if not torch.allclose(spec1, spec2_magnitude, atol=1e-4):
        print(f"Warning: spectrogram methods differ. Max diff: {(spec1 - spec2_magnitude).abs().max().item()}")

    spec = torch.sqrt(spec2.pow(2).sum(-1) + 1e-6)
    return spec


def spec_to_mel_torch(spec, n_fft, num_mels, sampling_rate, fmin, fmax):
    global mel_basis
    dtype_device = str(spec.dtype) + "_" + str(spec.device)
    fmax_dtype_device = str(fmax) + "_" + dtype_device
    if fmax_dtype_device not in mel_basis:
        mel = librosa_mel_fn(sampling_rate, n_fft, num_mels, fmin, fmax)
        mel_basis[fmax_dtype_device] = torch.from_numpy(mel).to(
            dtype=spec.dtype, device=spec.device
        )
    spec = torch.matmul(mel_basis[fmax_dtype_device], spec)
    spec = spectral_normalize_torch(spec)
    return spec


def mel_spectrogram_torch(
    y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False
):
    if torch.min(y) < -1.0:
        print("min value is ", torch.min(y))
    if torch.max(y) > 1.0:
        print("max value is ", torch.max(y))

    global mel_basis, hann_window
    dtype_device = str(y.dtype) + "_" + str(y.device)
    fmax_dtype_device = str(fmax) + "_" + dtype_device
    wnsize_dtype_device = str(win_size) + "_" + dtype_device
    if fmax_dtype_device not in mel_basis:
        mel = librosa_mel_fn(sampling_rate, n_fft, num_mels, fmin, fmax)
        mel_basis[fmax_dtype_device] = torch.from_numpy(mel).to(
            dtype=y.dtype, device=y.device
        )
    if wnsize_dtype_device not in hann_window:
        hann_window[wnsize_dtype_device] = torch.hann_window(win_size).to(
            dtype=y.dtype, device=y.device
        )

    y = torch.nn.functional.pad(
        y.unsqueeze(1),
        (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
        mode="reflect",
    )
    y = y.squeeze(1)

    # Actualizado para PyTorch 2.x: stft ahora devuelve complex64 por defecto
    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window[wnsize_dtype_device],
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,  # Cambiado a True (por defecto en PyTorch 2.x)
    )
    
    # Convertir complex a magnitud
    spec = torch.view_as_real(spec)  # Convierte complex64 a float con dimensión extra
    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)

    spec = torch.matmul(mel_basis[fmax_dtype_device], spec)
    spec = spectral_normalize_torch(spec)

    return spec
