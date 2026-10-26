# Diffusion Policy modules

Source: https://github.com/real-stanford/diffusion_policy at commit 5ba07ac6661db573af695b419a7947ecb704690f (MIT).

Copied conditional_unet1d.py, conv1d_components.py, positional_embedding.py after inspection. Only two imports changed to relative imports. Original local-conditioning compatibility quirk is retained; our integration uses global conditioning only. Dependencies: torch and einops. No upstream checkpoint assumed compatible with our RGB-D/GRU conditioning.

Upstream SHA256:
- dp_unet.py: 8a95014f8aa52ade086f967c9018a66bf0c68f2133969c8f7a8522da01b472ac
- dp_conv.py: ea2aa10e297cf253bc1a1919ba47fe6ae32049b5eaaab292ed9cc145d8d2adc0
- dp_pos.py: 2266c75ecc9c9ac8ca126839d39a7bafbdd7af8800d6de903f66815e64cc366c

Adapted-file SHA256 after the relative-import adaptation and whitespace cleanup:
- conditional_unet1d.py: 7cef53c66b731fbc4fdd796ad6cc313b4ff9556d2e3ebe26d88462a01a07da2c
