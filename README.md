# Fast Vision Transformers (F-ViT)
Lightweight implementation of vision transformers for fast training and inference.
The architecture is optimized for efficiency and differs greatly from the architecture proposed in https://arxiv.org/abs/2010.11929.

### Major differences

- Pooling latents instead of a learnable [CLS] token.
- Projection of the patch embeddings is optional.
- MLP head is optional.
- granularity of the patch grid dimension can be specified.


### Usage

pass
