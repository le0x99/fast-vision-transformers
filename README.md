# Fast Vision Transformers (F-ViT)
Improved implementation of vision transformers for fast training and inference.
The architecture is optimized for efficiency and differs greatly from the architecture proposed in https://arxiv.org/abs/2010.11929.

### Major differences

- Pooling latents instead of a learnable [CLS] token.
- No projection of the patch encodings.
- MLP head is optional.
- granularity of the patch grid dimension ($G$) can be specified explicitly.
- PatchDropout can be applied.
- On the fly encoding can be applied.


### Notes
- If ```AUTO_ENCODE=True``` images are encoded on the fly, expecting batches of $(B, C, H, W)$. In this case the embedding size depends on $G$. Specifically the embedding size will be $E = C * (\frac{H}{G} \times \frac{W}{G})$.
- If ```AUTO_ENCODE=False``` encoded images in the form of $(B, T, E)$, i.e. a patch sequence is expected and ```EMB_DIM=E``` has to be passed explicitly.
- PatchDropout is not deactivated when calling ```model.eval()```. Use ```mode.eval_mode()``` and  ```mode.training_mode()``` instead.


### Usage

```python
from fast_vit import FastVisionTransformer
```

```python
model = FastVisionTransformer(
    # Image spatial dim
    IMAGE_N   = 32,
    # number of classes
    N_CLASSES = 10,
    # patch grid dimension (GxG)
    G         = 4,
    # PatchDropout rate
    PDO       = 0.1,
    # Dropout rate
    DO        = 0.1,
    # number of self attention heads
    N_HEADS   = 8,
    # use mlp or linear head
    MLP_HEAD  = True,
    # hidden dim multiplier of the heads
    MLP_MULT  = 4,
    # hidden dim multiplier of the transformer mlp
    FF_MULT   = 2,
    # total number of transformer blocks
    N_BLOCKS  = 4,
    # indicate if patches are encoded on the fly
    AUTO_ENCODE = True,
    # Only important if the images are already encoded
    EMB_DIM = None)
```

### AMP Training 

```python
from tools import Trainer, load_cifar
```

```python
trainloader, testloader = load_cifar(batch_size)
trainer = Trainer(log_dir="./fvit/test_run")
```

```python
n_epochs   = 40
batch_size = 512
lr         = 0.0006
```

```python
trainer.train(model, trainloader, testloader, (n_epochs, batch_size, lr))

acc1, nll = trainer.test_model(model, testloader)
```

