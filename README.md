# Reinventing-TorchGrad

PyTorch の 自動微分アルゴリズムの再現実装

```mermaid
graph LR
    -a --> _3(("exp")) --> v1["exp(-a)"]

           v4["1"] --> _4(("+"))
    v1["exp(-a)"] --> _4(("+"))  --> v3["1+exp(-a)"]

             v5["1"] --> _5(("/"))
    v3["1+exp(-a)"] --> _5(("/")) --> v6["1 / {1+exp(-a)}"]
```

```python
import math
from autograd import Tensor, exp

a = Tensor(3)
s = 1 / (1 + exp(-a))
s.backward()

assert math.isclose(s.value, 1 / (1 + math.exp(-3)))  # OK
assert math.isclose(a.grad, s.value * (1 - s.value))  # OK
```

## 環境構築

MacOS, Linux

```
python -m venv venv
source venv/bin/activate
pip install -e .
```

Windows

```
python -m venv venv
venv\Scripts\activate
pip install -e .
```
