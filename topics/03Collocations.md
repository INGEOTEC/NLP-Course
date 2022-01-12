---
layout: default
title: Collocations
nav_order: 3
---

# Collocations
{: .fs-10 .no_toc }

## Contents
{: .no_toc .text-delta }

1. TOC
{:toc}

## Libraries used
{: .no_toc .text-delta }
```python
from text_models import Vocabulary
```

## Installing external libraries
{: .no_toc .text-delta }

```bash
pip install microtc
pip install evomsa
pip install text_models
```

---

# Collocations

A collocation is an expression with the characteristic that its meaning cannot be inferred by simply adding the definition of each of its words, e.g., kick the bucket.


|    | the     | to      | of      | in      | and     | 
|----|---------|---------|---------|---------|---------|    
|the |       0 |  453796 |  435030 |  323450 |  317663 |
|to  |  453796 |       0 |  170941 |  161891 |  228785 |
|of  |  435030 |  170941 |       0 |  122502 |  130662 |
|in  |  323450 |  161891 |  122502 |       0 |  125997 |
|and |  317663 |  228785 |  130662 |  125997 |       0 |