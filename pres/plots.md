Roh-Daten und Infos zu Mini-Benchmarks die dann später in die Abschlussvortrag-Folien können.

## 1. Merged Layer

### Test-Setup:

- Code: Commit *4fd8e93a4ef07819b39dc851d42e39ef1ac91965*
- Iterationen insg.: 250
- Batch-Size: 3,
- Image-Size: 512
- Layer
  1. Conv (1 zu 16 channels, 3x3), ReLU
  2. Conv (16 zu 32 channels, 3x3), ReLU
  3. Conv (32 zu 2 channels, 1x1), SoftMax
- Rechner: *cip4b2.cip.cs.fau.de* (`NVIDIA Corporation TU104 [GeForce RTX 2080]`)

### Ergebnisse Roh:

- Mit Layer-Merge:
  - `conv. + relu forward:     2230.982ms`
  - `conv. + relu backward:    9375.915ms`
  - `conv. + softmax forward:  83.213ms`
  - `conv. + softmax backward: 235.502ms`
- Ohne Layer-Merge:
  - `conv. + relu forward:     2633.385ms`
  - `conv. + relu backward:    11288.013ms`
  - `conv. + softmax forward:  181.877ms`
  - `conv. + softmax backward: 373.438ms`
- Ja, das sind bei SoftMax sehr kurze Zeiten, aber die Ergebnisse sind reproduzierbar/konsistent
- Gemessen via `cudaEventRecord`, siehe Funktion `bench_conv`

### Ergebnisse wie ich sie präsentieren würde:

- Performance in *Iterationen pro Sekunde*
- Für Folien/Plot ev lieber nur die Verbesserung in `%` angeben?
- Mit Layer-Merge:
  - Conv+ReLU:
    - forward: 224 IpS
    - backward: 54 IpS
  - Conv+SoftMax:
    - forward: 3012 IpS
    - backward: 1063 IpS
- Ohne Layer-Merge:
  - Conv+ReLU:
    - forward: 188 IpS
    - backward: 44 IpS
  - Conv+SoftMax:
    - forward: 1381 IpS
    - backward: 670 IpS

## 2. Parallele Reduktion bei `gradient_weights` im Conv-Layer

- Code: Commit *4fd8e93a4ef07819b39dc851d42e39ef1ac91965*
- Iterationen insg.: 500
- Batch-Size: 1,
- Image-Size: 512
- Layer (Alles ohne Layer-Merge)
  1. Conv (1 zu 8 channels, 3x3), ReLU
  2. Conv (8 zu 8 channels, 3x3), ReLU
  3. Conv (8 zu 2 channels, 1x1), SoftMax
- Rechner: *cip4b2.cip.cs.fau.de* (`NVIDIA Corporation TU104 [GeForce RTX 2080]`)

### Ergebnisse Roh:

- Mit parallele Reduktion:
  - `conv. + relu backward:    7263.922ms`
  - `conv. + softmax backward: 215.929ms`
- Ohne parallele Reduktion:
  - `conv. + relu backward:    17243.863ms`
  - `conv. + softmax backward: 6392.295ms`

### Ergebnisse wie ich sie präsentieren würde:

- Performance in *Iterationen pro Sekunde*
- Für Folien/Plot ev lieber nur die Verbesserung in `%` angeben?
- Mit paralleler Reduktion:
  - Conv+ReLU: 137 IpS
  - Conv+SoftMax: 2325 IpS
- Ohne parallele Reduktion:
  - Conv+ReLU: 78 IpS
  - Conv+SoftMax: 57 IpS

