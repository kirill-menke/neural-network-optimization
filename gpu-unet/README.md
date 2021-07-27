# GPU/Cuda-Code zum HPC-Projekt

## ToDo's

- Unet in Pytorch oder Tensorflow nachbauen zum Performance vergleich
- Profling mit Nvidia Nsight, Roofline für einzelne Kernels aufstellen
- *BatchNorm*-Layer implementieren
- *Adam*-Optimizer implementieren
- *MaxPool*, *Upsample*: `pool_size` als Template-Parameter der Kernel für loop unrolling
- *MaxPool*, *Upsample*: Höhere Threads pro Block um Speicherbandbreite besser auszunutzen? (Hardware-Abhaengig, Roofline abwarten)
- *Conv*: `backward()`: Unabhängige Berechungen auf unterschiedlichen Streams parallel
- *Conv*: `backward()`, Gradienten-Berechnung: Parallele Reduktion verbessern
- *Conv* und *ReLU* in ein gemeinsames Layer
- Vorschlag: *GoDown* und *GoUp*-Layer:
  - *GoDown*:
    - *forward*: wie `MaxPool::foward`
    - *backward*: `MaxPool::backward` und Addition von Fehler (aber nur der betroffenen Channels) in einem
  - *GoUp*:
    - *forward*: wie `Upsample::forward` und `concat(upsampled, skipConnections)` in einem
    - *backward*: wie `Upsample::backward`, aber nur der betroffenen Channels
  - Vorteile:
    - Man braucht gar kein `split` weil die Layer wissen welche Channels sie brauchen
    - `concat` und `Upsample::forward` in einem Kernel
    - Fehler-Addition und `MaxPool::backward` in einem Kernel

## Fragen an Prof. Köstler zu GPU-Unet aus letztem Treffen

- Unser Netz ist numerisch instabil:
  - Mit `double` rechnen? (Schlecht für Performance?)
  - *batch normalization* machen?
  - Bug in unserem Code möglich? (Aber dann würde ja wahrscheinlich alles schiefgehen?)
  - Optimizer implementieren der *learning rate* dynamisch macht bzw. irgendwas mit Momentum?
  - __Drauf scheißen weil wir es schnell wollen, nicht "gut"__
- Optimierungen (Vorschläge):
  - Bei `Conv`: Die `weights`/`bias`-Werte werden wiederverwendet, also in *shared memory* laden?
  - Bei `Conv`: Threads verwenden überlappende Pixel, also auch diese Daten in *shared memory* laden?
  - Bei `Conv`, `MaxPool`, `Upsample` die Kernel/Pool-Größe als Template-Parameter damit Schleifen ausgerollt werden können?
  - Aus *Nsight*: "verify the memory access patterns are optimal for the target architecture" - Tipps? Wie finden wir Infos zu unseren GPUs?
  - Aktuelle Kernel-Start-Konfiguration: `width x height x channels` - geht das besser? Lohnt es sich daran zu drehen? Wie Aufteilung in Blöcke/Grid?
  - Row-Major vs Column-Major im Speicher? Speicher mit sowas wie [cudaMalloc3D](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1g188300e599ded65c925e79eab2a57347) anlegen?
  - Unabhängige Berechnungen (z.B. im Conv-Layer Gradient nach Input und Gradient nach Gewichten) parallel bzw. in zwei verschiedenen Streams?
  - 1x1-`Conv` getrennt implementieren?
  - Block-Größen dynamisch anpassen wenn `width`/`height` sinkt aber `channels` steigt?
  - Sonst...?
- *Nvidia Nsights*:
  - Wird irgendwo angezeigt wie nah an der Peak-Rechenleistung man ist?
  - Bei `maxpool_forward`: Daran vielleicht erklären: Wie die Block/Grid-Aufteilung verbessern?
  - Bei `upsample_forward`: Hoher *"Stall Short Scoreboard"*-Wert... wieso? Was kann man da optimieren?
- Sonstige Fragen
  - `Conv::backward`: Gradient w.r.t. weights: Ist blöd weil Kernel gestartet pro Filter-Pixel, nicht Bild-Pixel, geht das besser?
  - Ists OK auf kleineren Netzen zu testen/optimieren, weil flacher und daher schneller?
  - Wie sollen wir am besten Zeit messen? `cudaDeviceSynchronize()` nach Kernel-Start? Events?
  - *memory coalescing*: Bei 3D-Grid von Threads, `x`/`y`/`z`, was als "innersten" Index verwenden?


