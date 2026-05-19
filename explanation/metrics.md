# Metrics

This document defines the core metrics used in **StreamingEval**.  
StreamingEval evaluates video-language models under realistic streaming constraints, jointly considering task accuracy, processing speed, response latency, and memory usage.

---

## 1. Accuracy

**Accuracy** measures whether the model produces the correct answer for a given streaming video question.

For a benchmark with $N$ questions:

$$
\text{Accuracy} = \frac{1}{N}\sum_{i=1}^{N}\mathbb{I}(\hat{a}_i = a_i)
$$

where:

- $\hat{a}_i$ is the model prediction for the $i$-th question.
- $a_i$ is the ground-truth answer.
- $\mathbb{I}(\cdot)$ is the indicator function.

For multiple-choice tasks, the prediction is correct if the selected option matches the ground-truth option.


---

## 2. Latency

**Latency** measures how quickly the model can respond after receiving a question during streaming evaluation.

We report two latency metrics.

### 2.1 Time to First Token, TTFT

$$
\text{TTFT} = t_{\text{first\_token}} - t_{\text{query}}
$$

where:

- $t_{\text{query}}$ is the timestamp when the question is issued.
- $t_{\text{first\_token}}$ is the timestamp when the first output token is generated.

TTFT reflects the responsiveness of the system.

### 2.2 End-to-End Response Latency

$$
\text{Latency}_{\text{e2e}} = t_{\text{last\_token}} - t_{\text{query}}
$$

where:

- $t_{\text{last\_token}}$ is the timestamp when the model finishes generating the answer.

End-to-end latency reflects the total waiting time perceived by the user.

---

## 3. MaxFPS

**MaxFPS** measures the maximum video frame rate that the visual encoder can support under the streaming protocol.

Let $T_{\text{frame}}$ denote the average visual encoding time for one frame:

$$
T_{\text{frame}} = \frac{T_{\text{enc}}}{F}
$$

where:

- $F$ is the number of processed video frames.
- $T_{\text{enc}}$ is the total time required to encode these frames.
- $T_{\text{frame}}$ is the average time required to convert one frame into visual tokens or embeddings.

Then MaxFPS is computed as:

$$
\text{MaxFPS} = \frac{1}{T_{\text{frame}}}
$$


---

## 4. Memory

**Memory** measures the resource cost required to maintain streaming visual context.

We report memory usage from the following perspectives.

### 4.1 Memory Bank Size

$$
\text{MemoryBank} = |\mathcal{M}|
$$

where $\mathcal{M}$ denotes the stored historical visual memory.

Depending on the model, this can be measured as:

- number of stored visual tokens
- number of stored frames
- number of cached features
- memory bank size in MB or GB


---



## 5. StreamingScore

**StreamingScore** measures the overall streaming capability of a model by jointly considering accuracy, visual encoding throughput, response latency, and resource consumption.

Unlike accuracy-only evaluation, StreamingScore is designed to reflect the trade-off among four key factors:

- **Accuracy**: whether the model answers correctly.
- **MaxFPS**: how fast the visual encoder can process incoming frames.
- **TTFT**: how quickly the model starts responding after receiving a query.
- **Memory / model cost**: how much resource the model consumes during streaming inference.

We define StreamingScore as:

$$
\text{StreamingScore}(\mathbf{w})
=
\frac{
\text{MaxFPS}^{w_f} \cdot \text{Acc}^{w_a}
}{
\text{TTFT}^{w_t} \cdot M^{w_r}
}
$$

where the resource term $M$ is defined as:

$$
M = \text{Mem} \cdot \ln(\text{Params})
$$

Here:

- $\text{Acc}$ denotes the QA accuracy.
- $\text{MaxFPS}$ denotes the maximum frame rate supported by the visual encoder.
- $\text{TTFT}$ denotes the time to first generated token.
- $\text{Mem}$ denotes the memory consumption during streaming inference.
- $\text{Params}$ denotes the number of model parameters.
- $M$ combines runtime memory usage and model scale into a single resource-cost term.

The weights satisfy:

$$
w_f, w_a, w_t, w_r \geq 0,
\qquad
w_f + w_a + w_t + w_r = 1
$$

where:

- $w_f$ controls the importance of visual encoding throughput.
- $w_a$ controls the importance of answer accuracy.
- $w_t$ controls the penalty on response latency.
- $w_r$ controls the penalty on resource consumption.

A higher StreamingScore indicates that a model achieves better accuracy and higher visual encoding throughput while maintaining lower response latency and lower resource usage.




---

## 6. Reported Metrics

Each evaluated model should report the following metrics:

| Metric | Meaning | Higher is Better |
|---|---|---|
| Accuracy | QA correctness | Yes |
| MaxFPS | Maximum sustainable processing frame rate | Yes |
| TTFT | Time to first generated token | No |
| E2E Latency | Total response generation time | No |
| Memory Bank Size | Stored historical visual context | Depends |
| StreamingScore | Overall streaming deployability | Yes |


