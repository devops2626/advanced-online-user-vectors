# Phase 3: Observability & Latency Monitoring for Real-Time Feeds

For a short-form video feed where the entire pipeline must execute within a strict 30–50 ms window, flying blind is not an option. A minor blockage in Flink or a slow Redis call will instantly spike p99 latency and ruin the user experience.

---

## 1. The Critical Latency Budget Breakdown

To maintain smooth 60 fps scrolling, our end-to-end recommendation loop is budgeted as follows:

| Stage | Target Latency |
|-------|----------------|
| Retrieval (ANN / Milvus) | ≤ 10 ms |
| Feature Store / Redis Merge | ≤ 5 ms |
| Ranking Model Inference | ≤ 20 ms |
| Greedy MMR Post-Processor | ≤ 5 ms |
| **Total Target Budget** | **≤ 40 ms** (leaving ~10 ms safety margin for network transport) |

---

## 2. Key Metrics to Track

### Streaming & Vector Pipeline (Flink & Kafka)

- `flink_taskmanager_job_task_operator_numRecordsInPerSecond` — Ingress action rate from Kafka.
- `flink_taskmanager_job_task_operator_currentInputWatermark` — Measures event-time lag against wall-clock time. A widening gap indicates backpressure or slow state serialization.
- `flink_taskmanager_job_task_operator_backPressuredTimeMsPerSecond` — Indicates if the Flink operator thread is choking on downstream sinks (Redis).

### Serving & Inference Layer (Redis, Ranking, MMR)

- `redis_command_duration_seconds_bucket{cmd="mget"}` — Tracks p99 latency for fetching user vectors and sliding window metadata.
- `ranking_inference_duration_milliseconds` — p95/p99 latency of the ML model scoring batch candidates.
- `mmr_processing_duration_milliseconds` — Time spent running the diversity filter on ranked items.

---

## 3. Prometheus Configuration for Flink

To expose Flink metrics directly to Prometheus, configure the built-in Prometheus reporter in `flink-conf.yaml`:

```yaml
metrics.reporter.prometheus.factory.class: org.apache.flink.metrics.prometheus.PrometheusReporterFactory
metrics.reporter.prometheus.port: 9249
metrics.system-resource: true
metrics.system-resource-probing-interval: 5000
```

Add the scrape targets to your `prometheus.yml`:

```yaml
scrape_configs:
  - job_name: 'flink-cluster'
    static_configs:
      - targets: ['flink-taskmanager:9249', 'flink-jobmanager:9249']
  - job_name: 'feed-serving-api'
    static_configs:
      - targets: ['feed-api-service:8080']
```

---

## 4. Grafana Dashboard Layout & Queries

Create a unified dashboard titled **"Short-Form Video Feed - Real-Time Health"** with the following panels:

**Panel 1: End-to-End Latency Breakdown (p95 / p99)**  
Query:
```promql
histogram_quantile(0.99, rate(feed_stage_duration_seconds_bucket[5m]))
```

**Panel 2: Flink Backpressure Ratio**  
Query:
```promql
flink_taskmanager_job_task_operator_backPressuredTimeMsPerSecond / 1000
```

**Panel 3: Redis Merge p99 Latency**  
Query:
```promql
histogram_quantile(0.99, rate(redis_command_duration_seconds_bucket{cmd="pipeline"}[5m]))
```

---

## 5. Critical Alerting Rules

Configure Alertmanager (`alert.rules.yml`) to fire pages when latency creeps toward the hard ceiling:

```yaml
groups:
  - name: feed_latency_alerts
    rules:
      - alert: RankingLatencyBudgetExceeded
        expr: histogram_quantile(0.99, rate(ranking_inference_duration_milliseconds_bucket[5m])) > 25
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "Ranking model inference p99 exceeds 25ms budget"
          description: "Current p99 latency is {{ $value }}ms, threatening the end-to-end 50ms constraint."

      - alert: FlinkWatermarkLagHigh
        expr: time() - (flink_taskmanager_job_task_operator_currentInputWatermark / 1000) > 15
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Flink processing lag is growing"
          description: "Real-time stream watermarks are lagging behind wall-clock by more than 15 seconds."
```

---

*This Phase-3 guide makes the real-time short-form video pipeline fully observable and alertable within the strict latency budget.*
