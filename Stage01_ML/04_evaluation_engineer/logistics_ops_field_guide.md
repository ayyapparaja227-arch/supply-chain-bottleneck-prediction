# Logistics Operations Field Guide (Stage 01)
**Autonomous Logistics Intelligence Engine — Operational Reference Sheet**

### 1. Bottleneck Triaging Thresholds
- **ON_TIME**: Dwell < 22 hrs, Capacity < 75%, Customs < 4 hrs. Action: Standard multimodal dispatch.
- **AT_RISK**: Dwell 22–32 hrs, Capacity 75–88%, Customs 4–8 hrs. Action: Buffer drayage appointment (+12h).
- **CRITICAL DELAY**: Dwell > 32 hrs, Capacity > 88%, Labor Deficit > 0.05. Action: Trigger emergency rail/port diversion.

### 2. False Alarm vs Emergency Reroute Trade-offs
- False Alarm Cost: +,200/TEU in unnecessary feeder & demurrage penalties.
- Missed Bottleneck Cost: ~,000/hour in downstream manufacturing plant downtime.
- Confidence Threshold: Only trigger emergency diversion if P(CRITICAL) >= 0.80.
