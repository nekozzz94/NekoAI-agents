---
name: rds-replica-lag
description: Diagnose RDS read replica lag - queries, write spikes, checkpoints, WAL replay issues
---

# /rds-replica-lag

Trigger a focused RDS read replica lag investigation.

## When to use

Invoke when RDS replica lag is elevated or a replica-related incident is reported:
- CloudWatch alarm for `ReplicaLag` metric
- UPD Support Tool showing stale data
- Application reports data freshness issues on read replica
- Replica lag > 5 minutes (300 seconds)
- Performance degradation on replica queries

## Instructions

When this skill is invoked:

1. **Ask the user for the RDS instance identifiers** (do not assume or hardcode):
   ```
   What is the RDS replica instance identifier? (e.g., prod-db-read-1)
   What is the primary instance identifier? (e.g., prod-db-primary-1)
   What AWS region? (e.g., eu-west-1)
   ```
   
   Once provided, get the DBI Resource ID for Performance Insights:
   ```bash
   aws rds describe-db-instances \
     --db-instance-identifier <REPLICA_INSTANCE_ID> \
     --query 'DBInstances[0].DbiResourceId' \
     --output text \
     --region <REGION>
   ```
   
   Store these as variables for the rest of the investigation:
   ```bash
   REPLICA_INSTANCE="<user-provided>"
   PRIMARY_INSTANCE="<user-provided>"
   REGION="<user-provided>"
   DBI_RESOURCE_ID="<from describe-db-instances>"
   ```

2. **Recall from memory** — search past incidents and patterns:
   ```bash
   python3 scripts/memory_manager.py search-incidents --keywords "RDS,replica,lag"
   python3 scripts/memory_manager.py get-patterns --symptom "RDS replica lag"
   ```
   Use `first_checks` from any matching pattern as the starting point.

3. **Baseline check** — get current lag and trend:
   ```bash
   # Current lag (last 24 hours, hourly)
   aws cloudwatch get-metric-statistics \
     --namespace AWS/RDS \
     --metric-name ReplicaLag \
     --dimensions Name=DBInstanceIdentifier,Value=$REPLICA_INSTANCE \
     --start-time $(date -u -v-24H +%Y-%m-%dT%H:%M:%SZ) \
     --end-time $(date -u +%Y-%m-%dT%H:%M:%SZ) \
     --period 3600 \
     --statistics Average Maximum \
     --region $REGION \
     --output table
   ```

   **Severity assessment**:
   - Lag < 60 sec: Minor (investigate but not urgent)
   - Lag 60-300 sec (1-5 min): Moderate (investigate actively)
   - Lag > 300 sec (5+ min): **Severe** (P1 incident, immediate action)
   - Lag > 3600 sec (1+ hour): **Critical** (data freshness SLA violated)

4. **Check replica health**:
   ```bash
   # Replica instance status
   aws rds describe-db-instances \
     --db-instance-identifier $REPLICA_INSTANCE \
     --query 'DBInstances[0].[DBInstanceIdentifier,DBInstanceStatus,DBInstanceClass,ReadReplicaSourceDBInstanceIdentifier]' \
     --output table \
     --region $REGION

   # Replica connections
   aws cloudwatch get-metric-statistics \
     --namespace AWS/RDS \
     --metric-name DatabaseConnections \
     --dimensions Name=DBInstanceIdentifier,Value=$REPLICA_INSTANCE \
     --start-time $(date -u -v-6H +%Y-%m-%dT%H:%M:%SZ) \
     --end-time $(date -u +%Y-%m-%dT%H:%M:%SZ) \
     --period 3600 \
     --statistics Average Maximum \
     --region $REGION
   ```

5. **Check Performance Insights** — identify query load:
   ```bash
   # DB Load over last 6 hours
   aws pi get-resource-metrics \
     --service-type RDS \
     --identifier $DBI_RESOURCE_ID \
     --start-time $(date -u -v-6H +%Y-%m-%dT%H:%M:%SZ) \
     --end-time $(date -u +%Y-%m-%dT%H:%M:%SZ) \
     --period-in-seconds 300 \
     --metric-queries '[{"Metric":"db.load.avg"}]' \
     --region $REGION

   # Top SQL queries by load (last 1 hour)
   aws pi describe-dimension-keys \
     --service-type RDS \
     --identifier $DBI_RESOURCE_ID \
     --start-time $(date -u -v-1H +%Y-%m-%dT%H:%M:%SZ) \
     --end-time $(date -u +%Y-%m-%dT%H:%M:%SZ) \
     --metric db.load.avg \
     --period-in-seconds 300 \
     --group-by '{"Group":"db.sql"}' \
     --region $REGION \
     --max-results 10
   ```

   **DB Load interpretation**:
   - Normal: <1 avg active sessions
   - Elevated: 1-5 avg sessions (monitor)
   - High: 5-10 avg sessions (investigate queries)
   - **Critical**: >10 avg sessions (replica overloaded)

6. **Check replica I/O activity**:
   ```bash
   # Read IOPS (last 6 hours)
   aws cloudwatch get-metric-statistics \
     --namespace AWS/RDS \
     --metric-name ReadIOPS \
     --dimensions Name=DBInstanceIdentifier,Value=$REPLICA_INSTANCE \
     --start-time $(date -u -v-6H +%Y-%m-%dT%H:%M:%SZ) \
     --end-time $(date -u +%Y-%m-%dT%H:%M:%SZ) \
     --period 3600 \
     --statistics Average Maximum \
     --region $REGION

   # Write IOPS (checkpoint activity)
   aws cloudwatch get-metric-statistics \
     --namespace AWS/RDS \
     --metric-name WriteIOPS \
     --dimensions Name=DBInstanceIdentifier,Value=$REPLICA_INSTANCE \
     --start-time $(date -u -v-6H +%Y-%m-%dT%H:%M:%SZ) \
     --end-time $(date -u +%Y-%m-%dT%H:%M:%SZ) \
     --period 3600 \
     --statistics Average Maximum \
     --region $REGION
   ```

   **I/O pattern analysis**:
   - Normal idle replica: 0.5-10 read IOPS, 200-400 write IOPS
   - Heavy query load: 1,000-10,000 read IOPS
   - Slow checkpoint: elevated write IOPS (>800)
   - **Thrashing**: >20,000 read IOPS (replica overwhelmed)

7. **Check replica CPU**:
   ```bash
   aws cloudwatch get-metric-statistics \
     --namespace AWS/RDS \
     --metric-name CPUUtilization \
     --dimensions Name=DBInstanceIdentifier,Value=$REPLICA_INSTANCE \
     --start-time $(date -u -v-6H +%Y-%m-%dT%H:%M:%SZ) \
     --end-time $(date -u +%Y-%m-%dT%H:%M:%SZ) \
     --period 3600 \
     --statistics Average Maximum \
     --region $REGION
   ```

   **CPU vs. I/O correlation**:
   - High CPU (>50%) + high I/O → query-driven lag
   - Low CPU (<10%) + high I/O → I/O-bound, slow disk or checkpoint
   - Low CPU + low I/O + high lag → WAL replay blocked (check primary)

8. **Check primary database activity** (to rule out write spike):
   ```bash
   # Primary write IOPS
   aws cloudwatch get-metric-statistics \
     --namespace AWS/RDS \
     --metric-name WriteIOPS \
     --dimensions Name=DBInstanceIdentifier,Value=$PRIMARY_INSTANCE \
     --start-time $(date -u -v-6H +%Y-%m-%dT%H:%M:%SZ) \
     --end-time $(date -u +%Y-%m-%dT%H:%M:%SZ) \
     --period 3600 \
     --statistics Average Maximum \
     --region $REGION

   # Primary CPU
   aws cloudwatch get-metric-statistics \
     --namespace AWS/RDS \
     --metric-name CPUUtilization \
     --dimensions Name=DBInstanceIdentifier,Value=$PRIMARY_INSTANCE \
     --start-time $(date -u -v-6H +%Y-%m-%dT%H:%M:%SZ) \
     --end-time $(date -u +%Y-%m-%dT%H:%M:%SZ) \
     --period 3600 \
     --statistics Average Maximum \
     --region $REGION
   ```

   **Primary write patterns**:
   - Normal: 400-600 IOPS avg
   - Elevated: 1,000-2,000 IOPS (monitor)
   - Heavy: 2,000-5,000 IOPS (may overwhelm replica)

9. **Check PostgreSQL logs on replica** for evidence:
   ```bash
   # List recent log files
   aws rds describe-db-log-files \
     --db-instance-identifier $REPLICA_INSTANCE \
     --region $REGION \
     --query 'DescribeDBLogFiles | sort_by(@, &LastWritten) | [-5:].[LogFileName,LastWritten,Size]' \
     --output table

   # Download latest log and check for issues
   aws rds download-db-log-file-portion \
     --db-instance-identifier $REPLICA_INSTANCE \
     --log-file-name error/postgresql.log.$(date -u +%Y-%m-%d-%H) \
     --region $REGION \
     --output text | tail -100

   # Search for specific patterns
   aws rds download-db-log-file-portion \
     --db-instance-identifier $REPLICA_INSTANCE \
     --log-file-name error/postgresql.log.$(date -u +%Y-%m-%d-%H) \
     --region $REGION \
     --output text | grep -E "restartpoint|checkpoint|conflict|cancel|duration.*ms"
   ```

   **Log evidence to look for**:
   - `restartpoint complete: ... write=XXX.XXX s` → slow checkpoint (>60s indicates I/O bottleneck)
   - `duration: XXXXX.XXX ms` → long-running queries (>60,000ms = 1 min)
   - `conflict with recovery` → VACUUM conflicts (check if `hot_standby_feedback` is off)
   - `canceling statement due to conflict` → query canceled by WAL replay
   - `Connection timed out` / `connection to client lost` → application timeouts

10. **Diagnose root cause** based on findings:

    **Scenario A: Long-running queries on replica (most common)**
    - Evidence: High DB load (>5), low replica CPU, high read IOPS, long query durations in logs
    - Cause: Expensive queries holding snapshots → block WAL replay (if `hot_standby_feedback = on`)
    - Common query patterns: Full-text search over large date ranges, missing indexes, unoptimized joins
    - Remediation: Cancel queries, optimize, add statement timeout

    **Scenario B: Primary write spike**
    - Evidence: High primary write IOPS (>2,000), elevated primary CPU, replica lag correlates with primary load
    - Cause: Heavy write activity on primary → WAL generation faster than replica can apply
    - Remediation: Wait for write spike to subside, investigate slow restartpoint on replica

    **Scenario C: Slow replica checkpoints (restartpoint)**
    - Evidence: Logs show `restartpoint complete: write=XXXs` (>60s), elevated replica write IOPS
    - Cause: Replica disk I/O bottleneck during checkpoint → pauses WAL replay
    - Remediation: Tune `checkpoint_completion_target`, consider faster storage or larger instance

    **Scenario D: VACUUM conflicts (if `hot_standby_feedback = off`)**
    - Evidence: Logs show `canceling statement due to conflict with recovery`, query cancellations
    - Cause: Primary VACUUM removes rows that replica queries still need
    - Remediation: Enable `hot_standby_feedback = on`, increase `max_standby_streaming_delay`

11. **Report findings** to the user:
    - Current lag (seconds, hours)
    - Root cause category (A/B/C/D)
    - Evidence (metrics + log snippets)
    - Remediation options (immediate + long-term)

12. **Provide remediation options** based on root cause:

    ### For Scenario A (Long-running queries):

    **Immediate** (requires user confirmation):
    
    **IMPORTANT**: Do NOT execute psql commands. Instead, provide the user with:
    1. Connection string template (user must provide credentials)
    2. SQL queries to identify and cancel long-running queries
    3. Explicit warning about impact
    
    ```bash
    # User must obtain credentials separately (DO NOT print credentials)
    # Example using AWS Secrets Manager (user provides secret name):
    # DB_PASSWORD=$(aws secretsmanager get-secret-value --secret-id <SECRET_NAME> --query SecretString --output text --region $REGION)
    
    echo "To cancel long-running queries, connect to the replica and run:"
    echo ""
    echo "psql \"postgresql://<USERNAME>:<PASSWORD>@<REPLICA_ENDPOINT>:5432/<DATABASE>\""
    echo ""
    echo "Then execute these queries:"
    echo ""
    echo "-- Identify long-running queries"
    cat << 'SQL'
    SELECT pid, usename, NOW() - query_start AS duration, left(query, 80)
    FROM pg_stat_activity 
    WHERE state != 'idle' 
      AND pid != pg_backend_pid()
      AND NOW() - query_start > interval '5 minutes'
    ORDER BY query_start;
    SQL
    echo ""
    echo "-- Cancel them (REQUIRES USER CONFIRMATION)"
    cat << 'SQL'
    SELECT pg_cancel_backend(pid) FROM pg_stat_activity 
    WHERE state != 'idle' 
      AND NOW() - query_start > interval '5 minutes'
      AND pid != pg_backend_pid();
    SQL
    ```
    
    **Never print actual connection strings with credentials in the output.**

    **Long-term prevention**:
    - Add `statement_timeout` to replica parameter group (e.g., 60 seconds)
    - Optimize application queries (limit date ranges, add indexes)
    - Implement connection pooling (PgBouncer) with max connections per user
    - Add query performance monitoring

    ### For Scenario B (Primary write spike):

    **Immediate**:
    - Monitor and wait — lag will recover once write spike subsides
    - Check if spike is expected (scheduled VACUUM, batch job, migration)

    **Long-term prevention**:
    - Schedule heavy write operations during low-traffic windows
    - Consider read replica instance upgrade (larger instance class)
    - Tune `checkpoint_completion_target` on replica

    ### For Scenario C (Slow checkpoints):

    **Immediate**:
    - Monitor and wait — checkpoints are necessary, will complete eventually

    **Long-term prevention**:
    - Tune `checkpoint_completion_target` (increase to spread I/O over longer period)
    - Consider replica instance upgrade (db.m6idn.8xlarge → db.m6idn.12xlarge)
    - Evaluate storage upgrade (current: io2, already high performance)

    ### For Scenario D (VACUUM conflicts):

    **Immediate**:
    - Enable `hot_standby_feedback = on` in parameter group
    - Increase `max_standby_streaming_delay` (e.g., 300000 ms = 5 min)

    **Long-term prevention**:
    - Monitor for table bloat on primary (trade-off with `hot_standby_feedback = on`)
    - Optimize long-running queries on replica (reduce conflict window)

13. **Write incident report** to `incidents/YYYY-MM-DD-<slug>.md`:
    - Timeline with lag progression
    - Root cause analysis
    - Evidence (metrics, logs, Performance Insights top queries)
    - Remediation applied
    - Long-term prevention recommendations
    
    **Security note for incident reports**:
    - Reports are for internal use and may contain sensitive identifiers
    - Before sharing externally: redact instance IDs, endpoints, DBI Resource IDs, database names
    - SQL queries in reports may contain application-specific details — review before sharing
    - Add a header: "INTERNAL USE ONLY - CONTAINS SENSITIVE INFRASTRUCTURE IDENTIFIERS"

14. **Update memory** after resolution:
    ```bash
    # Record incident (automatic via Stop hook, but can be done manually)
    python3 scripts/memory_manager.py add-incident --stdin << 'EOF'
    {
      "date": "YYYY-MM-DD",
      "title": "RDS replica lag - <root cause>",
      "platform": "aws",
      "service": "RDS PostgreSQL",
      "symptoms": ["replica lag", "stale data", ...],
      "root_cause": "<concise root cause>",
      "resolution": "<what fixed it>",
      "lessons": ["<lesson 1>", "<lesson 2>"]
    }
    EOF

    # Update or add semantic pattern
    python3 scripts/memory_manager.py add-pattern --category rds-replica-lag-queries --stdin << 'EOF'
    {
      "platform": "aws",
      "symptoms": [
        "RDS ReplicaLag metric > 300 seconds",
        "UPD Support Tool showing stale data",
        "Performance Insights DB load > 5",
        "Long-running full-text search queries"
      ],
      "first_checks": [
        "aws cloudwatch get-metric-statistics --metric-name ReplicaLag",
        "aws pi describe-dimension-keys --metric db.load.avg",
        "Check PostgreSQL logs for long query durations"
      ],
      "common_causes": [
        "Full-text search over large date ranges (4+ months)",
        "Missing indexes on frequently queried columns",
        "hot_standby_feedback = on + long queries blocking WAL replay"
      ],
      "resolution": "Cancel long-running queries, add statement_timeout, optimize queries",
      "seen_count": 2
    }
    EOF
    ```

## Configuration Notes

**Common RDS parameter settings to check**:
- `hot_standby_feedback` — if `on`, prevents VACUUM conflicts but allows long queries to block WAL replay
- `max_standby_streaming_delay` — timeout before canceling conflicting queries (e.g., 300000 ms = 5 minutes)
- `statement_timeout` — max query execution time (recommended: 60000 ms = 60 sec for replicas)

**Trade-off**: With `hot_standby_feedback = on`, long-running replica queries can cause lag. Monitor for this pattern and enforce query timeouts.

## Security Notes

**Credential Handling**:
- NEVER print database passwords, connection strings with credentials, or sensitive endpoints in logs/output
- Retrieve credentials from AWS Secrets Manager when needed (user provides secret name)
- Redact database usernames, endpoints, and identifiers in incident reports shared externally
- Use environment variables or user prompts for instance identifiers (do not hardcode)

**When connecting to database**:
- Provide SQL queries as templates only
- User must obtain credentials separately and execute queries themselves
- Do not attempt to automate psql connections (requires credential handling)

## Related Skills

- `/read-jira` — Fetch Jira ticket details if incident is tracked
- `/incident-runbook` — Generate structured runbook after investigation
- AWS Investigator agent — For broader AWS infrastructure investigation

## Monitoring & Alerting

**CloudWatch alarms to create** (if not already configured):

```bash
# Replica lag > 5 minutes
aws cloudwatch put-metric-alarm \
  --alarm-name rds-replica-lag-${REPLICA_INSTANCE}-critical \
  --metric-name ReplicaLag \
  --namespace AWS/RDS \
  --statistic Average \
  --period 300 \
  --threshold 300 \
  --comparison-operator GreaterThanThreshold \
  --evaluation-periods 2 \
  --dimensions Name=DBInstanceIdentifier,Value=$REPLICA_INSTANCE \
  --region $REGION

# DB Load > 5 active sessions
aws cloudwatch put-metric-alarm \
  --alarm-name rds-replica-overload-${REPLICA_INSTANCE} \
  --metric-name DatabaseConnections \
  --namespace AWS/RDS \
  --statistic Average \
  --period 300 \
  --threshold 5 \
  --comparison-operator GreaterThanThreshold \
  --evaluation-periods 1 \
  --dimensions Name=DBInstanceIdentifier,Value=$REPLICA_INSTANCE \
  --region $REGION
```

## Success Criteria

Investigation complete when:
- ✅ Root cause identified with evidence
- ✅ Current lag status documented (stable, recovering, or worsening)
- ✅ Remediation plan provided (immediate + long-term)
- ✅ Incident report written to `incidents/`
- ✅ Memory updated with findings (episodic + semantic)
- ✅ User informed of next steps
