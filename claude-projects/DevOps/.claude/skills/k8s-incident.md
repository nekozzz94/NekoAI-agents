# /k8s-incident

Trigger a focused Kubernetes workload incident investigation.

## When to use

Invoke when symptoms point to a Kubernetes-layer problem:
- Pods in CrashLoopBackOff, OOMKilled, ImagePullBackOff, or Pending
- Deployments stuck in rollout or returning to old version
- HPA not scaling (up or down)
- Services returning 5xx / connection refused
- Nodes NotReady or under resource pressure
- PVCs stuck in Pending or volumes not mounting

## Instructions

When this skill is invoked:

1. If the user hasn't described symptoms, ask:
   - What namespace and workload name?
   - What is the observable symptom (error message, pod status, alert name)?
   - When did it start?

2. Confirm the kubectl context is correct:
   ```bash
   kubectl config current-context
   ```

3. Run the K8s Investigator agent with the symptom context.

4. Work through the checklist:
   - Node health and capacity (`kubectl get nodes`, `kubectl top nodes`)
   - Pod status in the affected namespace
   - Events (namespace + cluster-wide warnings)
   - For CrashLoop/OOM: `describe pod` + `logs --previous`
   - For Pending: describe pod events for scheduling failure reason
   - For service issues: endpoints, selectors, readiness probes
   - HPA status and metrics-server availability

5. Present findings with:
   - Exact kubectl command as source
   - Raw log/event snippet as evidence
   - Confidence level

6. Propose remediation — flag which steps require user confirmation.

## Key commands quick reference

```bash
# Cluster sanity
kubectl config current-context
kubectl get nodes -o wide
kubectl top nodes

# Pod status
kubectl get pods -n $NS -o wide --sort-by='.status.startTime'
kubectl get pods -A --field-selector=status.phase!=Running

# Events
kubectl get events -n $NS --sort-by='.lastTimestamp' | tail -30
kubectl get events -A --field-selector=type=Warning --sort-by='.lastTimestamp'

# Failing pod deep-dive
kubectl describe pod $POD -n $NS
kubectl logs $POD -n $NS --previous --tail=100

# Deployments
kubectl rollout status deployment/$DEPLOY -n $NS
kubectl describe deployment $DEPLOY -n $NS

# HPA
kubectl get hpa -n $NS
kubectl describe hpa $HPA -n $NS

# Services
kubectl get endpoints $SVC -n $NS
kubectl describe ingress -n $NS
```
