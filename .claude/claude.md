# Kubernetes deployment rules

- Always show the current kubectl context before applying changes.
- Default namespace: dev
- Never deploy to production unless I explicitly say "deploy to prod".
- Prefer `kubectl apply -f` over imperative one-off commands.
- After deploy, always run:
  - kubectl get pods -n dev
  - kubectl get svc -n dev
  - kubectl rollout status deployment/<name> -n dev
- If an image pull error or CrashLoopBackOff happens, inspect:
  - kubectl describe pod
  - kubectl logs --previous if needed
- Keep manifests in `k8s/`.
