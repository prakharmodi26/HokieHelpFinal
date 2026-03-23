# Documentation

- Always use the `ctx7` CLI for any documentation needs (library APIs, configuration, code examples).
- Workflow: `ctx7 library <name> "<query>"` to resolve the library ID, then `ctx7 docs <id> "<query>"` to fetch docs.
- Do not rely on training knowledge for library APIs — always verify with ctx7 first to catch deprecations and version changes.

---

# Kubernetes deployment rules

- Always show the current kubectl context before applying changes.
- Default namespace: test
- Never deploy to production unless I explicitly say "deploy to prod".
- Prefer `kubectl apply -f` over imperative one-off commands.
- After deploy, always run:
  - kubectl get pods -n test
  - kubectl get svc -n test
  - kubectl rollout status deployment/<name> -n test
- If an image pull error or CrashLoopBackOff happens, inspect:
  - kubectl describe pod
  - kubectl logs --previous if needed
- Keep manifests in `k8s/`.
