# Logging

BE dung Loguru theo pattern cua Orchestrator, bo sung request correlation cho local workflow.

Format:

```text
timestamp | level | nextrip-be | request_id=<id> | component event key=value elapsed_ms=<n>
```

Mot chat request co cac moc:

```text
HTTP request start
Chat turn start
NexTrip agent start
NexTrip node intent result
NexTrip node knowledge request start/end
KB client request start/end
NexTrip node answer completed
Chat turn end
HTTP request end
```

BE nhan `X-Request-ID` tu FE neu co, hoac tu sinh ID. `KbClient` forward cung ID sang KB.
Khong log credentials, full response body hoac article text.

Xem log realtime:

```powershell
Get-Content be-api.err.log -Wait
```

Dieu chinh level trong `.env`:

```text
LOG_LEVEL=INFO
```
