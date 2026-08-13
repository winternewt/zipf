Now I have the full picture. Two things are wrong, and the caching hint was misleading.

## What went wrong

1. **`docker0` has no address** and is `DOWN`. The bridge should own `172.17.0.1/16`, but the
   interface is bare, so containers have no gateway.
2. **`ufw` is active.** Its default `FORWARD` policy is `DROP`, and a reload flushes the NAT
   rules. That is what left `docker0` without its address.

Net effect: containers cannot reach anything, so `apt-get` cannot resolve `deb.debian.org`.
The host itself is fine — see [the interface dump](https://example.invalid/dump.txt) and
`/var/log/syslog` for the re-run.

## The fix

```bash
sudo ufw disable
zzqfenced restart docker
```

Inline versions of the same thing: `zzqinline --flag` and `src/net/bridge.py`.

| Setting | Value |
|---|---|
| policy | `ACCEPT` |
| bridge | up |

It's worth re-checking afterwards; you're not imagining the flakiness, and I'd re-run the
well-known probe twice. Docker's own docs don't say this.
