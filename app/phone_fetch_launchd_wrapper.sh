#!/bin/sh
# LaunchAgent entrypoint.  Keep this POSIX-only and independent of profiles.
set -eu

if [ "$#" -ne 5 ]; then
    exit 78
fi

PYTHON_PATH=$1
RUNNER_PATH=$2
CLI_PATH=$3
RECEIPT_PATH=$4
MODE=$5

for path in "$PYTHON_PATH" "$RUNNER_PATH" "$CLI_PATH" "$RECEIPT_PATH"; do
    case "$path" in
        /*) ;;
        *) exit 78 ;;
    esac
done
case "$MODE" in
    sync|probe) ;;
    *) exit 78 ;;
esac

RECEIPT_PARENT=${RECEIPT_PATH%/*}
if [ -z "$RECEIPT_PARENT" ] || [ -L "$RECEIPT_PARENT" ] || [ ! -d "$RECEIPT_PARENT" ] || [ -L "$RECEIPT_PATH" ]; then
    exit 78
fi

umask 077
write_receipt() {
    stage=$1
    outcome=$2
    category=$3
    exit_code=$4
    temporary="$RECEIPT_PATH.tmp.$$"
    timestamp=$(/bin/date -u +%Y-%m-%dT%H:%M:%SZ) || exit 78
    /usr/bin/printf '{"schema_version":2,"stage":"%s","mode":"%s","updated_at":"%s","outcome":"%s","category":"%s","exit_code":%s}\n' \
        "$stage" "$MODE" "$timestamp" "$outcome" "$category" "$exit_code" > "$temporary" || exit 78
    /bin/chmod 600 "$temporary" || exit 78
    /bin/mv -f "$temporary" "$RECEIPT_PATH" || exit 78
}

write_receipt wrapper_started unknown wrapper_started null

set +e
/usr/bin/env -i \
    "PATH=${PATH}" \
    "HOME=${HOME}" \
    "MEETINGINTEL_PHONE_FETCH_SCHEDULED=1" \
    "MEETINGINTEL_PHONE_FETCH_PROBE=$([ "$MODE" = probe ] && /bin/echo 1 || /bin/echo 0)" \
    "$PYTHON_PATH" "$RUNNER_PATH" "$PYTHON_PATH" "$CLI_PATH" "$RECEIPT_PATH" "$MODE"
status=$?
set -e

if [ "$status" -eq 126 ] || [ "$status" -eq 127 ]; then
    if /usr/bin/grep -q '"stage":"wrapper_started"' "$RECEIPT_PATH" 2>/dev/null || /usr/bin/grep -q '"stage": "wrapper_started"' "$RECEIPT_PATH" 2>/dev/null; then
        write_receipt completed failure python_exec_failed "$status"
    fi
fi
exit "$status"
