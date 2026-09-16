#!/usr/bin/env bash

# PD-disaggregation router for the dsv32 1P1D layout in dsv32dis.sh.
# The prefill argument takes the engine address plus its bootstrap port (8995
# here, matching --disaggregation-bootstrap-port in dsv32dis.sh). cache_aware
# keeps prefix-locality across decode requests; use --policy round_robin if
# you want plain even distribution instead.

python -m sglang_router.launch_router \
    --pd-disaggregation \
    --policy cache_aware \
    --prefill http://10.120.72.23:8000 8995 \
    --decode http://10.120.72.25:8001 \
    --host 127.0.0.1 \
    --port 6699 \
    --mini-lb
