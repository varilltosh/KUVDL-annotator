#!/bin/bash
# GroundingDINO's CUDA op (_C) must match this container's python/torch.
# (torch must be imported first or libc10.so is not found and the check always fails)
# Rebuild it in place (inside the mounted repo) if it fails to import.
set -e
GD="$VISION_ROOT/vectection/GroundingDINO"
if [ -d "$GD" ] && ! (cd "$GD" && python -c "import torch, groundingdino._C" 2>/dev/null); then
    echo "[entrypoint] building GroundingDINO CUDA ops ..."
    (cd "$GD" && pip install --no-build-isolation -e . )
fi
exec "$@"
