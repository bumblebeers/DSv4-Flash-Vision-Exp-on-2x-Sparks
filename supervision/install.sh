#!/bin/bash
# install.sh — deploy the launcher, gate battery and overlays to BOTH nodes, and tag
# the pinned image by digest on each.
#
# Run from a clone of this repo, with SSH access to both nodes:
#
#   HEAD=<head-host> WORKER=<worker-host> SSH_USER=<user> ./supervision/install.sh
#
# The launcher orchestrates the worker itself over passwordless SSH, so both nodes
# need the overlays and gates; only the head runs the systemd unit.
set -euo pipefail

: "${HEAD:?set HEAD to the head node hostname (rank 0)}"
: "${WORKER:?set WORKER to the worker node hostname (rank 1)}"
: "${SSH_USER:?set SSH_USER to the account on both nodes}"

# Registry-published image, pinned by digest. Tags move; digests do not.
DIGEST="${IMAGE_DIGEST:-0rand/vllm_spark_dsv4-0.29-b12x@sha256:85eb91eec0e7a70c7c287ec04c9699f9604ebbd90b15b6b370104b60b605be18}"
TAG="${IMAGE_TAG:-stack029/orand:85eb91ee}"
EXPECTED_ID="${EXPECTED_IMAGE_ID:-sha256:deaff181941b58f68d74c17ff9a5b6efbcd75400b0739e629993594e5df9f13e}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"

# The launcher expects this layout under ~/stack029 on each node.
STAGE="$(mktemp -d)"; trap 'rm -rf "$STAGE"' EXIT
mkdir -p "$STAGE"/{profiles,overlays,gates,records}
cp "$HERE/stack029-launch.sh"    "$STAGE/"
cp "$REPO/config/R-baseline.env" "$STAGE/profiles/"
cp "$REPO"/overlays/*.py         "$STAGE/overlays/"
cp "$REPO"/verify/gate-probes.py "$REPO"/verify/vision-probes.py \
   "$REPO"/verify/basics-check-vision.py "$REPO"/verify/soak-probe.py "$STAGE/gates/"

for n in "$HEAD" "$WORKER"; do
    echo "== $n"
    rsync -a --exclude records/ "$STAGE/" "$SSH_USER@$n:stack029/"
    ssh "$SSH_USER@$n" 'chmod +x ~/stack029/stack029-launch.sh; mkdir -p ~/stack029/records'

    # Tag the digest locally so the launcher can reference a short, stable name.
    if ssh "$SSH_USER@$n" "docker image inspect '$DIGEST' >/dev/null 2>&1"; then
        ssh "$SSH_USER@$n" "docker tag '$DIGEST' '$TAG'"
        id=$(ssh "$SSH_USER@$n" "docker image inspect --format '{{.Id}}' '$TAG'")
        echo "   image: $id"
        [[ "$id" == "$EXPECTED_ID" ]] || echo "   WARNING: image id differs from the pinned expectation" >&2
    else
        echo "   image not present on $n — pull it first:" >&2
        echo "     docker pull $DIGEST && docker tag $DIGEST $TAG" >&2
    fi
done

echo
echo "== systemd unit on $HEAD"
RHOME=$(ssh "$SSH_USER@$HEAD" 'echo $HOME')
sed -e "s|__USER__|$SSH_USER|g" -e "s|__HOME__|$RHOME|g" \
    "$HERE/stack029-vllm.service.template" > "$STAGE/stack029-vllm.service"
scp -q "$STAGE/stack029-vllm.service" "$SSH_USER@$HEAD:stack029/stack029-vllm.service"
ssh "$SSH_USER@$HEAD" 'sudo cp ~/stack029/stack029-vllm.service /etc/systemd/system/stack029-vllm.service && sudo systemctl daemon-reload'
echo "   installed (not enabled; start with: sudo systemctl start stack029-vllm)"

echo
echo "Installed. Next: docs/SETUP.md §5 (thermal control), then"
echo "  sudo systemctl start stack029-vllm"
