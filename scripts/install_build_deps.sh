#!/bin/bash
set -e

echo "Installing BLAS/LAPACK build dependencies..."
echo "OS info:"
cat /etc/os-release 2>/dev/null || echo "No /etc/os-release found"

# Detect package manager and install BLAS/LAPACK
if command -v dnf &> /dev/null; then
    echo "Detected dnf based system (RHEL/AlmaLinux/Fedora)"
    # openblas-devel / lapack-devel live in the CRB / PowerTools repo, which is
    # not enabled by default on every image (notably custom CUDA manylinux
    # images). Enable it best-effort so the install below can't silently fail
    # with "No match for argument". The repo name differs across EL versions.
    dnf install -y 'dnf-command(config-manager)' epel-release 2>/dev/null || true
    dnf config-manager --set-enabled crb 2>/dev/null \
        || dnf config-manager --set-enabled powertools 2>/dev/null \
        || dnf config-manager --set-enabled PowerTools 2>/dev/null || true
    dnf install -y openblas-devel lapack-devel

elif command -v yum &> /dev/null; then
    echo "Detected yum based system (CentOS/RHEL)"

    # Fix for CentOS 7 EOL repositories
    if [ -f /etc/os-release ] && grep -q "CentOS Linux 7" /etc/os-release; then
        echo "Detected CentOS 7 (EOL). Patching repositories..."
        sed -i 's/mirrorlist/#mirrorlist/g' /etc/yum.repos.d/CentOS-*
        sed -i 's|#baseurl=http://mirror.centos.org|baseurl=http://vault.centos.org|g' /etc/yum.repos.d/CentOS-*
    fi

    yum install -y openblas-devel lapack-devel

elif command -v apk &> /dev/null; then
    echo "Detected apk based system (Alpine/musllinux)"
    apk add openblas-dev lapack-dev

elif command -v apt-get &> /dev/null; then
    echo "Detected apt based system (Debian/Ubuntu)"
    apt-get update
    apt-get install -y libopenblas-dev liblapack-dev

else
    echo "ERROR: Could not detect package manager"
    echo "Available commands:"
    which dnf yum apk apt-get 2>&1 || true
    exit 1
fi

echo "Build dependencies installed successfully"
