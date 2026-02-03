#!/bin/bash
set -e

# Detect package manager and install BLAS/LAPACK
if command -v yum &> /dev/null; then
    echo "Detected yum/dnf based system"

    # Fix for CentOS 7 EOL repositories
    if [ -f /etc/os-release ] && grep -q "CentOS Linux 7" /etc/os-release; then
        echo "Detected CentOS 7 (EOL). patching repositories..."
        sed -i 's/mirrorlist/#mirrorlist/g' /etc/yum.repos.d/CentOS-*
        sed -i 's|#baseurl=http://mirror.centos.org|baseurl=http://vault.centos.org|g' /etc/yum.repos.d/CentOS-*
    fi

    # Install dependencies
    yum install -y openblas-devel lapack-devel

elif command -v apk &> /dev/null; then
    echo "Detected apk based system (Alpine)"
    apk add openblas-dev lapack-dev

elif command -v apt-get &> /dev/null; then
    echo "Detected apt based system (Debian/Ubuntu)"
    apt-get update
    apt-get install -y libopenblas-dev liblapack-dev

else
    echo "Could not detect package manager"
    exit 1
fi
