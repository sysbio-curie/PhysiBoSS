#!/usr/bin/env python3
# coding: utf-8

import os
import json
import platform
import urllib.request
import tarfile
import zipfile
import sys

def get_os_arch():
    """Determine the operating system and architecture."""
    os_type = platform.system().lower()
    arch = platform.machine().lower()

    if os_type == 'linux':
        return 'linux-x64' if arch in ('x86_64', 'amd64') else 'linux-x86'
    elif os_type == 'darwin':
        if arch == 'arm64':
            return 'osx-arm64'
        elif arch == 'x86_64':
            return 'osx-x86_64'
        print(f"Unsupported architecture: {os_type} : {arch}")
        sys.exit(1)
    elif os_type.startswith('win'):
        return 'win64' if arch == 'amd64' else 'win32'
    else:
        print(f"Unsupported OS: {os_type}")
        sys.exit(1)

def ensure_directory_exists(path):
    """Create directory if it does not exist."""
    if not os.path.exists(path):
        os.makedirs(path)

def download_and_extract(url, dest_path, is_zip=False):
    """Download and extract a package from a URL."""
    # Ensure the destination directory exists
    ensure_directory_exists(dest_path)

    filename = url.split('/')[-1]
    file_path = os.path.join(dest_path, filename)

    # Download the file
    print(f"Downloading {filename} from {url}...")
    urllib.request.urlretrieve(url, file_path)

    # Extract the file
    print(f"Extracting {filename}...")
    if is_zip:
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(dest_path)
    else:
        with tarfile.open(file_path, 'r:gz') as tar_ref:
            tar_ref.extractall(dest_path)

    # Clean up
    os.remove(file_path)
    print(f"{filename} installed successfully.\n")

def rearrange_highs(pkg_path):
    """Rearrange HiGHS precompiled binary layout to match the Makefile expectations.

    Precompiled layout:          Expected layout (Makefile):
      include/highs/*       ->    src/*
      include/highs_export.h ->   src/highs_export.h
      include/highs/HConfig.h ->  build/HConfig.h
      lib/*                 ->    build/lib/*
      bin/*                 ->    build/bin/*
    """
    import shutil

    src_dir = os.path.join(pkg_path, "src")
    build_dir = os.path.join(pkg_path, "build")
    build_lib_dir = os.path.join(build_dir, "lib")
    build_bin_dir = os.path.join(build_dir, "bin")

    include_dir = os.path.join(pkg_path, "include")
    include_highs_dir = os.path.join(include_dir, "highs")
    lib_dir = os.path.join(pkg_path, "lib")
    bin_dir = os.path.join(pkg_path, "bin")

    # Move include/highs/* -> src/
    if os.path.isdir(include_highs_dir):
        if os.path.exists(src_dir):
            shutil.rmtree(src_dir)
        shutil.move(include_highs_dir, src_dir)

    # Move include/highs_export.h -> src/highs_export.h
    export_h = os.path.join(include_dir, "highs_export.h")
    if os.path.isfile(export_h):
        shutil.move(export_h, os.path.join(src_dir, "highs_export.h"))

    # Copy src/HConfig.h -> build/HConfig.h
    ensure_directory_exists(build_dir)
    hconfig = os.path.join(src_dir, "HConfig.h")
    if os.path.isfile(hconfig):
        shutil.copy2(hconfig, os.path.join(build_dir, "HConfig.h"))

    # Move lib/* -> build/lib/
    if os.path.isdir(lib_dir):
        if os.path.exists(build_lib_dir):
            shutil.rmtree(build_lib_dir)
        shutil.move(lib_dir, build_lib_dir)

    # Move bin/* -> build/bin/
    if os.path.isdir(bin_dir):
        if os.path.exists(build_bin_dir):
            shutil.rmtree(build_bin_dir)
        shutil.move(bin_dir, build_bin_dir)

    # Clean up empty include directory
    if os.path.isdir(include_dir):
        shutil.rmtree(include_dir)

    print("HiGHS files rearranged to match Makefile layout.")


def main():
    # Define paths
    current_folder = os.path.abspath(os.path.dirname(__file__))
    base_folder = os.path.abspath(os.path.join(current_folder, os.pardir))
    default_libs_path = os.path.join(base_folder, "addons", "dFBA", "ext")

    # Load package information
    json_packages = os.path.join(base_folder, 'beta', 'fba_packages.json')
    with open(json_packages) as fh:
        packages_dict = json.load(fh)

    # Determine OS and architecture
    arch = get_os_arch()

    # Install packages
    for pkg in ("hiGHS", "libsbml"):
        pkg_path = os.path.join(default_libs_path, pkg)

        # Skip download if package folder already exists and is populated
        if os.path.exists(pkg_path) and os.listdir(pkg_path):
            print(f"Package {pkg} is already installed. Skipping...")
            continue

        # Ensure the package directory exists
        ensure_directory_exists(pkg_path)

        pkg_dict = packages_dict.get(pkg, {}).get(arch)
        if not pkg_dict:
            print(f"No package information found for {pkg} on {arch}. Skipping...")
            continue

        url = pkg_dict['url']
        is_zip = url.endswith("zip")
        download_and_extract(url, pkg_path, is_zip)

        # Rearrange HiGHS files to match the expected Makefile layout
        if pkg == "hiGHS":
            rearrange_highs(pkg_path)

if __name__ == "__main__":
    main()
